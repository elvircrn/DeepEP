#!/usr/bin/env python3
"""Small DP4 x TP8 DeepEP v2 dispatch/combine repro.

This is intentionally not a MoE implementation.  It creates one fake decode
token per DP replica, routes only TP-rank 0 in each DP replica to 16 distinct
EP ranks, and fills the dispatched expert output with fake gate-weighted BF16
values before calling the real DeepEP v2 combine.  The dispatch, fake expert
write, and combine are captured into a CUDA graph and replayed for measurement.
Thus it exercises the real dispatch and combine kernels without loading model
weights or running an expert GEMM.

Expected launch on a four-node, eight-GPU-per-node Kermit allocation:

    torchrun --nnodes=4 --nproc-per-node=8 \
      --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
      tests/elastic/repro_dp4_tp8.py

Run one process per node and set NODE_RANK / MASTER_ADDR from the cluster
launcher.  The script requires WORLD_SIZE=32 and LOCAL_WORLD_SIZE=8.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist

import deep_ep


TP_SIZE = 8
DP_SIZE = 4
WORLD_SIZE = DP_SIZE * TP_SIZE
SP_SIZE = TP_SIZE

# Fixed repro parameters.  The only runtime inputs are the standard torchrun
# distributed environment variables.  This models one decode token per DP
# replica. vLLM pads that DP-local batch to 8 for SP, then each TP rank owns
# one row; only the first flattened row is real and the other seven are pad.
HIDDEN = 3584
NUM_EXPERTS = 896
TOPK = 16
# Mirrors vLLM GPUModelRunner._pad_for_sequence_parallelism and
# forward_context._compute_sp_num_tokens for one unpadded token per DP rank.
UNPADDED_TOKENS_PER_DP = 1
PADDED_TOKENS_PER_DP = (
    (UNPADDED_TOKENS_PER_DP + SP_SIZE - 1) // SP_SIZE
) * SP_SIZE
SP_LOCAL_TOKENS = (
    UNPADDED_TOKENS_PER_DP + SP_SIZE - 1
) // SP_SIZE
NUM_TOKENS_ACROSS_DP = [UNPADDED_TOKENS_PER_DP] * DP_SIZE
SP_LOCAL_SIZES = [
    (tokens + SP_SIZE - 1) // SP_SIZE
    for tokens in NUM_TOKENS_ACROSS_DP
    for _ in range(SP_SIZE)
]
NUM_MAX_TOKENS_PER_RANK = max(NUM_TOKENS_ACROSS_DP)
NUM_SMS = 132
NUM_QPS = 0
WARMUP = 5
ITERATIONS = 20
PROFILE_ALL_RANKS = os.environ.get("DEEPEP_PROFILE_ALL_RANKS", "0") == "1"
TRACE_DIR = Path("/tmp/deepep_dp4_tp8_trace")


def init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % TP_SIZE))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", TP_SIZE))

    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected WORLD_SIZE={WORLD_SIZE}, got {world_size}")
    if local_world_size != TP_SIZE:
        raise RuntimeError(
            f"expected LOCAL_WORLD_SIZE={TP_SIZE}, got {local_world_size}"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def make_fake_route(
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build the one-row TP shards produced by vLLM's SP padding."""
    tp_rank = rank % TP_SIZE
    dp_rank = rank // TP_SIZE
    local_experts = NUM_EXPERTS // WORLD_SIZE
    if NUM_EXPERTS % WORLD_SIZE != 0:
        raise ValueError("num_experts must be divisible by the 32-rank EP group")
    if TOPK > WORLD_SIZE:
        raise ValueError("this repro expects topk <= world size")
    if len(SP_LOCAL_SIZES) != WORLD_SIZE:
        raise ValueError("invalid DP x SP local-size layout")

    x = torch.zeros((SP_LOCAL_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_idx = torch.full(
        (SP_LOCAL_TOKENS, TOPK), -1, dtype=deep_ep.topk_idx_t, device=device
    )
    topk_weights = torch.zeros(
        (SP_LOCAL_TOKENS, TOPK), dtype=torch.float32, device=device
    )

    for local_token_idx in range(SP_LOCAL_TOKENS):
        # The flattened SP shard index within this DP replica. Rows after the
        # unpadded token count are vLLM padding rows and retain -1 routing IDs.
        token_idx_in_dp = tp_rank * SP_LOCAL_TOKENS + local_token_idx
        if token_idx_in_dp >= UNPADDED_TOKENS_PER_DP:
            continue

        # Route to distinct EP ranks 0..topk-1.  The expert index is the first
        # expert owned by each destination rank.  DP replicas intentionally
        # use the same route so each destination sees four real source tokens.
        destination_ranks = torch.arange(TOPK, device=device, dtype=torch.int64)
        topk_idx[local_token_idx] = (
            destination_ranks * local_experts
        ).to(deep_ep.topk_idx_t)
        topk_weights[local_token_idx].fill_(1.0 / TOPK)
        x[local_token_idx].fill_(float(dp_rank + 1))

    return x, topk_idx, topk_weights, tp_rank, dp_rank


def dispatch_and_fake_expert(
    buffer: deep_ep.ElasticBuffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, deep_ep.EPHandle]:
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=NUM_MAX_TOKENS_PER_RANK,
        num_sms=NUM_SMS,
        num_qps=NUM_QPS,
        # Match vLLM decode: fixed-size non-expanded layout and no CPU count
        # polling.  The all--1 padded rows remain present in the metadata.
        do_expand=False,
        do_cpu_sync=False,
        async_with_compute_stream=False,
    )
    if event.event is not None:
        event.current_stream_wait()

    if isinstance(recv_x, tuple):
        raise RuntimeError("this repro expects BF16 dispatch, not FP8 dispatch")

    return recv_x, recv_topk_idx, recv_topk_weights, handle


def run_once(
    buffer: deep_ep.ElasticBuffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[float, float, torch.Tensor]:
    comm_stream = buffer.get_comm_stream()
    dispatch_start = torch.cuda.Event(enable_timing=True)
    dispatch_end = torch.cuda.Event(enable_timing=True)
    combine_start = torch.cuda.Event(enable_timing=True)
    combine_end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(comm_stream):
        dispatch_start.record()
    recv_x, recv_topk_idx, recv_topk_weights, handle = dispatch_and_fake_expert(
        buffer, x, topk_idx, topk_weights
    )
    with torch.cuda.stream(comm_stream):
        dispatch_end.record()
        # DeepEP combine reduces expert outputs; it does not apply the gate
        # weights to x.  Model the expert/GEMM result by writing the sum of
        # valid local gate weights into each received row.  Each route has one
        # valid local expert here, so the 16 EP destinations sum back to one.
        if recv_topk_weights is None:
            raise RuntimeError("this repro requires dispatched top-k weights")
        valid_weight = recv_topk_weights.masked_fill(
            recv_topk_idx < 0, 0
        ).sum(dim=1, keepdim=True)
        recv_x.copy_(valid_weight.to(recv_x.dtype))
        combine_start.record()
    combined_x, _, _ = buffer.combine(
        x=recv_x,
        handle=handle,
        # vLLM's DeepEP v2 finalize applies router weights before combine.
        topk_weights=None,
        num_sms=NUM_SMS,
        num_qps=NUM_QPS,
        async_with_compute_stream=False,
    )
    with torch.cuda.stream(comm_stream):
        combine_end.record()
    combine_end.synchronize()
    return (
        dispatch_start.elapsed_time(dispatch_end),
        combine_start.elapsed_time(combine_end),
        combined_x,
    )


def capture_cuda_graphs(
    buffer: deep_ep.ElasticBuffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.CUDAGraph, torch.Tensor]:
    """Capture separate vLLM-style dispatch and combine graph segments."""
    dispatch_graph = torch.cuda.CUDAGraph()
    combine_graph = torch.cuda.CUDAGraph()

    # All ranks must enter capture with the same collective sequence.  The
    # warmup above has already compiled kernels and populated allocator state.
    torch.cuda.synchronize()
    dist.barrier()
    # Capture from the compute/default stream.  DeepEP's runtime launches its
    # communication work on a separate stream and inserts the required event
    # edges.  Capturing directly on the communication stream violates its
    # stream-control assertion because the two streams must differ.
    with torch.cuda.graph(dispatch_graph):
        recv_x, recv_topk_idx, recv_topk_weights, handle = \
            dispatch_and_fake_expert(buffer, x, topk_idx, topk_weights)

        if recv_topk_weights is None:
            raise RuntimeError("this repro requires dispatched top-k weights")
        valid_weight = recv_topk_weights.masked_fill(
            recv_topk_idx < 0, 0
        ).sum(dim=1, keepdim=True)
        recv_x.copy_(valid_weight.to(recv_x.dtype))

    # The combine graph consumes the stable dispatch outputs and handle.  This
    # mirrors vLLM's prepare/finalize split: dispatch is one graph segment and
    # the expert/finalize/combine path is a later segment.
    dispatch_graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.graph(combine_graph):
        combined_x, _, _ = buffer.combine(
            x=recv_x,
            handle=handle,
            topk_weights=None,
            num_sms=NUM_SMS,
            num_qps=NUM_QPS,
            async_with_compute_stream=False,
        )

    # Verify both graph segments can execute before using their output pointer.
    combine_graph.replay()
    torch.cuda.synchronize()
    return dispatch_graph, combine_graph, combined_x


def replay_cuda_graphs(
    buffer: deep_ep.ElasticBuffer,
    dispatch_graph: torch.cuda.CUDAGraph,
    combine_graph: torch.cuda.CUDAGraph,
) -> tuple[float, float]:
    comm_stream = buffer.get_comm_stream()
    default_stream = torch.cuda.current_stream()
    dispatch_start = torch.cuda.Event(enable_timing=True)
    dispatch_end = torch.cuda.Event(enable_timing=True)
    combine_start = torch.cuda.Event(enable_timing=True)
    combine_end = torch.cuda.Event(enable_timing=True)

    # DeepEP's kernels execute on its communication stream.  Timing the
    # default stream would measure only graph-launch overhead (~0.05 us), not
    # the dispatch/combine work.  Record on comm_stream and explicitly bridge
    # the fake expert write on the default stream before starting combine.
    with torch.cuda.stream(comm_stream):
        dispatch_start.record()
    dispatch_graph.replay()
    with torch.cuda.stream(comm_stream):
        dispatch_end.record()
    default_done = torch.cuda.Event(enable_timing=False)
    with torch.cuda.stream(default_stream):
        default_done.record()
    comm_stream.wait_event(default_done)
    with torch.cuda.stream(comm_stream):
        combine_start.record()
    combine_graph.replay()
    with torch.cuda.stream(comm_stream):
        combine_end.record()
    combine_end.synchronize()
    return (
        dispatch_start.elapsed_time(dispatch_end),
        combine_start.elapsed_time(combine_end),
    )


def main() -> None:
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if HIDDEN % 32 != 0:
        raise ValueError("hidden must be divisible by 32 for DeepEP BF16 vectors")

    x, topk_idx, topk_weights, tp_rank, dp_rank = make_fake_route(rank, device)

    buffer = deep_ep.ElasticBuffer(
        group=dist.group.WORLD,
        num_max_tokens_per_rank=NUM_MAX_TOKENS_PER_RANK,
        hidden=HIDDEN,
        num_topk=TOPK,
        allow_hybrid_mode=False,
        allow_multiple_reduction=False,
        sl_idx=3,
        explicitly_destroy=True,
    )

    if rank == 0:
        print(
            f"DeepEP={deep_ep.__file__} world={world_size} DP={DP_SIZE} TP={TP_SIZE} "
            f"SP={SP_SIZE} hidden={HIDDEN} experts={NUM_EXPERTS} topk={TOPK} "
            f"unpadded_dp_tokens={UNPADDED_TOKENS_PER_DP} "
            f"padded_dp_tokens={PADDED_TOKENS_PER_DP} "
            f"sp_local_tokens={SP_LOCAL_TOKENS} num_sms={NUM_SMS}",
            flush=True,
        )
    print(f"rank {rank}: dp={dp_rank} tp={tp_rank}", flush=True)

    # Warm up dispatch, JIT compilation, allocator state, and combine epilogue
    # before capturing.  These iterations are not included in measurements.
    for _ in range(WARMUP):
        run_once(buffer, x, topk_idx, topk_weights)
    dist.barrier()

    dispatch_graph, combine_graph, combined_x = capture_cuda_graphs(
        buffer, x, topk_idx, topk_weights
    )
    if rank == 0:
        print("cuda_graph=captured_and_replayed", flush=True)

    dispatch_ms: list[float] = []
    combine_ms: list[float] = []
    for _ in range(ITERATIONS):
        dist.barrier()
        d_ms, c_ms = replay_cuda_graphs(buffer, dispatch_graph, combine_graph)
        dispatch_ms.append(d_ms)
        combine_ms.append(c_ms)

    torch.cuda.synchronize()
    expected = 1 if tp_rank * SP_LOCAL_TOKENS < UNPADDED_TOKENS_PER_DP else 0
    max_error = (combined_x - expected).abs().max().to(torch.float32)
    dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"correctness max_abs_error={max_error.item():.6f}", flush=True)

    stats = torch.tensor(
        [sum(dispatch_ms) / len(dispatch_ms),
         sum(combine_ms) / len(combine_ms)],
        dtype=torch.float64,
        device=device,
    )
    all_stats = [torch.empty_like(stats) for _ in range(world_size)]
    dist.all_gather(all_stats, stats)
    if rank == 0:
        print("rank,dp,tp,dispatch_ms,combine_ms", flush=True)
        for r, value in enumerate(all_stats):
            print(
                f"{r},{r // TP_SIZE},{r % TP_SIZE},"
                f"{value[0].item():.4f},{value[1].item():.4f}",
                flush=True,
            )

    if PROFILE_ALL_RANKS:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
            replay_cuda_graphs(buffer, dispatch_graph, combine_graph)
        path = TRACE_DIR / f"deepep_dp4_tp8_rank{rank}.json"
        prof.export_chrome_trace(str(path))
        print(f"rank {rank}: wrote {path}", flush=True)
    else:
        replay_cuda_graphs(buffer, dispatch_graph, combine_graph)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
