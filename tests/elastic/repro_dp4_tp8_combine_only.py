#!/usr/bin/env python3
"""DP4 x TP8 DeepEP combine-only repro.

One real dispatch establishes the production EPHandle and receive layout.  The
measured loop then repeatedly calls only DeepEP v2 combine with the same
received metadata and fake expert output.  This removes dispatch, CUDA graph,
PyTorch profiler, and routing variability from the combine epilogue.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

import deep_ep


TP_SIZE = 8
DP_SIZE = 4
WORLD_SIZE = 32
HIDDEN = 3584
NUM_EXPERTS = 896
TOPK = 16
NUM_MAX_TOKENS_PER_RANK = 1
NUM_SMS = 132
NUM_QPS = 0
WARMUP_COMBINES = 10
MEASURED_COMBINES = 20


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % TP_SIZE))
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected WORLD_SIZE={WORLD_SIZE}, got {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")

    tp_rank = rank % TP_SIZE
    dp_rank = rank // TP_SIZE
    local_experts = NUM_EXPERTS // WORLD_SIZE

    # Exact vLLM TP8/SP one-token layout: one physical row per TP rank, with
    # only TP rank 0 real and TP ranks 1..7 carrying -1 padding routes.
    x = torch.zeros((1, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_idx = torch.full((1, TOPK), -1, dtype=deep_ep.topk_idx_t, device=device)
    topk_weights = torch.zeros((1, TOPK), dtype=torch.float32, device=device)
    if tp_rank == 0:
        topk_idx[0] = (
            torch.arange(TOPK, device=device, dtype=torch.int64) * local_experts
        ).to(deep_ep.topk_idx_t)
        topk_weights[0].fill_(1.0 / TOPK)
        x.fill_(float(dp_rank + 1))

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

    # One production dispatch only: this supplies the real handle, receive
    # metadata, communication slots, and token layout consumed by combine.
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=NUM_MAX_TOKENS_PER_RANK,
        num_sms=NUM_SMS,
        num_qps=NUM_QPS,
        do_expand=False,
        do_cpu_sync=False,
        async_with_compute_stream=False,
    )
    if event.event is not None:
        event.current_stream_wait()
    if isinstance(recv_x, tuple) or recv_topk_weights is None:
        raise RuntimeError("expected BF16 dispatch with received top-k weights")

    # Fake expert/GEMM output exactly as in the full repro: each valid received
    # row contains its valid local gate-weight sum; padded rows are zero.
    valid_weight = recv_topk_weights.masked_fill(
        recv_topk_idx < 0, 0
    ).sum(dim=1, keepdim=True)
    recv_x.copy_(valid_weight.to(recv_x.dtype))
    valid_recv_rows = int((recv_topk_idx >= 0).any(dim=1).sum().item())
    valid_recv_routes = int((recv_topk_idx >= 0).sum().item())
    torch.cuda.synchronize()
    dist.barrier()

    combined_x = None
    for _ in range(WARMUP_COMBINES):
        combined_x, _, event = buffer.combine(
            x=recv_x,
            handle=handle,
            topk_weights=None,
            num_sms=NUM_SMS,
            num_qps=NUM_QPS,
            async_with_compute_stream=False,
        )
        if event.event is not None:
            event.current_stream_wait()
        torch.cuda.synchronize()
    dist.barrier()

    comm_stream = buffer.get_comm_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(MEASURED_COMBINES):
        with torch.cuda.stream(comm_stream):
            start.record()
        combined_x, _, event = buffer.combine(
            x=recv_x,
            handle=handle,
            topk_weights=None,
            num_sms=NUM_SMS,
            num_qps=NUM_QPS,
            async_with_compute_stream=False,
        )
        if event.event is not None:
            event.current_stream_wait()
        with torch.cuda.stream(comm_stream):
            end.record()
        end.synchronize()

    expected = 1 if tp_rank == 0 else 0
    error = (combined_x - expected).abs().max().to(torch.float32)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    elapsed_us = start.elapsed_time(end) * 1000.0
    print(
        f"rank={rank} dp={dp_rank} tp={tp_rank} "
        f"received_rows={recv_x.shape[0]} valid_recv_rows={valid_recv_rows} "
        f"valid_recv_routes={valid_recv_routes} combined_rows={combined_x.shape[0]} "
        f"last_combine_us={elapsed_us:.2f} max_error={error.item():.6f}",
        flush=True,
    )

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
