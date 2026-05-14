# torchrun --nproc-per-node=4 tests/bench_fence_overhead.py
"""Measure the overhead of fence.proxy.async in combine_upstream (legacy)."""
import argparse
import os

os.environ.setdefault('NVSHMEM_QP_DEPTH', '4096')

import numpy as np
import torch
import torch.distributed as dist
import deep_ep

from deep_ep.utils.testing import bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tokens', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-warmups', type=int, default=50)
    parser.add_argument('--num-tests', type=int, default=1000)
    parser.add_argument('--num-rounds', type=int, default=3)
    parser.add_argument('--no-mnnvl', action='store_true')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.new_group(list(range(num_ranks)))

    T, H, E, K = args.num_tokens, args.hidden, args.num_experts, args.num_topk

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(T, H, num_ranks, E)
    buffer = deep_ep.Buffer(
        group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
        num_qps_per_rank=E // num_ranks,
        allow_mnnvl=not args.no_mnnvl, explicitly_destroy=True)

    torch.manual_seed(42 + rank)
    x = torch.randn((T, H), dtype=torch.bfloat16, device='cuda')
    topk_weights = torch.randn((T, K), dtype=torch.float32, device='cuda').abs()

    num_local_experts = E // num_ranks
    strategies = ['random', 'random-same', 'local-rand', 'local-same', 'remote-rand']
    formats = ['bf16', 'fp8']
    results = {}

    for strategy in strategies:
        torch.manual_seed(42 + rank)
        local_start = rank * num_local_experts
        if strategy == 'random':
            scores = torch.randn((T, E), dtype=torch.float32, device='cuda').abs() + 1
            topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1]
        elif strategy == 'random-same':
            scores = torch.randn((1, E), dtype=torch.float32, device='cuda').abs() + 1
            topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1].expand(T, K).contiguous()
        elif strategy == 'local-rand':
            local_scores = torch.randn((T, num_local_experts), dtype=torch.float32, device='cuda').abs() + 1
            topk_idx = torch.topk(local_scores, K, dim=-1, largest=True, sorted=True)[1] + local_start
        elif strategy == 'local-same':
            topk_idx = torch.arange(K, device='cuda', dtype=torch.int64).unsqueeze(0) + local_start
            topk_idx = topk_idx.expand(T, K).contiguous()
        else:
            scores = torch.randn((T, E), dtype=torch.float32, device='cuda').abs() + 1
            scores[:, local_start:local_start + num_local_experts] = -1
            topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1]

        buffer.clean_low_latency_buffer(T, H, E)
        probe_recv, _, _, probe_ev, _ = buffer.low_latency_dispatch(
            x, topk_idx, T, E, use_fp8=False,
            async_finish=True, return_recv_hook=False)
        probe_ev.current_stream_wait()
        torch.cuda.synchronize()
        expert_output = torch.randn_like(probe_recv)

        for fmt in formats:
            use_fp8 = (fmt == 'fp8')
            dispatch_kwargs = dict(use_fp8=use_fp8, async_finish=True, return_recv_hook=False)

            def make_full(fence_on, idx=topk_idx, kw=dispatch_kwargs):
                def fn():
                    buffer.clean_low_latency_buffer(T, H, E)
                    _, _, handle, event_d, _ = buffer.low_latency_dispatch(x, idx, T, E, **kw)
                    event_d.current_stream_wait()
                    _, event_c, _ = buffer.low_latency_combine(
                        expert_output, idx, topk_weights, handle,
                        async_finish=True, return_recv_hook=False,
                        use_upstream=True,
                        use_fence_proxy_async=fence_on)
                    event_c.current_stream_wait()
                return fn

            def make_dispatch(idx=topk_idx, kw=dispatch_kwargs):
                def fn():
                    buffer.clean_low_latency_buffer(T, H, E)
                    _, _, _, event_d, _ = buffer.low_latency_dispatch(x, idx, T, E, **kw)
                    event_d.current_stream_wait()
                return fn

            rounds = {'dispatch': [], 'fence_on': [], 'fence_off': []}

            for _ in range(args.num_rounds):
                dist.barrier()
                avg_d, _, _ = bench(make_dispatch(), args.num_warmups, args.num_tests)
                dist.barrier()
                avg_on, _, _ = bench(make_full(True), args.num_warmups, args.num_tests)
                dist.barrier()
                avg_off, _, _ = bench(make_full(False), args.num_warmups, args.num_tests)
                dist.barrier()

                for key, val in [('dispatch', avg_d), ('fence_on', avg_on), ('fence_off', avg_off)]:
                    t = torch.tensor([val], dtype=torch.float64, device='cuda')
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    rounds[key].append(t.item())

            results[(strategy, fmt)] = {k: np.median(v) for k, v in rounds.items()}

    if rank == 0:
        R = args.num_rounds
        print(f'\nlow-latency combine_upstream fence.proxy.async overhead')
        print(f'  median of {R} rounds, {args.num_tests} iters each')
        print(f'  T={T}  H={H}  E={E}  topk={K}  ranks={num_ranks}')
        print(f'  worst-case rank (max across ranks per round)\n')
        print(f'  strategies:')
        print(f'    random      = topk over random scores, different experts per token (cross-rank RDMA)')
        print(f'    random-same = topk over random scores, same K experts for all tokens (cross-rank RDMA)')
        print(f'    local-rand  = random offsets mapped to local rank experts (no RDMA, varied experts)')
        print(f'    local-same  = every token picks the same K local experts (no RDMA, hotspot)')
        print(f'    remote-rand = random experts, each from a different remote rank (all cross-rank RDMA)\n')
        print(f'  {"strategy":<12} {"fmt":<6} {"fence_on":>9} {"fence_off":>9} {"delta":>9} {"pct":>7}')
        print(f'  {"":12} {"":6} {"(us)":>9} {"(us)":>9} {"(us)":>9}')
        print(f'  {"-"*12} {"-"*6} {"-"*9} {"-"*9} {"-"*9} {"-"*7}')
        for strategy in strategies:
            for fmt in formats:
                r = results[(strategy, fmt)]
                disp = r['dispatch']
                on = (r['fence_on'] - disp) * 1e6
                off = (r['fence_off'] - disp) * 1e6
                delta = on - off
                pct = delta / off * 100 if off else 0
                print(f'  {strategy:<12} {fmt:<6} {on:>9.1f} {off:>9.1f} {delta:>+9.1f} {pct:>+6.1f}%')
        print()

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
