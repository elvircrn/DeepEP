# torchrun --nproc-per-node=4 tests/bench_fence.py
"""Benchmark low-latency dispatch + combine variants.

Measures dispatch-only and full dispatch+combine cycles for combine and
combine_upstream, all with fence.proxy.async enabled.
"""
import argparse
import os
import sys

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
    parser.add_argument('--allow-mnnvl', action='store_true')
    parser.add_argument('--pack-scale-writes', action='store_true')
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
        allow_mnnvl=args.allow_mnnvl, explicitly_destroy=True)

    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
    FLOAT4_E2M1_MAX = 6.0

    torch.manual_seed(42 + rank)
    x = torch.randn((T, H), dtype=torch.bfloat16, device='cuda')
    topk_weights = torch.randn((T, K), dtype=torch.float32, device='cuda').abs()

    x_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.max(torch.abs(x)).float()
    dist.all_reduce(x_global_scale, op=dist.ReduceOp.MIN, group=group)

    num_local_experts = E // num_ranks
    strategies = ['random', 'random-same', 'local-rand', 'local-same', 'remote-rand']
    formats = ['bf16', 'fp8', 'nvfp4']
    combine_variants = ['combine', 'upstream']
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
            offsets = torch.randint(0, E, (T, K), device='cuda', dtype=torch.int64)
            topk_idx = offsets % num_local_experts + local_start
        elif strategy == 'local-same':
            offsets = torch.arange(K, device='cuda', dtype=torch.int64).unsqueeze(0)
            topk_idx = (offsets % num_local_experts + local_start).expand(T, K).contiguous()
        else:
            remote_ranks = [r for r in range(num_ranks) if r != rank]
            per_token_idx = []
            for _ in range(T):
                picks = []
                for k in range(K):
                    r = remote_ranks[k % len(remote_ranks)]
                    e = torch.randint(0, num_local_experts, (1,)).item()
                    picks.append(r * num_local_experts + e)
                per_token_idx.append(picks)
            topk_idx = torch.tensor(per_token_idx, dtype=torch.int64, device='cuda')

        # Get expert output shape from a bf16 dispatch probe
        buffer.clean_low_latency_buffer(T, H, E)
        probe_recv, _, _, probe_ev, _ = buffer.low_latency_dispatch(
            x, topk_idx, T, E, use_fp8=False,
            async_finish=True, return_recv_hook=False)
        probe_ev.current_stream_wait()
        torch.cuda.synchronize()
        expert_output = torch.randn_like(probe_recv)

        for fmt in formats:
            use_fp8 = (fmt == 'fp8')
            use_nvfp4 = (fmt == 'nvfp4')

            dispatch_kwargs = dict(use_fp8=use_fp8, async_finish=True, return_recv_hook=False)
            if use_nvfp4:
                dispatch_kwargs['use_nvfp4'] = True
                dispatch_kwargs['x_global_scale'] = x_global_scale
                dispatch_kwargs['pack_scale_writes'] = args.pack_scale_writes
            elif fmt == 'bf16':
                dispatch_kwargs['use_fp8'] = False

            def make_clean_dispatch(idx=topk_idx, kw=dispatch_kwargs):
                def fn():
                    buffer.clean_low_latency_buffer(T, H, E)
                    _, _, _, event_d, _ = buffer.low_latency_dispatch(x, idx, T, E, **kw)
                    event_d.current_stream_wait()
                return fn

            def make_full(variant, idx=topk_idx, kw=dispatch_kwargs):
                use_upstream = (variant == 'upstream')
                def fn():
                    buffer.clean_low_latency_buffer(T, H, E)
                    _, _, handle, event_d, _ = buffer.low_latency_dispatch(x, idx, T, E, **kw)
                    event_d.current_stream_wait()
                    _, event_c, _ = buffer.low_latency_combine(
                        expert_output, idx, topk_weights, handle,
                        async_finish=True, return_recv_hook=False,
                        use_upstream=use_upstream)
                    event_c.current_stream_wait()
                return fn

            rounds = {'dispatch': []}
            for v in combine_variants:
                rounds[v] = []

            for _ in range(args.num_rounds):
                dist.barrier()
                avg_cd, _, _ = bench(make_clean_dispatch(), args.num_warmups, args.num_tests)
                for v in combine_variants:
                    dist.barrier()
                    avg_v, _, _ = bench(make_full(v), args.num_warmups, args.num_tests)
                    rounds[v].append(avg_v)
                dist.barrier()

                for key, val in [('dispatch', avg_cd)]:
                    t = torch.tensor([val], dtype=torch.float64, device='cuda')
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    rounds[key].append(t.item())
                for v in combine_variants:
                    t = torch.tensor([rounds[v][-1]], dtype=torch.float64, device='cuda')
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    rounds[v][-1] = t.item()

            results[(strategy, fmt)] = {k: np.median(v) for k, v in rounds.items()}

    if rank == 0:
        R = args.num_rounds
        print(f'\nlow-latency benchmark (median of {R} rounds, {args.num_tests} iters each)')
        print(f'  T={T}  H={H}  E={E}  topk={K}  ranks={num_ranks}')
        print(f'  worst-case rank (max across ranks per round)\n')
        print(f'  strategies:')
        print(f'    random      = topk over random scores, different experts per token (cross-rank RDMA)')
        print(f'    random-same = topk over random scores, same K experts for all tokens (cross-rank RDMA)')
        print(f'    local-rand  = random offsets mapped to local rank experts (no RDMA, varied experts)')
        print(f'    local-same  = every token picks the same K local experts (no RDMA, hotspot)')
        print(f'    remote-rand = random experts, each from a different remote rank (all cross-rank RDMA)\n')
        print(f'  {"strategy":<12} {"format":<8} {"dispatch":>10} {"combine":>10} {"upstream":>10}')
        print(f'  {"":12} {"":8} {"(us)":>10} {"(us)":>10} {"(us)":>10}')
        print(f'  {"-"*12} {"-"*8} {"-"*10} {"-"*10} {"-"*10}')
        for strategy in strategies:
            for fmt in formats:
                r = results[(strategy, fmt)]
                dispatch = r['dispatch'] * 1e6
                combine = (r['combine'] - r['dispatch']) * 1e6
                upstream = (r['upstream'] - r['dispatch']) * 1e6
                print(f'  {strategy:<12} {fmt:<8} {dispatch:>10.1f} {combine:>10.1f} {upstream:>10.1f}')
        print()

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
