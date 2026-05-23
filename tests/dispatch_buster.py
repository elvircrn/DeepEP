# torchrun --nproc-per-node=4 tests/dispatch_buster.py
"""Dispatch outlier detector.

Repeatedly calls dispatch under various routing strategies and data formats,
records every per-iteration latency, and flags statistical outliers.

Reports percentile distributions, outlier counts (IQR method), and tail-to-
median ratios so you can spot jitter, scheduling bubbles, or RDMA stalls.
"""
import argparse
import os
import sys

os.environ.setdefault('NVSHMEM_QP_DEPTH', '4096')

import numpy as np
import torch
import torch.distributed as dist
import deep_ep


FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FLOAT4_E2M1_MAX = 6.0


def timed_iters(fn, num_warmups: int, num_iters: int, flush_l2: bool = True):
    """Like deep_ep.utils.testing.bench but returns every per-iteration time."""
    flush_buf = torch.empty(40 * 1024 * 1024, dtype=torch.int8, device='cuda') if flush_l2 else None

    for _ in range(num_warmups):
        fn()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        if flush_buf is not None:
            flush_buf.zero_()
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    # elapsed_time returns milliseconds; convert to seconds
    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])
    return times


def compute_stats(times_sec):
    """Return a dict of percentile stats and outlier info (all in microseconds)."""
    us = times_sec * 1e6
    q1, median, q3 = np.percentile(us, [25, 50, 75])
    iqr = q3 - q1
    fence_lo = q1 - 1.5 * iqr
    fence_hi = q3 + 1.5 * iqr
    outliers = np.sum((us < fence_lo) | (us > fence_hi))
    severe_hi = q3 + 3.0 * iqr
    severe = np.sum(us > severe_hi)
    p95 = np.percentile(us, 95)
    p99 = np.percentile(us, 99)
    return {
        'median': median,
        'mean': np.mean(us),
        'std': np.std(us),
        'p95': p95,
        'p99': p99,
        'max': np.max(us),
        'outliers': int(outliers),
        'severe': int(severe),
        'tail_ratio': p99 / median if median > 0 else float('inf'),
    }


def build_topk(strategy, T, E, K, num_local_experts, local_start, rank):
    torch.manual_seed(42 + rank)
    if strategy == 'random':
        scores = torch.randn((T, E), dtype=torch.float32, device='cuda').abs() + 1
        return torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1]
    elif strategy == 'random-same':
        scores = torch.randn((1, E), dtype=torch.float32, device='cuda').abs() + 1
        return torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1].expand(T, K).contiguous()
    elif strategy == 'local-rand':
        local_scores = torch.randn((T, num_local_experts), dtype=torch.float32, device='cuda').abs() + 1
        return torch.topk(local_scores, K, dim=-1, largest=True, sorted=True)[1] + local_start
    elif strategy == 'local-same':
        topk_idx = torch.arange(K, device='cuda', dtype=torch.int64).unsqueeze(0) + local_start
        return topk_idx.expand(T, K).contiguous()
    elif strategy == 'remote-rand':
        scores = torch.randn((T, E), dtype=torch.float32, device='cuda').abs() + 1
        scores[:, local_start:local_start + num_local_experts] = -1
        return torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1]
    else:
        raise ValueError(f'unknown strategy: {strategy}')


def main():
    parser = argparse.ArgumentParser(description='Dispatch outlier detector')
    parser.add_argument('--num-tokens', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-warmups', type=int, default=50)
    parser.add_argument('--num-iters', type=int, default=2000,
                        help='iterations per config (more = better outlier detection)')
    parser.add_argument('--no-mnnvl', action='store_true')
    parser.add_argument('--pack-scale-writes', action='store_true')
    parser.add_argument('--skip-legacy', action='store_true', help='skip legacy (NVSHMEM) benchmarks')
    parser.add_argument('--sm-counts', type=int, nargs='+', default=None,
                        help='SM counts to sweep (default: auto-picked range)')
    parser.add_argument('--outlier-threshold', type=float, default=1.5,
                        help='IQR multiplier for outlier fence (default: 1.5)')
    parser.add_argument('--dump-csv', type=str, default=None,
                        help='dump raw per-iteration timings to CSV')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.new_group(list(range(num_ranks)))

    T, H, E, K = args.num_tokens, args.hidden, args.num_experts, args.num_topk
    num_local_experts = E // num_ranks
    local_start = rank * num_local_experts

    # --- Allocate buffers ---

    buffer = None
    if not args.skip_legacy:
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(T, H, num_ranks, E)
        buffer = deep_ep.Buffer(
            group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
            num_qps_per_rank=E // num_ranks,
            allow_mnnvl=not args.no_mnnvl, explicitly_destroy=True)

    # One buffer per format (standalone mode for max QP/capacity headroom)
    elastic_bufs = {}
    elastic_bufs['bf16'] = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_fp8_dispatch=False, prefer_overlap_with_compute=False, explicitly_destroy=True)
    elastic_bufs['fp8'] = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_fp8_dispatch=True, prefer_overlap_with_compute=False, explicitly_destroy=True)
    elastic_bufs['nvfp4'] = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_nvfp4_dispatch=True, prefer_overlap_with_compute=False, explicitly_destroy=True)

    # SM count sweep
    theoretical_sms = elastic_bufs['bf16'].get_theoretical_num_sms(E, K)
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if args.sm_counts:
        sm_counts = sorted(args.sm_counts)
    else:
        sm_counts = sorted(set([
            theoretical_sms,
            max(1, device_sms // 8),
            max(1, device_sms // 4),
            device_sms // 2,
            device_sms * 3 // 4,
            device_sms,
        ]))

    # --- Build inputs ---

    torch.manual_seed(42 + rank)
    x = torch.randn((T, H), dtype=torch.bfloat16, device='cuda')
    topk_weights = torch.randn((T, K), dtype=torch.float32, device='cuda').abs()

    x_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.max(torch.abs(x)).float()
    dist.all_reduce(x_global_scale, op=dist.ReduceOp.MIN, group=group)

    x_abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    x_fp8_scale = (FLOAT8_E4M3_MAX / x_abs_max).float()
    x_fp8 = (x.float() * x_fp8_scale).to(torch.float8_e4m3fn)
    x_fp8_sf = (1.0 / x_fp8_scale).view(T, -1)

    from deep_ep.utils.math import _float_to_e2m1_packed, align
    _aligned_n = align(H, 16)
    _x_padded = torch.nn.functional.pad(x, (0, _aligned_n - H), mode='constant', value=0)
    _x_view = _x_padded.view(T, -1, 16)
    _x_amax = _x_view.abs().float().amax(dim=2).view(T, -1).clamp(1e-4)
    _x_scaled = (_x_view * (6.0 / _x_amax.unsqueeze(2))).float()
    x_nvfp4 = _float_to_e2m1_packed(_x_scaled.view(T, _aligned_n))[:, :H // 2].contiguous()
    x_nvfp4_sf = (_x_amax / 6.0).view(T, -1)

    # --- Run dispatch under each config ---

    strategies = ['random', 'random-same', 'local-rand', 'local-same', 'remote-rand']
    formats = ['bf16', 'fp8', 'nvfp4']
    all_results = {}
    csv_rows = []

    for strategy in strategies:
        topk_idx = build_topk(strategy, T, E, K, num_local_experts, local_start, rank)

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

            if use_nvfp4:
                elastic_inp = (x_nvfp4, x_nvfp4_sf)
            elif use_fp8:
                elastic_inp = (x_fp8, x_fp8_sf)
            else:
                elastic_inp = x
            elastic_ebuf = elastic_bufs[fmt]

            kernels = {}

            # Legacy low-latency dispatch
            if not args.skip_legacy:
                def make_legacy(idx=topk_idx, kw=dispatch_kwargs):
                    def fn():
                        buffer.clean_low_latency_buffer(T, H, E)
                        _, _, _, event_d, _ = buffer.low_latency_dispatch(x, idx, T, E, **kw)
                        event_d.current_stream_wait()
                    return fn
                kernels['legacy'] = make_legacy()

            # Elastic dispatch (ablate num_sms x do_expand)
            def make_elastic(idx=topk_idx, ebuf=elastic_ebuf,
                             inp=elastic_inp, nsms=theoretical_sms,
                             expand=False):
                def fn():
                    _, _, _, _, ev = ebuf.dispatch(
                        inp, topk_idx=idx, topk_weights=topk_weights,
                        num_experts=E, num_max_tokens_per_rank=T,
                        num_sms=nsms, async_with_compute_stream=True,
                        do_expand=expand)
                    ev.current_stream_wait()
                return fn

            for nsms in sm_counts:
                for expand in [False, True]:
                    suffix = '+exp' if expand else ''
                    kernels[f'sm{nsms}{suffix}'] = make_elastic(nsms=nsms, expand=expand)

            for kname, kfn in kernels.items():
                dist.barrier()
                times = timed_iters(kfn, args.num_warmups, args.num_iters)

                # Reduce to worst-case rank per iteration
                times_t = torch.tensor(times, dtype=torch.float64, device='cuda')
                dist.all_reduce(times_t, op=dist.ReduceOp.MAX)
                times_max = times_t.cpu().numpy()

                stats = compute_stats(times_max)
                all_results[(strategy, fmt, kname)] = stats

                if args.dump_csv:
                    for i, t in enumerate(times_max):
                        csv_rows.append((strategy, fmt, kname, i, t * 1e6))

                dist.barrier()

    # --- Report ---

    if rank == 0:
        print(f'\n{"="*90}')
        print(f'  DISPATCH OUTLIER REPORT')
        print(f'  T={T}  H={H}  E={E}  topk={K}  ranks={num_ranks}')
        print(f'  {args.num_iters} iterations per config, {args.num_warmups} warmups')
        print(f'  outlier fence: Q1/Q3 +/- {args.outlier_threshold} * IQR')
        print(f'  all times worst-case across ranks')
        print(f'{"="*90}\n')

        print(f'  Strategies:')
        print(f'    random      = topk over random scores, varied experts per token')
        print(f'    random-same = same K experts for every token')
        print(f'    local-rand  = local-rank experts only, varied per token')
        print(f'    local-same  = every token picks same K local experts')
        print(f'    remote-rand = all experts from remote ranks\n')

        print(f'  Kernels:  smN = elastic dispatch with N SMs,  +exp = do_expand=True')
        print(f'  theoretical_sms={theoretical_sms}  device_sms={device_sms}')
        print(f'  SM sweep: {sm_counts}\n')

        # Header
        hdr = (f'  {"strategy":<12} {"fmt":<6} {"kernel":<17} '
               f'{"median":>8} {"p95":>8} {"p99":>8} {"max":>8} '
               f'{"std":>8} {"tail":>6} {"out":>5} {"sev":>5}')
        units = (f'  {"":12} {"":6} {"":17} '
                 f'{"(us)":>8} {"(us)":>8} {"(us)":>8} {"(us)":>8} '
                 f'{"(us)":>8} {"p99/m":>6} {"":>5} {"":>5}')
        sep = (f'  {"-"*12} {"-"*6} {"-"*17} '
               f'{"-"*8} {"-"*8} {"-"*8} {"-"*8} '
               f'{"-"*8} {"-"*6} {"-"*5} {"-"*5}')
        print(hdr)
        print(units)
        print(sep)

        flagged = []
        for strategy in strategies:
            for fmt in formats:
                elastic_knames = []
                for nsms in sm_counts:
                    for expand in [False, True]:
                        elastic_knames.append(f'sm{nsms}{"+exp" if expand else ""}')
                for kname in (['legacy'] if not args.skip_legacy else []) + elastic_knames:
                    key = (strategy, fmt, kname)
                    if key not in all_results:
                        continue
                    s = all_results[key]
                    flag = ''
                    if s['severe'] > 0:
                        flag = ' ***'
                        flagged.append(key)
                    elif s['outliers'] > 0:
                        flag = ' *'

                    print(f'  {strategy:<12} {fmt:<6} {kname:<17} '
                          f'{s["median"]:>8.1f} {s["p95"]:>8.1f} {s["p99"]:>8.1f} {s["max"]:>8.1f} '
                          f'{s["std"]:>8.1f} {s["tail_ratio"]:>6.2f} '
                          f'{s["outliers"]:>5d} {s["severe"]:>5d}{flag}')
            print(sep)

        print()
        print(f'  Legend: out = mild outliers (>{args.outlier_threshold}*IQR from Q1/Q3)')
        print(f'          sev = severe outliers (>3*IQR from Q3)')
        print(f'          tail = p99/median ratio (>1.5 suggests jitter)')
        print(f'          *  = has mild outliers    *** = has severe outliers')
        print()

        if flagged:
            print(f'  FLAGGED CONFIGS ({len(flagged)} with severe outliers):')
            for strategy, fmt, kname in flagged:
                s = all_results[(strategy, fmt, kname)]
                print(f'    {strategy}/{fmt}/{kname}: '
                      f'{s["severe"]} severe, max={s["max"]:.1f}us '
                      f'({s["max"]/s["median"]:.1f}x median)')
            print()

        if args.dump_csv:
            import csv
            with open(args.dump_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['strategy', 'format', 'kernel', 'iteration', 'time_us'])
                w.writerows(csv_rows)
            print(f'  Raw timings written to {args.dump_csv}\n')

    # --- Cleanup ---

    if buffer is not None:
        buffer.destroy()
    for ebuf in elastic_bufs.values():
        ebuf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
