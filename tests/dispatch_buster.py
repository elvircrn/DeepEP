# torchrun --nproc-per-node=4 tests/dispatch_buster.py
"""Dispatch outlier detector.

Repeatedly calls dispatch under various routing strategies and data formats,
records every per-iteration latency, and flags statistical outliers.

Reports percentile distributions, outlier counts (IQR method), and tail-to-
median ratios so you can spot jitter, scheduling bubbles, or RDMA stalls.

With --ablate-reuse-comm, runs everything twice (EP_REUSE_NCCL_COMM=0 then =1)
and prints a side-by-side comparison.
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


def timed_batches(fn, num_warmups: int, num_batches: int, batch_size: int):
    """Time batches of back-to-back calls — no L2 flush, no per-call event overhead.

    Returns amortized per-call times in seconds (batch_time / batch_size).
    """
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]

    for b in range(num_batches):
        start_events[b].record()
        for _ in range(batch_size):
            fn()
        end_events[b].record()

    torch.cuda.synchronize()
    return np.array([s.elapsed_time(e) / 1e3 / batch_size
                     for s, e in zip(start_events, end_events)])


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
    parser.add_argument('--ablate-reuse-comm', action='store_true',
                        help='ablate EP_REUSE_NCCL_COMM: run with =0 and =1, compare distributions')
    parser.add_argument('--ablate-expand', action='store_true',
                        help='print do_expand ablation summary: compare do_expand=False vs True')
    parser.add_argument('--sm-counts', type=int, nargs='+', default=None,
                        help='SM counts to sweep (default: auto-picked range)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='dispatches per timed batch (>1 = amortized timing, no L2 flush)')
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
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # --- Build inputs (independent of buffer / reuse mode) ---

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

    strategies = ['random', 'random-same', 'local-rand', 'local-same', 'remote-rand']
    formats = ['bf16', 'fp8', 'nvfp4']

    # --- Reuse-comm ablation ---

    ablating = args.ablate_reuse_comm
    reuse_modes = [0, 1] if ablating else [None]
    all_results = {}
    csv_rows = []
    sm_counts = None
    theoretical_sms = None

    for reuse_mode in reuse_modes:
        if reuse_mode is not None:
            os.environ['EP_REUSE_NCCL_COMM'] = str(reuse_mode)
            import deep_ep.utils.comm as comm_mod
            comm_mod._storage.clear()
            if rank == 0:
                print(f'\n=== EP_REUSE_NCCL_COMM={reuse_mode} ===', flush=True)

        # --- Allocate buffers ---

        buffer = None
        if not args.skip_legacy:
            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(T, H, num_ranks, E)
            buffer = deep_ep.Buffer(
                group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                num_qps_per_rank=E // num_ranks,
                allow_mnnvl=not args.no_mnnvl, explicitly_destroy=True)

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

        if sm_counts is None:
            theoretical_sms = elastic_bufs['bf16'].get_theoretical_num_sms(E, K)
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

        # --- Run dispatch under each config ---

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
                    if args.batch_size > 1:
                        times = timed_batches(kfn, args.num_warmups, args.num_iters, args.batch_size)
                    else:
                        times = timed_iters(kfn, args.num_warmups, args.num_iters)

                    # Reduce to worst-case rank per iteration
                    times_t = torch.tensor(times, dtype=torch.float64, device='cuda')
                    dist.all_reduce(times_t, op=dist.ReduceOp.MAX)
                    times_max = times_t.cpu().numpy()

                    stats = compute_stats(times_max)
                    all_results[(reuse_mode, strategy, fmt, kname)] = stats

                    if args.dump_csv:
                        reuse_label = '' if reuse_mode is None else str(reuse_mode)
                        for i, t in enumerate(times_max):
                            csv_rows.append((reuse_label, strategy, fmt, kname, i, t * 1e6))

                    dist.barrier()

        # --- Destroy buffers before next reuse mode ---

        if buffer is not None:
            buffer.destroy()
        for ebuf in elastic_bufs.values():
            ebuf.destroy()

    # --- Report ---

    if rank == 0:
        print(f'\n{"="*100}')
        print(f'  DISPATCH OUTLIER REPORT')
        print(f'  T={T}  H={H}  E={E}  topk={K}  ranks={num_ranks}')
        if args.batch_size > 1:
            print(f'  {args.num_iters} batches of {args.batch_size} dispatches, {args.num_warmups} warmups')
            print(f'  times are amortized per-dispatch (batch_time / {args.batch_size})')
        else:
            print(f'  {args.num_iters} iterations per config, {args.num_warmups} warmups')
        print(f'  outlier fence: Q1/Q3 +/- {args.outlier_threshold} * IQR')
        print(f'  all times worst-case across ranks')
        if ablating:
            print(f'  EP_REUSE_NCCL_COMM ablation: comparing reuse=0 vs reuse=1')
        print(f'{"="*100}\n')

        print(f'  Strategies:')
        print(f'    random      = topk over random scores, varied experts per token')
        print(f'    random-same = same K experts for every token')
        print(f'    local-rand  = local-rank experts only, varied per token')
        print(f'    local-same  = every token picks same K local experts')
        print(f'    remote-rand = all experts from remote ranks\n')

        print(f'  Kernels:  smN = elastic dispatch with N SMs,  +exp = do_expand=True')
        print(f'  theoretical_sms={theoretical_sms}  device_sms={device_sms}')
        print(f'  SM sweep: {sm_counts}\n')

        # Header — add reuse column when ablating
        if ablating:
            hdr = (f'  {"strategy":<12} {"fmt":<6} {"reuse":>5} {"kernel":<17} '
                   f'{"median":>8} {"p95":>8} {"p99":>8} {"max":>8} '
                   f'{"std":>8} {"tail":>6} {"out":>5} {"sev":>5}')
            units = (f'  {"":12} {"":6} {"":>5} {"":17} '
                     f'{"(us)":>8} {"(us)":>8} {"(us)":>8} {"(us)":>8} '
                     f'{"(us)":>8} {"p99/m":>6} {"":>5} {"":>5}')
            sep = (f'  {"-"*12} {"-"*6} {"-"*5} {"-"*17} '
                   f'{"-"*8} {"-"*8} {"-"*8} {"-"*8} '
                   f'{"-"*8} {"-"*6} {"-"*5} {"-"*5}')
        else:
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
                knames = (['legacy'] if not args.skip_legacy else []) + elastic_knames
                for kname in knames:
                    for reuse_mode in reuse_modes:
                        key = (reuse_mode, strategy, fmt, kname)
                        if key not in all_results:
                            continue
                        s = all_results[key]
                        flag = ''
                        if s['severe'] > 0:
                            flag = ' ***'
                            flagged.append(key)
                        elif s['outliers'] > 0:
                            flag = ' *'

                        if ablating:
                            print(f'  {strategy:<12} {fmt:<6} {reuse_mode:>5} {kname:<17} '
                                  f'{s["median"]:>8.1f} {s["p95"]:>8.1f} {s["p99"]:>8.1f} {s["max"]:>8.1f} '
                                  f'{s["std"]:>8.1f} {s["tail_ratio"]:>6.2f} '
                                  f'{s["outliers"]:>5d} {s["severe"]:>5d}{flag}')
                        else:
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
            for reuse_mode, strategy, fmt, kname in flagged:
                s = all_results[(reuse_mode, strategy, fmt, kname)]
                reuse_str = f' reuse={reuse_mode}' if ablating else ''
                print(f'    {strategy}/{fmt}/{kname}{reuse_str}: '
                      f'{s["severe"]} severe, max={s["max"]:.1f}us '
                      f'({s["max"]/s["median"]:.1f}x median)')
            print()

        # Ablation summary: side-by-side delta table
        if ablating:
            print(f'  {"="*90}')
            print(f'  EP_REUSE_NCCL_COMM ABLATION SUMMARY (R=0 vs R=1)')
            print(f'  {"="*90}\n')
            print(f'  {"strategy":<12} {"fmt":<6} {"kernel":<17} '
                  f'{"R0 med":>8} {"R1 med":>8} {"delta":>8} '
                  f'{"R0 p99":>8} {"R1 p99":>8} {"delta":>8} '
                  f'{"R0 sev":>6} {"R1 sev":>6}')
            print(f'  {"-"*12} {"-"*6} {"-"*17} '
                  f'{"-"*8} {"-"*8} {"-"*8} '
                  f'{"-"*8} {"-"*8} {"-"*8} '
                  f'{"-"*6} {"-"*6}')
            for strategy in strategies:
                for fmt in formats:
                    elastic_knames = []
                    for nsms in sm_counts:
                        for expand in [False, True]:
                            elastic_knames.append(f'sm{nsms}{"+exp" if expand else ""}')
                    knames = (['legacy'] if not args.skip_legacy else []) + elastic_knames
                    for kname in knames:
                        k0 = (0, strategy, fmt, kname)
                        k1 = (1, strategy, fmt, kname)
                        if k0 not in all_results or k1 not in all_results:
                            continue
                        s0, s1 = all_results[k0], all_results[k1]
                        d_med = (s1['median'] - s0['median']) / s0['median'] * 100 if s0['median'] > 0 else 0
                        d_p99 = (s1['p99'] - s0['p99']) / s0['p99'] * 100 if s0['p99'] > 0 else 0
                        print(f'  {strategy:<12} {fmt:<6} {kname:<17} '
                              f'{s0["median"]:>8.1f} {s1["median"]:>8.1f} {d_med:>+7.1f}% '
                              f'{s0["p99"]:>8.1f} {s1["p99"]:>8.1f} {d_p99:>+7.1f}% '
                              f'{s0["severe"]:>6d} {s1["severe"]:>6d}')
            print()

        if args.ablate_expand:
            print(f'  {"="*90}')
            print(f'  do_expand ABLATION SUMMARY (expand=False vs expand=True)')
            print(f'  {"="*90}\n')
            print(f'  {"strategy":<12} {"fmt":<6} {"sms":<6} '
                  f'{"noexp":>8} {"+exp":>8} {"delta":>8} '
                  f'{"noexp99":>8} {"+exp99":>8} {"delta":>8} '
                  f'{"noexp_s":>7} {"+exp_s":>7}')
            print(f'  {"-"*12} {"-"*6} {"-"*6} '
                  f'{"-"*8} {"-"*8} {"-"*8} '
                  f'{"-"*8} {"-"*8} {"-"*8} '
                  f'{"-"*7} {"-"*7}')
            for strategy in strategies:
                for fmt in formats:
                    for nsms in sm_counts:
                        for reuse_mode in reuse_modes:
                            k_no = (reuse_mode, strategy, fmt, f'sm{nsms}')
                            k_ex = (reuse_mode, strategy, fmt, f'sm{nsms}+exp')
                            if k_no not in all_results or k_ex not in all_results:
                                continue
                            s0, s1 = all_results[k_no], all_results[k_ex]
                            d_med = (s1['median'] - s0['median']) / s0['median'] * 100 if s0['median'] > 0 else 0
                            d_p99 = (s1['p99'] - s0['p99']) / s0['p99'] * 100 if s0['p99'] > 0 else 0
                            reuse_tag = f' R={reuse_mode}' if ablating else ''
                            print(f'  {strategy:<12} {fmt:<6} {nsms:<6} '
                                  f'{s0["median"]:>8.1f} {s1["median"]:>8.1f} {d_med:>+7.1f}% '
                                  f'{s0["p99"]:>8.1f} {s1["p99"]:>8.1f} {d_p99:>+7.1f}% '
                                  f'{s0["severe"]:>7d} {s1["severe"]:>7d}{reuse_tag}')
            print()

        if args.dump_csv:
            import csv
            with open(args.dump_csv, 'w', newline='') as f:
                w = csv.writer(f)
                header = ['strategy', 'format', 'kernel', 'iteration', 'time_us']
                if ablating:
                    header = ['reuse_comm'] + header
                w.writerow(header)
                if ablating:
                    w.writerows(csv_rows)
                else:
                    w.writerows([row[1:] for row in csv_rows])
            print(f'  Raw timings written to {args.dump_csv}\n')

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
