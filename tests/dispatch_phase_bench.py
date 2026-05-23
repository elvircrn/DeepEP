# torchrun --nproc-per-node=4 tests/dispatch_phase_bench.py
"""Focused dispatch benchmark with per-phase clock64() instrumentation.

Runs a single config (random/fp8 by default) and reports which kernel phase
caused each outlier iteration.

Phases:
  0: kernel entry (before entry barrier)
  1: after entry barrier
  2: after SM 0 reduction wait
  3: after send rank/expert counts
  4: after receive peer counts
  5: after prefix sums
  6: after dispatch warp token loop
  7: after exit barrier
"""
import argparse
import os

os.environ.setdefault('NVSHMEM_QP_DEPTH', '4096')

import numpy as np
import torch
import torch.distributed as dist
import deep_ep


FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
NUM_PHASES = 8
PHASE_NAMES = [
    'entry',
    'entry_barrier',
    'sm0_reduce',
    'send_counts',
    'recv_counts',
    'prefix_sums',
    'token_loop',
    'exit_barrier',
]


def main():
    parser = argparse.ArgumentParser(description='Phase-instrumented dispatch benchmark')
    parser.add_argument('--num-tokens', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-warmups', type=int, default=50)
    parser.add_argument('--num-iters', type=int, default=2000)
    parser.add_argument('--num-sms', type=int, default=0,
                        help='SM count (0 = theoretical default)')
    parser.add_argument('--strategy', type=str, default='random',
                        choices=['random', 'random-same', 'local-rand', 'local-same', 'remote-rand'])
    parser.add_argument('--fmt', type=str, default='fp8', choices=['bf16', 'fp8', 'nvfp4'])
    parser.add_argument('--do-expand', action='store_true')
    parser.add_argument('--outlier-threshold', type=float, default=1.5)
    parser.add_argument('--dump-csv', type=str, default=None)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.new_group(list(range(num_ranks)))

    T, H, E, K = args.num_tokens, args.hidden, args.num_experts, args.num_topk
    num_local_experts = E // num_ranks
    local_start = rank * num_local_experts

    use_fp8 = (args.fmt == 'fp8')
    use_nvfp4 = (args.fmt == 'nvfp4')

    ebuf = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_fp8_dispatch=use_fp8,
        use_nvfp4_dispatch=use_nvfp4,
        prefer_overlap_with_compute=False, explicitly_destroy=True)

    num_sms = args.num_sms if args.num_sms > 0 else ebuf.get_theoretical_num_sms(E, K)

    # Legacy buffer for control comparison
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(T, H, num_ranks, E)
    legacy_buf = deep_ep.Buffer(
        group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
        num_qps_per_rank=E // num_ranks, explicitly_destroy=True)

    # --- Build inputs ---

    torch.manual_seed(42 + rank)
    x = torch.randn((T, H), dtype=torch.bfloat16, device='cuda')
    topk_weights = torch.randn((T, K), dtype=torch.float32, device='cuda').abs()

    # Routing
    if args.strategy == 'random':
        scores = torch.randn((T, E), dtype=torch.float32, device='cuda').abs() + 1
        topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1]
    elif args.strategy == 'random-same':
        scores = torch.randn((1, E), dtype=torch.float32, device='cuda').abs() + 1
        topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1].expand(T, K).contiguous()
    elif args.strategy == 'local-rand':
        local_scores = torch.randn((T, num_local_experts), dtype=torch.float32, device='cuda').abs() + 1
        topk_idx = torch.topk(local_scores, K, dim=-1, largest=True, sorted=True)[1] + local_start
    elif args.strategy == 'local-same':
        topk_idx = (torch.arange(K, device='cuda', dtype=torch.int64).unsqueeze(0) + local_start).expand(T, K).contiguous()
    elif args.strategy == 'remote-rand':
        scores = torch.randn((T, E), dtype=torch.float32, device='cuda').abs() + 1
        scores[:, local_start:local_start + num_local_experts] = -1
        topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1]

    # Format-specific input
    if use_nvfp4:
        x_global_scale = (FLOAT8_E4M3_MAX * 6.0) / torch.max(torch.abs(x)).float()
        dist.all_reduce(x_global_scale, op=dist.ReduceOp.MIN, group=group)
        from deep_ep.utils.math import _float_to_e2m1_packed, align
        _aligned_n = align(H, 16)
        _x_padded = torch.nn.functional.pad(x, (0, _aligned_n - H), mode='constant', value=0)
        _x_view = _x_padded.view(T, -1, 16)
        _x_amax = _x_view.abs().float().amax(dim=2).view(T, -1).clamp(1e-4)
        _x_scaled = (_x_view * (6.0 / _x_amax.unsqueeze(2))).float()
        x_nvfp4 = _float_to_e2m1_packed(_x_scaled.view(T, _aligned_n))[:, :H // 2].contiguous()
        x_nvfp4_sf = (_x_amax / 6.0).view(T, -1)
        inp = (x_nvfp4, x_nvfp4_sf)
    elif use_fp8:
        x_abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        x_fp8_scale = (FLOAT8_E4M3_MAX / x_abs_max).float()
        x_fp8 = (x.float() * x_fp8_scale).to(torch.float8_e4m3fn)
        x_fp8_sf = (1.0 / x_fp8_scale).view(T, -1)
        inp = (x_fp8, x_fp8_sf)
    else:
        inp = x

    # --- Run with per-iteration phase timestamps ---

    N = args.num_iters
    W = args.num_warmups
    flush_buf = torch.empty(40 * 1024 * 1024, dtype=torch.int8, device='cuda')

    dispatch_kwargs = dict(use_fp8=use_fp8, async_finish=True, return_recv_hook=False)
    if use_nvfp4:
        dispatch_kwargs['use_nvfp4'] = True
        dispatch_kwargs['x_global_scale'] = x_global_scale
    elif not use_fp8:
        dispatch_kwargs['use_fp8'] = False

    # Warmup (no timestamps)
    for _ in range(W):
        ebuf.dispatch(
            inp, topk_idx=topk_idx, topk_weights=topk_weights,
            num_experts=E, num_max_tokens_per_rank=T,
            num_sms=num_sms, async_with_compute_stream=True,
            do_expand=args.do_expand)
    torch.cuda.synchronize()

    # Timed iterations with phase timestamps
    phase_ts = torch.zeros((N, NUM_PHASES), dtype=torch.int64, device='cuda')
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(N)]

    for i in range(N):
        torch.cuda.synchronize()
        dist.barrier()
        flush_buf.zero_()
        torch.cuda.synchronize()
        start_events[i].record()
        _, _, _, _, ev = ebuf.dispatch(
            inp, topk_idx=topk_idx, topk_weights=topk_weights,
            num_experts=E, num_max_tokens_per_rank=T,
            num_sms=num_sms, async_with_compute_stream=True,
            do_expand=args.do_expand,
            phase_timestamps=phase_ts[i])
        ev.current_stream_wait()
        end_events[i].record()

    torch.cuda.synchronize()

    # Collect wall-clock times (seconds)
    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])

    # Reduce to worst-case rank per iteration
    times_t = torch.tensor(times, dtype=torch.float64, device='cuda')
    dist.all_reduce(times_t, op=dist.ReduceOp.MAX)
    times_max = times_t.cpu().numpy()

    # Compute per-phase deltas per rank first (clock64 is per-GPU, not cross-GPU comparable)
    ts = phase_ts.cpu().numpy()  # (N, 8)
    deltas = np.zeros((N, NUM_PHASES - 1), dtype=np.int64)
    for p in range(5):
        deltas[:, p] = ts[:, p + 1] - ts[:, p]
    deltas[:, 5] = ts[:, 7] - ts[:, 5]
    deltas[:, 6] = ts[:, 6] - ts[:, 1]

    # Now reduce deltas (max across ranks) — these are durations, not absolute timestamps
    deltas_t = torch.tensor(deltas, dtype=torch.int64, device='cuda')
    dist.all_reduce(deltas_t, op=dist.ReduceOp.MAX)
    deltas = deltas_t.cpu().numpy()

    delta_names = [
        '0->1 entry_barrier',
        '1->2 sm0_reduce',
        '2->3 send_counts',
        '3->4 recv_counts',
        '4->5 prefix_sums',
        '5->7 exit+dispatch',
        '1->6 token_loop',
    ]

    # --- Outlier detection ---

    us = times_max * 1e6
    q1, median, q3 = np.percentile(us, [25, 50, 75])
    iqr = q3 - q1
    fence_hi = q3 + args.outlier_threshold * iqr
    severe_hi = q3 + 3.0 * iqr
    outlier_mask = us > fence_hi
    severe_mask = us > severe_hi
    num_outliers = int(np.sum(outlier_mask))
    num_severe = int(np.sum(severe_mask))

    if rank == 0:
        print(f'\n{"="*80}')
        print(f'  PHASE-INSTRUMENTED DISPATCH BENCHMARK')
        print(f'  {args.strategy}/{args.fmt}/sm{num_sms}'
              f'{"(+exp)" if args.do_expand else ""}  T={T} H={H} E={E} K={K} ranks={num_ranks}')
        print(f'  {N} iterations, {W} warmups')
        print(f'{"="*80}\n')

        # Overall stats
        p95 = np.percentile(us, 95)
        p99 = np.percentile(us, 99)
        print(f'  Wall-clock (us):  median={median:.1f}  p95={p95:.1f}  p99={p99:.1f}'
              f'  max={np.max(us):.1f}  std={np.std(us):.1f}')
        print(f'  Outliers: {num_outliers} mild (>{fence_hi:.1f}us)  {num_severe} severe (>{severe_hi:.1f}us)\n')

        # Per-phase delta stats (cycles)
        print(f'  {"phase delta":<22} {"median":>10} {"p95":>10} {"p99":>10} {"max":>10} {"std":>10}  (GPU cycles)')
        print(f'  {"-"*22} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
        for j, name in enumerate(delta_names):
            d = deltas[:, j].astype(np.float64)
            dm = np.median(d)
            print(f'  {name:<22} {dm:>10.0f} {np.percentile(d, 95):>10.0f}'
                  f' {np.percentile(d, 99):>10.0f} {np.max(d):>10.0f} {np.std(d):>10.0f}')
        print()

        # For each outlier iteration, show which phase spiked
        if num_outliers > 0:
            outlier_indices = np.where(outlier_mask)[0]
            # Compute normal median per phase
            normal_mask = ~outlier_mask
            if np.sum(normal_mask) > 10:
                normal_medians = np.median(deltas[normal_mask], axis=0)
            else:
                normal_medians = np.median(deltas, axis=0)

            print(f'  OUTLIER ITERATIONS ({num_outliers} total, showing up to 30):')
            print(f'  {"iter":>6} {"time_us":>10} {"worst_phase":<22} {"spike_cycles":>12} {"normal_cycles":>14} {"ratio":>6}')
            print(f'  {"-"*6} {"-"*10} {"-"*22} {"-"*12} {"-"*14} {"-"*6}')

            shown = 0
            for idx in outlier_indices:
                if shown >= 30:
                    break
                ratios = np.zeros(len(delta_names))
                for j in range(len(delta_names)):
                    if normal_medians[j] > 0:
                        ratios[j] = deltas[idx, j] / normal_medians[j]
                worst_j = np.argmax(ratios)
                sev = ' ***' if severe_mask[idx] else ''
                print(f'  {idx:>6} {us[idx]:>10.1f} {delta_names[worst_j]:<22}'
                      f' {deltas[idx, worst_j]:>12} {normal_medians[worst_j]:>14.0f}'
                      f' {ratios[worst_j]:>5.1f}x{sev}')
                shown += 1

            # Phase blame summary
            blame = np.zeros(len(delta_names), dtype=int)
            for idx in outlier_indices:
                ratios = np.zeros(len(delta_names))
                for j in range(len(delta_names)):
                    if normal_medians[j] > 0:
                        ratios[j] = deltas[idx, j] / normal_medians[j]
                blame[np.argmax(ratios)] += 1

            print(f'\n  PHASE BLAME SUMMARY (which phase spiked most per outlier):')
            for j, name in enumerate(delta_names):
                if blame[j] > 0:
                    pct = 100.0 * blame[j] / num_outliers
                    print(f'    {name:<22}  {blame[j]:>4} ({pct:>5.1f}%)')
            print()

        if args.dump_csv:
            import csv
            with open(args.dump_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['iteration', 'time_us', 'outlier', 'severe'] +
                           [f'delta_{j}' for j in range(len(delta_names))] +
                           [f'ts_{j}' for j in range(NUM_PHASES)])
                for i in range(N):
                    w.writerow([i, us[i], int(outlier_mask[i]), int(severe_mask[i])] +
                               [int(deltas[i, j]) for j in range(len(delta_names))] +
                               [int(ts[i, j]) for j in range(NUM_PHASES)])
            print(f'  Raw data written to {args.dump_csv}\n')

    # --- Legacy control benchmark (same measurement methodology) ---

    legacy_dispatch_kwargs = dict(use_fp8=use_fp8, async_finish=True, return_recv_hook=False)
    if use_nvfp4:
        legacy_dispatch_kwargs['use_nvfp4'] = True
        legacy_dispatch_kwargs['x_global_scale'] = x_global_scale
    elif not use_fp8:
        legacy_dispatch_kwargs['use_fp8'] = False

    for _ in range(W):
        legacy_buf.clean_low_latency_buffer(T, H, E)
        _, _, _, event_d, _ = legacy_buf.low_latency_dispatch(x, topk_idx, T, E, **legacy_dispatch_kwargs)
        event_d.current_stream_wait()
    torch.cuda.synchronize()

    legacy_start = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    legacy_end = [torch.cuda.Event(enable_timing=True) for _ in range(N)]

    for i in range(N):
        torch.cuda.synchronize()
        dist.barrier()
        flush_buf.zero_()
        torch.cuda.synchronize()
        legacy_start[i].record()
        legacy_buf.clean_low_latency_buffer(T, H, E)
        _, _, _, event_d, _ = legacy_buf.low_latency_dispatch(x, topk_idx, T, E, **legacy_dispatch_kwargs)
        event_d.current_stream_wait()
        legacy_end[i].record()

    torch.cuda.synchronize()

    legacy_times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(legacy_start, legacy_end)])
    legacy_t = torch.tensor(legacy_times, dtype=torch.float64, device='cuda')
    dist.all_reduce(legacy_t, op=dist.ReduceOp.MAX)
    legacy_max = legacy_t.cpu().numpy()
    legacy_us = legacy_max * 1e6

    if rank == 0:
        lq1, lmed, lq3 = np.percentile(legacy_us, [25, 50, 75])
        liqr = lq3 - lq1
        lfence = lq3 + args.outlier_threshold * liqr
        lsevere = lq3 + 3.0 * liqr
        lout = int(np.sum(legacy_us > lfence))
        lsev = int(np.sum(legacy_us > lsevere))
        lp95 = np.percentile(legacy_us, 95)
        lp99 = np.percentile(legacy_us, 99)

        print(f'{"="*80}')
        print(f'  LEGACY CONTROL (same loop, same barrier, same flush)')
        print(f'{"="*80}')
        print(f'  Wall-clock (us):  median={lmed:.1f}  p95={lp95:.1f}  p99={lp99:.1f}'
              f'  max={np.max(legacy_us):.1f}  std={np.std(legacy_us):.1f}')
        print(f'  Outliers: {lout} mild (>{lfence:.1f}us)  {lsev} severe (>{lsevere:.1f}us)\n')

    ebuf.destroy()
    legacy_buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
