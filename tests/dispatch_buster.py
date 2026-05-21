# torchrun --nproc-per-node=4 tests/bench_fence.py
"""Benchmark low-latency dispatch + combine variants alongside elastic (v2) kernels.

Measures dispatch-only and full dispatch+combine cycles for:
  - dispatch_v2 / combine_v2 / combine_legacy  (legacy low-latency, NVSHMEM)
  - elastic_dispatch / elastic_combine          (ElasticBuffer, NCCL Gin)
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
    parser.add_argument('--no-mnnvl', action='store_true')
    parser.add_argument('--pack-scale-writes', action='store_true')
    parser.add_argument('--skip-legacy', action='store_true', help='Skip legacy (NVSHMEM) benchmarks')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.new_group(list(range(num_ranks)))

    T, H, E, K = args.num_tokens, args.hidden, args.num_experts, args.num_topk

    # Legacy low-latency buffer (NVSHMEM)
    buffer = None
    if not args.skip_legacy:
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(T, H, num_ranks, E)
        buffer = deep_ep.Buffer(
            group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
            num_qps_per_rank=E // num_ranks,
            allow_mnnvl=not args.no_mnnvl, explicitly_destroy=True)

    # Elastic buffer (NCCL Gin) — overlap mode (fewer SMs, leaves room for compute)
    elastic_buf_bf16 = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_fp8_dispatch=False, prefer_overlap_with_compute=True, explicitly_destroy=True)
    elastic_num_sms = elastic_buf_bf16.get_theoretical_num_sms(E, K)

    elastic_buf_fp8 = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_fp8_dispatch=True, prefer_overlap_with_compute=True, explicitly_destroy=True)

    elastic_buf_nvfp4 = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_nvfp4_dispatch=True, prefer_overlap_with_compute=True, explicitly_destroy=True)

    # Elastic buffer (NCCL Gin) — standalone mode (more SMs, no overlap)
    elastic_buf_bf16_full = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_fp8_dispatch=False, prefer_overlap_with_compute=False, explicitly_destroy=True)
    elastic_num_sms_full = elastic_buf_bf16_full.get_theoretical_num_sms(E, K)

    elastic_buf_fp8_full = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_fp8_dispatch=True, prefer_overlap_with_compute=False, explicitly_destroy=True)

    elastic_buf_nvfp4_full = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=T, hidden=H, num_topk=K,
        use_nvfp4_dispatch=True, prefer_overlap_with_compute=False, explicitly_destroy=True)

    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
    FLOAT4_E2M1_MAX = 6.0

    torch.manual_seed(42 + rank)
    x = torch.randn((T, H), dtype=torch.bfloat16, device='cuda')
    topk_weights = torch.randn((T, K), dtype=torch.float32, device='cuda').abs()

    x_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.max(torch.abs(x)).float()
    dist.all_reduce(x_global_scale, op=dist.ReduceOp.MIN, group=group)

    # Precompute FP8 input for elastic fp8 dispatch
    x_abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    x_fp8_scale = (FLOAT8_E4M3_MAX / x_abs_max).float()
    x_fp8 = (x.float() * x_fp8_scale).to(torch.float8_e4m3fn)
    x_fp8_sf = (1.0 / x_fp8_scale).view(T, -1)

    # Precompute NVFP4 input for elastic nvfp4 dispatch
    from deep_ep.utils.math import _float_to_e2m1_packed, align
    _aligned_n = align(H, 16)
    _x_padded = torch.nn.functional.pad(x, (0, _aligned_n - H), mode='constant', value=0)
    _x_view = _x_padded.view(T, -1, 16)
    _x_amax = _x_view.abs().float().amax(dim=2).view(T, -1).clamp(1e-4)
    _x_scaled = (_x_view * (6.0 / _x_amax.unsqueeze(2))).float()
    x_nvfp4 = _float_to_e2m1_packed(_x_scaled.view(T, _aligned_n))[:, :H // 2].contiguous()
    x_nvfp4_sf = (_x_amax / 6.0).view(T, -1)

    num_local_experts = E // num_ranks
    strategies = ['random', 'random-same', 'local-rand', 'local-same', 'remote-rand']
    formats = ['bf16', 'fp8', 'nvfp4']
    combine_variants = ['combine_v2', 'combine_v2_nf', 'combine_legacy', 'combine_legacy_nf']
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
            # Pick K unique local experts per token via topk on random scores over local experts
            local_scores = torch.randn((T, num_local_experts), dtype=torch.float32, device='cuda').abs() + 1
            topk_idx = torch.topk(local_scores, K, dim=-1, largest=True, sorted=True)[1] + local_start
        elif strategy == 'local-same':
            # Every token picks the same K local experts
            topk_idx = torch.arange(K, device='cuda', dtype=torch.int64).unsqueeze(0) + local_start
            topk_idx = topk_idx.expand(T, K).contiguous()
        else:
            # Pick K unique experts from remote ranks via topk on scores zeroed for local rank
            scores = torch.randn((T, E), dtype=torch.float32, device='cuda').abs() + 1
            scores[:, local_start:local_start + num_local_experts] = -1
            topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1]

        # Get expert output shape from a bf16 dispatch probe
        expert_output = None
        if not args.skip_legacy:
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

            # --- Legacy low-latency benchmarks ---

            if not args.skip_legacy:
                def make_clean_dispatch(idx=topk_idx, kw=dispatch_kwargs):
                    def fn():
                        buffer.clean_low_latency_buffer(T, H, E)
                        _, _, _, event_d, _ = buffer.low_latency_dispatch(x, idx, T, E, **kw)
                        event_d.current_stream_wait()
                    return fn

                def make_full(variant, idx=topk_idx, kw=dispatch_kwargs):
                    use_upstream = variant.startswith('combine_legacy')
                    use_fence = not variant.endswith('_nf')
                    def fn():
                        buffer.clean_low_latency_buffer(T, H, E)
                        _, _, handle, event_d, _ = buffer.low_latency_dispatch(x, idx, T, E, **kw)
                        event_d.current_stream_wait()
                        _, event_c, _ = buffer.low_latency_combine(
                            expert_output, idx, topk_weights, handle,
                            async_finish=True, return_recv_hook=False,
                            use_upstream=use_upstream,
                            use_fence_proxy_async=use_fence)
                        event_c.current_stream_wait()
                    return fn

            # --- Elastic (v2) benchmarks ---

            if use_nvfp4:
                elastic_inp = (x_nvfp4, x_nvfp4_sf)
                elastic_ebuf = elastic_buf_nvfp4
                elastic_ebuf_full = elastic_buf_nvfp4_full
            elif use_fp8:
                elastic_inp = (x_fp8, x_fp8_sf)
                elastic_ebuf = elastic_buf_fp8
                elastic_ebuf_full = elastic_buf_fp8_full
            else:
                elastic_inp = x
                elastic_ebuf = elastic_buf_bf16
                elastic_ebuf_full = elastic_buf_bf16_full

            def make_elastic_dispatch(idx=topk_idx, ebuf=elastic_ebuf,
                                      inp=elastic_inp, nsms=elastic_num_sms):
                def fn():
                    _, _, _, _, ev = ebuf.dispatch(
                        inp, topk_idx=idx, topk_weights=topk_weights,
                        num_experts=E, num_max_tokens_per_rank=T,
                        num_sms=nsms,
                        async_with_compute_stream=True)
                    ev.current_stream_wait()
                return fn

            def make_elastic_full(idx=topk_idx, ebuf=elastic_ebuf,
                                  inp=elastic_inp, nsms=elastic_num_sms):
                def fn():
                    recv_x, _, recv_topk_w, ehandle, ev_d = ebuf.dispatch(
                        inp, topk_idx=idx, topk_weights=topk_weights,
                        num_experts=E, num_max_tokens_per_rank=T,
                        num_sms=nsms,
                        async_with_compute_stream=True)
                    ev_d.current_stream_wait()
                    rx = recv_x[0] if isinstance(recv_x, tuple) else recv_x
                    expert_out_e = torch.empty_like(rx, dtype=torch.bfloat16)
                    _, _, ev_c = ebuf.combine(
                        expert_out_e, ehandle, topk_weights=recv_topk_w,
                        num_sms=nsms,
                        async_with_compute_stream=True)
                    ev_c.current_stream_wait()
                return fn

            rounds = {'elastic_dispatch': [], 'elastic_full_dispatch': [],
                      'elastic_combine': [], 'elastic_full_combine': []}
            if not args.skip_legacy:
                rounds['dispatch_v2'] = []
                for v in combine_variants:
                    rounds[v] = []

            for _ in range(args.num_rounds):
                # Legacy dispatch
                if not args.skip_legacy:
                    dist.barrier()
                    avg_cd, _, _ = bench(make_clean_dispatch(), args.num_warmups, args.num_tests)

                    # Legacy combine variants
                    for v in combine_variants:
                        dist.barrier()
                        avg_v, _, _ = bench(make_full(v), args.num_warmups, args.num_tests)
                        rounds[v].append(avg_v)

                # Elastic dispatch + combine
                # Overlap mode (fewer SMs)
                dist.barrier()
                avg_ed, _, _ = bench(make_elastic_dispatch(), args.num_warmups, args.num_tests)
                dist.barrier()
                avg_ef, _, _ = bench(make_elastic_full(), args.num_warmups, args.num_tests)
                # Standalone mode (more SMs)
                dist.barrier()
                avg_ed_full, _, _ = bench(make_elastic_dispatch(ebuf=elastic_ebuf_full, inp=elastic_inp, nsms=elastic_num_sms_full), args.num_warmups, args.num_tests)
                dist.barrier()
                avg_ef_full, _, _ = bench(make_elastic_full(ebuf=elastic_ebuf_full, inp=elastic_inp, nsms=elastic_num_sms_full), args.num_warmups, args.num_tests)

                dist.barrier()

                # Reduce across ranks (worst-case)
                if not args.skip_legacy:
                    for key, val in [('dispatch_v2', avg_cd)]:
                        t = torch.tensor([val], dtype=torch.float64, device='cuda')
                        dist.all_reduce(t, op=dist.ReduceOp.MAX)
                        rounds[key].append(t.item())
                    for v in combine_variants:
                        t = torch.tensor([rounds[v][-1]], dtype=torch.float64, device='cuda')
                        dist.all_reduce(t, op=dist.ReduceOp.MAX)
                        rounds[v][-1] = t.item()
                for key, val in [('elastic_dispatch', avg_ed), ('elastic_full_dispatch', avg_ed_full)]:
                    t = torch.tensor([val], dtype=torch.float64, device='cuda')
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    rounds[key].append(t.item())
                for ckey, cval in [('elastic_combine', avg_ef), ('elastic_full_combine', avg_ef_full)]:
                    t = torch.tensor([cval], dtype=torch.float64, device='cuda')
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    rounds[ckey].append(t.item())

            with np.errstate(all='ignore'):
                results[(strategy, fmt)] = {k: np.nanmedian(v) for k, v in rounds.items()}

    if rank == 0:
        R = args.num_rounds
        print(f'\nbenchmark (median of {R} rounds, {args.num_tests} iters each)')
        print(f'  T={T}  H={H}  E={E}  topk={K}  ranks={num_ranks}')
        print(f'  worst-case rank (max across ranks per round)\n')
        print(f'  strategies:')
        print(f'    random      = topk over random scores, different experts per token (cross-rank RDMA)')
        print(f'    random-same = topk over random scores, same K experts for all tokens (cross-rank RDMA)')
        print(f'    local-rand  = random offsets mapped to local rank experts (no RDMA, varied experts)')
        print(f'    local-same  = every token picks the same K local experts (no RDMA, hotspot)')
        print(f'    remote-rand = random experts, each from a different remote rank (all cross-rank RDMA)\n')

        print(f'  elastic overlap mode:    num_sms={elastic_num_sms} (prefer_overlap_with_compute=True)')
        print(f'  elastic standalone mode: num_sms={elastic_num_sms_full} (prefer_overlap_with_compute=False)\n')

        # --- Main table ---
        if args.skip_legacy:
            hdr = (f'  {"strategy":<12} {"fmt":<6} '
                   f'{"e_disp":>9} {"e_comb":>9} '
                   f'{"ef_disp":>9} {"ef_comb":>9}')
            uline = (f'  {"":12} {"":6} '
                     f'{"(us)":>9} {"(us)":>9} '
                     f'{"(us)":>9} {"(us)":>9}')
            sep = (f'  {"-"*12} {"-"*6} '
                   f'{"-"*9} {"-"*9} '
                   f'{"-"*9} {"-"*9}')
        else:
            hdr = (f'  {"strategy":<12} {"fmt":<6} '
                   f'{"disp_v2":>9} {"comb_v2":>9} {"comb_leg":>9} '
                   f'{"e_disp":>9} {"e_comb":>9} '
                   f'{"ef_disp":>9} {"ef_comb":>9}')
            uline = (f'  {"":12} {"":6} '
                     f'{"(us)":>9} {"(us)":>9} {"(us)":>9} '
                     f'{"(us)":>9} {"(us)":>9} '
                     f'{"(us)":>9} {"(us)":>9}')
            sep = (f'  {"-"*12} {"-"*6} '
                   f'{"-"*9} {"-"*9} {"-"*9} '
                   f'{"-"*9} {"-"*9} '
                   f'{"-"*9} {"-"*9}')
        print(f'  legend: e_=elastic overlap, ef_=elastic standalone (full SMs)\n')
        print(hdr)
        print(uline)
        print(sep)
        for strategy in strategies:
            for fmt in formats:
                r = results[(strategy, fmt)]
                ed = r['elastic_dispatch'] * 1e6
                ec = (r['elastic_combine'] - r['elastic_dispatch']) * 1e6
                efd = r['elastic_full_dispatch'] * 1e6
                efc = (r['elastic_full_combine'] - r['elastic_full_dispatch']) * 1e6

                if args.skip_legacy:
                    print(f'  {strategy:<12} {fmt:<6} '
                          f'{ed:>9.1f} {ec:>9.1f} '
                          f'{efd:>9.1f} {efc:>9.1f}')
                else:
                    d_v2 = r['dispatch_v2'] * 1e6
                    c_v2 = (r['combine_v2'] - r['dispatch_v2']) * 1e6
                    c_leg = (r['combine_legacy'] - r['dispatch_v2']) * 1e6
                    print(f'  {strategy:<12} {fmt:<6} '
                          f'{d_v2:>9.1f} {c_v2:>9.1f} {c_leg:>9.1f} '
                          f'{ed:>9.1f} {ec:>9.1f} '
                          f'{efd:>9.1f} {efc:>9.1f}')
        print()

        # --- Fence overhead table ---
        if not args.skip_legacy:
            print(f'  fence.proxy.async overhead (combine only, fence ON vs OFF)')
            print(f'  positive delta = fence adds latency\n')
            fhdr = (f'  {"strategy":<12} {"fmt":<6} '
                    f'{"cv2_on":>9} {"cv2_off":>9} {"delta":>9} {"pct":>7} '
                    f'{"cleg_on":>9} {"cleg_off":>9} {"delta":>9} {"pct":>7}')
            fuline = (f'  {"":12} {"":6} '
                      f'{"(us)":>9} {"(us)":>9} {"(us)":>9} {"":>7} '
                      f'{"(us)":>9} {"(us)":>9} {"(us)":>9} {"":>7}')
            fsep = (f'  {"-"*12} {"-"*6} '
                    f'{"-"*9} {"-"*9} {"-"*9} {"-"*7} '
                    f'{"-"*9} {"-"*9} {"-"*9} {"-"*7}')
            print(fhdr)
            print(fuline)
            print(fsep)
            for strategy in strategies:
                for fmt in formats:
                    r = results[(strategy, fmt)]
                    cv2_on  = (r['combine_v2'] - r['dispatch_v2']) * 1e6
                    cv2_off = (r['combine_v2_nf'] - r['dispatch_v2']) * 1e6
                    cv2_delta = cv2_on - cv2_off
                    cv2_pct = cv2_delta / cv2_off * 100 if cv2_off else 0

                    cleg_on  = (r['combine_legacy'] - r['dispatch_v2']) * 1e6
                    cleg_off = (r['combine_legacy_nf'] - r['dispatch_v2']) * 1e6
                    cleg_delta = cleg_on - cleg_off
                    cleg_pct = cleg_delta / cleg_off * 100 if cleg_off else 0

                    print(f'  {strategy:<12} {fmt:<6} '
                          f'{cv2_on:>9.1f} {cv2_off:>9.1f} {cv2_delta:>+9.1f} {cv2_pct:>+6.1f}% '
                          f'{cleg_on:>9.1f} {cleg_off:>9.1f} {cleg_delta:>+9.1f} {cleg_pct:>+6.1f}%')
            print()

    if buffer is not None:
        buffer.destroy()
    elastic_buf_bf16.destroy()
    elastic_buf_fp8.destroy()
    elastic_buf_nvfp4.destroy()
    elastic_buf_bf16_full.destroy()
    elastic_buf_fp8_full.destroy()
    elastic_buf_nvfp4_full.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
