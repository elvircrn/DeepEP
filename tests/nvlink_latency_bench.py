# torchrun --nproc-per-node=2 tests/nvlink_latency_bench.py
"""NVLink ping-pong latency benchmark: NCCL Gin vs NVSHMEM.

Measures raw one-way and round-trip NVLink write latency between two GPUs
using both the elastic (NCCL Gin) and legacy (NVSHMEM) transport layers.
"""
import argparse
import os

os.environ.setdefault('NVSHMEM_QP_DEPTH', '4096')

import numpy as np
import torch
import torch.distributed as dist
import deep_ep


def format_stats(cycles, clock_ghz):
    us = cycles / (clock_ghz * 1e3)
    q1, med, q3 = np.percentile(us, [25, 50, 75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    return {
        'median': med,
        'p95': np.percentile(us, 95),
        'p99': np.percentile(us, 99),
        'max': np.max(us),
        'std': np.std(us),
        'outlier_pct': 100.0 * np.sum(us > fence) / len(us),
    }


def main():
    parser = argparse.ArgumentParser(description='NVLink ping-pong latency')
    parser.add_argument('--num-iters', type=int, default=10000)
    parser.add_argument('--num-warmups', type=int, default=500)
    parser.add_argument('--num-sms', type=int, default=2)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--skip-legacy', action='store_true',
                        help='Skip NVSHMEM benchmark')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.new_group(list(range(num_ranks)))

    assert num_ranks >= 2, 'Need at least 2 ranks'
    if rank == 0:
        peer = 1
    elif rank == 1:
        peer = 0
    else:
        peer = -1  # non-participating ranks

    # --- NCCL Gin (elastic) ---

    ebuf = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=64, hidden=128, num_topk=1,
        use_fp8_dispatch=False, use_nvfp4_dispatch=False,
        prefer_overlap_with_compute=False, explicitly_destroy=True)

    # Warmup
    dist.barrier()
    for _ in range(args.num_warmups):
        ebuf.runtime.ping_pong(peer, 1, args.num_sms, args.num_experts)
    torch.cuda.synchronize()

    # Timed
    dist.barrier()
    torch.cuda.synchronize()
    gin_ts = ebuf.runtime.ping_pong(peer, args.num_iters, args.num_sms, args.num_experts)
    torch.cuda.synchronize()

    gin_cycles = gin_ts.cpu().numpy()
    gin_rt = gin_cycles[:, 1] - gin_cycles[:, 0]

    # --- NVSHMEM (legacy) ---

    nvshmem_rt = None
    if not args.skip_legacy:
        H, T, E = 128, 64, args.num_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(T, H, num_ranks, E)
        lbuf = deep_ep.Buffer(
            group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
            num_qps_per_rank=E // num_ranks, explicitly_destroy=True)

        # Warmup
        dist.barrier()
        for _ in range(args.num_warmups):
            lbuf.runtime.ping_pong(peer, 1)
        torch.cuda.synchronize()

        # Timed
        dist.barrier()
        torch.cuda.synchronize()
        nvshmem_ts = lbuf.runtime.ping_pong(peer, args.num_iters)
        torch.cuda.synchronize()

        nvshmem_cycles = nvshmem_ts.cpu().numpy()
        nvshmem_rt = nvshmem_cycles[:, 1] - nvshmem_cycles[:, 0]
        lbuf.destroy()

    # --- Report (rank 0 = pinger, measures full round-trip) ---

    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        clock_ghz = props.clock_rate / 1e6

        gin_stats = format_stats(gin_rt.astype(np.float64), clock_ghz)

        print(f'\n{"="*70}')
        print(f'  NVLink Ping-Pong Latency  (GPU {rank} <-> GPU {peer})')
        print(f'  {args.num_iters} iterations, {args.num_warmups} warmups')
        print(f'  GPU clock: {clock_ghz:.3f} GHz')
        print(f'{"="*70}\n')

        header = f'  {"metric":<16}'
        gin_col = f'{"NCCL Gin":>12}'
        nvshmem_col = f'{"NVSHMEM":>12}' if nvshmem_rt is not None else ''
        print(f'{header}{gin_col}{nvshmem_col}')
        print(f'  {"-"*16}{"-"*12}{"" if nvshmem_rt is None else "-"*12}')

        nv_stats = format_stats(nvshmem_rt.astype(np.float64), clock_ghz) if nvshmem_rt is not None else None

        for label, key, div in [
            ('round-trip (us)', 'median', 1),
            ('one-way (us)', 'median', 2),
            ('p95 RT (us)', 'p95', 1),
            ('p99 RT (us)', 'p99', 1),
            ('max RT (us)', 'max', 1),
            ('std RT (us)', 'std', 1),
            ('outlier %', 'outlier_pct', 1),
        ]:
            gv = gin_stats[key] / div
            line = f'  {label:<16}{gv:>12.2f}'
            if nv_stats is not None:
                nv = nv_stats[key] / div
                line += f'{nv:>12.2f}'
            print(line)

        print()

    ebuf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
