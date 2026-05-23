# torchrun --nproc-per-node=2 tests/nvlink_latency_bench.py
"""NVLink ping-pong latency benchmark with payload size sweep.

Measures round-trip NVLink write latency between two GPUs using NCCL Gin,
sweeping over payload sizes from 8B (signal only) to 64KB.
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


def run_ping_pong(ebuf, peer, num_iters, num_warmups, num_sms, num_experts, num_payload_bytes):
    for _ in range(num_warmups):
        ebuf.runtime.ping_pong(peer, 1, num_sms, num_experts, num_payload_bytes)
    torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()
    ts = ebuf.runtime.ping_pong(peer, num_iters, num_sms, num_experts, num_payload_bytes)
    torch.cuda.synchronize()

    cycles = ts.cpu().numpy()
    return cycles[:, 1] - cycles[:, 0]


def main():
    parser = argparse.ArgumentParser(description='NVLink ping-pong latency sweep')
    parser.add_argument('--num-iters', type=int, default=10000)
    parser.add_argument('--num-warmups', type=int, default=500)
    parser.add_argument('--num-sms', type=int, default=2)
    parser.add_argument('--num-experts', type=int, default=256)
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
        peer = -1

    # Buffer large enough for max payload
    ebuf = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=256, hidden=512, num_topk=1,
        use_fp8_dispatch=False, use_nvfp4_dispatch=False,
        prefer_overlap_with_compute=False, explicitly_destroy=True)

    payload_sizes = [0, 64, 256, 1024, 4096, 16384, 65536]

    results = {}
    for nbytes in payload_sizes:
        rt = run_ping_pong(ebuf, peer, args.num_iters, args.num_warmups,
                           args.num_sms, args.num_experts, nbytes)
        results[nbytes] = rt

    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        clock_ghz = props.clock_rate / 1e6

        print(f'\n{"="*78}')
        print(f'  NVLink Ping-Pong Latency Sweep  (GPU 0 <-> GPU 1)')
        print(f'  {args.num_iters} iterations, {args.num_warmups} warmups')
        print(f'  GPU clock: {clock_ghz:.3f} GHz')
        print(f'{"="*78}\n')

        print(f'  {"payload":>10} {"RT med":>10} {"one-way":>10} {"RT p95":>10} {"RT p99":>10}'
              f' {"RT max":>10} {"std":>10} {"outlier%":>10}')
        print(f'  {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')

        for nbytes in payload_sizes:
            s = format_stats(results[nbytes].astype(np.float64), clock_ghz)
            label = '8B (sig)' if nbytes == 0 else f'{nbytes}B' if nbytes < 1024 else f'{nbytes//1024}KB'
            print(f'  {label:>10} {s["median"]:>10.2f} {s["median"]/2:>10.2f} {s["p95"]:>10.2f}'
                  f' {s["p99"]:>10.2f} {s["max"]:>10.2f} {s["std"]:>10.2f} {s["outlier_pct"]:>10.1f}')

        print(f'\n  All times in microseconds (us)\n')

    ebuf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
