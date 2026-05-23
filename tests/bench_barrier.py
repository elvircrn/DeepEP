# torchrun --nproc-per-node=4 tests/bench_barrier.py
"""Minimal gin barrier latency test."""
import argparse
import os
os.environ.setdefault('NVSHMEM_QP_DEPTH', '4096')

import numpy as np
import torch
import torch.distributed as dist
import deep_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-iters', type=int, default=2000)
    parser.add_argument('--num-sms', type=int, default=64)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    torch.cuda.set_device(rank)
    group = dist.new_group(list(range(num_ranks)))

    ebuf = deep_ep.ElasticBuffer(
        group, num_max_tokens_per_rank=1024, hidden=7168, num_topk=args.num_topk,
        use_fp8_dispatch=True, prefer_overlap_with_compute=False, explicitly_destroy=True)

    # Warmup
    ebuf.runtime.barrier_test(50, args.num_sms, args.num_experts)
    torch.cuda.synchronize()

    # Timed run
    ts = ebuf.runtime.barrier_test(args.num_iters, args.num_sms, args.num_experts)
    torch.cuda.synchronize()

    ts = ts.cpu().numpy()  # (N, 2) — before/after each barrier
    deltas = ts[:, 1] - ts[:, 0]

    # Reduce deltas across ranks
    deltas_t = torch.tensor(deltas, dtype=torch.int64, device='cuda')
    dist.all_reduce(deltas_t, op=dist.ReduceOp.MAX)
    deltas = deltas_t.cpu().numpy().astype(np.float64)

    if rank == 0:
        q1, med, q3 = np.percentile(deltas, [25, 50, 75])
        iqr = q3 - q1
        fence = q3 + 1.5 * iqr
        severe = q3 + 3.0 * iqr
        nout = int(np.sum(deltas > fence))
        nsev = int(np.sum(deltas > severe))

        print(f'\n  GIN BARRIER TEST  sms={args.num_sms}  ranks={num_ranks}  iters={args.num_iters}')
        print(f'  median={med:.0f}  p95={np.percentile(deltas, 95):.0f}'
              f'  p99={np.percentile(deltas, 99):.0f}  max={np.max(deltas):.0f}'
              f'  std={np.std(deltas):.0f}  (cycles)')
        print(f'  outliers={nout}  severe={nsev}\n')

    ebuf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
