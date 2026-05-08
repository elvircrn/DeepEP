# torchrun --nproc-per-node=4 tests/ncu_fence.py --fence / --no-fence
"""Minimal script for ncu profiling of combine_v2 with/without fence."""
import argparse
import os
import sys

os.environ.setdefault('NVSHMEM_QP_DEPTH', '4096')

import torch
import torch.distributed as dist
import deep_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-tokens', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--fence', action='store_true', default=True)
    parser.add_argument('--no-fence', dest='fence', action='store_false')
    parser.add_argument('--allow-mnnvl', action='store_true')
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

    torch.manual_seed(42 + rank)
    x = torch.randn((T, H), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((T, E), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, K, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((T, K), dtype=torch.float32, device='cuda').abs()

    def run_cycle():
        buffer.clean_low_latency_buffer(T, H, E)
        recv_x, _, handle, event_d, _ = buffer.low_latency_dispatch(
            x, topk_idx, T, E, use_fp8=False,
            async_finish=True, return_recv_hook=False)
        event_d.current_stream_wait()
        _, event_c, _ = buffer.low_latency_combine(
            recv_x, topk_idx, topk_weights, handle,
            async_finish=True, return_recv_hook=False,
            use_fence_proxy_async=args.fence)
        event_c.current_stream_wait()
        torch.cuda.synchronize()

    for _ in range(5):
        run_cycle()

    dist.barrier()
    run_cycle()
    dist.barrier()

    if rank == 0:
        print(f'done (fence={"on" if args.fence else "off"})')

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
