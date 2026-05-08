# Profile local-same dispatch with ncu.
#
# Usage:
#   MASTER_ADDR=localhost MASTER_PORT=29700 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
#     ncu -f --set full --replay-mode application -k dispatch \
#     --launch-skip 3 --launch-count 1 --import-source yes \
#     --source-folders csrc/kernels,csrc \
#     -o /workspace/ncu_local_same python tests/ncu_local_same.py [--nvfp4]
"""Local-same dispatch for ncu profiling."""
import argparse
import os
os.environ.setdefault('NVSHMEM_QP_DEPTH', '4096')

import torch
import torch.distributed as dist
import deep_ep

parser = argparse.ArgumentParser()
parser.add_argument('--nvfp4', action='store_true')
args = parser.parse_args()

if not dist.is_initialized():
    dist.init_process_group(backend='nccl')
rank = dist.get_rank()
num_ranks = dist.get_world_size()
torch.cuda.set_device(rank)
group = dist.new_group(list(range(num_ranks)))

T, H, E, K = 1024, 7168, 256, 8
num_local_experts = E // num_ranks

num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(T, H, num_ranks, E)
buffer = deep_ep.Buffer(
    group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
    num_qps_per_rank=E // num_ranks, explicitly_destroy=True)

torch.manual_seed(42 + rank)
x = torch.randn((T, H), dtype=torch.bfloat16, device='cuda')

local_start = rank * num_local_experts
offsets = torch.arange(K, device='cuda', dtype=torch.int64).unsqueeze(0)
topk_idx = (offsets % num_local_experts + local_start).expand(T, K).contiguous()

dispatch_kwargs = dict(async_finish=True, return_recv_hook=False)
if args.nvfp4:
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
    FLOAT4_E2M1_MAX = 6.0
    x_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.max(torch.abs(x)).float()
    dist.all_reduce(x_global_scale, op=dist.ReduceOp.MIN, group=group)
    dispatch_kwargs.update(use_fp8=False, use_nvfp4=True, x_global_scale=x_global_scale)
else:
    dispatch_kwargs['use_fp8'] = False

for _ in range(3):
    buffer.clean_low_latency_buffer(T, H, E)
    _, _, _, ev, _ = buffer.low_latency_dispatch(
        x, topk_idx, T, E, **dispatch_kwargs)
    ev.current_stream_wait()
    torch.cuda.synchronize()

dist.barrier()
buffer.clean_low_latency_buffer(T, H, E)
_, _, _, ev, _ = buffer.low_latency_dispatch(
    x, topk_idx, T, E, **dispatch_kwargs)
ev.current_stream_wait()
torch.cuda.synchronize()

if rank == 0:
    print('done')

buffer.destroy()
dist.barrier()
dist.destroy_process_group()
