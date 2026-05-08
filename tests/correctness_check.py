# torchrun --nproc-per-node=4 correctness_check.py
# A/B test: fence.proxy.async on vs off in combine_v2
import torch, torch.distributed as dist, deep_ep

dist.init_process_group(backend='nccl')
rank, W = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

topk = 8
T, H, E = 1024, 7168, 256
local_E = E // W

buf = deep_ep.Buffer(
    dist.new_group(list(range(W))),
    num_nvl_bytes=0,
    num_rdma_bytes=deep_ep.Buffer.get_low_latency_rdma_size_hint(T, H, W, E),
    low_latency_mode=True, num_qps_per_rank=E//W,
    allow_mnnvl=True, explicitly_destroy=True)

torch.manual_seed(42 + rank)
x = torch.randint(0, 2, (T, H), dtype=torch.bfloat16, device="cuda")

ids = torch.arange(topk, device='cuda', dtype=torch.int64).unsqueeze(0).expand(T, -1).contiguous()
w = torch.ones(T, topk, dtype=torch.float32, device='cuda')

expected = (float(topk) * x.float()).bfloat16()

N = 1024 * 4

def run_test(label, use_fence_proxy_async):
    n_wrong = 0
    total_bad_rows = 0
    total_bad_cols = 0
    for it in range(N):
        buf.clean_low_latency_buffer(T, H, E)
        torch.cuda.synchronize()
        dist.barrier()

        ex, _, h, _, hk = buf.low_latency_dispatch(
            x, ids, T, E, use_fp8=False, async_finish=False, return_recv_hook=True)
        hk()
        torch.cuda.synchronize()

        out = torch.zeros(T, H, dtype=torch.bfloat16, device='cuda')
        _, _, hk_c = buf.low_latency_combine(
            ex.clone(), ids, w, h, async_finish=False,
            return_recv_hook=True, out=out, use_fence_proxy_async=use_fence_proxy_async)
        hk_c()
        torch.cuda.synchronize()

        if not torch.equal(out, expected):
            n_wrong += 1
            diff = (out != expected)
            total_bad_rows += diff.any(dim=1).sum().item()
            total_bad_cols += diff.any(dim=0).sum().item()

    if rank == 0:
        n_bad = max(n_wrong, 1)
        print(f'  {label}: correct={N-n_wrong}/{N}')
        if n_wrong > 0:
            print(f'    avg bad rows: {total_bad_rows/n_bad:.1f}/{T}  avg bad cols: {total_bad_cols/n_bad:.1f}/{H}')
    dist.barrier()


if rank == 0:
    print(f'\n=== fence.proxy.async A/B (T={T} H={H} E={E} W={W} topk={topk}) ===\n')

run_test('no fence  (use_fence_proxy_async=False)', use_fence_proxy_async=False)
run_test('with fence (use_fence_proxy_async=True) ', use_fence_proxy_async=True)

dist.barrier()
buf.destroy()
dist.destroy_process_group()
