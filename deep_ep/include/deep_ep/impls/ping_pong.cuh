#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic {

template <bool kIsScaleupNVLink,
          int kNumSMs, int kNumThreads,
          int kNumRanks, int kNumExperts,
          int kNumQPs, int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumThreads, 1)
ping_pong_gin_impl(
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
    void* workspace,
    const int rank_idx,
    const int peer_rank_idx,
    int64_t* timestamps,
    const int num_iters) {

    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);

    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumThreads / 32, false>(
        sm_idx, 0, false);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Use notify reduction workspace as mailbox slots (safe: no dispatch running)
    auto* mailbox = workspace_layout.get_notify_reduction_workspace_ptr();

    // Zero our mailbox
    if (sm_idx == 0 && thread_idx == 0)
        mailbox[rank_idx] = 0;

    // Barrier so all ranks have zeroed before any sends
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles,
                      comm::kDeviceBarrierTag, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Only SM 0 thread 0 does the ping-pong; others idle
    if (sm_idx != 0 || thread_idx != 0)
        return;

    // Non-participating ranks (peer_rank_idx < 0) skip the loop
    if (peer_rank_idx < 0)
        return;

    const bool is_pinger = (rank_idx < peer_rank_idx);

    for (int i = 0; i < num_iters; ++i) {
        const int64_t val = static_cast<int64_t>(i + 1);

        if (is_pinger) {
            // Send to peer, wait for response
            timestamps[i * 2] = clock64();
            gin.put_value<ncclTeamTagLsa>(&mailbox[peer_rank_idx], val, peer_rank_idx);
            while (ptx::ld_acquire_sys(&mailbox[rank_idx]) < val) {}
            timestamps[i * 2 + 1] = clock64();
        } else {
            // Wait for peer, then respond
            timestamps[i * 2] = clock64();
            while (ptx::ld_acquire_sys(&mailbox[rank_idx]) < val) {}
            gin.put_value<ncclTeamTagLsa>(&mailbox[peer_rank_idx], val, peer_rank_idx);
            timestamps[i * 2 + 1] = clock64();
        }
    }
}

}  // namespace deep_ep::elastic
