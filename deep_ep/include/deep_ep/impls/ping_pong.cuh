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
    const int num_iters,
    void* data_buffer,
    const int num_payload_bytes) {

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

    // Get symmetric pointer to peer's data buffer for bulk writes
    auto* peer_data = gin.template get_sym_ptr<ncclTeamTagLsa>(
        static_cast<int64_t*>(data_buffer), peer_rank_idx);
    const int num_words = num_payload_bytes / 8;

    const bool is_pinger = (rank_idx < peer_rank_idx);

    for (int i = 0; i < num_iters; ++i) {
        const int64_t val = static_cast<int64_t>(i + 1);

        if (is_pinger) {
            timestamps[i * 2] = clock64();

            // Write payload to peer's data region
            for (int j = 0; j < num_words; ++j)
                ptx::st_relaxed_sys(peer_data + j, val);
            if (num_words > 0)
                __threadfence_system();

            // Signal
            gin.template put_value<ncclTeamTagLsa>(&mailbox[peer_rank_idx], val, peer_rank_idx);

            // Wait for response
            while (ptx::ld_acquire_sys(&mailbox[rank_idx]) < val) {}
            timestamps[i * 2 + 1] = clock64();
        } else {
            timestamps[i * 2] = clock64();

            // Wait for peer's signal
            while (ptx::ld_acquire_sys(&mailbox[rank_idx]) < val) {}

            // Write payload back
            for (int j = 0; j < num_words; ++j)
                ptx::st_relaxed_sys(peer_data + j, val);
            if (num_words > 0)
                __threadfence_system();

            // Signal response
            gin.template put_value<ncclTeamTagLsa>(&mailbox[peer_rank_idx], val, peer_rank_idx);

            timestamps[i * 2 + 1] = clock64();
        }
    }
}

}  // namespace deep_ep::elastic
