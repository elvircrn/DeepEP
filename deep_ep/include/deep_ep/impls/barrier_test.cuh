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
barrier_test_impl(
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
    void* workspace,
    const int rank_idx,
    int64_t* timestamps,
    const int num_iters) {

    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);

    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumThreads / 32, false>(
        sm_idx, 0, false);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    for (int i = 0; i < num_iters; ++i) {
        if (sm_idx == 0 && thread_idx == 0)
            timestamps[i * 2] = clock64();

        comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                          kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles,
                          comm::kDispatchTag0, false, false, true>(
            gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

        if (sm_idx == 0 && thread_idx == 0)
            timestamps[i * 2 + 1] = clock64();
    }
}

}  // namespace deep_ep::elastic
