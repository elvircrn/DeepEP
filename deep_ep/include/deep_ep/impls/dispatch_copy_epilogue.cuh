#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

// Device-wide arrive-and-wait barrier. Safe without cooperative launch because
// this kernel runs with `__launch_bounds__(kNumThreads, 1)` and a grid of
// exactly `kNumSMs` blocks -> at most one block per SM, so every block is
// co-resident and no block can be waiting on an unscheduled peer.
// `counter` must be host-zeroed before launch; use a distinct counter per call.
__device__ __forceinline__ void grid_arrive_and_wait(int* counter, int expected) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        atomicAdd(counter, 1);
        while (atomicAdd(counter, 0) < expected) { /* spin */ }
    }
    __syncthreads();
}

template <bool kDoExpand, bool kCachedMode,
          // NOTES: this channel concept only applies for scale-out ranks
          int kNumSMs, int kNumChannels, int kNumWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          // Fused MoE-align (decode / non-expand only). When enabled, the
          // epilogue additionally emits DeepGEMM/Triton grouped-GEMM metadata
          // (sorted_token_ids / expert_ids / num_tokens_post_pad), folding away
          // the separate globalize + moe_align_block_size + count_and_sort
          // kernels. `kAlignM` is the per-expert padding divisibility (each
          // expert region is padded up to a multiple of it), NOT the runtime
          // GEMM tile: the GEMM is chosen by the runtime heuristic after
          // dispatch. The chosen GEMM's tile M must divide / be compatible with
          // kAlignM. 16 is a safe default granularity.
          bool kFuseMoeAlign = false, int kAlignM = 16,
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumThreads = kNumWarps * 32,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          bool kDoCreateLinkedList = (kNumScaleoutRanks > 1 and not kCachedMode),
          int kNumExpertsPerRank = kNumExperts / kNumRanks>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_copy_epilogue_impl(void* buffer, void* workspace,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* psum_num_recv_tokens_per_expert,
                            void* recv_x, sf_pack_t* recv_sf,
                            topk_idx_t* recv_topk_idx, float* recv_topk_weights,
                            int* recv_src_metadata,
                            int* channel_linked_list,
                            int num_recv_tokens,
                            const int recv_sf_token_stride, const int recv_sf_hidden_stride,
                            const int scaleout_rank_idx, const int scaleup_rank_idx,
                            // Fused MoE-align outputs / scratch (unused unless kFuseMoeAlign)
                            int* sorted_token_ids = nullptr,
                            int* fused_expert_ids = nullptr,
                            int* num_tokens_post_pad = nullptr,
                            int* expert_count = nullptr,   // [kNumExpertsPerRank], host-zeroed
                            int* grid_barrier = nullptr,   // [>=2] ints, host-zeroed
                            int num_sorted_slots = 0) {
    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;

    // For top-k index transformations (kNumExpertsPerRank is a template param)
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;
    const auto expert_start_idx = kNumExpertsPerRank * rank_idx, expert_end_idx = kNumExpertsPerRank * (rank_idx + 1);

    // Buffer layouts
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    const auto scaleup_buffer = layout::BufferLayout<false>(token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);

    // Init TMA
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // Will block until the main dispatch kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();

    // For no CPU sync case, the number of received tokens should be read from the GPU tensor.
    // Capture the allocated (pre-reread) row count first: it bounds the recv_topk_idx
    // buffer, so the fused padding-tail sanitization stays in-bounds.
    [[maybe_unused]] const int num_allocated_recv_tokens = num_recv_tokens;
    if (num_recv_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[kNumScaleupRanks - 1];

    // Current rank indices should be maintained
    int current_rank_idx = -1, stored_psum_num_recv_tokens;
    int current_rank_start = 0, current_rank_end = 0;
    #pragma unroll
    for (int i = global_warp_idx; i < num_recv_tokens; i += kNumWarps * kNumSMs) {
        // Calculate token index in the buffer
        while (i >= current_rank_end) {
            current_rank_idx += 1;
            EP_DEVICE_ASSERT(current_rank_idx < kNumScaleupRanks);
            const auto stored_lane_idx = current_rank_idx % 32;
            if (stored_lane_idx == 0 and current_rank_idx + lane_idx < kNumScaleupRanks)
                stored_psum_num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[current_rank_idx + lane_idx];
            current_rank_start = current_rank_end;
            current_rank_end = ptx::exchange(stored_psum_num_recv_tokens, stored_lane_idx);
        }
        const auto buffer_token = scaleup_buffer.get_rank_buffer(current_rank_idx).get_token_buffer(i - current_rank_start);

        // Wait buffer releases
        ptx::tma_store_wait();
        __syncwarp();

        // Issue TMA loads
        // Including all stuffs: data, SF, top-k metadata
        if (ptx::elect_one_sync()) {
            ptx::tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(),
                             mbarrier_ptr, tma_buffer.get_num_bytes<false>());
            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, tma_buffer.get_num_bytes<false>());
        }
        __syncwarp();

        // Load target expert indices separately to tolerate TMA load latency
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        int dst_expert_idx = -1;
        if (lane_idx < kNumTopk)
            dst_expert_idx = buffer_token.get_topk_idx_ptr()[lane_idx];
        __syncwarp();

        // Validate target expert indices and store for non-expand mode
        const auto in_range = expert_start_idx <= dst_expert_idx and dst_expert_idx < expert_end_idx;
        const auto master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(in_range));
        // Global expert id (or -1) — used only by the fused path, which stores the
        // globalized value so it can also fold away `_globalize_recv_topk_idx`.
        const int global_expert_idx = in_range ? dst_expert_idx : -1;
        dst_expert_idx = in_range ? dst_expert_idx - expert_start_idx : -1;
        EP_DEVICE_ASSERT(ptx::deduplicate(dst_expert_idx, lane_idx) or dst_expert_idx == -1);
        if (not kDoExpand and lane_idx < kNumTopk) {
            // Non-fused: store the local id (Python `_globalize_recv_topk_idx`
            // converts + sanitizes). Fused: store the global id directly so the
            // globalize kernel can be skipped entirely.
            const int stored_idx = kFuseMoeAlign ? global_expert_idx : dst_expert_idx;
            recv_topk_idx[i * kNumTopk + lane_idx] = static_cast<topk_idx_t>(stored_idx);
        }
        // Fused MoE-align Phase 1: tally per-(local)-expert recv counts. Uses the
        // local `dst_expert_idx` (0..kNumExpertsPerRank) or -1, independent of the
        // globalized value stored above.
        if constexpr (kFuseMoeAlign) {
            if (lane_idx < kNumTopk and dst_expert_idx >= 0)
                atomicAdd(expert_count + dst_expert_idx, 1);
        }
        __syncwarp();

        // Calculate target indices in the tensor
        int dst_tensor_idx = -1;
        if (not kDoExpand and ptx::elect_one_sync()) {
            dst_tensor_idx = i;
        } else if (kDoExpand and dst_expert_idx >= 0) {
            dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);
        }
        __syncwarp();

        // Wait for TMA arrival
        if (ptx::elect_one_sync())
            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
        __syncwarp();

        // Maintain linked list
        if constexpr (kDoCreateLinkedList) {
            if (ptx::elect_one_sync())
                channel_linked_list[tma_buffer.get_linked_list_idx_ptr()[master_src_topk_idx]] = i;
            __syncwarp();
        }

        // Issue TMA stores for data
        if (kDoExpand ? (dst_tensor_idx >= 0) : ptx::elect_one_sync()) {
            ptx::tma_store_1d(math::advance_ptr(recv_x, static_cast<int64_t>(dst_tensor_idx) * kNumHiddenBytes),
                              tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
            ptx::tma_store_commit();
        }
        __syncwarp();

        // Store SF
        if constexpr (kNumSFPacks > 0) {
            constexpr auto kNumFullIters = kNumSFPacks / 32;
            const bool do_last_iter = (kNumSFPacks % 32 != 0) and (kNumFullIters * 32 + lane_idx < kNumSFPacks);
            EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");

            // Load into registers
            const auto smem_src_ptr = tma_buffer.get_sf_ptr();
            sf_pack_t reg_src[kNumFullIters + 1];
            #pragma unroll
            for (int k = 0; k < kNumFullIters; ++ k)
                reg_src[k] = smem_src_ptr[k * 32 + lane_idx];
            if (do_last_iter)
                reg_src[kNumFullIters] = smem_src_ptr[kNumFullIters * 32 + lane_idx];

            // Prepare strides
            const auto recv_sf_token_stride_i64 = static_cast<int64_t>(recv_sf_token_stride);
            const auto recv_sf_hidden_stride_i64 = static_cast<int64_t>(recv_sf_hidden_stride);

            // Iterate through all valid indices and store into output buffer
            auto mask = kDoExpand ? ptx::gather(dst_tensor_idx >= 0) : 1;
            while (mask) {
                const int valid_lane_idx = __ffs(mask) - 1;
                const auto gmem_dst = math::advance_ptr<sf_pack_t>(recv_sf,
                    ptx::exchange(dst_tensor_idx, valid_lane_idx) * (recv_sf_token_stride_i64 * sizeof(sf_pack_t)));
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k)
                    gmem_dst[(k * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[k];
                if (do_last_iter)
                    gmem_dst[(kNumFullIters * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[kNumFullIters];
                mask ^= 1 << valid_lane_idx;
            }
        }

        // Store the top-k weights
        if (kDoExpand and recv_topk_weights != nullptr and dst_tensor_idx >= 0) {
            recv_topk_weights[dst_tensor_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        } else if (not kDoExpand and recv_topk_weights != nullptr and lane_idx < kNumTopk) {
            // For backward, weights are optional
            recv_topk_weights[i * kNumTopk + lane_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        }
        __syncwarp();

        // Write source token index
        // And:
        //   - Non-hybrid mode: the source scaleup peer rank index and master top-k lane index
        //   - Hybrid mode: the slot index and master top-k lane index
        constexpr int kMetadataStride = 2 + kNumTopk;
        if (ptx::elect_one_sync()) {
            recv_src_metadata[i * kMetadataStride + 0] = *tma_buffer.get_src_token_global_idx_ptr();
            if constexpr (kNumScaleoutRanks == 1) {
                recv_src_metadata[i * kMetadataStride + 1] = current_rank_idx * kNumTopk + master_src_topk_idx;
            } else {
                recv_src_metadata[i * kMetadataStride + 1] = (i - current_rank_start) * kNumTopk + master_src_topk_idx;
            }
        }
        __syncwarp();

        // Write reduction source indices
        if (kDoExpand and lane_idx < kNumTopk)
            recv_src_metadata[i * kMetadataStride + 2 + lane_idx] = dst_tensor_idx;
        __syncwarp();
    }

    // Maintain linked list's ending
    // Or you can understand it as writing the tail at once
    if constexpr (kDoCreateLinkedList) {
        constexpr int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32);
        const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
        for (int i = global_warp_idx; i < kNumChannels; i += kNumSMs * kNumWarps) {
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                if (const auto k = j * 32 + lane_idx; j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks) {
                    channel_linked_list[
                        *workspace_layout.get_channel_scaleup_tail_ptr(i, k)
                    ] = -1;

                    // Clean for combine usages
                    *workspace_layout.get_channel_scaleup_tail_ptr(i, k) = 0;
                }
            }
            __syncwarp();
        }
    }

    // ============================================================
    // Fused MoE-align (decode / non-expand). Emits grouped-GEMM metadata
    // equivalent to vLLM's globalize + moe_align_block_size + count_and_sort:
    //   - sorted_token_ids: flattened (token*kNumTopk + k) ids grouped by local
    //     expert, each expert region padded up to kAlignM. Dead slots = sentinel
    //     (num_recv*kNumTopk), which the GEMM masks via `offs_token < num_valid`.
    //   - fused_expert_ids: local expert id per kAlignM block, -1 for dead blocks.
    //   - num_tokens_post_pad: padded total.
    // recv_topk_idx[j] already holds the *global* expert id (or -1) written in the
    // copy loop above (globalized there so `_globalize_recv_topk_idx` is folded
    // away too); Phase 3 reads it back and converts to local — no buffer re-walk.
    // ============================================================
    if constexpr (kFuseMoeAlign) {
        static_assert(not kDoExpand, "Fused MoE-align is non-expand only");
        const int n_flat = num_recv_tokens * kNumTopk;
        const int sentinel = n_flat;
        const int num_blocks = num_sorted_slots / kAlignM;
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = kNumSMs * blockDim.x;

        // Barrier 0: Phase 1 (recv_topk_idx + expert_count) done on all blocks.
        grid_arrive_and_wait(grid_barrier + 0, kNumSMs);

        // Phase 2a: sentinel-fill sorted_token_ids and clear expert_ids.
        for (int t = tid; t < num_sorted_slots; t += stride)
            sorted_token_ids[t] = sentinel;
        for (int b = tid; b < num_blocks; b += stride)
            fused_expert_ids[b] = -1;

        // Sanitize the uninitialized recv_topk_idx padding tail [num_recv, allocated)
        // to -1 (folds `_globalize_recv_topk_idx`'s row-mask): the copy loop only
        // wrote rows [0, num_recv); stale rows would otherwise alias valid experts
        // downstream. `n_flat` is the real (actual) count; the buffer is allocated
        // for `num_allocated_recv_tokens` rows (worst case in the cudagraph path).
        const int num_allocated_flat = num_allocated_recv_tokens * kNumTopk;
        for (int t = n_flat + tid; t < num_allocated_flat; t += stride)
            recv_topk_idx[t] = static_cast<topk_idx_t>(-1);

        // Barrier 1: fills visible before block 0 overwrites assigned blocks.
        grid_arrive_and_wait(grid_barrier + 1, kNumSMs);

        // Phase 2b: block 0 computes kAlignM-aligned exclusive prefix offsets,
        // stamps expert_ids per assigned block, and num_tokens_post_pad; then
        // overwrites expert_count[e] with the base offset (Phase 3 write cursor).
        if (blockIdx.x == 0 and threadIdx.x == 0) {
            int base = 0;
            #pragma unroll 1
            for (int e = 0; e < kNumExpertsPerRank; ++ e) {
                const int cnt = expert_count[e];
                const int aligned = ((cnt + kAlignM - 1) / kAlignM) * kAlignM;
                const int start_block = base / kAlignM;
                const int nb = aligned / kAlignM;
                for (int b = 0; b < nb; ++ b)
                    fused_expert_ids[start_block + b] = e;
                expert_count[e] = base;   // cursor start for Phase 3
                base += aligned;
            }
            *num_tokens_post_pad = base;
        }

        // Barrier 2: offsets/cursors visible to all blocks before scatter.
        grid_arrive_and_wait(grid_barrier + 2, kNumSMs);

        // Phase 3: scatter flattened (token,k) ids into their expert's region.
        // recv_topk_idx now holds the *global* expert id (see copy loop); convert
        // back to local for the per-expert write cursor.
        for (int j = tid; j < n_flat; j += stride) {
            const int ge = static_cast<int>(recv_topk_idx[j]);
            if (ge >= 0)
                sorted_token_ids[atomicAdd(expert_count + (ge - expert_start_idx), 1)] = j;
        }
    }
}

}  // namespace deep_ep::elastic
