#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/layout.cuh>

#include <deep_ep/impls/combine_utils.cuh>


namespace deep_ep::elastic {

template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          // TODO: merge these two variables into one (ensure the whole file does not contain "scaleup")
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          int kNumRanks = kNumScaleoutRanks == 1 ? kNumScaleupRanks : kNumScaleoutRanks,
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>()>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,
                             float* combined_topk_weights,
                             topk_idx_t* combined_topk_idx,
                             void* recv_buffer,
                             void* bias_0, void* bias_1,
                             const int num_combined_tokens,
                             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    constexpr int kNumExpertsPerRank = kNumExperts / (kNumScaleupRanks * kNumScaleoutRanks);
    EP_STATIC_ASSERT(kNumExperts % (kNumScaleupRanks * kNumScaleoutRanks) == 0, "Invalid number of experts or ranks");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;   // NOTES: Here we prioritize distributing tasks to different SMs to ensure that the last wave is evenly concentrated on each SM.

    using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;
    constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);
    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
    constexpr int kHiddenVecPerWarp = kUnrollFactor * 32;
    constexpr int kNumHiddenStages = kHiddenVec / kHiddenVecPerWarp;

    // For a small number of output tokens, the original mapping assigns one
    // token to one warp.  A Kimi K3 decode step can have one real token, so
    // that leaves almost the entire 32-warp epilogue idle while one warp
    // reduces the complete hidden vector.  Give one token one warp per
    // hidden stage instead.  Each stage is written to a disjoint output
    // slice, so the warps do not need to communicate with one another.
    constexpr bool kCanSplitToken = kNumHiddenStages > 1 and kNumHiddenStages <= kNumWarps;
    constexpr int kWarpsPerSplitToken = kCanSplitToken ? kNumHiddenStages : 1;
    constexpr int kTotalWarps = kNumSMs * kNumWarps;
    const bool use_split_tokens = kCanSplitToken and
        num_combined_tokens * kWarpsPerSplitToken <= kTotalWarps;

    // Load buffers from scale-out or scale-up ranks
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto comm_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    const auto comm_buffer = layout::BufferLayout<false>(
        comm_token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, recv_buffer);

    // Store buffers
    const auto output_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);
    const auto output_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, combined_x);
    const auto tma_buffer = layout::BufferLayout<false>(output_token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);

    // Bias layout
    const auto bias_0_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_0);
    const auto bias_1_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_1);

    // Will block until the main combine kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();

    auto prepare_topk = [&](const int& token_idx, int (&topk_slot_idx)[kNumTokensInLayout],
                            int& stored_dst_rank_idx) {
        // Preprocess all indices
        stored_dst_rank_idx = -1;
        int stored_dst_expert_idx = -1;
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        if (lane_idx < kNumTopk) {
            stored_dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);
            stored_dst_rank_idx = stored_dst_expert_idx >= 0 ?
                stored_dst_expert_idx / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : kNumExpertsPerScaleout) : -1;
        }
        __syncwarp();

        // Sort valid top-k indices to front
        const auto [should_deduplicate, deduplicate_key] = [&]() -> std::pair<bool, int> {
            if constexpr (kUseExpandedLayout and not kAllowMultipleReduction) {
                // Activations are never reduced before
                return {false, 0};
            } else if constexpr (kNumScaleoutRanks != 1 and not kUseExpandedLayout and not kAllowMultipleReduction) {
                // Hybrid mode without expanded layout and multiple reduction. Should deduplicate on a per-rank basis
                return {true, stored_dst_expert_idx >= 0 ? stored_dst_expert_idx / kNumExpertsPerRank : -1};
            } else {
                // Should deduplicate on a per-rank (for non-hybrid mode) or a per-scale-rank (for hybrid mode) basis
                return {true, stored_dst_rank_idx};
            }
        }();
        auto reduce_valid_mask = should_deduplicate ?
            ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
            ptx::gather(stored_dst_rank_idx >= 0);
        compute_topk_slots(
            topk_slot_idx, reduce_valid_mask,
            [=](const int& idx) {
                return kUseRankLayout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;
            }
        );
    };

    auto write_topk_weights = [&](const int& token_idx, const int& stored_dst_rank_idx) {
        if (combined_topk_weights != nullptr) {
            const auto master_lane_idx = ptx::get_master_lane_idx(ptx::match(stored_dst_rank_idx));
            if (lane_idx < kNumTopk) {
                float value = 0;
                if (stored_dst_rank_idx >= 0) {
                    const auto dst_ptr = comm_buffer
                        .get_rank_buffer(kUseRankLayout ? stored_dst_rank_idx : master_lane_idx)
                        .get_token_buffer(token_idx).get_topk_weights_ptr() + lane_idx;
                    value = *dst_ptr;
                }
                combined_topk_weights[token_idx * kNumTopk + lane_idx] = value;
            }
            __syncwarp();
        }
    };

    if (use_split_tokens) {
        // Consecutive global warps cooperate on one token.  The current
        // global-warp ordering intentionally puts these warps on different
        // SM blocks, which gives the small reduction more memory bandwidth.
        const int token_idx = global_warp_idx / kWarpsPerSplitToken;
        if (token_idx < num_combined_tokens) {
            const int hidden_stage_idx = global_warp_idx % kWarpsPerSplitToken;
            const int hidden_vec_offset = hidden_stage_idx * kHiddenVecPerWarp;
            const int hidden_byte_offset = hidden_vec_offset * sizeof(combine_vec_t);
            int stored_dst_rank_idx = -1;
            int topk_slot_idx[kNumTokensInLayout];
            prepare_topk(token_idx, topk_slot_idx, stored_dst_rank_idx);

            auto tma_dst = math::advance_ptr<combine_vec_t>(
                tma_buffer.get_base_ptr(), hidden_byte_offset);
            combine_reduce<kHiddenVecPerWarp, kUnrollFactor, kNumTokensInLayout>(
                lane_idx, topk_slot_idx, tma_dst,
                /* Get source base */ [=](const int& slot_idx) {
                    return math::advance_ptr<combine_vec_t>(
                        static_cast<combine_vec_t*>(comm_buffer.get_rank_buffer(slot_idx)
                            .get_token_buffer(token_idx).get_base_ptr()), hidden_byte_offset);
                },
                /* Wait buffer release */ [=]() {
                    ptx::tma_store_wait();
                    __syncwarp();
                },
                /* Bias 0 */ bias_0 == nullptr ? nullptr : math::advance_ptr<combine_vec_t>(
                    bias_0_buffer.get_token_buffer(token_idx).get_base_ptr(), hidden_byte_offset),
                /* Bias 1 */ bias_1 == nullptr ? nullptr : math::advance_ptr<combine_vec_t>(
                    bias_1_buffer.get_token_buffer(token_idx).get_base_ptr(), hidden_byte_offset)
            );
            ptx::tma_store_fence();
            __syncwarp();

            if (ptx::elect_one_sync()) {
                ptx::tma_store_1d(
                    math::advance_ptr(output_buffer.get_token_buffer(token_idx).get_base_ptr(), hidden_byte_offset),
                    tma_dst, kHiddenVecPerWarp * sizeof(combine_vec_t));
                ptx::tma_store_commit();
            }
            __syncwarp();

            // Only the first stage writes the token's top-k weights.
            if (hidden_stage_idx == 0)
                write_topk_weights(token_idx, stored_dst_rank_idx);
        }
    } else {
        // Original one-warp-per-token path for sufficiently large batches.
        for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; token_idx += kNumWarps * kNumSMs) {
            int stored_dst_rank_idx = -1;
            int topk_slot_idx[kNumTokensInLayout];
            prepare_topk(token_idx, topk_slot_idx, stored_dst_rank_idx);

            combine_reduce<kHiddenVec, kUnrollFactor, kNumTokensInLayout>(
            lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
            /* Get source base */ [=](const int& slot_idx) {
                return static_cast<combine_vec_t*>(
                    comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr());
            },
            /* Wait buffer release */ [=]() {
                ptx::tma_store_wait();
                __syncwarp();
            },
            /* Bias 0 */ bias_0 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_0_buffer.get_token_buffer(token_idx).get_base_ptr()),
            /* Bias 1 */ bias_1 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_1_buffer.get_token_buffer(token_idx).get_base_ptr())
            );
            ptx::tma_store_fence();
            __syncwarp();

            // Issue TMA copy
            if (ptx::elect_one_sync()) {
                ptx::tma_store_1d(output_buffer.get_token_buffer(token_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), kNumHiddenBytes);
                ptx::tma_store_commit();
            }
            __syncwarp();

            write_topk_weights(token_idx, stored_dst_rank_idx);
        }
    }
}

}  // deep_ep::elastic
