#!/usr/bin/env bash
set -euo pipefail

if [[ "${RANK}" == "8" ]]; then
    exec /opt/nvidia/nsight-compute/2025.4.1/ncu \
        --metrics "gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_active.avg,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,smsp__warp_issue_stalled_barrier_per_warp_active,smsp__warp_issue_stalled_branch_resolving_per_warp_active,smsp__warp_issue_stalled_dispatch_stall_per_warp_active,smsp__warp_issue_stalled_drain_per_warp_active,smsp__warp_issue_stalled_gmma_per_warp_active,smsp__warp_issue_stalled_imc_miss_per_warp_active,smsp__warp_issue_stalled_lg_throttle_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active,smsp__warp_issue_stalled_membar_per_warp_active,smsp__warp_issue_stalled_mio_throttle_per_warp_active,smsp__warp_issue_stalled_misc_per_warp_active,smsp__warp_issue_stalled_no_instruction_per_warp_active,smsp__warp_issue_stalled_not_selected_per_warp_active,smsp__warp_issue_stalled_selected_per_warp_active,smsp__warp_issue_stalled_short_scoreboard_per_warp_active,smsp__warp_issue_stalled_sleeping_per_warp_active,smsp__warp_issue_stalled_tex_throttle_per_warp_active,smsp__warp_issue_stalled_wait_per_warp_active" \
        --kernel-name 'regex:.*combine_reduce_epilogue_impl.*' \
        --launch-skip 5 \
        --launch-count 1 \
        --replay-mode kernel \
        --cache-control none \
        --clock-control none \
        --force-overwrite \
        --export /tmp/deepep-ncu/combine-rank8-stalls \
        /usr/bin/python3 /tmp/repro_dp4_tp8_combine_only.py
fi

exec /usr/bin/python3 /tmp/repro_dp4_tp8_combine_only.py
