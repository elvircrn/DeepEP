#!/bin/bash
# Profile rank 0 of a multi-rank local-same dispatch with ncu.
# Other ranks run un-profiled but participate in the distributed group.
#
# Usage: bash tests/ncu_local_same.sh [num_gpus] [output_path]
#
# This avoids the "Unexpected number of profiled kernels" error that
# occurs with --target-processes all, because only rank 0 is replayed.
set -euo pipefail

NGPU="${1:-4}"
OUTPUT="${2:-/workspace/ncu_local_same_${NGPU}r}"
PORT=29600
SCRIPT="tests/ncu_local_same.py"

# Launch non-profiled ranks in the background
pids=()
for ((i=1; i<NGPU; i++)); do
    MASTER_ADDR=localhost MASTER_PORT=$PORT \
    RANK=$i LOCAL_RANK=$i WORLD_SIZE=$NGPU \
    CUDA_VISIBLE_DEVICES=$i \
    python "$SCRIPT" &
    pids+=($!)
done

# Profile rank 0
MASTER_ADDR=localhost MASTER_PORT=$PORT \
RANK=0 LOCAL_RANK=0 WORLD_SIZE=$NGPU \
CUDA_VISIBLE_DEVICES=0 \
ncu -f --set full --replay-mode application \
    -k dispatch --launch-skip 3 --launch-count 1 \
    -o "$OUTPUT" \
    python "$SCRIPT"

# Wait for background ranks
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo "Report: ${OUTPUT}.ncu-rep"
