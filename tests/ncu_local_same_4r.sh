#!/bin/bash
# Profile rank 0 of a 4-rank local-same dispatch with ncu.
#
# Usage: bash tests/ncu_local_same_4r.sh [output_path]
#
# Approach: launch torchrun normally but wrap only rank 0 with ncu
# by using a helper that checks LOCAL_RANK.
set -euo pipefail

OUTPUT="${1:-/workspace/ncu_local_same_4r}"

torchrun --nproc-per-node=4 tests/ncu_local_same_4r_wrapper.py "$OUTPUT"
