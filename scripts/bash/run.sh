#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

time_step=$1

python run.py \
        --run-id "$time_step" \
        --start-xyz-file "xyz/start.xyz" \
        --target-file "data/eirik_data_$time_step.dat" \
        --reference-dat-file "data/chd_reference.dat" \
        --excitation-factor 0.628
