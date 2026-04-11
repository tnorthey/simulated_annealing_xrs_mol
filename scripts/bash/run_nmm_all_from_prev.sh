#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

#for i in {19..42}
for i in {43..60}
do
    RESULTS_DIR="results_nmm_phi0p3_from_prev" TUNING_RATIO=0.30 START_FROM_PREV_MEAN=1 TOP_N=200 N_WORKERS=64 EXCITATION_FACTOR=0.057 TARGET_FILE="nmm_data/target_"$i".dat" "$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$i"
done
