#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

traj=094

for i in {75..01}
do
    RESULTS_DIR="results_chd_traj_"$traj"_phi0p5_from_prev" TUNING_RATIO=0.50 START_FROM_PREV_MEAN=1 TOP_N=200 N_WORKERS=64 EXCITATION_FACTOR=1.0 TARGET_FILE="target_traj"$traj"/target_"$i".xyz" "$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$i"
done
