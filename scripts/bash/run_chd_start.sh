#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

i=00
traj=090
qmax=8

RESULTS_DIR="results_chd_traj_"$traj"_phi0p5_from_prev_nr2_qmax"$qmax"" STARTING_XYZ="xyz/start.xyz" TUNING_RATIO=0.50 N_WORKERS=64 EXCITATION_FACTOR=1.0 TARGET_FILE="target_traj"$traj"/target_"$i".xyz" "$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$i"
