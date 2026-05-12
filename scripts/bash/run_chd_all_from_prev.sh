#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

number_flag=$1  # e.g. "64_1" "64_2" ...

topn=50
traj=099
qmax=8
nr=4
desc="sdf_boltzmann_1000K_"$number_flag""
results_dir="results_chd_traj_"$traj"_phi0p5_from_prev_nr"$nr"_qmax"$qmax"_"$desc""
phi=0.50

i=00 # Run the first step (00)
RESULTS_DIR=$results_dir STARTING_XYZ="xyz/start.xyz" TUNING_RATIO=$phi N_WORKERS=64 EXCITATION_FACTOR=1.0 TARGET_FILE="target_traj"$traj"/target_"$i".xyz" "$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$i"

# Run subsequent steps
for i in {01..75}
do
    RESULTS_DIR=$results_dir TUNING_RATIO=$phi START_FROM_PREV_MEAN=1 TOP_N=$topn N_WORKERS=64 EXCITATION_FACTOR=1.0 TARGET_FILE="target_traj"$traj"/target_"$i".xyz" "$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$i"
done
