#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for i in {17..42}
do
    RESULTS_DIR="results_abi_phi0p1_from_gs" TUNING_RATIO=0.10 STARTING_XYZ="chd+_data/CCSD-neut.xyz" N_WORKERS=64 EXCITATION_FACTOR=0.628 TARGET_FILE="chd+_data/eirik_data_"$i".dat" "$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$i"
done
