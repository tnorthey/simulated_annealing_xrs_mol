#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

i=18

RESULTS_DIR="results_nmm_phi0p3_from_prev" STARTING_XYZ="xyz/nmm_opt.xyz" TUNING_RATIO=0.30 N_WORKERS=64 EXCITATION_FACTOR=0.057 TARGET_FILE="nmm_data/target_"$i".dat" "$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$i"
