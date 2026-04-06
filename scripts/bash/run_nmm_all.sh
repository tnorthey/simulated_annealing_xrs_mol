#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXCITATION_FACTOR="${EXCITATION_FACTOR:-0.057}"
TUNING_RATIO="${TUNING_RATIO:-0.5}"
STEP_START="${STEP_START:-18}"
STEP_END="${STEP_END:-39}"

usage() {
    cat <<EOF
Usage: $0

Runs run_parallel.sh for each time-step in [STEP_START..STEP_END].
Each invocation auto-picks a random starting xyz (see run_parallel.sh --help).

Environment variables (defaults shown):
  EXCITATION_FACTOR  $EXCITATION_FACTOR
  TUNING_RATIO       $TUNING_RATIO
  STEP_START         $STEP_START
  STEP_END           $STEP_END

Passed through to run_parallel.sh:
  XYZ_SOURCE_STEP    source time-step for starting xyz selection
  TOP_N              how many top files to consider (default 20)
  RESULTS_DIR        directory to search (default results)
  N_WORKERS          parallel workers per step (default 64)
EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

for time_step in $(seq "$STEP_START" "$STEP_END"); do
    echo "=== time-step $time_step ==="
    python3 run.py --run-id "$time_step" --target-file "nmm_data/target_"$time_step".dat" --excitation-factor "$EXCITATION_FACTOR" --tuning-ratio-target "$TUNING_RATIO"
done
