#!/bin/bash
set -euo pipefail

EXCITATION_RATIO="${EXCITATION_RATIO:-0.057}"
TUNING_RATIO="${TUNING_RATIO:-0.9}"
RESULTS_DIR="${RESULTS_DIR:-results}"
STEP_START="${STEP_START:-18}"
STEP_END="${STEP_END:-48}"

usage() {
    cat <<EOF
Usage: $0

Runs run_parallel.sh for each time-step in [STEP_START..STEP_END].
For each step, a random starting xyz is chosen from the first 20 files
matching \${RESULTS_DIR}/<zero-padded step>_???.????????.xyz.

Environment variables (defaults shown):
  EXCITATION_RATIO  $EXCITATION_RATIO
  TUNING_RATIO      $TUNING_RATIO
  RESULTS_DIR       $RESULTS_DIR
  STEP_START        $STEP_START
  STEP_END          $STEP_END
EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

pick_random_xyz() {
    local ts_padded="$1"
    local pattern="${RESULTS_DIR}/${ts_padded}_???.????????.xyz"

    local candidates
    candidates=( $(ls -1 $pattern 2>/dev/null | head -n 20) ) || true

    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "ERROR: no xyz files matching '$pattern' in ${RESULTS_DIR}/" >&2
        exit 1
    fi

    local idx=$(( RANDOM % ${#candidates[@]} ))
    echo "${candidates[$idx]}"
}

for time_step in $(seq "$STEP_START" "$STEP_END"); do
    ts_padded=$(printf '%02d' "$time_step")
    starting_xyz=$(pick_random_xyz "$ts_padded")

    echo "=== time-step $ts_padded | xyz: $starting_xyz ==="
    ./run_parallel.sh "$time_step" "$starting_xyz" "$EXCITATION_RATIO" "$TUNING_RATIO"
done
