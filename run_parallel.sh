#!/bin/bash
set -euo pipefail

N_WORKERS="${N_WORKERS:-64}"
RESULTS_DIR="${RESULTS_DIR:-results}"
XYZ_SOURCE_STEP="${XYZ_SOURCE_STEP:-01}"
TOP_N="${TOP_N:-20}"

usage() {
    cat <<EOF
Usage: $0 <time_step> [excitation_factor] [tuning_ratio_target]

Launch N_WORKERS parallel instances of run.py for a single time-step.
A random starting xyz is chosen from the first TOP_N files (by ls order)
matching \${RESULTS_DIR}/<XYZ_SOURCE_STEP padded>_???.????????.xyz.

Positional arguments:
  time_step           Time-step identifier (passed as --run-id)
  excitation_factor   (default: 1.0)
  tuning_ratio_target (default: 0.5)

Environment variables (defaults shown):
  N_WORKERS          $N_WORKERS
  RESULTS_DIR        $RESULTS_DIR
  XYZ_SOURCE_STEP    $XYZ_SOURCE_STEP   (which time-step to pick starting xyz from)
  TOP_N              $TOP_N    (consider first N files by ls order)
EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

if [[ $# -lt 1 ]]; then
    echo "ERROR: time_step is required" >&2
    usage
fi

time_step="$1"
excitation_factor="${2:-1.0}"
tuning_ratio_target="${3:-0.5}"

# Pick a random starting xyz from the source time-step's results
ts_padded=$(printf '%02d' "$XYZ_SOURCE_STEP")
pattern="${RESULTS_DIR}/${ts_padded}_???.????????.xyz"

candidates=( $(ls -1 $pattern 2>/dev/null | head -n "$TOP_N") ) || true

if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "ERROR: no xyz files matching '$pattern' in ${RESULTS_DIR}/" >&2
    exit 1
fi

echo "  pool=${#candidates[@]}/${TOP_N} (from step $ts_padded)"
echo "  workers=$N_WORKERS | excitation=$excitation_factor | tuning=$tuning_ratio_target"

for (( worker=0; worker<N_WORKERS; worker++ )); do
    idx=$(( RANDOM % ${#candidates[@]} ))
    starting_xyz="${candidates[$idx]}"
    echo "  worker $worker: xyz=$starting_xyz"
    python run.py \
        --run-id "$time_step" \
        --start-xyz-file "$starting_xyz" \
        --target-file "nmm_data/target_${time_step}.dat" \
        --excitation-factor "$excitation_factor" \
        --tuning-ratio-target "$tuning_ratio_target" &
done

wait
echo "  All $N_WORKERS workers finished for time-step $time_step"
