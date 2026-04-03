#!/bin/bash
set -euo pipefail

N_WORKERS="${N_WORKERS:-64}"
RESULTS_DIR="${RESULTS_DIR:-results}"
XYZ_SOURCE_STEP="${XYZ_SOURCE_STEP:-01}"
TOP_N="${TOP_N:-20}"
STARTING_XYZ="${STARTING_XYZ:-}"
PYTHON="${PYTHON:-python3}"
# Override the --target-file path passed to run.py for every worker.
# If unset, the default uses TARGET_FILE_TEMPLATE.
TARGET_FILE="${TARGET_FILE:-}"
TARGET_FILE_TEMPLATE="${TARGET_FILE_TEMPLATE:-nmm_data/target_{time_step}.dat}"
EXTRA_RUN_PY_ARGS="${EXTRA_RUN_PY_ARGS:-}"

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
  STARTING_XYZ       (unset)  (if set, use this xyz file for all workers
                               instead of picking randomly from the pool)
  TARGET_FILE       (unset)  (if set, passed as --target-file for all workers)
  TARGET_FILE_TEMPLATE $TARGET_FILE_TEMPLATE (used when TARGET_FILE is unset)
  EXTRA_RUN_PY_ARGS  (unset)  extra args appended to run.py (e.g. --gpu-backend cpu)
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

if [[ -n "$STARTING_XYZ" ]]; then
    if [[ ! -f "$STARTING_XYZ" ]]; then
        echo "ERROR: STARTING_XYZ='$STARTING_XYZ' does not exist" >&2
        exit 1
    fi
    echo "  starting_xyz=$STARTING_XYZ (fixed for all workers)"
    echo "  workers=$N_WORKERS | excitation=$excitation_factor | tuning=$tuning_ratio_target"
else
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
fi

for (( worker=0; worker<N_WORKERS; worker++ )); do
    if [[ -n "$STARTING_XYZ" ]]; then
        starting_xyz="$STARTING_XYZ"
    else
        idx=$(( RANDOM % ${#candidates[@]} ))
        starting_xyz="${candidates[$idx]}"
    fi
    echo "  worker $worker: xyz=$starting_xyz"
    if [[ -n "$TARGET_FILE" ]]; then
        target_file="$TARGET_FILE"
    else
        target_file="${TARGET_FILE_TEMPLATE//{time_step}/$time_step}"
    fi
    "$PYTHON" run.py \
        --run-id "$time_step" \
        --start-xyz-file "$starting_xyz" \
        --target-file "$target_file" \
        --excitation-factor "$excitation_factor" \
        --tuning-ratio-target "$tuning_ratio_target" \
        ${EXTRA_RUN_PY_ARGS} &
done

wait
echo "  All $N_WORKERS workers finished for time-step $time_step"
