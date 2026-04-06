#!/usr/bin/env bash
# Initial timestep: always start from a user-chosen XYZ (no pool from a prior step).
# Copies it to ${RESULTS_DIR}/${time_step}_mean.xyz and runs one CUDA job (default 1024 chains).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CONFIG="${CONFIG:-input.toml}"
GPU_CHAINS="${GPU_CHAINS:-1024}"
EXCITATION_FACTOR="${EXCITATION_FACTOR:-1.0}"
TUNING_RATIO="${TUNING_RATIO:-0.5}"
STARTING_XYZ="${STARTING_XYZ:-}"

usage() {
    cat <<EOF
Usage:
  $0 <time_step> <starting_xyz> [excitation_factor] [tuning_ratio_target]
  STARTING_XYZ=path $0 <time_step> [excitation_factor] [tuning_ratio_target]

Run the first (or any) timestep from a fixed starting geometry — same GPU invocation
as run_gpu_from_previous_timestep.sh (CUDA, GPU_CHAINS).

Copies the starting structure to:
  \${RESULTS_DIR}/<time_step>_mean.xyz
then runs run.py with --start-xyz-file pointing at that file.

Positional:
  time_step           Passed as --run-id (e.g. 01)
  starting_xyz       Path to initial XYZ (optional if env STARTING_XYZ is set to an existing file)

Environment (defaults):
  STARTING_XYZ      Alternative to the second positional argument
  RESULTS_DIR=${RESULTS_DIR}
  TARGET_FILE         If unset, nmm_data/target_<time_step>.dat (two-digit padded when numeric)
  CONFIG=${CONFIG}
  GPU_CHAINS=${GPU_CHAINS}
  EXCITATION_FACTOR=${EXCITATION_FACTOR}
  TUNING_RATIO=${TUNING_RATIO}
  PYTHON=${PYTHON}
EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

if [[ $# -lt 1 ]]; then
    echo "ERROR: time_step is required" >&2
    usage
fi

time_step="$1"

if [[ -n "${2:-}" && -f "$2" ]]; then
    start_xyz="$2"
    excitation_factor="${3:-$EXCITATION_FACTOR}"
    tuning_ratio_target="${4:-$TUNING_RATIO}"
elif [[ -n "$STARTING_XYZ" && -f "$STARTING_XYZ" ]]; then
    start_xyz="$STARTING_XYZ"
    excitation_factor="${2:-$EXCITATION_FACTOR}"
    tuning_ratio_target="${3:-$TUNING_RATIO}"
else
    echo "ERROR: provide an initial structure: second positional argument, or set STARTING_XYZ to an existing file." >&2
    exit 1
fi

if [[ "$time_step" =~ ^[0-9]+$ ]]; then
    ts_padded=$(printf '%02d' "$((10#$time_step))")
else
    ts_padded="$time_step"
fi

TARGET_FILE="${TARGET_FILE:-nmm_data/target_${ts_padded}.dat}"
mean_out="${RESULTS_DIR}/${ts_padded}_mean.xyz"

mkdir -p "$RESULTS_DIR"

echo "=== GPU initial run (fixed STARTING_XYZ) ==="
echo "  time_step (run-id)=$ts_padded"
echo "  source: $start_xyz"
echo "  mean copy: $mean_out"

cp -f "$start_xyz" "$mean_out"

echo "  launching: $PYTHON run.py --gpu-backend cuda --gpu-chains $GPU_CHAINS ..."
"$PYTHON" run.py \
    --config "$CONFIG" \
    --run-id "$ts_padded" \
    --start-xyz-file "$mean_out" \
    --target-file "$TARGET_FILE" \
    --excitation-factor "$excitation_factor" \
    --tuning-ratio-target "$tuning_ratio_target" \
    --gpu-backend cuda \
    --gpu-chains "$GPU_CHAINS"
