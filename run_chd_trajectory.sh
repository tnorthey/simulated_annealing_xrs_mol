#!/usr/bin/env bash
# Full CHD trajectory minimisation on the GPU.
#
# One shared results directory. Outer loop is decreasing tuning_ratio_target
# (phi); inner loop is trajectory frames. Later phi passes keep earlier .xyz
# files and start from the best structures found so far.
#
# The first frame of the first phi starts from STARTING_XYZ via run_gpu_start.sh.
# Every other launch uses run_gpu_from_previous_timestep.sh with
# START_FROM_TRADIUS_RANDOM=1 and TRADIUS_N=2 (each CUDA chain draws a real
# low-f structure from timesteps t+/-1..t+/-N).
#
# Usage:
#   ./run_chd_trajectory.sh [description]
#   ./run_chd_trajectory.sh --help
#
# Environment (defaults shown):
#   TRAJ=094
#   PHI_VALUES="0.5 0.45 0.4 0.35 0.3"   space- or comma-separated, decreasing
#   EXCITATION=1.0
#   QMAX=8                                qlen = int(10*QMAX+1)
#   STEP_START=0
#   STEP_END=75
#   TOP_N=200
#   TRADIUS_N=2
#   GPU_CHAINS=512
#   C_TUNING_INITIAL=0.1                  passed as --c-tuning-initial
#   STARTING_XYZ=xyz/start.xyz            first frame of the first phi only
#   CONFIG=input.toml
#   EXTRA_RUN_PY_ARGS                     appended after --qmax/--qlen/--c-tuning-initial
#
# Results directory (phi is not part of the name):
#   results_chd_traj_${TRAJ}_qmax${QMAX}[_${description}]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
    sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

DESCRIPTION="${1:-}"

TRAJ="${TRAJ:-094}"
PHI_VALUES="${PHI_VALUES:-0.5 0.45 0.4 0.35 0.3}"
EXCITATION="${EXCITATION:-1.0}"
QMAX="${QMAX:-8}"
STEP_START="${STEP_START:-0}"
STEP_END="${STEP_END:-75}"
TOP_N="${TOP_N:-200}"
TRADIUS_N="${TRADIUS_N:-2}"
GPU_CHAINS="${GPU_CHAINS:-512}"
C_TUNING_INITIAL="${C_TUNING_INITIAL:-0.1}"
STARTING_XYZ="${STARTING_XYZ:-xyz/start.xyz}"
CONFIG="${CONFIG:-input.toml}"
USER_EXTRA_RUN_PY_ARGS="${EXTRA_RUN_PY_ARGS:-}"

PHI_VALUES="${PHI_VALUES//,/ }"
# shellcheck disable=SC2206
PHI_LIST=($PHI_VALUES)
if [[ ${#PHI_LIST[@]} -eq 0 ]]; then
    echo "ERROR: PHI_VALUES is empty" >&2
    exit 1
fi

if [[ ! "$STEP_START" =~ ^[0-9]+$ || ! "$STEP_END" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STEP_START and STEP_END must be non-negative integers" >&2
    exit 1
fi
if [[ "$STEP_START" -gt "$STEP_END" ]]; then
    echo "ERROR: STEP_START ($STEP_START) is greater than STEP_END ($STEP_END)" >&2
    exit 1
fi
if [[ ! "$TRADIUS_N" =~ ^[0-9]+$ || "$TRADIUS_N" -lt 1 ]]; then
    echo "ERROR: TRADIUS_N must be an integer >= 1 (got $TRADIUS_N)" >&2
    exit 1
fi
if [[ ! -f "$STARTING_XYZ" ]]; then
    echo "ERROR: STARTING_XYZ='$STARTING_XYZ' not found" >&2
    exit 1
fi

QLEN=$(awk -v q="$QMAX" 'BEGIN { print int(10 * q + 1) }')
export GPU_CHAINS
export CONFIG
export EXTRA_RUN_PY_ARGS="--qmax ${QMAX} --qlen ${QLEN} --c-tuning-initial ${C_TUNING_INITIAL}${USER_EXTRA_RUN_PY_ARGS:+ ${USER_EXTRA_RUN_PY_ARGS}}"

if [[ -n "$DESCRIPTION" ]]; then
    RESULTS_DIR="results_chd_traj_${TRAJ}_qmax${QMAX}_${DESCRIPTION}"
else
    RESULTS_DIR="results_chd_traj_${TRAJ}_qmax${QMAX}"
fi
export RESULTS_DIR

first_phi="${PHI_LIST[0]}"
first_step=$(printf '%02d' "$((10#$STEP_START))")

echo "=== CHD trajectory minimisation (GPU) ==="
echo "  results:     $RESULTS_DIR"
echo "  traj:        $TRAJ  frames ${first_step}..$(printf '%02d' "$((10#$STEP_END))")"
echo "  phi values:  ${PHI_LIST[*]}"
echo "  excitation:  $EXCITATION"
echo "  qmax/qlen:   $QMAX / $QLEN"
echo "  c_tuning:    $C_TUNING_INITIAL"
echo "  GPU_CHAINS:  $GPU_CHAINS"
echo "  TOP_N:       $TOP_N"
echo "  TRADIUS_N:   $TRADIUS_N  (START_FROM_TRADIUS_RANDOM)"
echo "  first start: $STARTING_XYZ  (phi=$first_phi frame=$first_step only)"
echo "  config:      $CONFIG"
echo "  extra args:  $EXTRA_RUN_PY_ARGS"

for step in $(seq "$STEP_START" "$STEP_END"); do
    ts=$(printf '%02d' "$((10#$step))")
    target="xyz/target_traj${TRAJ}/target_${ts}.xyz"
    if [[ ! -f "$target" ]]; then
        echo "ERROR: missing target file $target" >&2
        exit 1
    fi
done

for phi in "${PHI_LIST[@]}"; do
    echo "=== phi (tuning_ratio_target)=$phi ==="
    for step in $(seq "$STEP_START" "$STEP_END"); do
        ts=$(printf '%02d' "$((10#$step))")
        target="xyz/target_traj${TRAJ}/target_${ts}.xyz"
        if [[ "$phi" == "$first_phi" && "$ts" == "$first_step" ]]; then
            echo "--- phi=$phi frame=$ts  run_gpu_start.sh  STARTING_XYZ=$STARTING_XYZ ---"
            STARTING_XYZ="$STARTING_XYZ" TARGET_FILE="$target" \
                "$REPO_ROOT/scripts/bash/run_gpu_start.sh" "$ts" "$EXCITATION" "$phi"
        else
            echo "--- phi=$phi frame=$ts  run_gpu_from_previous_timestep.sh  TRADIUS_N=$TRADIUS_N ---"
            # Clear STARTING_XYZ so the helper does not ignore the tradius pool.
            STARTING_XYZ="" START_FROM_TRADIUS_RANDOM=1 TRADIUS_N="$TRADIUS_N" \
                TOP_N="$TOP_N" TARGET_FILE="$target" \
                "$REPO_ROOT/scripts/bash/run_gpu_from_previous_timestep.sh" "$ts" "$EXCITATION" "$phi"
        fi
    done
done
