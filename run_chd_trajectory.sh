#!/usr/bin/env bash
# Full CHD trajectory minimisation on the GPU.
#
# One shared results directory. Outer loop is decreasing tuning_ratio_target
# (phi); inner loop is trajectory frames. Later phi passes keep earlier results
# and start from the best structures found so far.
#
# The first frame of the first phi starts from STARTING_XYZ via run_gpu_start.sh.
# Every other launch uses run_gpu_from_previous_timestep.sh with
# START_FROM_TRADIUS_RANDOM=1. The first phi uses TRADIUS_FIRST (default 1);
# every later phi uses TRADIUS_LATER (default 4). Each CUDA chain draws a real
# low-f structure from timesteps t+/-1..t+/-N.
#
# Usage:
#   ./run_chd_trajectory.sh [--phi 0.5,0.4,0.3] [description]
#   ./run_chd_trajectory.sh --help
#
# Options:
#   --phi LIST    tuning_ratio_target values, comma- or space-separated.
#                 Overrides PHI_VALUES. Default: 0.5 0.45 0.4 0.35 0.3
#
# Environment (defaults shown):
#   TRAJ=094
#   PHI_VALUES="0.5 0.45 0.4 0.35 0.3"   used when --phi is omitted
#   EXCITATION=1.0
#   QMAX=8                                qlen = int(10*QMAX+1)
#   STEP_START=0
#   STEP_END=75
#   TOP_N=200
#   TRADIUS_FIRST=1                       radius for the first phi pass
#   TRADIUS_LATER=4                       radius for every later phi pass
#   GPU_CHAINS=512
#   C_TUNING_INITIAL=0.1                  passed as --c-tuning-initial
#   STARTING_XYZ=xyz/start.xyz            first frame of the first phi only
#   CONFIG=input.toml
#   EXTRA_RUN_PY_ARGS                     appended after --qmax/--qlen/--c-tuning-initial
#
# NPZ mode (options.npz_mode_bool in CONFIG): chain xyz/dat files are not kept.
# Structures go to ${RESULTS_DIR}/trajectory.npz (archive_top_m lowest f_xray per frame).
# TOP_N must be <= archive_top_m, because later frames restart from that archive.
#
# Results directory (phi is not part of the name):
#   results_chd_traj_${TRAJ}_qmax${QMAX}[_${description}]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
    sed -n '2,43p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

DESCRIPTION=""
PHI_FROM_CLI=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --phi)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --phi requires a value (e.g. --phi 0.5,0.4,0.3)" >&2
                exit 1
            fi
            PHI_VALUES="$2"
            PHI_FROM_CLI=1
            shift 2
            ;;
        --phi=*)
            PHI_VALUES="${1#--phi=}"
            PHI_FROM_CLI=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -n "$DESCRIPTION" ]]; then
                echo "ERROR: unexpected argument: $1" >&2
                exit 1
            fi
            DESCRIPTION="$1"
            shift
            ;;
    esac
done
# A positional description may follow --.
if [[ $# -gt 0 ]]; then
    if [[ -n "$DESCRIPTION" || $# -gt 1 ]]; then
        echo "ERROR: unexpected argument: $1" >&2
        exit 1
    fi
    DESCRIPTION="$1"
fi

TRAJ="${TRAJ:-094}"
if [[ "$PHI_FROM_CLI" -eq 0 ]]; then
    PHI_VALUES="${PHI_VALUES:-0.5 0.45 0.4 0.35 0.3}"
fi
EXCITATION="${EXCITATION:-1.0}"
QMAX="${QMAX:-8}"
STEP_START="${STEP_START:-0}"
STEP_END="${STEP_END:-75}"
TOP_N="${TOP_N:-200}"
TRADIUS_FIRST="${TRADIUS_FIRST:-1}"
TRADIUS_LATER="${TRADIUS_LATER:-4}"
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
if [[ ! "$TRADIUS_FIRST" =~ ^[0-9]+$ || "$TRADIUS_FIRST" -lt 1 ]]; then
    echo "ERROR: TRADIUS_FIRST must be an integer >= 1 (got $TRADIUS_FIRST)" >&2
    exit 1
fi
if [[ ! "$TRADIUS_LATER" =~ ^[0-9]+$ || "$TRADIUS_LATER" -lt 1 ]]; then
    echo "ERROR: TRADIUS_LATER must be an integer >= 1 (got $TRADIUS_LATER)" >&2
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

PYTHON="${PYTHON:-python3}"
ARCHIVE_TOP_M="$("$PYTHON" "$REPO_ROOT/modules/trajectory_archive.py" archive-top-m --config "$CONFIG")"
if [[ -n "$ARCHIVE_TOP_M" ]]; then
    if [[ ! "$TOP_N" =~ ^[0-9]+$ ]]; then
        echo "ERROR: TOP_N must be an integer >= 1 when NPZ mode is on (got $TOP_N)" >&2
        exit 1
    fi
    if [[ "$TOP_N" -gt "$ARCHIVE_TOP_M" ]]; then
        echo "ERROR: TOP_N ($TOP_N) is larger than archive_top_m ($ARCHIVE_TOP_M) in $CONFIG." >&2
        echo "       The NPZ archive drops structures outside archive_top_m, so a later frame cannot draw them." >&2
        exit 1
    fi
fi

first_phi="${PHI_LIST[0]}"
first_step=$(printf '%02d' "$((10#$STEP_START))")

RUN_STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
SECONDS=0
finish_timer() {
    local status=$?
    local elapsed=$SECONDS
    local h=$((elapsed / 3600))
    local m=$(((elapsed % 3600) / 60))
    local s=$((elapsed % 60))
    printf '=== full run time: %dh %02dm %02ds (started %s) ===\n' \
        "$h" "$m" "$s" "$RUN_STARTED_AT"
    exit "$status"
}
trap finish_timer EXIT

echo "=== CHD trajectory minimisation (GPU) ==="
echo "  results:     $RESULTS_DIR"
echo "  traj:        $TRAJ  frames ${first_step}..$(printf '%02d' "$((10#$STEP_END))")"
echo "  phi values:  ${PHI_LIST[*]}"
echo "  excitation:  $EXCITATION"
echo "  qmax/qlen:   $QMAX / $QLEN"
echo "  c_tuning:    $C_TUNING_INITIAL"
echo "  GPU_CHAINS:  $GPU_CHAINS"
echo "  TOP_N:       $TOP_N"
if [[ -n "$ARCHIVE_TOP_M" ]]; then
    echo "  npz archive: ${RESULTS_DIR}/trajectory.npz  (archive_top_m=$ARCHIVE_TOP_M)"
fi
echo "  tradius:     first phi=$TRADIUS_FIRST  later phi=$TRADIUS_LATER  (START_FROM_TRADIUS_RANDOM)"
echo "  first start: $STARTING_XYZ  (phi=$first_phi frame=$first_step only)"
echo "  config:      $CONFIG"
echo "  extra args:  $EXTRA_RUN_PY_ARGS"
echo "  started:     $RUN_STARTED_AT"

for step in $(seq "$STEP_START" "$STEP_END"); do
    ts=$(printf '%02d' "$((10#$step))")
    target="xyz/target_traj${TRAJ}/target_${ts}.xyz"
    if [[ ! -f "$target" ]]; then
        echo "ERROR: missing target file $target" >&2
        exit 1
    fi
done

phi_index=0
for phi in "${PHI_LIST[@]}"; do
    if [[ "$phi_index" -eq 0 ]]; then
        radius="$TRADIUS_FIRST"
    else
        radius="$TRADIUS_LATER"
    fi
    echo "=== phi (tuning_ratio_target)=$phi  TRADIUS_N=$radius ==="
    for step in $(seq "$STEP_START" "$STEP_END"); do
        ts=$(printf '%02d' "$((10#$step))")
        target="xyz/target_traj${TRAJ}/target_${ts}.xyz"
        if [[ "$phi_index" -eq 0 && "$ts" == "$first_step" ]]; then
            echo "--- phi=$phi frame=$ts  run_gpu_start.sh  STARTING_XYZ=$STARTING_XYZ ---"
            STARTING_XYZ="$STARTING_XYZ" TARGET_FILE="$target" \
                "$REPO_ROOT/scripts/bash/run_gpu_start.sh" "$ts" "$EXCITATION" "$phi"
        else
            echo "--- phi=$phi frame=$ts  run_gpu_from_previous_timestep.sh  TRADIUS_N=$radius ---"
            # Clear STARTING_XYZ so the helper does not ignore the tradius pool.
            STARTING_XYZ="" START_FROM_TRADIUS_RANDOM=1 TRADIUS_N="$radius" \
                TOP_N="$TOP_N" TARGET_FILE="$target" \
                "$REPO_ROOT/scripts/bash/run_gpu_from_previous_timestep.sh" "$ts" "$EXCITATION" "$phi"
        fi
    done
    phi_index=$((phi_index + 1))
done
