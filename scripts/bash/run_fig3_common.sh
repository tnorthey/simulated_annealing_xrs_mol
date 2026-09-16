#!/usr/bin/env bash
# ============================================================================
# Figure 3 CHD test-mode GPU runs.
#
# Uses repo input.toml for MM/molecule defaults, plus experiment CLI:
# q-grid, open/closed C1-C6, files, GPU, PCD, and the Figure 3 SA schedule
# (step 0.025, GA 8000, n_tuning 100, nrestarts 1). Those SA values are
# NOT taken from input.toml (CHD+ production defaults are smaller/shorter).
#
# Usage:
#   ./scripts/bash/run_fig3_common.sh --qmax 4 --qlen 41 --ring open
#   ./scripts/bash/run_fig3_common.sh --qmax 8 --qlen 81 --ring closed
#   ./scripts/bash/run_fig3_qmax4_open.sh --comment rerun2
#
# Results directory (default results_fig3_qmax<X>_<open|closed>):
#   --comment DETAIL   append _DETAIL (e.g. results_fig3_qmax4_open_rerun2)
#   RESULTS_DIR=path   full override (ignores --comment)
#
# Environment overrides (defaults shown):
#   PYTHON             python3
#   CONFIG             input.toml
#   GPU_CHAINS         128
#   STARTING_XYZ       xyz/start.xyz
#   TARGET_FILE        xyz/target_20.xyz
#   COMMENT            (unset) same as --comment
#   RESULTS_DIR        (unset) auto from qmax/ring/comment
#   RESTART_RATIO      (unset) inherit from CONFIG
#   SA_NSTEPS          4000
#   GA_NSTEPS          8000
#   SA_STEP_SIZE       0.025
#   GA_STEP_SIZE       0.025
#   NRESTARTS          1
#   N_TUNING_UPDATE_FREQ 100
#   C_TUNING_INITIAL   0.1
#   TUNING_RATIO_TARGET 0.5
#   EXTRA_RUN_PY_ARGS  (unset) extra args appended to run.py
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-input.toml}"
GPU_CHAINS="${GPU_CHAINS:-128}"
STARTING_XYZ="${STARTING_XYZ:-xyz/start.xyz}"
TARGET_FILE="${TARGET_FILE:-xyz/target_20.xyz}"
START_SDF="${START_SDF:-sdf/chd_start.sdf}"
REFERENCE_XYZ="${REFERENCE_XYZ:-xyz/chd_reference.xyz}"
EXTRA_RUN_PY_ARGS="${EXTRA_RUN_PY_ARGS:-}"
COMMENT="${COMMENT:-}"
RESULTS_DIR_OVERRIDE="${RESULTS_DIR:-}"
RESTART_RATIO="${RESTART_RATIO:-}"
SA_NSTEPS="${SA_NSTEPS:-4000}"
GA_NSTEPS="${GA_NSTEPS:-8000}"
SA_STEP_SIZE="${SA_STEP_SIZE:-0.025}"
GA_STEP_SIZE="${GA_STEP_SIZE:-0.025}"
NRESTARTS="${NRESTARTS:-1}"
N_TUNING_UPDATE_FREQ="${N_TUNING_UPDATE_FREQ:-100}"
C_TUNING_INITIAL="${C_TUNING_INITIAL:-0.1}"
TUNING_RATIO_TARGET="${TUNING_RATIO_TARGET:-0.5}"

QMAX=""
QLEN=""
RING=""

usage() {
    sed -n '3,40p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qmax)
            QMAX="${2:-}"
            shift 2
            ;;
        --qlen)
            QLEN="${2:-}"
            shift 2
            ;;
        --ring)
            RING="${2:-}"
            shift 2
            ;;
        --comment)
            COMMENT="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$QMAX" || -z "$QLEN" || -z "$RING" ]]; then
    echo "ERROR: --qmax, --qlen, and --ring are required" >&2
    usage >&2
    exit 1
fi

case "$RING" in
    open)
        BOND_IGNORE='[[0, 5]]'
        ;;
    closed)
        BOND_IGNORE='[]'
        ;;
    *)
        echo "ERROR: --ring must be 'open' or 'closed' (got '$RING')" >&2
        exit 1
        ;;
esac

COMMENT="${COMMENT// /_}"
if [[ -n "$COMMENT" && ! "$COMMENT" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: --comment must be a simple token (letters, digits, . _ -); got '$COMMENT'" >&2
    exit 1
fi

if [[ -n "$RESULTS_DIR_OVERRIDE" ]]; then
    RESULTS_DIR="$RESULTS_DIR_OVERRIDE"
else
    RESULTS_DIR="results_fig3_qmax${QMAX}_${RING}"
    if [[ -n "$COMMENT" ]]; then
        RESULTS_DIR="${RESULTS_DIR}_${COMMENT}"
    fi
fi
RUN_ID="fig3_qmax${QMAX}_${RING}${COMMENT:+_${COMMENT}}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: CONFIG='$CONFIG' does not exist" >&2
    exit 1
fi
if [[ ! -f "$STARTING_XYZ" ]]; then
    echo "ERROR: STARTING_XYZ='$STARTING_XYZ' does not exist" >&2
    exit 1
fi
if [[ ! -f "$TARGET_FILE" ]]; then
    echo "ERROR: TARGET_FILE='$TARGET_FILE' does not exist" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "=== Figure 3 run ==="
echo "  config         = $CONFIG  (MM/molecule defaults; this script does not generate a toml)"
echo "  ring           = $RING  (--bond-ignore-array $BOND_IGNORE)"
echo "  q              = 0.0 .. ${QMAX}.0  (${QLEN} pts)"
echo "  pcd_mode       = true  (IAM vs $REFERENCE_XYZ, inelastic Compton in I_ref)"
echo "  comment        = ${COMMENT:-<none>}"
echo "  gpu_chains     = $GPU_CHAINS"
echo "  restart_ratio  = ${RESTART_RATIO:-<from $CONFIG>}"
echo "  sa/ga steps    = $SA_NSTEPS / $GA_NSTEPS  (step $SA_STEP_SIZE / $GA_STEP_SIZE)"
echo "  nrestarts      = $NRESTARTS  n_tuning_update_freq=$N_TUNING_UPDATE_FREQ"
echo "  c_tuning       = $C_TUNING_INITIAL  tuning_ratio_target=$TUNING_RATIO_TARGET"
echo "  target         = $TARGET_FILE"
echo "  start          = $STARTING_XYZ"
echo "  results_dir    = $RESULTS_DIR"

RUN_CMD=(
    "$PYTHON" run.py
    --config "$CONFIG"
    --mode test
    --run-id "$RUN_ID"
    --results-dir "$RESULTS_DIR"
    --start-xyz-file "$STARTING_XYZ"
    --start-sdf-file "$START_SDF"
    --reference-xyz-file "$REFERENCE_XYZ"
    --target-file "$TARGET_FILE"
    --ab-initio-scattering-file ""
    --reference-dat-file ""
    --gpu-backend cuda
    --gpu-chains "$GPU_CHAINS"
    --sampling
    --pcd-mode
    --inelastic
    --qmin 0.0
    --qmax "$QMAX"
    --qlen "$QLEN"
    --bond-ignore-array "$BOND_IGNORE"
    --sa-nsteps "$SA_NSTEPS"
    --ga-nsteps "$GA_NSTEPS"
    --sa-step-size "$SA_STEP_SIZE"
    --ga-step-size "$GA_STEP_SIZE"
    --nrestarts "$NRESTARTS"
    --n-tuning-update-freq "$N_TUNING_UPDATE_FREQ"
    --c-tuning-initial "$C_TUNING_INITIAL"
    --tuning-ratio-target "$TUNING_RATIO_TARGET"
)

if [[ -n "$RESTART_RATIO" ]]; then
    RUN_CMD+=(--restart-ratio "$RESTART_RATIO")
fi

# shellcheck disable=SC2206
RUN_CMD+=( ${EXTRA_RUN_PY_ARGS} )
"${RUN_CMD[@]}"
