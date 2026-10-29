#!/usr/bin/env bash
# Single-frame CHD minimisation on the 3D Ewald grid, CPU workers.
#
# One trajectory frame, one phi. Default phi is 0 with signal-only annealing
# (c_tuning 0, no bond/angle/torsion terms). --phi sets a single value in
# (0, 1]; that run keeps geometry terms and uses C_TUNING_INITIAL (default 0.1).
#
# Workers are independent run.py processes via run_cpu_parallel.sh. By default
# every worker starts from STARTING_XYZ. --from-top-m M instead draws each
# worker uniformly from the M lowest-f_xray structures already in the results
# directory. There is no phi sweep and no restart from a previous frame.
# The target geometry is copied to ${RESULTS_DIR}/${FRAME}_target.xyz before
# the workers start.
#
# Usage:
#   ./run_chd_ewald.sh [--phi 0.5] [--from-top-m 200] [description]
#   ./run_chd_ewald.sh --help
#
# Options:
#   --phi VALUE   tuning_ratio_target in [0, 1]. Overrides PHI.
#                 Default: 0 (signal-only, pure x-ray).
#   --from-top-m M
#                 Each worker draws uniformly from the M lowest-f_xray xyz
#                 files already in the results directory. Overrides TOP_M.
#                 STARTING_XYZ is not used. M must be an integer >= 1.
#
# Environment (defaults shown):
#   TRAJ=094
#   FRAME=00                              one frame; target_traj${TRAJ}/target_${FRAME}.xyz
#   PHI=0                                 used when --phi is omitted
#   EXCITATION=1.0
#   QMAX=4                                qlen = int(10*QMAX+1)
#   N_WORKERS=64
#   C_TUNING_INITIAL=0.1                  used only when PHI > 0
#                                         (PHI=0 forces c_tuning 0 and signal-only)
#   TOP_M                                 used when --from-top-m is omitted; unset disables the pool
#   STARTING_XYZ=xyz/start.xyz            ignored when --from-top-m / TOP_M is set
#   TARGET_FILE                           default: xyz/target_traj${TRAJ}/target_${FRAME}.xyz
#   EXTRA_RUN_PY_ARGS                     appended after the Ewald grid flags
#
# Results directory:
#   results_chd_ewald_traj_${TRAJ}_frame${FRAME}_qmax${QMAX}_phi${PHI}[_${description}]
#   The dot in phi is written as p (0 -> phi0, 0.5 -> phi0p5).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
    awk 'NR==1 {next} /^[^#]/ {exit} {sub(/^# ?/, ""); print}' "$0"
    exit 0
}

DESCRIPTION=""
PHI_FROM_CLI=0
TOP_M_FROM_CLI=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --from-top-m)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --from-top-m requires an integer >= 1 (e.g. --from-top-m 200)" >&2
                exit 1
            fi
            TOP_M="$2"
            TOP_M_FROM_CLI=1
            shift 2
            ;;
        --from-top-m=*)
            TOP_M="${1#--from-top-m=}"
            TOP_M_FROM_CLI=1
            shift
            ;;
        --phi)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --phi requires a value (e.g. --phi 0.5)" >&2
                exit 1
            fi
            PHI="$2"
            PHI_FROM_CLI=1
            shift 2
            ;;
        --phi=*)
            PHI="${1#--phi=}"
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
FRAME="${FRAME:-00}"
if [[ "$PHI_FROM_CLI" -eq 0 ]]; then
    PHI="${PHI:-0}"
fi
EXCITATION="${EXCITATION:-1.0}"
QMAX="${QMAX:-4}"
N_WORKERS="${N_WORKERS:-64}"
C_TUNING_INITIAL="${C_TUNING_INITIAL:-0.1}"
STARTING_XYZ="${STARTING_XYZ:-xyz/start.xyz}"
if [[ "$TOP_M_FROM_CLI" -eq 0 ]]; then
    TOP_M="${TOP_M:-}"
fi
USER_EXTRA_RUN_PY_ARGS="${EXTRA_RUN_PY_ARGS:-}"

if [[ "$PHI" == *","* || "$PHI" == *" "* ]]; then
    echo "ERROR: PHI must be a single number in [0, 1] (got '$PHI')" >&2
    exit 1
fi
if ! awk -v p="$PHI" 'BEGIN { exit !(p ~ /^([0-9]+([.][0-9]+)?|[.][0-9]+)$/) }'; then
    echo "ERROR: PHI must be a number in [0, 1] (got '$PHI')" >&2
    exit 1
fi
if ! awk -v p="$PHI" 'BEGIN { exit !((p+0) >= 0 && (p+0) <= 1) }'; then
    echo "ERROR: PHI must be in [0, 1] (got '$PHI')" >&2
    exit 1
fi
PHI_IS_ZERO=0
if awk -v p="$PHI" 'BEGIN { exit !((p+0) == 0) }'; then
    PHI_IS_ZERO=1
fi

if [[ ! "$FRAME" =~ ^[0-9]+$ ]]; then
    echo "ERROR: FRAME must be a non-negative integer (got '$FRAME')" >&2
    exit 1
fi
FRAME_PADDED=$(printf '%02d' "$((10#$FRAME))")

if [[ ! "$N_WORKERS" =~ ^[0-9]+$ || "$N_WORKERS" -lt 1 ]]; then
    echo "ERROR: N_WORKERS must be an integer >= 1 (got '$N_WORKERS')" >&2
    exit 1
fi
if ! awk -v q="$QMAX" 'BEGIN { exit !(q ~ /^([0-9]+([.][0-9]+)?|[.][0-9]+)$/ && (q+0) > 0) }'; then
    echo "ERROR: QMAX must be a number > 0 (got '$QMAX')" >&2
    exit 1
fi

if [[ "$PHI_IS_ZERO" -eq 1 ]]; then
    C_TUNING_EFFECTIVE=0
    SIGNAL_ONLY_ARGS=(--signal-only-mode)
else
    if ! awk -v c="$C_TUNING_INITIAL" 'BEGIN { exit !(c ~ /^([0-9]+([.][0-9]+)?|[.][0-9]+)$/) }'; then
        echo "ERROR: C_TUNING_INITIAL must be a number > 0 when PHI > 0 (got '$C_TUNING_INITIAL')" >&2
        exit 1
    fi
    if awk -v c="$C_TUNING_INITIAL" 'BEGIN { exit !((c+0) == 0) }'; then
        echo "ERROR: C_TUNING_INITIAL=0 with PHI=$PHI forces signal-only mode, so phi is not applied." >&2
        echo "       Use PHI=0 for pure x-ray, or set C_TUNING_INITIAL > 0." >&2
        exit 1
    fi
    if ! awk -v c="$C_TUNING_INITIAL" 'BEGIN { exit !((c+0) > 0) }'; then
        echo "ERROR: C_TUNING_INITIAL must be > 0 when PHI > 0 (got '$C_TUNING_INITIAL')" >&2
        exit 1
    fi
    C_TUNING_EFFECTIVE="$C_TUNING_INITIAL"
    SIGNAL_ONLY_ARGS=()
fi

if [[ -z "${TARGET_FILE:-}" ]]; then
    TARGET_FILE="xyz/target_traj${TRAJ}/target_${FRAME_PADDED}.xyz"
fi
START_FROM_TOP_RANDOM=0
if [[ -n "$TOP_M" ]]; then
    if [[ ! "$TOP_M" =~ ^[0-9]+$ || "$TOP_M" -lt 1 ]]; then
        echo "ERROR: TOP_M must be an integer >= 1 (got '$TOP_M')" >&2
        exit 1
    fi
    START_FROM_TOP_RANDOM=1
elif [[ ! -f "$STARTING_XYZ" ]]; then
    echo "ERROR: STARTING_XYZ='$STARTING_XYZ' not found" >&2
    exit 1
fi
if [[ ! -f "$TARGET_FILE" ]]; then
    echo "ERROR: missing target file $TARGET_FILE" >&2
    exit 1
fi

QLEN=$(awk -v q="$QMAX" 'BEGIN { print int(10 * q + 1) }')

if [[ "$PHI_IS_ZERO" -eq 1 ]]; then
    PHI_TAG="0"
else
    PHI_TAG="${PHI//./p}"
fi

if [[ -n "$DESCRIPTION" ]]; then
    RESULTS_DIR="results_chd_ewald_traj_${TRAJ}_frame${FRAME_PADDED}_qmax${QMAX}_phi${PHI_TAG}_${DESCRIPTION}"
else
    RESULTS_DIR="results_chd_ewald_traj_${TRAJ}_frame${FRAME_PADDED}_qmax${QMAX}_phi${PHI_TAG}"
fi

GRID_ARGS=(
    --config input_chd_ewald.toml
    --mode test
    --ewald-mode
    --gpu-backend cpu
    --gpu-chains 1
    --qmin 0.0
    --qmax "$QMAX"
    --qlen "$QLEN"
    --tmin 0.0
    --tmax 1.0
    --tlen 21
    --pmin 0.0
    --pmax 2.0
    --plen 21
    --tuning-ratio-target "$PHI"
    --c-tuning-initial "$C_TUNING_EFFECTIVE"
)
if [[ ${#SIGNAL_ONLY_ARGS[@]} -gt 0 ]]; then
    GRID_ARGS+=("${SIGNAL_ONLY_ARGS[@]}")
fi
EXTRA_RUN_PY_ARGS="${GRID_ARGS[*]}"
if [[ -n "$USER_EXTRA_RUN_PY_ARGS" ]]; then
    EXTRA_RUN_PY_ARGS="${EXTRA_RUN_PY_ARGS} ${USER_EXTRA_RUN_PY_ARGS}"
fi

export RESULTS_DIR
export N_WORKERS
export TARGET_FILE
export EXTRA_RUN_PY_ARGS
if [[ "$START_FROM_TOP_RANDOM" == "1" ]]; then
    export START_FROM_TOP_RANDOM
    export TOP_M
    unset STARTING_XYZ
else
    export START_FROM_TOP_RANDOM=0
    export STARTING_XYZ
fi

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

echo "=== CHD Ewald single-frame minimisation (CPU) ==="
echo "  results:     $RESULTS_DIR"
echo "  traj/frame:  $TRAJ / $FRAME_PADDED"
echo "  target:      $TARGET_FILE"
if [[ "$START_FROM_TOP_RANDOM" == "1" ]]; then
    echo "  start:       uniform draw from lowest TOP_M=$TOP_M in $RESULTS_DIR"
else
    echo "  start:       $STARTING_XYZ"
fi
echo "  phi:         $PHI"
if [[ "$PHI_IS_ZERO" -eq 1 ]]; then
    echo "  objective:   signal-only (c_tuning=0, pure x-ray)"
else
    echo "  objective:   x-ray + geometry (c_tuning=$C_TUNING_EFFECTIVE)"
fi
echo "  excitation:  $EXCITATION"
echo "  qmax/qlen:   $QMAX / $QLEN"
echo "  grid:        q($QMAX, $QLEN) theta(0-pi, 21) phi(0-2pi, 21)"
echo "  workers:     $N_WORKERS"
echo "  extra args:  $EXTRA_RUN_PY_ARGS"
echo "  started:     $RUN_STARTED_AT"

mkdir -p "$RESULTS_DIR"
TARGET_COPY="${RESULTS_DIR}/${FRAME_PADDED}_target.xyz"
cp -f "$TARGET_FILE" "$TARGET_COPY"
echo "  target copy: $TARGET_COPY"

"$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$FRAME_PADDED" "$EXCITATION" "$PHI"
