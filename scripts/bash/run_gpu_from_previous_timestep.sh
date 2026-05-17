#!/usr/bin/env bash
# Starting geometry for run-id N from timestep N-1 (default: Kabsch mean of top TOP_N),
# or optional random pick from top TOP_N at t-1 / t+1. Then one CUDA run.py job.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RESULTS_DIR="${RESULTS_DIR:-results}"
TOP_N="${TOP_N:-250}"
CONFIG="${CONFIG:-input.toml}"
GPU_CHAINS="${GPU_CHAINS:-1024}"
EXCITATION_FACTOR="${EXCITATION_FACTOR:-0.628}"
TUNING_RATIO="${TUNING_RATIO:-0.5}"
EXTRA_RUN_PY_ARGS="${EXTRA_RUN_PY_ARGS:-}"
# Override previous source step (default: current - 1, two-digit)
PREV_STEP="${PREV_STEP:-}"
# Override next source step (default: current + 1, two-digit; START_FROM_NEXT_RANDOM only)
NEXT_STEP="${NEXT_STEP:-}"
# If set, skip pooling/averaging and use this XYZ as the starting geometry (still copied to staging)
STARTING_XYZ="${STARTING_XYZ:-}"
# Random pick from top TOP_N at t-1 (mutually exclusive with START_FROM_NEXT_RANDOM)
START_FROM_PREVIOUS_RANDOM="${START_FROM_PREVIOUS_RANDOM:-}"
# Random pick from top TOP_N at t+1 (requires results for step t+1)
START_FROM_NEXT_RANDOM="${START_FROM_NEXT_RANDOM:-}"

usage() {
    cat <<EOF
Usage: $0 <time_step> [excitation_factor] [tuning_ratio_target]

Starting geometry (first match wins):
  1. STARTING_XYZ set                    - copy fixed file to staging
  2. START_FROM_PREVIOUS_RANDOM=1        - random pick from top TOP_N at t-1
  3. START_FROM_NEXT_RANDOM=1            - random pick from top TOP_N at t+1
  4. (default)                           - Kabsch mean of top TOP_N at t-1

Default writes \${RESULTS_DIR}/<t-1>_mean.xyz; next-random writes
  \${RESULTS_DIR}/<t>_start_from_<t+1>_random.xyz

Then runs: $PYTHON run.py --gpu-backend cuda --gpu-chains ${GPU_CHAINS} ...

Positional:
  time_step           Passed as --run-id (e.g. 02)
  excitation_factor   (default: ${EXCITATION_FACTOR})
  tuning_ratio_target (default: ${TUNING_RATIO})

Environment (defaults):
  RESULTS_DIR=${RESULTS_DIR}
  TOP_N=${TOP_N}
  TARGET_FILE         If unset, chd+_data/eirik_data_<time_step>.dat
  CONFIG=${CONFIG}
  GPU_CHAINS=${GPU_CHAINS}
  PREV_STEP           If unset, use (time_step - 1) padded to 2 digits
  NEXT_STEP           If unset, use (time_step + 1) padded (next-random mode)
  STARTING_XYZ        Fixed starting xyz (e.g. first timestep)
  START_FROM_PREVIOUS_RANDOM  If 1, random from top TOP_N at t-1
  START_FROM_NEXT_RANDOM      If 1, random from top TOP_N at t+1
  PYTHON=${PYTHON}
  ALIGN_INDICES       Space-separated atom indices for average_xyz.py --align-indices
  EXTRA_RUN_PY_ARGS   Extra args appended to run.py (e.g. --qmax 8 --qlen 81)
EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

if [[ $# -lt 1 ]]; then
    echo "ERROR: time_step is required" >&2
    usage
fi

if [[ "${START_FROM_PREVIOUS_RANDOM:-0}" == "1" && "${START_FROM_NEXT_RANDOM:-0}" == "1" ]]; then
    echo "ERROR: START_FROM_PREVIOUS_RANDOM and START_FROM_NEXT_RANDOM cannot both be 1" >&2
    exit 1
fi

time_step="$1"
excitation_factor="${2:-$EXCITATION_FACTOR}"
tuning_ratio_target="${3:-$TUNING_RATIO}"

# Two-digit time_step for filenames / run-id
if [[ "$time_step" =~ ^[0-9]+$ ]]; then
    ts_padded=$(printf '%02d' "$((10#$time_step))")
else
    ts_padded="$time_step"
fi

if [[ -n "$PREV_STEP" ]]; then
    prev_step=$(printf '%02d' "$((10#$PREV_STEP))")
else
    if [[ "$time_step" =~ ^[0-9]+$ ]]; then
        prev=$((10#$time_step - 1))
        prev_step=$(printf '%02d' "$prev")
    else
        echo "ERROR: non-numeric time_step requires PREV_STEP to be set" >&2
        exit 1
    fi
fi

if [[ -n "$NEXT_STEP" ]]; then
    next_step=$(printf '%02d' "$((10#$NEXT_STEP))")
else
    if [[ "$time_step" =~ ^[0-9]+$ ]]; then
        next=$((10#$time_step + 1))
        next_step=$(printf '%02d' "$next")
    else
        next_step=""
    fi
fi

TARGET_FILE="${TARGET_FILE:-chd+_data/eirik_data_${ts_padded}.dat}"

mkdir -p "$RESULTS_DIR"

# List best TOP_N paths (lowest f in filename) for pool_step; exclude *_mean.xyz
collect_top_pool() {
    local pool_step="$1"
    mapfile -t TOP_FILES < <("$PYTHON" - "$RESULTS_DIR" "$pool_step" "$TOP_N" <<'PY'
import glob, os, re, sys

def parse(path):
    base = os.path.basename(path)
    if base.endswith("_mean.xyz"):
        return None
    m = re.match(r"^(\d+)_(.+)\.xyz$", base)
    if not m:
        return None
    ts, mid = m.group(1), m.group(2)
    mid = re.sub(r"_dup\d+$", "", mid)
    m2 = re.search(r"\d+\.\d+", mid)
    if not m2:
        return None
    return float(m2.group(0)), path, ts

def main():
    rd, prev, top_n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    pat = os.path.join(rd, f"{prev}_*.xyz")
    rows = []
    for p in sorted(glob.glob(pat)):
        if not os.path.isfile(p):
            continue
        r = parse(p)
        if r is None:
            continue
        f, path, ts = r
        if ts != prev:
            continue
        rows.append((f, path))
    rows.sort(key=lambda x: x[0])
    if not rows:
        return
    take = min(top_n, len(rows))
    sys.stderr.write(
        f"  pooled {take} / {len(rows)} structures from {pat} (lowest f_xray first)\n"
    )
    for _, path in rows[:take]:
        print(path)

if __name__ == "__main__":
    main()
PY
    )
}

random_pick_from_pool() {
    local pool_step="$1"
    local dest="$2"
    local hint="$3"
    collect_top_pool "$pool_step"
    if [[ ${#TOP_FILES[@]} -eq 0 ]]; then
        echo "ERROR: no pool files for step '$pool_step' under ${RESULTS_DIR}/" >&2
        echo "  Expected glob: ${RESULTS_DIR}/${pool_step}_*.xyz (excluding *_mean.xyz)" >&2
        echo "  $hint" >&2
        exit 1
    fi
    idx=$(( RANDOM % ${#TOP_FILES[@]} ))
    picked="${TOP_FILES[$idx]}"
    cp -f "$picked" "$dest"
    echo "  random pick [$idx/${#TOP_FILES[@]}] from step $pool_step: $picked -> $dest"
}

echo "=== GPU run from previous timestep ==="
echo "  time_step (run-id)=$ts_padded  prev=$prev_step  TOP_N=$TOP_N"

if [[ -n "$STARTING_XYZ" ]]; then
    start_out="${RESULTS_DIR}/${prev_step}_mean.xyz"
    echo "  mode: STARTING_XYZ -> $start_out"
    if [[ ! -f "$STARTING_XYZ" ]]; then
        echo "ERROR: STARTING_XYZ='$STARTING_XYZ' not found" >&2
        exit 1
    fi
    cp -f "$STARTING_XYZ" "$start_out"

elif [[ "${START_FROM_PREVIOUS_RANDOM:-0}" == "1" ]]; then
    start_out="${RESULTS_DIR}/${prev_step}_mean.xyz"
    echo "  mode: random from previous step $prev_step -> $start_out"
    random_pick_from_pool "$prev_step" "$start_out" \
        "For the first timestep, set STARTING_XYZ to an initial structure."

elif [[ "${START_FROM_NEXT_RANDOM:-0}" == "1" ]]; then
    if [[ -z "$next_step" ]]; then
        echo "ERROR: non-numeric time_step requires NEXT_STEP for START_FROM_NEXT_RANDOM" >&2
        exit 1
    fi
    start_out="${RESULTS_DIR}/${ts_padded}_start_from_${next_step}_random.xyz"
    echo "  mode: random from next step $next_step -> $start_out"
    random_pick_from_pool "$next_step" "$start_out" \
        "Run step $next_step first so ${RESULTS_DIR}/${next_step}_*.xyz exist."

else
    start_out="${RESULTS_DIR}/${prev_step}_mean.xyz"
    echo "  mode: Kabsch mean from step $prev_step -> $start_out"
    collect_top_pool "$prev_step"
    if [[ ${#TOP_FILES[@]} -eq 0 ]]; then
        echo "ERROR: no pool files for previous step '$prev_step' under ${RESULTS_DIR}/" >&2
        echo "  Expected glob: ${RESULTS_DIR}/${prev_step}_*.xyz (excluding *_mean.xyz)" >&2
        echo "  For the first timestep, set STARTING_XYZ to an initial structure." >&2
        exit 1
    fi
    AVG_CMD=("$PYTHON" "$REPO_ROOT/scripts/python/average_xyz.py" "${TOP_FILES[@]}" --align kabsch -o "$start_out")
    if [[ -n "${ALIGN_INDICES:-}" ]]; then
        # shellcheck disable=SC2206
        AI=($ALIGN_INDICES)
        AVG_CMD+=(--align-indices "${AI[@]}")
    fi
    echo "  averaging ${#TOP_FILES[@]} structures -> $start_out"
    "${AVG_CMD[@]}"
fi

echo "  launching: $PYTHON run.py --gpu-backend cuda --gpu-chains $GPU_CHAINS ..."
RUN_CMD=(
    "$PYTHON" run.py
    --config "$CONFIG"
    --run-id "$ts_padded"
    --results-dir "$RESULTS_DIR"
    --start-xyz-file "$start_out"
    --target-file "$TARGET_FILE"
    --excitation-factor "$excitation_factor"
    --tuning-ratio-target "$tuning_ratio_target"
    --gpu-backend cuda
    --gpu-chains "$GPU_CHAINS"
)
# shellcheck disable=SC2206
RUN_CMD+=( ${EXTRA_RUN_PY_ARGS} )
"${RUN_CMD[@]}"
