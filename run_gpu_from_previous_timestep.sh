#!/usr/bin/env bash
# Build a Kabsch mean of the best-fitting N structures from the previous timestep,
# save as ${RESULTS_DIR}/${time_step}_mean.xyz, then run one CUDA job with many GPU chains.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RESULTS_DIR="${RESULTS_DIR:-results}"
TOP_N="${TOP_N:-20}"
CONFIG="${CONFIG:-input.toml}"
GPU_CHAINS="${GPU_CHAINS:-1024}"
EXCITATION_FACTOR="${EXCITATION_FACTOR:-1.0}"
TUNING_RATIO="${TUNING_RATIO:-0.5}"
# Override previous source step (default: current - 1, two-digit)
PREV_STEP="${PREV_STEP:-}"
# If set, skip pooling/averaging and use this XYZ as the starting geometry (still copied to NN_mean.xyz)
STARTING_XYZ="${STARTING_XYZ:-}"

usage() {
    cat <<EOF
Usage: $0 <time_step> [excitation_factor] [tuning_ratio_target]

Kabsch-average the TOP_N lowest-f(xray) structures from the previous timestep in
RESULTS_DIR (files named like NN_<padded f>.xyz from wrap.py), write
  \${RESULTS_DIR}/<time_step>_mean.xyz
then run a single:
  $PYTHON run.py --gpu-backend cuda --gpu-chains ${GPU_CHAINS} ...

Positional:
  time_step           Passed as --run-id (e.g. 02)
  excitation_factor   (default: ${EXCITATION_FACTOR})
  tuning_ratio_target (default: ${TUNING_RATIO})

Environment (defaults):
  RESULTS_DIR=${RESULTS_DIR}
  TOP_N=${TOP_N}
  TARGET_FILE         If unset, nmm_data/target_<time_step>.dat
  CONFIG=${CONFIG}
  GPU_CHAINS=${GPU_CHAINS}
  PREV_STEP           If unset, use (time_step - 1) padded to 2 digits
  STARTING_XYZ      If set, use this file instead of pooling (e.g. first timestep)
  PYTHON=${PYTHON}
  ALIGN_INDICES       Space-separated atom indices for average_xyz.py --align-indices
EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

if [[ $# -lt 1 ]]; then
    echo "ERROR: time_step is required" >&2
    usage
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

TARGET_FILE="${TARGET_FILE:-nmm_data/target_${ts_padded}.dat}"
mean_out="${RESULTS_DIR}/${ts_padded}_mean.xyz"

mkdir -p "$RESULTS_DIR"

echo "=== GPU run from previous timestep ==="
echo "  time_step (run-id)=$ts_padded  previous pool=$prev_step  TOP_N=$TOP_N"
echo "  mean output: $mean_out"

if [[ -n "$STARTING_XYZ" ]]; then
    if [[ ! -f "$STARTING_XYZ" ]]; then
        echo "ERROR: STARTING_XYZ='$STARTING_XYZ' not found" >&2
        exit 1
    fi
    echo "  using STARTING_XYZ=$STARTING_XYZ (skip pool / average)"
    cp -f "$STARTING_XYZ" "$mean_out"
else
    # List best TOP_N paths (lowest f in filename), exclude *_mean.xyz
    mapfile -t TOP_FILES < <("$PYTHON" - "$RESULTS_DIR" "$prev_step" "$TOP_N" <<'PY'
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

    if [[ ${#TOP_FILES[@]} -eq 0 ]]; then
        echo "ERROR: no pool files for previous step '$prev_step' under ${RESULTS_DIR}/" >&2
        echo "  Expected glob: ${RESULTS_DIR}/${prev_step}_*.xyz (excluding *_mean.xyz)" >&2
        echo "  For the first timestep, set STARTING_XYZ to an initial structure." >&2
        exit 1
    fi

    AVG_CMD=("$PYTHON" "$REPO_ROOT/average_xyz.py" "${TOP_FILES[@]}" --align kabsch -o "$mean_out")
    if [[ -n "${ALIGN_INDICES:-}" ]]; then
        # shellcheck disable=SC2206
        AI=($ALIGN_INDICES)
        AVG_CMD+=(--align-indices "${AI[@]}")
    fi
    echo "  averaging ${#TOP_FILES[@]} structures -> $mean_out"
    "${AVG_CMD[@]}"
fi

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
