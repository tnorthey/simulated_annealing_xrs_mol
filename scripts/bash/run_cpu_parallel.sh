#!/bin/bash
# Launch N_WORKERS parallel instances of run.py for a single time-step.
#
# Four mutually exclusive modes for choosing the starting geometry:
#   1. START_FROM_TOP_RANDOM=1  -> each worker draws uniformly from the TOP_M
#                                  lowest-f_xray structures in RESULTS_DIR
#   2. STARTING_XYZ set         -> fixed file for every worker
#   3. START_FROM_PREV_MEAN=1   -> Kabsch mean of TOP_N best from previous step
#   4. (default)                -> random pick from XYZ_SOURCE_STEP pool
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

N_WORKERS="${N_WORKERS:-64}"
RESULTS_DIR="${RESULTS_DIR:-results}"
XYZ_SOURCE_STEP="${XYZ_SOURCE_STEP:-01}"
TOP_N="${TOP_N:-20}"
STARTING_XYZ="${STARTING_XYZ:-}"
START_FROM_TOP_RANDOM="${START_FROM_TOP_RANDOM:-}"
TOP_M="${TOP_M:-}"
START_FROM_PREV_MEAN="${START_FROM_PREV_MEAN:-}"
PREV_STEP="${PREV_STEP:-}"
ALIGN_INDICES="${ALIGN_INDICES:-}"
PYTHON="${PYTHON:-python3}"
EXCITATION_FACTOR="${EXCITATION_FACTOR:-1.0}"
TUNING_RATIO="${TUNING_RATIO:-0.5}"
TARGET_FILE="${TARGET_FILE:-}"
TARGET_FILE_TEMPLATE="${TARGET_FILE_TEMPLATE:-nmm_data/target_{time_step}.dat}"
EXTRA_RUN_PY_ARGS="${EXTRA_RUN_PY_ARGS:-}"

usage() {
    cat <<EOF
Usage: $0 <time_step> [excitation_factor] [tuning_ratio_target]

Launch N_WORKERS parallel instances of run.py for a single time-step.

Starting geometry is chosen by one of four modes (first match wins):
  1. START_FROM_TOP_RANDOM=1 - each worker draws uniformly from the TOP_M
                               lowest-f_xray xyz files in RESULTS_DIR
  2. STARTING_XYZ set        - use that file for every worker
  3. START_FROM_PREV_MEAN=1  - Kabsch-average the TOP_N best structures from
                               the previous timestep, then use that mean
  4. (default)               - pick randomly from XYZ_SOURCE_STEP pool

Positional arguments:
  time_step           Time-step identifier (passed as --run-id)
  excitation_factor   (default: \$EXCITATION_FACTOR=$EXCITATION_FACTOR)
  tuning_ratio_target (default: \$TUNING_RATIO=$TUNING_RATIO)

Environment variables (defaults shown):
  N_WORKERS            $N_WORKERS
  RESULTS_DIR          $RESULTS_DIR   (also passed as --results-dir to run.py)
  EXCITATION_FACTOR    $EXCITATION_FACTOR  (used when positional arg is omitted)
  TUNING_RATIO         $TUNING_RATIO  (used when positional arg is omitted)
  STARTING_XYZ         (unset)  fixed starting xyz for all workers
  START_FROM_TOP_RANDOM (unset) if 1, uniform draw per worker from the lowest TOP_M
  TOP_M                (unset)  pool size for START_FROM_TOP_RANDOM (required then)
  START_FROM_PREV_MEAN (unset)  if 1, build mean from previous step
  PREV_STEP            (unset)  override previous step (default: time_step - 1)
  TOP_N                $TOP_N   structures to pool (mean mode or random-pool mode)
  ALIGN_INDICES        (unset)  space-separated atom indices for Kabsch alignment
  XYZ_SOURCE_STEP      $XYZ_SOURCE_STEP   (random-pool mode only)
  TARGET_FILE          (unset)  override --target-file for all workers
  TARGET_FILE_TEMPLATE nmm_data/target_{time_step}.dat (used when TARGET_FILE is unset)
  EXTRA_RUN_PY_ARGS    (unset)  extra args appended to run.py (e.g. --gpu-backend cpu)
  PYTHON               $PYTHON
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

# ── Mode 1: uniform draw from the lowest TOP_M in RESULTS_DIR ───────────────
if [[ "${START_FROM_TOP_RANDOM:-0}" == "1" ]]; then
    if [[ ! "$TOP_M" =~ ^[0-9]+$ || "$TOP_M" -lt 1 ]]; then
        echo "ERROR: START_FROM_TOP_RANDOM requires TOP_M to be an integer >= 1 (got '${TOP_M}')" >&2
        exit 1
    fi
    echo "  START_FROM_TOP_RANDOM: lowest TOP_M=$TOP_M in ${RESULTS_DIR}/"
    echo "  (STARTING_XYZ, XYZ_SOURCE_STEP, and START_FROM_PREV_MEAN ignored)"

    mapfile -t candidates < <("$PYTHON" - "$RESULTS_DIR" "$TOP_M" <<'PY'
import glob, os, re, sys

def parse(path):
    base = os.path.basename(path)
    if base.endswith("_mean.xyz") or base.endswith("_target.xyz"):
        return None
    m = re.match(r"^(\d+)_(.+)\.xyz$", base)
    if not m:
        return None
    mid = re.sub(r"_dup\d+$", "", m.group(2))
    m2 = re.search(r"\d+\.\d+", mid)
    if not m2:
        return None
    return float(m2.group(0)), path

def main():
    rd, top_m = sys.argv[1], int(sys.argv[2])
    pat = os.path.join(rd, "*.xyz")
    rows = []
    for p in sorted(glob.glob(pat)):
        if not os.path.isfile(p):
            continue
        r = parse(p)
        if r is None:
            continue
        rows.append(r)
    rows.sort(key=lambda x: x[0])
    if not rows:
        return
    take = min(top_m, len(rows))
    chosen = rows[:take]
    sys.stderr.write(
        f"  pooled {take} / {len(rows)} structures from {pat} "
        f"(lowest f_xray {chosen[0][0]:.8f} .. {chosen[-1][0]:.8f})\n"
    )
    for _, path in chosen:
        print(path)

if __name__ == "__main__":
    main()
PY
    )

    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "ERROR: no scored xyz files under ${RESULTS_DIR}/" >&2
        echo "  Expected names like <run_id>_<f_xray>.xyz (not *_target.xyz or *_mean.xyz)." >&2
        echo "  Run once from STARTING_XYZ before using START_FROM_TOP_RANDOM." >&2
        exit 1
    fi
    echo "  workers=$N_WORKERS | excitation=$excitation_factor | tuning=$tuning_ratio_target"
    echo "  each worker draws uniformly from ${#candidates[@]} structures"

# ── Mode 2: fixed STARTING_XYZ ──────────────────────────────────────────────
elif [[ -n "$STARTING_XYZ" ]]; then
    if [[ ! -f "$STARTING_XYZ" ]]; then
        echo "ERROR: STARTING_XYZ='$STARTING_XYZ' does not exist" >&2
        exit 1
    fi
    effective_start="$STARTING_XYZ"
    echo "  starting_xyz=$effective_start (fixed for all workers)"
    echo "  (XYZ_SOURCE_STEP and START_FROM_PREV_MEAN ignored)"
    echo "  workers=$N_WORKERS | excitation=$excitation_factor | tuning=$tuning_ratio_target"

# ── Mode 3: Kabsch mean of previous timestep ────────────────────────────────
elif [[ "${START_FROM_PREV_MEAN:-0}" == "1" ]]; then
    if [[ -n "$PREV_STEP" ]]; then
        prev_step=$(printf '%02d' "$((10#$PREV_STEP))")
    else
        if [[ "$time_step" =~ ^[0-9]+$ ]]; then
            prev_step=$(printf '%02d' "$(( 10#$time_step - 1 ))")
        else
            echo "ERROR: non-numeric time_step requires PREV_STEP to be set" >&2
            exit 1
        fi
    fi
    mean_out="${RESULTS_DIR}/${prev_step}_mean.xyz"
    mean_out_sdf="${RESULTS_DIR}/${prev_step}_mean.sdf"
    mkdir -p "$RESULTS_DIR"

    echo "  START_FROM_PREV_MEAN: pooling TOP_N=$TOP_N from step $prev_step"
    echo "  (XYZ_SOURCE_STEP ignored)"

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
        echo "  For the first timestep, use STARTING_XYZ instead." >&2
        exit 1
    fi

    AVG_CMD=("$PYTHON" "$REPO_ROOT/scripts/python/average_xyz.py" "${TOP_FILES[@]}" --align kabsch -o "$mean_out")
    if [[ -n "$ALIGN_INDICES" ]]; then
        # shellcheck disable=SC2206
        AI=($ALIGN_INDICES)
        AVG_CMD+=(--align-indices "${AI[@]}")
    fi
    echo "  averaging ${#TOP_FILES[@]} structures -> $mean_out"
    "${AVG_CMD[@]}"

    effective_start="$mean_out"
    effective_start_sdf="$mean_out_sdf"
    echo "  creating mean SDF -> $mean_out_sdf"
    if command -v obabel >/dev/null 2>&1; then
        # OpenBabel CLI is the most portable option (no Python deps needed).
        # -xk: attempt to perceive/assign bond orders where possible.
        obabel -ixyz "$mean_out" -osdf -O "$mean_out_sdf" -xk >/dev/null
        echo "  mean SDF created via obabel"
    else
        echo "ERROR: obabel not found; cannot create mean SDF from $mean_out" >&2
        echo "  Hint: install OpenBabel (obabel) or provide Python openbabel/rdkit deps." >&2
        exit 1
    fi
    echo "  workers=$N_WORKERS | excitation=$excitation_factor | tuning=$tuning_ratio_target"

# ── Mode 4: random pick from XYZ_SOURCE_STEP pool ───────────────────────────
else
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
    if [[ -n "${effective_start:-}" ]]; then
        starting_xyz="$effective_start"
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
    RUN_CMD=(
        "$PYTHON" run.py
        --run-id "$time_step"
        --results-dir "$RESULTS_DIR"
        --start-xyz-file "$starting_xyz"
        --target-file "$target_file"
        --excitation-factor "$excitation_factor"
        --tuning-ratio-target "$tuning_ratio_target"
    )
    if [[ -n "${effective_start_sdf:-}" ]]; then
        RUN_CMD+=(--start-sdf-file "$effective_start_sdf")
    fi
    # shellcheck disable=SC2206
    RUN_CMD+=( ${EXTRA_RUN_PY_ARGS} )
    "${RUN_CMD[@]}" &
done

wait
echo "  All $N_WORKERS workers finished for time-step $time_step"
