#!/usr/bin/env bash
# Starting geometry for run-id N from timestep N-1 (default: Kabsch mean of top TOP_N),
# or optional random from top TOP_N, tradius structure union, or tradius neighbor means.
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
# Override previous source step (default: current - PREV_DISTANCE, two-digit)
PREV_STEP="${PREV_STEP:-}"
# Offset for START_FROM_PREVIOUS_RANDOM when PREV_STEP is unset (default: 1 => t-1)
PREV_DISTANCE="${PREV_DISTANCE:-1}"
# Override next source step (default: current + NEXT_DISTANCE, two-digit)
NEXT_STEP="${NEXT_STEP:-}"
# Offset for START_FROM_NEXT_RANDOM when NEXT_STEP is unset (default: 1 => t+1)
NEXT_DISTANCE="${NEXT_DISTANCE:-1}"
# Radius N for tradius modes: neighbors t+/-1 .. t+/-N
TRADIUS_N="${TRADIUS_N:-1}"
# If set, skip pooling/averaging and use this XYZ as the starting geometry (still copied to staging)
STARTING_XYZ="${STARTING_XYZ:-}"
# Random pick from top TOP_N at t-1 (mutually exclusive with other START_FROM_*_RANDOM)
START_FROM_PREVIOUS_RANDOM="${START_FROM_PREVIOUS_RANDOM:-}"
# Random pick from top TOP_N at t+1 (requires results for step t+1)
START_FROM_NEXT_RANDOM="${START_FROM_NEXT_RANDOM:-}"
# Random pick uniformly from union of top TOP_N at t±1..t±N (no global f sort)
START_FROM_TRADIUS_RANDOM="${START_FROM_TRADIUS_RANDOM:-}"
# Random pick uniformly from existing {step}_mean.xyz at t+/-1..t+/-N
START_FROM_TRADIUS_MEAN="${START_FROM_TRADIUS_MEAN:-}"

usage() {
    cat <<EOF
Usage: $0 <time_step> [excitation_factor] [tuning_ratio_target]

Starting geometry (first match wins):
  1. STARTING_XYZ set                    - copy fixed file to staging
  2. START_FROM_PREVIOUS_RANDOM=1        - random from top TOP_N at t-PREV_DISTANCE
  3. START_FROM_NEXT_RANDOM=1            - random from top TOP_N at t+NEXT_DISTANCE
  4. START_FROM_TRADIUS_RANDOM=1         - random from union of top TOP_N at t+/-1..t+/-N
  5. START_FROM_TRADIUS_MEAN=1          - random from {step}_mean.xyz at t+/-1..t+/-N
     (missing means auto-built from each step's top TOP_N pool)
  6. (default)                           - Kabsch mean of top TOP_N at t-1

Before each run, {prev}_mean.xyz is created if missing and {prev}_*.xyz exist.

With GPU_CHAINS>1, random/tradius modes write a pool manifest; each CUDA chain gets
its own random draw (--gpu-per-chain-random-pool-file). Prev/next multi-chain use the
best (lowest f) pool member for --start-xyz-file (MM only). Tradius modes use a
random pool member for MM. No global sort across timesteps in tradius-random mode.

Then runs: $PYTHON run.py --gpu-backend cuda --gpu-chains ${GPU_CHAINS} ...

Positional:
  time_step           Passed as --run-id (e.g. 02)
  excitation_factor   (default: ${EXCITATION_FACTOR})
  tuning_ratio_target (default: ${TUNING_RATIO})

Environment (defaults):
  RESULTS_DIR=${RESULTS_DIR}
  TOP_N=${TOP_N}
  TRADIUS_N=${TRADIUS_N}      For tradius modes: include t+/-1..t+/-N
  TARGET_FILE         If unset, chd+_data/eirik_data_<time_step>.dat
  CONFIG=${CONFIG}
  GPU_CHAINS=${GPU_CHAINS}
  PREV_STEP           Override pool step for prev-random / default mean (default: t-1)
  PREV_DISTANCE       If PREV_STEP unset, prev-random pools from t-PREV_DISTANCE (default: 1)
  NEXT_STEP           Override pool step for next-random (default: t+NEXT_DISTANCE)
  NEXT_DISTANCE       If NEXT_STEP unset, next-random pools from t+NEXT_DISTANCE (default: 1)
  STARTING_XYZ        Fixed starting xyz (e.g. first timestep)
  START_FROM_PREVIOUS_RANDOM  If 1, random from top TOP_N at t-PREV_DISTANCE
  START_FROM_NEXT_RANDOM      If 1, random from top TOP_N at t+NEXT_DISTANCE
  START_FROM_TRADIUS_RANDOM   If 1, uniform random from union top TOP_N at t+/-1..t+/-N
  START_FROM_TRADIUS_MEAN     If 1, uniform random from {step}_mean.xyz at t+/-1..t+/-N
                              (missing {step}_mean.xyz built from top TOP_N pool automatically)
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

_random_mode_count=0
[[ "${START_FROM_PREVIOUS_RANDOM:-0}" == "1" ]] && _random_mode_count=$((_random_mode_count + 1))
[[ "${START_FROM_NEXT_RANDOM:-0}" == "1" ]] && _random_mode_count=$((_random_mode_count + 1))
[[ "${START_FROM_TRADIUS_RANDOM:-0}" == "1" ]] && _random_mode_count=$((_random_mode_count + 1))
[[ "${START_FROM_TRADIUS_MEAN:-0}" == "1" ]] && _random_mode_count=$((_random_mode_count + 1))
if [[ "$_random_mode_count" -gt 1 ]]; then
    echo "ERROR: only one of START_FROM_PREVIOUS_RANDOM, START_FROM_NEXT_RANDOM," >&2
    echo "       START_FROM_TRADIUS_RANDOM, START_FROM_TRADIUS_MEAN may be set to 1" >&2
    exit 1
fi

time_step="$1"
excitation_factor="${2:-$EXCITATION_FACTOR}"
tuning_ratio_target="${3:-$TUNING_RATIO}"

# Two-digit time_step for filenames / run-id
if [[ "$time_step" =~ ^[0-9]+$ ]]; then
    ts_padded=$(printf '%02d' "$((10#$time_step))")
    ts_int=$((10#$time_step))
else
    ts_padded="$time_step"
    ts_int=""
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

TARGET_FILE="${TARGET_FILE:-chd+_data/eirik_data_${ts_padded}.dat}"

# Pool step for prev/next random modes (PREV_STEP / NEXT_STEP override distance).
resolve_previous_random_pool_step() {
    if [[ -n "$PREV_STEP" ]]; then
        printf '%02d' "$((10#$PREV_STEP))"
        return
    fi
    if [[ -z "$ts_int" ]]; then
        echo "ERROR: non-numeric time_step requires PREV_STEP for START_FROM_PREVIOUS_RANDOM" >&2
        exit 1
    fi
    local dist=$((10#${PREV_DISTANCE:-1}))
    if [[ "$dist" -lt 1 ]]; then
        echo "ERROR: PREV_DISTANCE must be >= 1 (got $PREV_DISTANCE)" >&2
        exit 1
    fi
    local step_num=$((ts_int - dist))
    if [[ "$step_num" -lt 0 ]]; then
        echo "ERROR: PREV_DISTANCE=$dist yields step $step_num < 0 for time_step $ts_padded" >&2
        exit 1
    fi
    printf '%02d' "$step_num"
}

resolve_next_random_pool_step() {
    if [[ -n "$NEXT_STEP" ]]; then
        printf '%02d' "$((10#$NEXT_STEP))"
        return
    fi
    if [[ -z "$ts_int" ]]; then
        echo "ERROR: non-numeric time_step requires NEXT_STEP for START_FROM_NEXT_RANDOM" >&2
        exit 1
    fi
    local dist=$((10#${NEXT_DISTANCE:-1}))
    if [[ "$dist" -lt 1 ]]; then
        echo "ERROR: NEXT_DISTANCE must be >= 1 (got $NEXT_DISTANCE)" >&2
        exit 1
    fi
    printf '%02d' "$((ts_int + dist))"
}

mkdir -p "$RESULTS_DIR"

# Populate TOP_FILES with up to TOP_N lowest-f(xray) paths for one timestep.
collect_top_pool_for_step() {
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

# Kabsch-average TOP_FILES into dest (uses ALIGN_INDICES when set).
average_top_files_to_mean() {
    local dest="$1"
    if [[ ${#TOP_FILES[@]} -eq 0 ]]; then
        echo "ERROR: average_top_files_to_mean: TOP_FILES is empty" >&2
        return 1
    fi
    local AVG_CMD=(
        "$PYTHON" "$REPO_ROOT/scripts/python/average_xyz.py"
        "${TOP_FILES[@]}"
        --align kabsch
        -o "$dest"
    )
    if [[ -n "${ALIGN_INDICES:-}" ]]; then
        # shellcheck disable=SC2206
        local AI=($ALIGN_INDICES)
        AVG_CMD+=(--align-indices "${AI[@]}")
    fi
    "${AVG_CMD[@]}"
}

# Create ${RESULTS_DIR}/${step}_mean.xyz from top TOP_N pool if missing (no-op if exists).
ensure_step_mean_xyz() {
    local step="$1"
    local mean_path="${RESULTS_DIR}/${step}_mean.xyz"
    if [[ -f "$mean_path" ]]; then
        return 0
    fi
    collect_top_pool_for_step "$step"
    if [[ ${#TOP_FILES[@]} -eq 0 ]]; then
        echo "  WARNING: cannot create ${mean_path}: no ${step}_*.xyz pool under ${RESULTS_DIR}/" >&2
        return 1
    fi
    echo "  creating ${step}_mean.xyz from ${#TOP_FILES[@]} structures (TOP_N=$TOP_N)" >&2
    average_top_files_to_mean "$mean_path"
}

# Union of top TOP_N per step at t±1..t±N (concatenated; no global f sort). Dedupe by path.
collect_tradius_pool() {
    if [[ -z "$ts_int" ]]; then
        echo "ERROR: START_FROM_TRADIUS_RANDOM requires numeric time_step" >&2
        exit 1
    fi
    mapfile -t TOP_FILES < <("$PYTHON" - "$RESULTS_DIR" "$ts_padded" "$TRADIUS_N" "$TOP_N" <<'PY'
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

def pool_one_step(rd, step, top_n):
    pat = os.path.join(rd, f"{step}_*.xyz")
    rows = []
    for p in sorted(glob.glob(pat)):
        if not os.path.isfile(p):
            continue
        r = parse(p)
        if r is None:
            continue
        f, path, ts = r
        if ts != step:
            continue
        rows.append((f, path))
    rows.sort(key=lambda x: x[0])
    take = min(top_n, len(rows))
    if take:
        sys.stderr.write(
            f"  pooled {take} / {len(rows)} from step {step} ({pat})\n"
        )
    return [path for _, path in rows[:take]]

def main():
    rd, ts_str, radius, top_n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    t = int(ts_str)
    union = []
    seen = set()
    used_steps = []
    skipped_steps = []
    for k in range(1, radius + 1):
        for step_num in (t - k, t + k):
            if step_num < 0:
                continue
            step = f"{step_num:02d}"
            paths = pool_one_step(rd, step, top_n)
            if not paths:
                skipped_steps.append(step)
                sys.stderr.write(
                    f"  WARNING: no pool for step {step} (t{'+' if step_num > t else '-'}{k}), skipping\n"
                )
                continue
            used_steps.append(step)
            for path in paths:
                if path in seen:
                    continue
                seen.add(path)
                union.append(path)
    if not union:
        return
    skipped_msg = f" (skipped: {', '.join(skipped_steps)})" if skipped_steps else ""
    sys.stderr.write(
        f"  tradius N={radius}: {len(union)} unique structures in union from "
        f"steps {', '.join(used_steps)}{skipped_msg}\n"
    )
    for path in union:
        print(path)

if __name__ == "__main__":
    main()
PY
    )
}

apply_random_start_from_top_files() {
    local dest="$1"
    local hint="$2"
    local pool_label="$3"
    if [[ ${#TOP_FILES[@]} -eq 0 ]]; then
        echo "ERROR: pool is empty under ${RESULTS_DIR}/" >&2
        echo "  $hint" >&2
        exit 1
    fi
    if [[ "$GPU_CHAINS" -gt 1 ]]; then
        GPU_POOL_FILE="${RESULTS_DIR}/${ts_padded}_pool_${pool_label}.lst"
        printf '%s\n' "${TOP_FILES[@]}" >"$GPU_POOL_FILE"
        cp -f "${TOP_FILES[0]}" "$dest"
        echo "  multi-chain (${GPU_CHAINS} CUDA chains): pool size=${#TOP_FILES[@]} (TOP_N=$TOP_N)"
        echo "  pool manifest -> $GPU_POOL_FILE"
        echo "  MM / --start-xyz-file (best in pool): ${TOP_FILES[0]} -> $dest"
        echo "  run.py will print one line per chain: random XYZ from this pool"
    else
        idx=$(( RANDOM % ${#TOP_FILES[@]} ))
        picked="${TOP_FILES[$idx]}"
        cp -f "$picked" "$dest"
        echo "  random pick [$idx/${#TOP_FILES[@]}]: $picked -> $dest"
    fi
}

apply_tradius_random_start() {
    local dest="$1"
    local hint="$2"
    if [[ ${#TOP_FILES[@]} -eq 0 ]]; then
        echo "ERROR: tradius union pool is empty under ${RESULTS_DIR}/" >&2
        echo "  $hint" >&2
        exit 1
    fi
    idx=$(( RANDOM % ${#TOP_FILES[@]} ))
    picked="${TOP_FILES[$idx]}"
    if [[ "$GPU_CHAINS" -gt 1 ]]; then
        GPU_POOL_FILE="${RESULTS_DIR}/${ts_padded}_pool_tradius_N${TRADIUS_N}.lst"
        printf '%s\n' "${TOP_FILES[@]}" >"$GPU_POOL_FILE"
        cp -f "$picked" "$dest"
        echo "  multi-chain (${GPU_CHAINS} CUDA chains): tradius union size=${#TOP_FILES[@]} (TOP_N/step=$TOP_N, N=$TRADIUS_N)"
        echo "  pool manifest -> $GPU_POOL_FILE"
        echo "  MM / --start-xyz-file (random from union [$idx/${#TOP_FILES[@]}]): $picked -> $dest"
        echo "  run.py will print one line per chain: uniform random XYZ from union pool"
    else
        cp -f "$picked" "$dest"
        echo "  tradius random pick [$idx/${#TOP_FILES[@]}] from union: $picked -> $dest"
    fi
}

random_pick_from_pool() {
    local pool_step="$1"
    local dest="$2"
    local hint="$3"
    collect_top_pool_for_step "$pool_step"
    apply_random_start_from_top_files "$dest" "$hint" "$pool_step"
}

# Collect existing {step}_mean.xyz for neighbors t+/-1..t+/-N into MEAN_FILES.
collect_tradius_means() {
    if [[ -z "$ts_int" ]]; then
        echo "ERROR: START_FROM_TRADIUS_MEAN requires numeric time_step" >&2
        exit 1
    fi
    MEAN_FILES=()
    local k step_num step mean_path
    for ((k = 1; k <= TRADIUS_N; k++)); do
        for step_num in $((ts_int - k)) $((ts_int + k)); do
            if [[ "$step_num" -lt 0 ]]; then
                continue
            fi
            step=$(printf '%02d' "$step_num")
            mean_path="${RESULTS_DIR}/${step}_mean.xyz"
            if ensure_step_mean_xyz "$step"; then
                MEAN_FILES+=("$mean_path")
                echo "  using mean for step $step: $mean_path" >&2
            fi
        done
    done
    if [[ ${#MEAN_FILES[@]} -gt 0 ]]; then
        echo "  tradius mean N=${TRADIUS_N}: ${#MEAN_FILES[@]} mean structure(s) available" >&2
    fi
}

apply_tradius_mean_start() {
    local dest="$1"
    local hint="$2"
    if [[ ${#MEAN_FILES[@]} -eq 0 ]]; then
        echo "ERROR: no neighbor mean files found under ${RESULTS_DIR}/" >&2
        echo "  $hint" >&2
        exit 1
    fi
    idx=$(( RANDOM % ${#MEAN_FILES[@]} ))
    picked="${MEAN_FILES[$idx]}"
    if [[ "$GPU_CHAINS" -gt 1 ]]; then
        GPU_POOL_FILE="${RESULTS_DIR}/${ts_padded}_pool_tradius_means_N${TRADIUS_N}.lst"
        printf '%s\n' "${MEAN_FILES[@]}" >"$GPU_POOL_FILE"
        cp -f "$picked" "$dest"
        echo "  multi-chain (${GPU_CHAINS} CUDA chains): tradius mean pool size=${#MEAN_FILES[@]} (N=$TRADIUS_N)"
        echo "  pool manifest -> $GPU_POOL_FILE"
        echo "  MM / --start-xyz-file (random mean [$idx/${#MEAN_FILES[@]}]): $picked -> $dest"
        echo "  run.py will print one line per chain: uniform random mean from pool"
    else
        cp -f "$picked" "$dest"
        echo "  tradius mean pick [$idx/${#MEAN_FILES[@]}]: $picked -> $dest"
    fi
}

GPU_POOL_FILE=""

echo "=== GPU run from previous timestep ==="
echo "  time_step (run-id)=$ts_padded  prev=$prev_step  TOP_N=$TOP_N  GPU_CHAINS=$GPU_CHAINS"

# Ensure previous-step mean exists when pool is available (for tradius-mean and reuse).
if [[ -n "$ts_int" ]]; then
    ensure_step_mean_xyz "$prev_step" || true
fi

if [[ -n "$STARTING_XYZ" ]]; then
    start_out="${RESULTS_DIR}/${prev_step}_mean.xyz"
    echo "  mode: STARTING_XYZ -> $start_out"
    if [[ ! -f "$STARTING_XYZ" ]]; then
        echo "ERROR: STARTING_XYZ='$STARTING_XYZ' not found" >&2
        exit 1
    fi
    cp -f "$STARTING_XYZ" "$start_out"

elif [[ "${START_FROM_PREVIOUS_RANDOM:-0}" == "1" ]]; then
    pool_step=$(resolve_previous_random_pool_step)
    if [[ "$pool_step" == "$prev_step" ]]; then
        start_out="${RESULTS_DIR}/${prev_step}_mean.xyz"
    else
        start_out="${RESULTS_DIR}/${ts_padded}_start_from_${pool_step}_random.xyz"
    fi
    if [[ -n "$PREV_STEP" ]]; then
        echo "  mode: random from step $pool_step (PREV_STEP) -> $start_out"
    else
        echo "  mode: random from step $pool_step (t-$PREV_DISTANCE) -> $start_out"
    fi
    random_pick_from_pool "$pool_step" "$start_out" \
        "Run step $pool_step first so ${RESULTS_DIR}/${pool_step}_*.xyz exist, or set STARTING_XYZ for the first timestep."

elif [[ "${START_FROM_NEXT_RANDOM:-0}" == "1" ]]; then
    pool_step=$(resolve_next_random_pool_step)
    start_out="${RESULTS_DIR}/${ts_padded}_start_from_${pool_step}_random.xyz"
    if [[ -n "$NEXT_STEP" ]]; then
        echo "  mode: random from step $pool_step (NEXT_STEP) -> $start_out"
    else
        echo "  mode: random from step $pool_step (t+$NEXT_DISTANCE) -> $start_out"
    fi
    random_pick_from_pool "$pool_step" "$start_out" \
        "Run step $pool_step first so ${RESULTS_DIR}/${pool_step}_*.xyz exist."

elif [[ "${START_FROM_TRADIUS_RANDOM:-0}" == "1" ]]; then
    start_out="${RESULTS_DIR}/${ts_padded}_start_tradius_N${TRADIUS_N}.xyz"
    echo "  mode: tradius random N=$TRADIUS_N (union top TOP_N at t+/-1..t+/-N) -> $start_out"
    collect_tradius_pool
    apply_tradius_random_start "$start_out" \
        "Need results for at least one timestep in t+/-1..t+/-${TRADIUS_N} under ${RESULTS_DIR}/."

elif [[ "${START_FROM_TRADIUS_MEAN:-0}" == "1" ]]; then
    start_out="${RESULTS_DIR}/${ts_padded}_start_tradius_mean_N${TRADIUS_N}.xyz"
    echo "  mode: tradius mean N=$TRADIUS_N (random {step}_mean.xyz at t+/-1..t+/-N) -> $start_out"
    collect_tradius_means
    apply_tradius_mean_start "$start_out" \
        "Need at least one {step}_mean.xyz in t+/-1..t+/-${TRADIUS_N} under ${RESULTS_DIR}/."

else
    start_out="${RESULTS_DIR}/${prev_step}_mean.xyz"
    echo "  mode: Kabsch mean from step $prev_step -> $start_out"
    collect_top_pool_for_step "$prev_step"
    if [[ ${#TOP_FILES[@]} -eq 0 ]]; then
        echo "ERROR: no pool files for previous step '$prev_step' under ${RESULTS_DIR}/" >&2
        echo "  Expected glob: ${RESULTS_DIR}/${prev_step}_*.xyz (excluding *_mean.xyz)" >&2
        echo "  For the first timestep, set STARTING_XYZ to an initial structure." >&2
        exit 1
    fi
    echo "  averaging ${#TOP_FILES[@]} structures -> $start_out"
    average_top_files_to_mean "$start_out"
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
if [[ -n "${GPU_POOL_FILE:-}" ]]; then
    RUN_CMD+=(--gpu-per-chain-random-pool-file "$GPU_POOL_FILE")
fi
# shellcheck disable=SC2206
RUN_CMD+=( ${EXTRA_RUN_PY_ARGS} )
"${RUN_CMD[@]}"
