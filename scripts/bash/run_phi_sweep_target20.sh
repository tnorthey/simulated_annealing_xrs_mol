#!/bin/bash
# ============================================================================
# Phi sweep: tuning_ratio_target on xyz/target_20.xyz in test mode.
#
# Temporarily patches input.toml (mode=test, target_file, tuning_ratio_target),
# runs SA for Phi = 0, 0.05, 0.1, ..., 1.0, then restores input.toml.
#
# Usage:
#   ./scripts/bash/run_phi_sweep_target20.sh
#   N_WORKERS=64 ./scripts/bash/run_phi_sweep_target20.sh
#
# Environment overrides (defaults shown):
#   N_WORKERS          4       parallel workers per Phi (use 64 for production)
#   RESULTS_DIR        results_phi_sweep   parent dir; each Phi -> phi_<value>/
#   STARTING_XYZ       xyz/start.xyz   fixed start geometry for all workers
#   EXTRA_RUN_PY_ARGS  (unset) extra args appended to run.py
#
# Patches only three fields in input.toml; all other settings are unchanged.
# Original input.toml is backed up to input.toml.bak and restored on exit.
#
# Expected outputs (per Phi under RESULTS_DIR/phi_<value>/):
#   <run_id>_<fxray>.xyz   best structure per worker (8-field header line 2)
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_TOML="input.toml"
INPUT_BACKUP="${INPUT_TOML}.bak"
TARGET_XYZ="xyz/target_20.xyz"

N_WORKERS="${N_WORKERS:-4}"
RESULTS_PARENT="${RESULTS_DIR:-results_phi_sweep}"
STARTING_XYZ="${STARTING_XYZ:-xyz/start.xyz}"

restore_input_toml() {
    if [[ -f "$INPUT_BACKUP" ]]; then
        cp "$INPUT_BACKUP" "$INPUT_TOML"
        echo "Restored $INPUT_TOML from $INPUT_BACKUP"
    fi
}

patch_input_toml() {
    local phi="$1"
    sed -i \
        -e 's/^mode = .*/mode = "test"/' \
        -e "s|^target_file = .*|target_file = \"${TARGET_XYZ}\"|" \
        -e "s/^tuning_ratio_target = .*/tuning_ratio_target = ${phi}/" \
        "$INPUT_TOML"
}

if [[ ! -f "$INPUT_BACKUP" ]]; then
    cp "$INPUT_TOML" "$INPUT_BACKUP"
    echo "Backed up $INPUT_TOML -> $INPUT_BACKUP"
fi

trap restore_input_toml EXIT

echo "=== Phi sweep on ${TARGET_XYZ} ==="
echo "  results_parent = $RESULTS_PARENT"
echo "  workers        = $N_WORKERS"
echo "  starting_xyz   = $STARTING_XYZ"
echo "  phi grid       = 0.00 .. 1.00 step 0.05 (21 values)"

if [[ ! -f "$STARTING_XYZ" ]]; then
    echo "ERROR: STARTING_XYZ='$STARTING_XYZ' does not exist" >&2
    exit 1
fi

for phi in $(python3 -c 'print(" ".join(f"{i/20:.2f}" for i in range(21)))'); do
    run_id="phi_${phi}"
    results_subdir="${RESULTS_PARENT}/phi_${phi}"
    mkdir -p "$results_subdir"

    echo ""
    echo "--- Phi (tuning_ratio_target) = ${phi} ---"
    patch_input_toml "$phi"

    TARGET_FILE="$TARGET_XYZ" \
    TUNING_RATIO="$phi" \
    RESULTS_DIR="$results_subdir" \
    N_WORKERS="$N_WORKERS" \
    STARTING_XYZ="$STARTING_XYZ" \
    EXTRA_RUN_PY_ARGS="${EXTRA_RUN_PY_ARGS:-}" \
    "$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$run_id"
done

echo ""
echo "=== Phi sweep complete ==="
echo "  Results under: ${RESULTS_PARENT}/phi_*/"
