#!/bin/bash
# ============================================================================
# Run CHD simulated annealing in test + Ewald mode.
#
# Target signal I(q,theta,phi) is generated from xyz/target_20.xyz via IAM
# on a 3D Ewald grid (q: 0-4 Å⁻¹, 41 pts; theta: 0-π, 21 pts; phi: 0-2π,
# 21 pts).  SA starts every worker from xyz/start.xyz.
#
# Uses input_chd_ewald.toml as the base config; CLI flags in EXTRA_RUN_PY_ARGS
# enforce the same grid so edits to the TOML cannot silently change the grid.
#
# Usage:
#   ./scripts/bash/run_chd_ewald_test.sh
#   N_WORKERS=8 ./scripts/bash/run_chd_ewald_test.sh
#
# Environment overrides (defaults shown):
#   N_WORKERS          64      parallel workers
#   RESULTS_DIR        results_chd_ewald   output directory
#   TUNING_RATIO       0.5
#
# Excitation factor is not exposed here: this workflow uses test mode without PCD,
# so --excitation-factor has no effect on the target; run_cpu_parallel.sh keeps
# its default (1.0) for the positional argument it passes through.
#
# Expected outputs (inside RESULTS_DIR):
#   TARGET_FUNCTION_<run_id>.dat   rotational average I(q)
#   target_function.npy            full 3D target I(q,theta,phi)  shape (41,21,21)
#   <run_id>_<worker>.<fxray>.xyz  best structure per worker
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export STARTING_XYZ="${STARTING_XYZ:-xyz/start.xyz}"
export TARGET_FILE="${TARGET_FILE:-xyz/target_20.xyz}"
export RESULTS_DIR="${RESULTS_DIR:-results_chd_ewald}"
export N_WORKERS="${N_WORKERS:-64}"
export TUNING_RATIO="${TUNING_RATIO:-0.5}"

export EXTRA_RUN_PY_ARGS="${EXTRA_RUN_PY_ARGS:---config input_chd_ewald.toml --mode test --ewald-mode --qmin 0.0 --qmax 4.0 --qlen 41 --tmin 0.0 --tmax 1.0 --tlen 21 --pmin 0.0 --pmax 2.0 --plen 21}"

RUN_ID="${RUN_ID:-chd_ewald}"

mkdir -p "$RESULTS_DIR"

echo "=== CHD Ewald test-mode run ==="
echo "  config       = input_chd_ewald.toml"
echo "  target       = $TARGET_FILE"
echo "  start        = $STARTING_XYZ"
echo "  results_dir  = $RESULTS_DIR"
echo "  workers      = $N_WORKERS"
echo "  grid         = q(0.0-4.0, 41) theta(0-π, 21) phi(0-2π, 21)"

"$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$RUN_ID"
