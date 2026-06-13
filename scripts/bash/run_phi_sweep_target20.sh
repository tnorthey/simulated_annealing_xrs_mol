#!/bin/bash
# ============================================================================
# Phi sweep: tuning_ratio_target on xyz/target_20.xyz in test mode.
#
# Reads input.toml unchanged and passes sweep overrides via run.py CLI flags
# (--mode test, --target-file, --tuning-ratio-target). Does not modify
# input.toml or input.toml.bak.
#
# Usage:
#   ./scripts/bash/run_phi_sweep_target20.sh              # CPU (default)
#   ./scripts/bash/run_phi_sweep_target20.sh --gpu        # single CUDA job per Phi
#   N_WORKERS=64 ./scripts/bash/run_phi_sweep_target20.sh --cpu
#   GPU_CHAINS=1024 ./scripts/bash/run_phi_sweep_target20.sh --gpu
#
# Options:
#   --cpu   Launch N_WORKERS parallel CPU jobs per Phi (default)
#   --gpu   Launch one CUDA job per Phi (GPU_CHAINS independent chains)
#   -h, --help
#
# Environment overrides (defaults shown):
#   BACKEND            cpu     cpu or gpu (overridden by --cpu / --gpu)
#   N_WORKERS          4       CPU only: parallel workers per Phi
#   GPU_CHAINS         1024    GPU only: chains per Phi run
#   BOLTZMANN_SAMPLING 1       GPU only: pass --sampling for per-chain Boltzmann (0 to disable)
#   RESULTS_DIR        results_phi_sweep   parent dir; each Phi -> phi_<value>/
#   STARTING_XYZ       xyz/start.xyz   fixed start geometry
#   EXTRA_RUN_PY_ARGS  (unset) extra args appended to run.py
#
# Expected outputs (per Phi under RESULTS_DIR/phi_<value>/):
#   CPU: <run_id>_<fxray>.xyz per worker
#   GPU: <run_id>_<fxray>.xyz per chain (multi-chain mode)
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TARGET_XYZ="xyz/target_20.xyz"

BACKEND="${BACKEND:-cpu}"
N_WORKERS="${N_WORKERS:-4}"
GPU_CHAINS="${GPU_CHAINS:-1024}"
RESULTS_PARENT="${RESULTS_DIR:-results_phi_sweep}"
STARTING_XYZ="${STARTING_XYZ:-xyz/start.xyz}"

usage() {
    sed -n '3,32p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            BACKEND=cpu
            shift
            ;;
        --gpu)
            BACKEND=gpu
            shift
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

case "$BACKEND" in
    cpu|gpu) ;;
    *)
        echo "ERROR: BACKEND must be 'cpu' or 'gpu' (got '$BACKEND')" >&2
        exit 1
        ;;
esac

echo "=== Phi sweep on ${TARGET_XYZ} ==="
echo "  backend        = $BACKEND"
echo "  config         = input.toml (read-only)"
echo "  results_parent = $RESULTS_PARENT"
echo "  starting_xyz   = $STARTING_XYZ"
echo "  phi grid       = 0.00 .. 1.00 step 0.05 (21 values)"
if [[ "$BACKEND" == "cpu" ]]; then
    echo "  workers        = $N_WORKERS"
else
    echo "  gpu_chains     = $GPU_CHAINS"
    echo "  boltzmann      = ${BOLTZMANN_SAMPLING:-1} (per-chain when GPU_CHAINS > 1)"
fi

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

    # Sweep overrides via CLI; input.toml is not modified.
    sweep_run_py_args="--mode test ${EXTRA_RUN_PY_ARGS:-}"

    if [[ "$BACKEND" == "cpu" ]]; then
        TARGET_FILE="$TARGET_XYZ" \
        TUNING_RATIO="$phi" \
        RESULTS_DIR="$results_subdir" \
        N_WORKERS="$N_WORKERS" \
        STARTING_XYZ="$STARTING_XYZ" \
        EXTRA_RUN_PY_ARGS="$sweep_run_py_args" \
        "$REPO_ROOT/scripts/bash/run_cpu_parallel.sh" "$run_id"
    else
        CONFIG="input.toml" \
        TARGET_FILE="$TARGET_XYZ" \
        TUNING_RATIO="$phi" \
        RESULTS_DIR="$results_subdir" \
        GPU_CHAINS="$GPU_CHAINS" \
        STARTING_XYZ="$STARTING_XYZ" \
        EXTRA_RUN_PY_ARGS="$sweep_run_py_args" \
        "$REPO_ROOT/scripts/bash/run_gpu_start.sh" "$run_id"
    fi
done

echo ""
echo "=== Phi sweep complete ==="
echo "  Results under: ${RESULTS_PARENT}/phi_*/"
