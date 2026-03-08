#!/bin/bash
set -euo pipefail

N_WORKERS="${N_WORKERS:-64}"

usage() {
    cat <<EOF
Usage: $0 <time_step> <starting_xyz> [excitation_factor] [tuning_ratio_target]

Launch N_WORKERS parallel instances of run.py for a single time-step.

Positional arguments:
  time_step           Time-step identifier (passed as --run-id)
  starting_xyz        Path to starting xyz file
  excitation_factor   (default: 1.0)
  tuning_ratio_target (default: 0.5)

Environment variables (defaults shown):
  N_WORKERS  $N_WORKERS
EOF
    exit 0
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

if [[ $# -lt 2 ]]; then
    echo "ERROR: time_step and starting_xyz are required" >&2
    usage
fi

time_step="$1"
starting_xyz="$2"
excitation_factor="${3:-1.0}"
tuning_ratio_target="${4:-0.5}"

if [[ ! -f "$starting_xyz" ]]; then
    echo "ERROR: starting xyz file not found: $starting_xyz" >&2
    exit 1
fi

echo "  workers=$N_WORKERS | excitation=$excitation_factor | tuning=$tuning_ratio_target"

for (( worker=0; worker<N_WORKERS; worker++ )); do
    python run.py \
        --run-id "$time_step" \
        --start-xyz-file "$starting_xyz" \
        --target-file "nmm_data/target_${time_step}.dat" \
        --excitation-factor "$excitation_factor" \
        --tuning-ratio-target "$tuning_ratio_target" &
done

wait
echo "  All $N_WORKERS workers finished for time-step $time_step"
