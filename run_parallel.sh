#!/bin/bash
#
# Parallel execution script for run.py
# Runs N instances in parallel, where N = number of CPU cores
#
# Usage: ./run_parallel.sh <time_step> <starting_xyz> [tuning_ratio_target]
#
# Example: ./run_parallel.sh 01 xyz/start.xyz 0.5

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <time_step> <starting_xyz> [tuning_ratio_target]"
    echo "  time_step: Timestep identifier (e.g., 01, 02, ...)"
    echo "  starting_xyz: Path to starting XYZ file"
    echo "  tuning_ratio_target: Target tuning ratio (default: 0.5)"
    exit 1
fi

time_step=$1
starting_xyz=$2
tuning_ratio_target=${3:-0.5}  # Default to 0.5 if not provided

# Detect number of CPU cores
N=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Running $N parallel jobs (using $N CPU cores)"
echo "Time step: $time_step"
echo "Starting XYZ: $starting_xyz"
echo "Tuning ratio target: $tuning_ratio_target"
echo ""

# Run N parallel jobs
for i in $(seq 0 $((N - 1)))
do
    python3 run.py \
        --start-xyz-file "$starting_xyz" \
        --target-file "data_/eirik_data/eirik_data_${time_step}.dat" \
        --tuning-ratio-target "$tuning_ratio_target" &
done

# Wait for all background jobs to complete
echo "Waiting for all processes to finish..."
wait
echo "All processes finished"
