#!/bin/bash
#
time_step=$1
starting_xyz=$2
excitation_factor=${3:-1.0}  # Default to 1.0 if not provided
tuning_ratio_target=${4:-0.5}  # Default to 0.5 if not provided

# run multiple times to make a larger initial pool
for i in {0..63}
do
    python run.py \
        --run-id "$time_step" \
        --start-xyz-file "$starting_xyz" \
        --target-file "data_/eirik_data/eirik_data_${time_step}.dat" \
        --excitation-factor "$excitation_factor" \
        --tuning-ratio-target "$tuning_ratio_target" &
done

sleep 0.1 # For sequential output
echo "Waiting for processes to finish" 
wait $(jobs -p)
echo "All processes finished"


