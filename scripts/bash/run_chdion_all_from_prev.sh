#!/bin/bash

for i in {18..42}
do
    RESULTS_DIR="results_abi_phi0p1_from_prev" TUNING_RATIO=0.10 START_FROM_PREV_MEAN=1 TOP_N=200 N_WORKERS=64 EXCITATION_FACTOR=0.628 TARGET_FILE="chd+_data/eirik_data_"$i".dat" ./run_parallel.sh $i
done
