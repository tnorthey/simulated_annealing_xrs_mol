#!/bin/bash

for i in {17..42}
do
    RESULTS_DIR="results_abi_phi0p1_from_gs" TUNING_RATIO=0.10 STARTING_XYZ="chd+_data/CCSD-neut.xyz" N_WORKERS=64 EXCITATION_FACTOR=0.628 TARGET_FILE="chd+_data/eirik_data_"$i".dat" ./run_parallel.sh $i
done
