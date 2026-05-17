#!/bin/bash

number_flag=$1  # 0,1,2, ...

Traj=094
Phi=0.4
Excitation=1.0
Qmax=8
Qlen=$(awk -v q="$Qmax" 'BEGIN { print int(10 * q + 1) }')

Top_N=200
Description=""$number_flag""
Results_dir="results_chd_traj_"$Traj"_phi"$Phi"_qmax"$Qmax"_"$Description""

export EXTRA_RUN_PY_ARGS="--qmax ${Qmax} --qlen ${Qlen}"
export GPU_CHAINS=512

# Initialisation: Time-step 00
i=00
RESULTS_DIR=$Results_dir STARTING_XYZ="xyz/start.xyz" TARGET_FILE="xyz/target_traj"$Traj"/target_"$i".xyz" ./scripts/bash/run_gpu_start.sh $i $Excitation $Phi

for i in {01..75}
do
    RESULTS_DIR=$Results_dir TOP_N=$Top_N TARGET_FILE="xyz/target_traj"$Traj"/target_"$i".xyz" ./scripts/bash/run_gpu_from_previous_timestep.sh $i $Excitation $Phi
done

