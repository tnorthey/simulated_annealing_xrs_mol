#!/bin/bash
 
number_flag=$1  # e.g. "64_1" "64_2" ...
traj=001
nr=1
desc="sdf_ds_0p025_N8000_"$number_flag""
phi="0p4"
topM=50
#for i in qmax4 qmax8
for i in qmax8
do
# bond 0 5
python3 scripts/python/topM_geometry_statistics.py --skip-closest --closest-selection target_rmsd --rmsd-indices 0,1,2,3,4,5 --time-units fs --time-origin 0 --time-file chd_results/time.dat --topM $topM --bond 0 5 --ymin 1.25 --ymax 2.25 --xmin -1 --xmax 201 --recompute --output-dir chd_results/plots_chd_traj_"$traj"_phi"$phi"_from_prev_nr"$nr"_"$i"_"$desc" chd_results/results_chd_traj_"$traj"_phi"$phi"_from_prev_nr"$nr"_"$i"_"$desc"
# dihedral 0 1 4 5
python3 scripts/python/topM_geometry_statistics.py --skip-closest --closest-selection target_rmsd --rmsd-indices 0,1,2,3,4,5 --time-units fs --time-origin 0 --time-file chd_results/time.dat --topM $topM --dihedral 0 1 4 5 --ymin -40 --ymax 100 --xmin -1 --xmax 201 --recompute --output-dir chd_results/plots_chd_traj_"$traj"_phi"$phi"_from_prev_nr"$nr"_"$i"_"$desc" chd_results/results_chd_traj_"$traj"_phi"$phi"_from_prev_nr"$nr"_"$i"_"$desc"
# dihedral 1 2 3 4
python3 scripts/python/topM_geometry_statistics.py --skip-closest --closest-selection target_rmsd --rmsd-indices 0,1,2,3,4,5 --time-units fs --time-origin 0 --time-file chd_results/time.dat --topM $topM --dihedral 1 2 3 4 --ymin -40 --ymax 100 --xmin -1 --xmax 201 --recompute --output-dir chd_results/plots_chd_traj_"$traj"_phi"$phi"_from_prev_nr"$nr"_"$i"_"$desc" chd_results/results_chd_traj_"$traj"_phi"$phi"_from_prev_nr"$nr"_"$i"_"$desc"
done
