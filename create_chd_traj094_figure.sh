#!/bin/bash

number_flag=$1  # e.g. "64_1" "64_2" ...
traj=094
qmax=8
phi="0p4"
nr=4
topM=50
desc="sdf_ds_0p012_N8000_"$number_flag""
#desc="sdf_boltzmann_"$number_flag""
#desc="sdf_boltzmann"
results_dir="chd_results"
time_file="$results_dir/time.dat"
sa_bond_file="$results_dir/plots_chd_traj_"$traj"_phi"$phi"_from_prev_nr"$nr"_qmax"$qmax"_"$desc"/topM_geometry_bond-0-5_topM-"$topM".csv"
echo $sa_bond_file
vmd_bond_file="$results_dir/vmd_c1c6_traj"$traj".csv"
sa_dihedral1_file="$results_dir/plots_chd_traj_"$traj"_phi"$phi"_from_prev_nr"$nr"_qmax"$qmax"_"$desc"/topM_geometry_dihedral-0-1-4-5_topM-"$topM".csv"
vmd_dihedral1_file="$results_dir/vmd_dihdral0145_traj"$traj".csv"
sa_dihedral2_file="$results_dir/plots_chd_traj_"$traj"_phi"$phi"_from_prev_nr"$nr"_qmax"$qmax"_"$desc"/topM_geometry_dihedral-1-2-3-4_topM-"$topM".csv"
vmd_dihedral2_file="$results_dir/vmd_dihdral1234_traj"$traj".csv"

pdfout_file="chd_traj"$traj"_qmax"$qmax"_phi_"$phi"_nr"$nr"_"$desc"_figure.pdf"

W=4 H=4 RELH1=0.33 RELH2=0.33 RELH3=0.34 \
XMIN=-1 XMAX=201 Y1MIN=1.25 Y1MAX=6.5 Y2MIN=-15 Y2MAX=115 Y3MIN=-85 Y3MAX=125 \
XTIC_STEP=50 YTIC_STEP1=1 YTIC_STEP2=50 YTIC_STEP3=50 \
SHOW_KEY=0 \
XLABEL='$t$ (fs)' Y1LABEL='\ce{C1-C6} (\AA)' Y2LABEL='$\phi_{(\ce{C2-C3-C4-C5})}\\ (^\circ)$' Y3LABEL='$\phi_{(\ce{C1-C2-C5-C6})}\\ (^\circ)$' \
./scripts/bash/plot_bond_dihedral_3stack.sh \
  $sa_bond_file $vmd_bond_file \
  $sa_dihedral1_file $sa_dihedral2_file \
  $vmd_dihedral1_file $vmd_dihedral2_file \
  $pdfout_file $time_file
