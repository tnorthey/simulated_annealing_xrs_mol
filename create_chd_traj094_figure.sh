#!/bin/bash
 
number_flag=$1  # 0,1,2, ...

Traj=094
Phi=0.4
Excitation=1.0
Qmax=8
Qlen=$(awk -v q="$Qmax" 'BEGIN { print int(10 * q + 1) }')

Top_N=1024
Description="basic_random_"$number_flag""
Results_dir="results_chd_traj_"$Traj"_phi"$Phi"_qmax"$Qmax"_"$Description""

time_file="chd_results/time.dat"
sa_bond_file="$Results_dir/plots_"$Results_dir"/topM_geometry_bond-0-5_topM-"$Top_N".csv"
echo $sa_bond_file
vmd_bond_file="chd_results/vmd_c1c6_traj"$Traj".csv"
sa_dihedral1_file="$Results_dir/plots_"$Results_dir"/topM_geometry_dihedral-0-1-4-5_topM-"$Top_N".csv"
vmd_dihedral1_file="chd_results/vmd_dihdral0145_traj"$Traj".csv"
sa_dihedral2_file="$Results_dir/plots_"$Results_dir"/topM_geometry_dihedral-1-2-3-4_topM-"$Top_N".csv"
vmd_dihedral2_file="chd_results/vmd_dihdral1234_traj"$Traj".csv"

pdfout_file="figure_"$Results_dir".pdf"

W=4 H=4 RELH1=0.33 RELH2=0.33 RELH3=0.34 \
XMIN=-1 XMAX=201 Y1MIN=1.25 Y1MAX=6.5 Y2MIN=-60 Y2MAX=200 Y3MIN=-165 Y3MAX=165 \
XTIC_STEP=50 YTIC_STEP1=1 YTIC_STEP2=50 YTIC_STEP3=50 \
SHOW_KEY=0 \
XLABEL='$t$ (fs)' Y1LABEL='\ce{C1-C6} (\AA)' Y2LABEL='$\phi_{(\ce{C2-C3-C4-C5})}\\ (^\circ)$' Y3LABEL='$\phi_{(\ce{C1-C2-C5-C6})}\\ (^\circ)$' \
./scripts/bash/plot_bond_dihedral_3stack.sh \
  $sa_bond_file $vmd_bond_file \
  $sa_dihedral1_file $sa_dihedral2_file \
  $vmd_dihedral1_file $vmd_dihedral2_file \
  $pdfout_file $time_file
