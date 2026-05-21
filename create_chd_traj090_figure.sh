#!/bin/bash
 
number_flag=$1  # 0,1,2, ...

Traj=090
Phi=0.4
Excitation=1.0
Qmax=8
Qlen=$(awk -v q="$Qmax" 'BEGIN { print int(10 * q + 1) }')

Top_N=128
Description=""$number_flag""
Results_dir="results_chd_traj_"$Traj"_qmax"$Qmax"_"$Description""

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
XMIN=-1 XMAX=201 Y1MIN=1.25 Y1MAX=2.25 Y2MIN=-15 Y2MAX=55 Y3MIN=-40 Y3MAX=70 \
DIHEDRAL_TO3602=0 DIHEDRAL_TO3602B=0 DIHEDRAL_TO3603=0 DIHEDRAL_TO3603B=0 \
XTIC_STEP=50 YTIC_STEP1=0.5 YTIC_STEP2=50 YTIC_STEP3=50 \
SHADEDERRORS1=0 \
PLOTMODE='POINTS' PLOTMODE2='LINES' \
PLOTMODE_DIH='POINTS' PLOTMODEB_DIH='LINES' \
SHOW_KEY=0 \
XLABEL='$t$ (fs)' Y1LABEL='\ce{C1-C6} (\AA)' Y2LABEL='$\phi_{\ce{C1C2C5C6}}\\ (^\circ)$' Y3LABEL='$\phi_{\ce{C2C3C4C5}}\\ (^\circ)$' \
./scripts/bash/plot_bond_dihedral_3stack.sh \
  $sa_bond_file $vmd_bond_file \
  $sa_dihedral1_file $sa_dihedral2_file \
  $vmd_dihedral1_file $vmd_dihedral2_file \
  $pdfout_file $time_file
