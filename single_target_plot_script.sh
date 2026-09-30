#!/bin/bash

if false; then
comment_a=phi0p5_nr2
comment_b=phi0p5_nr2_tournament

PS1A=1.2
PS1B=1.2
PS2A=1.0
PS2B=1.0
LW1A=2.4
LW1B=7.2
LW2A=0.6
LW2B=1.8

NAME1='C$_1-$C$_6$ open'
style="PS1A=$PS1A;PS1B=$PS1B;PS2A=$PS2A;PS2B=$PS2B;LW1A=$LW1A;LW1B=$LW1B;LW2A=$LW2A;LW2B=$LW2B;NAME1='$NAME1'"

gnuplot -e "XMIN=0.00;XMAX=0.95;YMIN=2e-6;YMAX=0.1;$style;RESULTS_DIR_1A='results_fig3_qmax4_open_"$comment_a"';RESULTS_DIR_1B='results_fig3_qmax4_open_"$comment_b"';RESULTS_DIR_2A='results_fig3_qmax4_closed_"$comment_a"';RESULTS_DIR_2B='results_fig3_qmax4_closed_"$comment_b"'" \
  ./scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
gnuplot -e "XMIN=0.00;XMAX=0.95;YMIN=2e-5;YMAX=0.1;$style;RESULTS_DIR_1A='results_fig3_qmax8_open_"$comment_a"';RESULTS_DIR_1B='results_fig3_qmax8_open_"$comment_b"';RESULTS_DIR_2A='results_fig3_qmax8_closed_"$comment_a"';RESULTS_DIR_2B='results_fig3_qmax8_closed_"$comment_b"'" \
  ./scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp

#pdflatex figure_results_fig3_qmax4_open_"$comment".tex
#pdflatex figure_results_fig3_qmax8_open_"$comment".tex
fi

comment_a=phi0p5_nr2

PS1A=1.2
PS2A=1.0
LW1A=2.4
LW2A=0.6

NAME1='C$_1-$C$_6$ open'
style="PS1A=$PS1A;PS2A=$PS2A;LW1A=$LW1A;LW2A=$LW2A;NAME1='$NAME1'"

gnuplot -e "XMIN=0.00;XMAX=0.95;YMIN=2e-6;YMAX=0.1;$style;RESULTS_DIR_1A='results_fig3_qmax4_open_"$comment_a"';RESULTS_DIR_2A='results_fig3_qmax4_closed_"$comment_a"'" \
  ./scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
gnuplot -e "XMIN=0.00;XMAX=0.95;YMIN=2e-5;YMAX=0.1;$style;RESULTS_DIR_1A='results_fig3_qmax8_open_"$comment_a"';RESULTS_DIR_2A='results_fig3_qmax8_closed_"$comment_a"'" \
  ./scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
