#!/bin/bash

gnuplot -e "RESULTS_DIR='results_single_target_qmax4_c1c6_open';RESULTS_DIR2='results_single_target_qmax4_c1c6_closed'" ./scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
gnuplot -e "RESULTS_DIR='results_single_target_qmax8_c1c6_open';RESULTS_DIR2='results_single_target_qmax8_c1c6_closed'" ./scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp

pdflatex figure_results_single_target_qmax4_c1c6_open.tex
pdflatex figure_results_single_target_qmax8_c1c6_open.tex

