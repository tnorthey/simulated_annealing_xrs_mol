#!/usr/bin/env gnuplot
#
# Publication figure: overlay two PCD curves from calculate_iam.py (see run_iam_pcd_comparison.sh).
#
# Usage (from repository root):
#   gnuplot plot_iam_pcd_compare.gp
#
# Override paths or output prefix:
#   gnuplot -e "FILE_A='out_a.dat'; FILE_B='out_b.dat'; OUT_PREFIX='myfig'" plot_iam_pcd_compare.gp
#
# Produces: <OUT_PREFIX>.png

if (!exists("FILE_A")) FILE_A = "results/iam_pcd_corr_plus44.dat"
if (!exists("FILE_B")) FILE_B = "results/iam_pcd_ion_neut_iam.dat"
if (!exists("OUT_PREFIX")) OUT_PREFIX = "iam_pcd_compare"

# ~4.2 x 2.8 in at 300 dpi (publication-friendly); adjust size or add -e "DPI=..." if needed
if (!exists("PNG_W")) PNG_W = 1260
if (!exists("PNG_H")) PNG_H = 840

set terminal pngcairo enhanced color font "Helvetica,12" size PNG_W,PNG_H linewidth 2
set output OUT_PREFIX . ".png"
set encoding utf8

# Line styles: distinct colors (color-blind friendly pair)
set style line 1 lw 2.2 lc rgb "#0173B2"
set style line 2 lw 2.2 lc rgb "#DE8F05"

set border lw 1
set grid lw 0.5 lt 1 lc rgb "#cccccc"
set tics nomirror
set mxtics 2
set mytics 2

# Enhanced text: superscript on inverse Å
set xlabel 'q (Å^{-1})'
set ylabel 'PCD (%)'

set key top right samplen 2.5 spacing 1.15 width -4

# Match input.toml q range (adjust if you change --qmin/--qmax in calculate_iam.py)
set xrange [0.001:4.0]

plot FILE_A using 1:2 with lines ls 1 \
        title 'PCD + correction; ref. CCSD neut. +44', \
     FILE_B using 1:2 with lines ls 2 \
        title 'PCD + ion; ref. CCSD neut. IAM'

unset output
