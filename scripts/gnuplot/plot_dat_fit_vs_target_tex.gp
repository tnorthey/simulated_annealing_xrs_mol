#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# One panel: target scattering curve vs one x-ray fit (.dat).
#
# Both files are whitespace-separated columns q, %Delta I(q).
# Output: standalone LaTeX <OUTBASE>.tex (epslatex), same terminal as
# plot_chi2_rmsd_scatter_tex.gp.
#
# Example (from repo root):
#   gnuplot -e "XMIN=0;XMAX=4;TARGET='results_dir/TARGET_FUNCTION_run.dat';FIT='results_dir/run_000.00100000.dat';OUTBASE='figure_xray_qmax4'" \
#       scripts/gnuplot/plot_dat_fit_vs_target_tex.gp
#   pdflatex figure_xray_qmax4.tex
# ------------------------------------------------------------------------------

if (!exists("TARGET")) TARGET = "target.dat"
if (!exists("FIT")) FIT = "fit.dat"
if (!exists("OUTBASE")) OUTBASE = "figure_xray_fit"

reset

# Same width as the previous 0.8-scaled default canvas. Height is 0.8 of 1.665 in.
set terminal epslatex standalone color colortext 10 font "Helvetica,12" \
    header "\\usepackage{amsmath}" size 4.0, 1.332

# pt 6: open circle. pt 2: cross. Target is black and solid; fit is grey and dashed.
PS_TARGET = 1.05
PS_FIT = 1.125
LW_PT = 1.6

set style line 1 pt 6 ps PS_TARGET lw LW_PT lc rgb '#000000' dt 1
set style line 2 pt 2 ps PS_FIT lw LW_PT lc rgb '#666666' dt 2

set style line 102 lc rgb '#808080' lt 0 lw 2
set grid back ls 102

set output OUTBASE . ".tex"

set xlabel "q (\\AA$^{-1}$)" offset 0,0.4
set ylabel "$\\%\\Delta I(q)$" offset 1,0
set mxtics 2
unset mytics
set ytics 10
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics 1

set key bottom right opaque nobox spacing 2.6 font ',9'

if (!exists("YMIN")) YMIN = -16
if (!exists("YMAX")) YMAX = 12
if (exists("XMIN") && exists("XMAX")) set xrange [XMIN+0.0 : XMAX+0.0]
set yrange [YMIN+0.0 : YMAX+0.0]

plot TARGET using 1:2 with linespoints ls 1 title "$I_\\mathrm{target}(q)$", \
     FIT using 1:2 with linespoints ls 2 title "$\\chi^2 = 10^{-3}$ fit"
