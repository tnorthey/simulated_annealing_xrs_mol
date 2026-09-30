#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# One panel: target scattering curve vs one x-ray fit (.dat).
#
# Both files are whitespace-separated columns q, %Delta I(q).
# Output: standalone LaTeX <OUTBASE>.tex (epslatex), same terminal as
# plot_chi2_rmsd_scatter_tex.gp.
#
# Example (from repo root):
#   gnuplot -e "XMIN=0;XMAX=4;TARGET='results_dir/TARGET_FUNCTION_run.dat';FIT='results_dir/run_000.00100000.dat';CHI2=0.001;OUTBASE='figure_xray_qmax4'" \
#       scripts/gnuplot/plot_dat_fit_vs_target_tex.gp
#   pdflatex figure_xray_qmax4.tex
# ------------------------------------------------------------------------------

if (!exists("TARGET")) TARGET = "target.dat"
if (!exists("FIT")) FIT = "fit.dat"
if (!exists("CHI2")) CHI2 = 1e-3
if (!exists("OUTBASE")) OUTBASE = "figure_xray_fit"
CHI2 = CHI2 + 0

# Scientific notation that stays on a decade boundary (1.000 x 10^n, not 10.000 x 10^{n-1}).
logc = log10(CHI2)
expn = floor(logc + 1e-9)
mant = CHI2 / (10**expn)
FIT_TITLE = sprintf("$\\chi^2 = %.3f\\times 10^{%d}$", mant, expn)

reset

# latex .eps output
set terminal epslatex standalone color colortext 10 font "Helvetica,12" \
    header "\\usepackage{amsmath}"

PAL_LW = 4.0

set style line 1 lt 1 lw PAL_LW lc rgb '#a2142f' dt 1
set style line 2 lt 1 lw PAL_LW lc rgb '#0072bd' dt 2

set style line 102 lc rgb '#808080' lt 0 lw 3
set grid back ls 102

set size 0.8, 0.8

set output OUTBASE . ".tex"

set xlabel "q (\\AA$^{-1}$)" offset 0,0.4
set ylabel "$\\%\\Delta I(q)$" offset 0,0
set mxtics 2
set mytics 2
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics 1

set key top right opaque nobox samplen 2 font ',10'

if (exists("XMIN") && exists("XMAX")) set xrange [XMIN+0.0 : XMAX+0.0]
if (exists("YMIN") && exists("YMAX")) set yrange [YMIN+0.0 : YMAX+0.0]

plot TARGET using 1:2 with lines ls 1 title 'target', \
     FIT using 1:2 with lines ls 2 title FIT_TITLE
