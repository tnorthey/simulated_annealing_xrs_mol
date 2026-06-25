#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Scatter: chi^2 vs RMSD from chi2_rmsd.dat (from extract_chi2_rmsd.sh).
#
# Default: one series from ./chi2_rmsd.dat (repo root when run from there).
# Optional second series via RESULTS_DIR2.
#
# Example (from repo root):
#   scripts/bash/extract_chi2_rmsd.sh results_chd_ewald_smoke
#   gnuplot scripts/gnuplot/PLOT_SCATTER_SINGLE_TARGET_QMAX4_NOISE0p0_RMSD_FXRAY_new_tex.gp
#
# Two result directories at repo root:
#   gnuplot -e "RESULTS_DIR='results_open';RESULTS_DIR2='results_closed';NAME1='open';NAME2='closed'" \
#       scripts/gnuplot/PLOT_SCATTER_SINGLE_TARGET_QMAX4_NOISE0p0_RMSD_FXRAY_new_tex.gp
#
# Legacy multi-column analysis files (chi2 in col 4, RMSD in col 5):
#   gnuplot -e "CHI2_COL=4;RMSD_COL=5;DATA='analysis_qmax4_no_constraints.dat'" \
#       scripts/gnuplot/PLOT_SCATTER_SINGLE_TARGET_QMAX4_NOISE0p0_RMSD_FXRAY_new_tex.gp
# ------------------------------------------------------------------------------

if (!exists("RESULTS_DIR")) RESULTS_DIR = "."
if (!exists("RESULTS_DIR2")) RESULTS_DIR2 = ""
if (!exists("DATA")) DATA = RESULTS_DIR . "/chi2_rmsd.dat"
if (!exists("DATA2")) DATA2 = (RESULTS_DIR2 ne "") ? RESULTS_DIR2 . "/chi2_rmsd.dat" : ""
if (!exists("OUTBASE")) OUTBASE = "PLOT_SCATTER_SINGLE_TARGET_QMAX4"
if (!exists("NAME1")) NAME1 = "C1-C6 open"
if (!exists("NAME2")) NAME2 = "C1-C6 closed"

# extract_chi2_rmsd.sh: col1 = chi^2, col2 = RMSD
if (!exists("CHI2_COL")) CHI2_COL = 1
if (!exists("RMSD_COL")) RMSD_COL = 2
CHI2_COL = CHI2_COL + 0
RMSD_COL = RMSD_COL + 0

is_nonempty_file(f) = int(system(sprintf("bash -lc \"test -s '%s' && echo 1 || echo 0\" ", f)))
HAS_DATA = is_nonempty_file(DATA)
HAS_DATA2 = (DATA2 ne "") ? is_nonempty_file(DATA2) : 0
if (!HAS_DATA) print sprintf("ERROR: DATA missing or empty: %s", DATA)
if (!HAS_DATA) exit
if (DATA2 ne "" && !HAS_DATA2) print sprintf("ERROR: DATA2 missing or empty: %s", DATA2)
if (DATA2 ne "" && !HAS_DATA2) exit

reset

# latex .eps output
set terminal epslatex standalone color colortext 10 font "Helvetica,12" \
    header "\\usepackage{amsmath}"

# Custom line styles

LW1= 4.0
LW2 = 0.0
PS = 1.0
PS2 = 1.2

set style line 1 lt 1 pt 7 ps PS lw LW1 lc rgb '#0072bd' # blue
set style line 2 lt 1 pt 7 ps PS lw LW1 lc rgb '#d95319' # orange
set style line 3 lt 1 pt 7 ps PS lw LW1 lc rgb '#edb120' # yellow
set style line 4 lt 1 pt 7 ps PS lw LW1 lc rgb '#7e2f8e' # purple
set style line 5 lt 1 pt 7 ps PS lw LW1 lc rgb '#77ac30' # green
set style line 6 lt 1 pt 7 ps PS lw LW1 lc rgb '#4dbeee' # light-blue
set style line 7 lt 1 pt 6 ps PS2 lw LW1 lc rgb '#a2142f' # red
set style line 8 lt 1 pt 7 ps PS lw LW1 lc rgb '#666666' # grey
set style line 9 lt 1 pt 7 ps PS lw LW1 lc rgb '#99ae52' # olive
set style line 10 lt 1 pt 7 ps PS lw LW1 lc rgb '#000000' # black

set style line 102 lc rgb '#808080' lt 0 lw 3
set grid back ls 102

set size 0.8, 0.8   # Scale up the plot instead

set output OUTBASE . ".tex"

set xtics 0, 0.2, 1.6
set xlabel "RMSD (\\AA)" offset 0,0.4
set mxtics 2

set ytics ("" 10, "" 1, "" 0.1, "$10^{-2}$" 0.01, "$10^{-3}$" 0.001, "$10^{-4}$" 0.0001, "$10^{-5}$" 0.00001, "$10^{-6}$" 0.000001)
set mytics 10 
set ylabel "$\\chi^2$" offset 1.0,-3

#set key bottom right
unset key

if (!exists("XMIN")) XMIN = 0.000
if (!exists("XMAX")) XMAX = 0.44
if (!exists("YMIN")) YMIN = 0.000050
if (!exists("YMAX")) YMAX = 5
XMIN = XMIN + 0
XMAX = XMAX + 0
YMIN = YMIN + 0
YMAX = YMAX + 0
set yrange [YMIN : YMAX]
set xrange [XMIN : XMAX]

#set label 1 'q_{max} = 4 Å^{-1}' @POS
#set logscale x 10
set logscale y 10

USING = sprintf("%d:%d", RMSD_COL, CHI2_COL)
PLOT_CMD = "'".DATA."' u ".USING." w p ls 7 t '".NAME1."'"
if (HAS_DATA2) PLOT_CMD = PLOT_CMD . ", '".DATA2."' u ".USING." w p ls 1 t '".NAME2."'"
eval "plot ".PLOT_CMD

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)

### End
