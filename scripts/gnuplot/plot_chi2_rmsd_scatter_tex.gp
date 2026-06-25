#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Scatter: chi^2 vs RMSD from chi2_rmsd.dat (from extract_chi2_rmsd.sh).
#
# Default: one series from ./chi2_rmsd.dat (repo root when run from there).
# Optional second series via RESULTS_DIR2.
# Output: figure_<RESULTS_DIR>.tex (override with OUTBASE).
#
# Example (from repo root):
#   scripts/bash/extract_chi2_rmsd.sh results_chd_ewald_smoke
#   gnuplot -e "RESULTS_DIR='results_chd_ewald_smoke'" scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
#   pdflatex figure_results_chd_ewald_smoke.tex
#
# Two result directories at repo root:
#   gnuplot -e "RESULTS_DIR='results_open';RESULTS_DIR2='results_closed';NAME1='open';NAME2='closed'" \
#       scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
#
# Legacy multi-column analysis files (chi2 in col 4, RMSD in col 5):
#   gnuplot -e "CHI2_COL=4;RMSD_COL=5;DATA='analysis_qmax4_no_constraints.dat'" \
#       scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
#
# Restore fixed publication axis ranges:
#   gnuplot -e "XMIN=0;XMAX=0.44;YMIN=5e-5;YMAX=5;RESULTS_DIR='...';RESULTS_DIR2='...'" \
#       scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
# ------------------------------------------------------------------------------

if (!exists("RESULTS_DIR")) RESULTS_DIR = "."
if (!exists("RESULTS_DIR2")) RESULTS_DIR2 = ""
if (!exists("DATA")) DATA = RESULTS_DIR . "/chi2_rmsd.dat"
if (!exists("DATA2")) DATA2 = (RESULTS_DIR2 ne "") ? RESULTS_DIR2 . "/chi2_rmsd.dat" : ""
if (!exists("OUTBASE") && ((RESULTS_DIR eq ".") || (RESULTS_DIR eq "./"))) OUTBASE = "figure_chi2_rmsd"
if (!exists("OUTBASE")) OUTBASE = "figure_" . system(sprintf("bash -lc \"printf '%%s' $(basename '%s')\"", RESULTS_DIR))
if (!exists("NAME1")) NAME1 = "C1-C6 open"
if (!exists("NAME2")) NAME2 = "C1-C6 closed"
if (!exists("COL1")) COL1 = "#a2142f"
if (!exists("COL2")) COL2 = "#0072bd"
if (!exists("PT1")) PT1 = 7
if (!exists("PT2")) PT2 = 5
if (!exists("PS1")) PS1 = 1.2
if (!exists("PS2")) PS2 = 1.0
PT1 = PT1 + 0
PT2 = PT2 + 0
PS1 = PS1 + 0
PS2 = PS2 + 0

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

if (!exists("SHOW_KEY")) SHOW_KEY = HAS_DATA2
SHOW_KEY = SHOW_KEY + 0

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

if (SHOW_KEY) set key bottom right opaque box lw 0.8 spacing 1.1 font ',10'
if (!SHOW_KEY) unset key

# Fixed ranges only when passed via -e (e.g. XMIN=0;XMAX=0.44;YMIN=5e-5;YMAX=5).
# Omit them to autoscale from data so both series stay visible.
if (exists("XMIN") && exists("XMAX")) set xrange [XMIN+0.0 : XMAX+0.0]
if (exists("YMIN") && exists("YMAX")) set yrange [YMIN+0.0 : YMAX+0.0]

#set label 1 'q_{max} = 4 Å^{-1}' @POS
#set logscale x 10
set logscale y 10

USING = sprintf("%d:%d", RMSD_COL, CHI2_COL)
STYLE1 = sprintf("w p pt %d ps %g lc rgb '%s' lw 0", PT1, PS1, COL1)
STYLE2 = sprintf("w p pt %d ps %g lc rgb '%s' lw 0", PT2, PS2, COL2)
PLOT_CMD = "'".DATA."' u ".USING." ".STYLE1." t '".NAME1."'"
if (HAS_DATA2) PLOT_CMD = PLOT_CMD . ", '".DATA2."' u ".USING." ".STYLE2." t '".NAME2."'"

stats DATA using RMSD_COL nooutput
print sprintf("Series 1: %s (%d points)", DATA, STATS_records)
if (HAS_DATA2) stats DATA2 using RMSD_COL nooutput
if (HAS_DATA2) print sprintf("Series 2: %s (%d points)", DATA2, STATS_records)
eval "plot ".PLOT_CMD

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
