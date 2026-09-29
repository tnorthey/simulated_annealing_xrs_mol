#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Scatter: chi^2 vs RMSD from chi2_rmsd.dat (from extract_chi2_rmsd.sh).
#
# Default: one series from ./chi2_rmsd.dat (repo root when run from there).
# Optional series via RESULTS_DIR_1B, RESULTS_DIR_2A, and RESULTS_DIR_2B.
# 1A/1B share one style; 2A/2B share another. B uses a larger point and a
# slightly thicker stroke. Output: figure_<RESULTS_DIR_1A>.tex (override with OUTBASE).
#
# Example (from repo root):
#   scripts/bash/extract_chi2_rmsd.sh results_chd_ewald_smoke
#   gnuplot -e "RESULTS_DIR='results_chd_ewald_smoke'" scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
#   pdflatex figure_results_chd_ewald_smoke.tex
#
# Four result directories (open = series 1, closed = series 2; A uses comment_a, B uses comment_b):
#   gnuplot -e "XMIN=0.00;XMAX=0.95;YMIN=2e-5;YMAX=0.1;RESULTS_DIR_1A='results_fig3_qmax8_open_"$comment_a"';RESULTS_DIR_1B='results_fig3_qmax8_open_"$comment_b"';RESULTS_DIR_2A='results_fig3_qmax8_closed_"$comment_a"';RESULTS_DIR_2B='results_fig3_qmax8_closed_"$comment_b"'" \
#       scripts/gnuplot/plot_chi2_rmsd_scatter_tex.gp
#
# Legacy two result directories (RESULTS_DIR -> 1A, RESULTS_DIR2 -> 2A):
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
if (!exists("RESULTS_DIR_1A")) RESULTS_DIR_1A = RESULTS_DIR
if (!exists("RESULTS_DIR_1B")) RESULTS_DIR_1B = ""
if (!exists("RESULTS_DIR_2A")) RESULTS_DIR_2A = RESULTS_DIR2
if (!exists("RESULTS_DIR_2B")) RESULTS_DIR_2B = ""

if (!exists("DATA_1A")) DATA_1A = exists("DATA") ? DATA : RESULTS_DIR_1A . "/chi2_rmsd.dat"
if (!exists("DATA_1B")) DATA_1B = (RESULTS_DIR_1B ne "") ? RESULTS_DIR_1B . "/chi2_rmsd.dat" : ""
if (!exists("DATA_2A")) DATA_2A = exists("DATA2") ? DATA2 : ((RESULTS_DIR_2A ne "") ? RESULTS_DIR_2A . "/chi2_rmsd.dat" : "")
if (!exists("DATA_2B")) DATA_2B = (RESULTS_DIR_2B ne "") ? RESULTS_DIR_2B . "/chi2_rmsd.dat" : ""

if (!exists("OUTBASE") && ((RESULTS_DIR_1A eq ".") || (RESULTS_DIR_1A eq "./") || (RESULTS_DIR_1A eq ""))) OUTBASE = "figure_chi2_rmsd"
if (!exists("OUTBASE")) OUTBASE = "figure_" . system(sprintf("bash -lc \"printf '%%s' $(basename '%s')\"", RESULTS_DIR_1A))
if (!exists("NAME1")) NAME1 = "C1-C6 open"
if (!exists("NAME2")) NAME2 = "C1-C6 closed"
if (!exists("COL1")) COL1 = "#a2142f"
if (!exists("COL2")) COL2 = "#0072bd"
if (!exists("PT1")) PT1 = 7
if (!exists("PT2")) PT2 = 5
if (!exists("PS1")) PS1 = 1.2
if (!exists("PS1B")) PS1B = 1.8
if (!exists("PS2")) PS2 = 1.0
if (!exists("PS2B")) PS2B = 1.6
if (!exists("LW1")) LW1 = 0
if (!exists("LW1B")) LW1B = 1.2
if (!exists("LW2")) LW2 = 0
if (!exists("LW2B")) LW2B = 1.2
PT1 = PT1 + 0
PT2 = PT2 + 0
PS1 = PS1 + 0
PS1B = PS1B + 0
PS2 = PS2 + 0
PS2B = PS2B + 0
LW1 = LW1 + 0
LW1B = LW1B + 0
LW2 = LW2 + 0
LW2B = LW2B + 0

# extract_chi2_rmsd.sh: col1 = chi^2, col2 = RMSD
if (!exists("CHI2_COL")) CHI2_COL = 1
if (!exists("RMSD_COL")) RMSD_COL = 2
CHI2_COL = CHI2_COL + 0
RMSD_COL = RMSD_COL + 0

is_nonempty_file(f) = int(system(sprintf("bash -lc \"test -s '%s' && echo 1 || echo 0\" ", f)))
HAS_1A = (DATA_1A ne "") ? is_nonempty_file(DATA_1A) : 0
HAS_1B = (DATA_1B ne "") ? is_nonempty_file(DATA_1B) : 0
HAS_2A = (DATA_2A ne "") ? is_nonempty_file(DATA_2A) : 0
HAS_2B = (DATA_2B ne "") ? is_nonempty_file(DATA_2B) : 0
if (!HAS_1A) print sprintf("ERROR: DATA_1A missing or empty: %s", DATA_1A)
if (!HAS_1A) exit
if (DATA_1B ne "" && !HAS_1B) print sprintf("ERROR: DATA_1B missing or empty: %s", DATA_1B)
if (DATA_1B ne "" && !HAS_1B) exit
if (DATA_2A ne "" && !HAS_2A) print sprintf("ERROR: DATA_2A missing or empty: %s", DATA_2A)
if (DATA_2A ne "" && !HAS_2A) exit
if (DATA_2B ne "" && !HAS_2B) print sprintf("ERROR: DATA_2B missing or empty: %s", DATA_2B)
if (DATA_2B ne "" && !HAS_2B) exit

if (!exists("SHOW_KEY")) SHOW_KEY = (HAS_1B || HAS_2A || HAS_2B)
SHOW_KEY = SHOW_KEY + 0

reset

# latex .eps output
set terminal epslatex standalone color colortext 10 font "Helvetica,12" \
    header "\\usepackage{amsmath}"

# Custom line styles

PAL_LW = 4.0
PAL_PS = 1.0
PAL_PS2 = 1.2

set style line 1 lt 1 pt 7 ps PAL_PS lw PAL_LW lc rgb '#0072bd' # blue
set style line 2 lt 1 pt 7 ps PAL_PS lw PAL_LW lc rgb '#d95319' # orange
set style line 3 lt 1 pt 7 ps PAL_PS lw PAL_LW lc rgb '#edb120' # yellow
set style line 4 lt 1 pt 7 ps PAL_PS lw PAL_LW lc rgb '#7e2f8e' # purple
set style line 5 lt 1 pt 7 ps PAL_PS lw PAL_LW lc rgb '#77ac30' # green
set style line 6 lt 1 pt 7 ps PAL_PS lw PAL_LW lc rgb '#4dbeee' # light-blue
set style line 7 lt 1 pt 6 ps PAL_PS2 lw PAL_LW lc rgb '#a2142f' # red
set style line 8 lt 1 pt 7 ps PAL_PS lw PAL_LW lc rgb '#666666' # grey
set style line 9 lt 1 pt 7 ps PAL_PS lw PAL_LW lc rgb '#99ae52' # olive
set style line 10 lt 1 pt 7 ps PAL_PS lw PAL_LW lc rgb '#000000' # black

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
# Omit them to autoscale from data so every series stays visible.
if (exists("XMIN") && exists("XMAX")) set xrange [XMIN+0.0 : XMAX+0.0]
if (exists("YMIN") && exists("YMAX")) set yrange [YMIN+0.0 : YMAX+0.0]

#set label 1 'q_{max} = 4 Å^{-1}' @POS
#set logscale x 10
set logscale y 10

USING = sprintf("%d:%d", RMSD_COL, CHI2_COL)
STYLE1A = sprintf("w p pt %d ps %g lc rgb '%s' lw %g", PT1, PS1, COL1, LW1)
STYLE1B = sprintf("w p pt %d ps %g lc rgb '%s' lw %g", PT1, PS1B, COL1, LW1B)
STYLE2A = sprintf("w p pt %d ps %g lc rgb '%s' lw %g", PT2, PS2, COL2, LW2)
STYLE2B = sprintf("w p pt %d ps %g lc rgb '%s' lw %g", PT2, PS2B, COL2, LW2B)
clause(data, style, title) = "'" . data . "' u " . USING . " " . style . " " . title

# B before the matching A series so the larger markers sit behind.
PLOT_CMD = ""
if (HAS_1B) PLOT_CMD = clause(DATA_1B, STYLE1B, "notitle")
PLOT_CMD = PLOT_CMD . (PLOT_CMD eq "" ? "" : ", ") . clause(DATA_1A, STYLE1A, "t '" . NAME1 . "'")
if (HAS_2B) PLOT_CMD = PLOT_CMD . ", " . clause(DATA_2B, STYLE2B, "notitle")
if (HAS_2A) PLOT_CMD = PLOT_CMD . ", " . clause(DATA_2A, STYLE2A, "t '" . NAME2 . "'")

if (HAS_1B) stats DATA_1B using RMSD_COL nooutput
if (HAS_1B) print sprintf("Series 1B: %s (%d points)", DATA_1B, STATS_records)
stats DATA_1A using RMSD_COL nooutput
print sprintf("Series 1A: %s (%d points)", DATA_1A, STATS_records)
if (HAS_2B) stats DATA_2B using RMSD_COL nooutput
if (HAS_2B) print sprintf("Series 2B: %s (%d points)", DATA_2B, STATS_records)
if (HAS_2A) stats DATA_2A using RMSD_COL nooutput
if (HAS_2A) print sprintf("Series 2A: %s (%d points)", DATA_2A, STATS_records)
eval "plot ".PLOT_CMD

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
