#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Dual-axis plot: chi2 (left y, log) and RMSD (right y2) vs Phi.
#
# Input CSVs from aggregate_phi_sweep_stats.py (phi, mean, std; skip 1 header):
#   phi_chi2_stats.csv — phi, mean_chi2, std_chi2
#   phi_rmsd_stats.csv — phi, mean_rmsd, std_rmsd
#
# Output: standalone LaTeX (<OUTBASE>.tex) -> PDF via pdflatex.
#
# Requirements: gnuplot 5.4+, LaTeX
#
# Example (from repo root):
#   python3 scripts/python/aggregate_phi_sweep_stats.py results_phi_sweep
#   gnuplot scripts/gnuplot/plot_phi_sweep_stats_tex.gp
#   pdflatex figure_phi_sweep.tex
#
# Override paths and ranges:
#   gnuplot -e "RESULTS_DIR='results_phi_sweep';OUTBASE='fig_phi_sweep';XMIN=0;XMAX=1" \
#       scripts/gnuplot/plot_phi_sweep_stats_tex.gp
#
# Chi2 uses log y-axis by default (Y1LOG=0 for linear). Y2 is always linear.
# ------------------------------------------------------------------------------

if (!exists("RESULTS_DIR")) RESULTS_DIR = "results_phi_sweep"
if (!exists("DATA_CHI2")) DATA_CHI2 = RESULTS_DIR . "/phi_chi2_stats.csv"
if (!exists("DATA_RMSD")) DATA_RMSD = RESULTS_DIR . "/phi_rmsd_stats.csv"
if (!exists("OUTBASE")) OUTBASE = "figure_phi_sweep"

if (!exists("XLABEL")) XLABEL = '$\Phi$'
if (!exists("Y1LABEL")) Y1LABEL = '$\chi^2$'
if (!exists("Y2LABEL")) Y2LABEL = 'RMSD (\\AA)'

if (!exists("NAME_CHI2")) NAME_CHI2 = '$\chi^2$'
if (!exists("NAME_RMSD")) NAME_RMSD = 'RMSD'

if (!exists("COL_CHI2")) COL_CHI2 = "#0072B2"
if (!exists("COL_RMSD")) COL_RMSD = "#D55E00"

if (!exists("LW")) LW = 1.8
if (!exists("LW_ERR")) LW_ERR = 1.0
if (!exists("PS")) PS = 1.0
if (!exists("PT_CHI2")) PT_CHI2 = 7
if (!exists("PT_RMSD")) PT_RMSD = 5
LW = LW + 0
LW_ERR = LW_ERR + 0
PS = PS + 0
PT_CHI2 = PT_CHI2 + 0
PT_RMSD = PT_RMSD + 0

if (!exists("SHOW_KEY")) SHOW_KEY = 1
SHOW_KEY = SHOW_KEY + 0
if (!exists("KEY_POS")) KEY_POS = "top left"

if (!exists("Y1LOG")) Y1LOG = 1
Y1LOG = Y1LOG + 0

# Single-column journal width (~8.5 cm); modest height for dual-axis overlay.
if (!exists("W")) W = 3.35
if (!exists("H")) H = 2.75
if (!exists("MLEFT")) MLEFT = 0.18
if (!exists("MRIGHT")) MRIGHT = 0.84
if (!exists("MBOTTOM")) MBOTTOM = 0.16
if (!exists("MTOP")) MTOP = 0.92
if (!exists("FONT")) FONT = "Latin Modern Roman,11"
if (!exists("LATEX_HEADER")) LATEX_HEADER = ""

if (!exists("MIRROR_TICS")) MIRROR_TICS = 0
MIRROR_TICS = MIRROR_TICS + 0

is_nonempty_file(f) = int(system(sprintf("bash -lc \"test -s '%s' && echo 1 || echo 0\" ", f)))
HAS_CHI2 = is_nonempty_file(DATA_CHI2)
HAS_RMSD = is_nonempty_file(DATA_RMSD)
if (!HAS_CHI2) print sprintf("ERROR: DATA_CHI2 missing or empty: %s", DATA_CHI2)
if (!HAS_CHI2) exit
if (!HAS_RMSD) print sprintf("ERROR: DATA_RMSD missing or empty: %s", DATA_RMSD)
if (!HAS_RMSD) exit

set datafile separator comma
set encoding utf8

LATEX_HDR = (LATEX_HEADER ne "") ? " header '".LATEX_HEADER."'" : ""
eval "set terminal cairolatex pdf standalone size ".sprintf("%g",W).",".sprintf("%g",H)." font '".FONT."' dashed color".LATEX_HDR
set output sprintf("%s.tex", OUTBASE)

set border linewidth 1.4 front
set tics out scale 0.8
set tics nomirror
if (MIRROR_TICS) set xtics mirror
if (MIRROR_TICS) set ytics mirror
set mxtics 2
set mytics 2
set my2tics 2
set grid back xtics ytics lw 0.5 lc rgb "#E6E6E6"

set lmargin at screen MLEFT
set rmargin at screen MRIGHT
set tmargin at screen MTOP
set bmargin at screen MBOTTOM

set xlabel XLABEL offset 0,0.8
set ylabel Y1LABEL offset 1.2,0
set y2label Y2LABEL offset -1.2,0

eval "set ylabel textcolor rgb '".COL_CHI2."'"
eval "set ytics textcolor rgb '".COL_CHI2."'"
eval "set y2label textcolor rgb '".COL_RMSD."'"
eval "set y2tics textcolor rgb '".COL_RMSD."'"

if (Y1LOG) set logscale y
if (!Y1LOG) unset logscale y
unset logscale y2

if (Y1LOG) set format y "$10^{%L}$"
if (!Y1LOG) set format y "%g"
set format y2 "%.2f"
set format x "%g"

unset title
if (SHOW_KEY) eval "set key ".KEY_POS." opaque box lw 0.8 spacing 1.1 font ',10'"
if (!SHOW_KEY) unset key

if (exists("XMIN")) XMIN = XMIN + 0
if (exists("XMAX")) XMAX = XMAX + 0
if (exists("XMIN") && exists("XMAX")) set xrange [XMIN:XMAX]
if (!(exists("XMIN") && exists("XMAX"))) unset xrange

if (exists("Y1MIN")) Y1MIN = Y1MIN + 0
if (exists("Y1MAX")) Y1MAX = Y1MAX + 0
if (exists("Y2MIN")) Y2MIN = Y2MIN + 0
if (exists("Y2MAX")) Y2MAX = Y2MAX + 0
if (exists("Y1MIN") && exists("Y1MAX")) set yrange [Y1MIN:Y1MAX]
if (!(exists("Y1MIN") && exists("Y1MAX"))) unset yrange
if (exists("Y2MIN") && exists("Y2MAX")) set y2range [Y2MIN:Y2MAX]
if (!(exists("Y2MIN") && exists("Y2MAX"))) unset y2range

TITLE_CHI2 = (SHOW_KEY != 0) ? " title '".NAME_CHI2."'" : " notitle"
TITLE_RMSD = (SHOW_KEY != 0) ? " title '".NAME_RMSD."'" : " notitle"

PLOT_CMD = \
    "'".DATA_CHI2."' using 1:2:3 skip 1 axes x1y1 with yerrorbars lw LW_ERR lc rgb COL_CHI2 pt -1 notitle, " \
  . "'".DATA_CHI2."' using 1:2 skip 1 axes x1y1 with linespoints lw LW pt PT_CHI2 ps PS lc rgb COL_CHI2".TITLE_CHI2.", " \
  . "'".DATA_RMSD."' using 1:2:3 skip 1 axes x1y2 with yerrorbars lw LW_ERR lc rgb COL_RMSD pt -1 notitle, " \
  . "'".DATA_RMSD."' using 1:2 skip 1 axes x1y2 with linespoints lw LW dt 2 pt PT_RMSD ps PS lc rgb COL_RMSD".TITLE_RMSD

eval "plot ".PLOT_CMD

unset output
set terminal pop

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
