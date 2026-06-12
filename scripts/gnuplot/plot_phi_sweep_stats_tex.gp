#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Two-panel plot: chi2 and RMSD vs Phi from aggregate_phi_sweep_stats.py CSVs.
#
# Each CSV has columns: phi, mean, std (skip 1 header row).
#   phi_chi2_stats.csv — phi, mean_chi2, std_chi2
#   phi_rmsd_stats.csv — phi, mean_rmsd, std_rmsd
#
# Output: standalone LaTeX (<OUTBASE>.tex) -> PDF via pdflatex.
#
# Requirements: gnuplot 5.4+, LaTeX
#
# Example (from repo root, after aggregation):
#   python3 scripts/python/aggregate_phi_sweep_stats.py results_phi_sweep
#   gnuplot scripts/gnuplot/plot_phi_sweep_stats_tex.gp
#
# Override paths and labels:
#   gnuplot -e "RESULTS_DIR='results_phi_sweep';OUTBASE='fig_phi_sweep'" \
#       scripts/gnuplot/plot_phi_sweep_stats_tex.gp
#
# Optional ranges (Y1MIN/Y1MAX apply on log scale when Y1LOG=1):
#   gnuplot -e "XMIN=0;XMAX=1;Y1MIN=0.1;Y1MAX=10" scripts/gnuplot/plot_phi_sweep_stats_tex.gp
#
# Chi2 panel uses log y-axis by default; disable with Y1LOG=0.
# ------------------------------------------------------------------------------

if (!exists("RESULTS_DIR")) RESULTS_DIR = "results_phi_sweep"
if (!exists("DATA_CHI2")) DATA_CHI2 = RESULTS_DIR . "/phi_chi2_stats.csv"
if (!exists("DATA_RMSD")) DATA_RMSD = RESULTS_DIR . "/phi_rmsd_stats.csv"
if (!exists("OUTBASE")) OUTBASE = "figure_phi_sweep"

if (!exists("XLABEL")) XLABEL = 'Phi'
if (!exists("Y1LABEL")) Y1LABEL = 'chi$^2$'
if (!exists("Y2LABEL")) Y2LABEL = 'RMSD (\\AA)'

if (!exists("NAME_CHI2")) NAME_CHI2 = 'mean $\\chi^2$'
if (!exists("NAME_RMSD")) NAME_RMSD = 'mean RMSD'

if (!exists("COL_CHI2")) COL_CHI2 = "#1b9e77"
if (!exists("COL_RMSD")) COL_RMSD = "#7570b3"

if (!exists("LW")) LW = 2.0
if (!exists("PS")) PS = 0.75
if (!exists("PT")) PT = 7
LW = LW + 0
PS = PS + 0
PT = PT + 0
eblw = (LW < 1.0 ? 1.0 : 0.5*LW)

if (!exists("SHOW_KEY")) SHOW_KEY = 1
SHOW_KEY = SHOW_KEY + 0

if (!exists("Y1LOG")) Y1LOG = 1
Y1LOG = Y1LOG + 0

if (!exists("W")) W = 3.35
if (!exists("H")) H = 4.5
if (!exists("MLEFT")) MLEFT = 0.14
if (!exists("MRIGHT")) MRIGHT = 0.98
if (!exists("MBOTTOM")) MBOTTOM = 0.12
if (!exists("MTOP")) MTOP = 0.95
if (!exists("FONT")) FONT = "Latin Modern Roman,10"
if (!exists("LATEX_HEADER")) LATEX_HEADER = ""

if (!exists("RELH1")) RELH1 = 0.5
if (!exists("RELH2")) RELH2 = 0.5
RELH1 = RELH1 + 0
RELH2 = RELH2 + 0
if (RELH1 <= 0 || RELH2 <= 0) print "ERROR: RELH1 and RELH2 must be > 0"
if (RELH1 <= 0 || RELH2 <= 0) exit
RELHSUM = RELH1 + RELH2

if (!exists("MIRROR_TICS")) MIRROR_TICS = 1
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

set border linewidth 1.2
set tics scale 0.75
if (MIRROR_TICS) set xtics mirror
if (MIRROR_TICS) set ytics mirror
if (!MIRROR_TICS) set xtics nomirror
if (!MIRROR_TICS) set ytics nomirror
set mxtics 2
set mytics 2
set grid back xtics ytics lw 0.6 lc rgb "#D0D0D0"

unset title
if (SHOW_KEY) set key top right opaque box lw 0.6
if (!SHOW_KEY) unset key

if (exists("XMIN")) XMIN = XMIN + 0
if (exists("XMAX")) XMAX = XMAX + 0
if (exists("XMIN") && exists("XMAX")) set xrange [XMIN:XMAX]
if (!(exists("XMIN") && exists("XMAX"))) unset xrange

if (exists("Y1MIN")) Y1MIN = Y1MIN + 0
if (exists("Y1MAX")) Y1MAX = Y1MAX + 0
if (exists("Y2MIN")) Y2MIN = Y2MIN + 0
if (exists("Y2MAX")) Y2MAX = Y2MAX + 0

P_CHI2_ERR = "'".DATA_CHI2."' using 1:2:3 skip 1 with yerrorbars lw eblw lc rgb COL_CHI2 pt -1 notitle"
P_CHI2_LINE = "'".DATA_CHI2."' using 1:2 skip 1 with linespoints lw LW pt PT ps PS lc rgb COL_CHI2"
P_RMSD_ERR = "'".DATA_RMSD."' using 1:2:3 skip 1 with yerrorbars lw eblw lc rgb COL_RMSD pt -1 notitle"
P_RMSD_LINE = "'".DATA_RMSD."' using 1:2 skip 1 with linespoints lw LW pt PT ps PS lc rgb COL_RMSD"

if (SHOW_KEY) P_CHI2_LINE = P_CHI2_LINE . " title '".NAME_CHI2."'"
if (!SHOW_KEY) P_CHI2_LINE = P_CHI2_LINE . " notitle"
if (SHOW_KEY) P_RMSD_LINE = P_RMSD_LINE . " title '".NAME_RMSD."'"
if (!SHOW_KEY) P_RMSD_LINE = P_RMSD_LINE . " notitle"

P1_CMD = "plot ".P_CHI2_ERR.", ".P_CHI2_LINE
P2_CMD = "plot ".P_RMSD_ERR.", ".P_RMSD_LINE

set lmargin at screen MLEFT
set rmargin at screen MRIGHT

set multiplot

AVAILH = MTOP - MBOTTOM
H2 = AVAILH * (RELH2 / RELHSUM)
YSPLIT = MBOTTOM + H2

# ---- Panel 1: chi2 (log y by default) ----
set tmargin at screen MTOP
set bmargin at screen YSPLIT
set ylabel Y1LABEL
unset xlabel
set format x ""
if (Y1LOG) set logscale y
if (!Y1LOG) unset logscale y
if (exists("Y1MIN") && exists("Y1MAX")) set yrange [Y1MIN:Y1MAX]
if (!(exists("Y1MIN") && exists("Y1MAX"))) unset yrange
eval P1_CMD

# ---- Panel 2: RMSD ----
unset logscale y
set tmargin at screen YSPLIT
set bmargin at screen MBOTTOM
set ylabel Y2LABEL
set xlabel XLABEL
set format x "%g"
unset key
if (exists("Y2MIN") && exists("Y2MAX")) set yrange [Y2MIN:Y2MAX]
if (!(exists("Y2MIN") && exists("Y2MAX"))) unset yrange
eval P2_CMD

unset multiplot
unset output
set terminal pop

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
