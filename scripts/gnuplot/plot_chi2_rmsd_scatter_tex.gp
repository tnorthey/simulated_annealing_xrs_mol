#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Scatter: chi^2 vs RMSD from chi2_rmsd.dat (from extract_chi2_rmsd.sh).
#
# Default: one series from ./chi2_rmsd.dat (repo root when run from there).
# Optional series via RESULTS_DIR_1B, RESULTS_DIR_2A, and RESULTS_DIR_2B.
# Series 1 (open) is open circles with a center dot. Series 2 (closed) is crosses.
# Point size is the same for A and B. A uses a thinner stroke than B.
# Open strokes are 4x the forcefield-default strokes.
# Each series also gets an RMSD box (min, Q1, median, Q3, max) above the axes.
# CHI2_RATIO selects which points enter the boxes (the scatter still shows all):
#   1    every in-range point (default)
#   0.5  chi^2 at or below the median (STATS_median)
#   0.25 chi^2 at or below the lower quartile (STATS_lo_quartile)
# Override per series with PS1A PS1B PS2A PS2B and LW1A LW1B LW2A LW2B.
# Output: figure_<RESULTS_DIR_1A>.tex (override with OUTBASE).
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
if (!exists("NAME1")) NAME1 = 'C$_1-$C$_6$ open'
if (!exists("NAME2")) NAME2 = "Forcefield default"
if (!exists("COL1")) COL1 = "#a2142f"
if (!exists("COL2")) COL2 = "#0072bd"
# pt 6: open circle with a center dot. pt 2: cross.
if (!exists("PT1")) PT1 = 6
if (!exists("PT1B")) PT1B = 6
if (!exists("PT2")) PT2 = 2
if (!exists("PT2B")) PT2B = 2
if (!exists("PS1A")) PS1A = exists("PS1") ? PS1 : 1.2
if (!exists("PS1B")) PS1B = 1.2
if (!exists("PS2A")) PS2A = exists("PS2") ? PS2 : 1.0
if (!exists("PS2B")) PS2B = 1.0
if (!exists("LW1A")) LW1A = exists("LW1") ? LW1 : 2.4
if (!exists("LW1B")) LW1B = 7.2
if (!exists("LW2A")) LW2A = exists("LW2") ? LW2 : 0.6
if (!exists("LW2B")) LW2B = 1.8
PT1 = PT1 + 0
PT2 = PT2 + 0
PT1B = PT1B + 0
PT2B = PT2B + 0
PS1A = PS1A + 0
PS1B = PS1B + 0
PS2A = PS2A + 0
PS2B = PS2B + 0
LW1A = LW1A + 0
LW1B = LW1B + 0
LW2A = LW2A + 0
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

# 1: every in-range point. 0.5: chi^2 <= median. 0.25: chi^2 <= lower quartile.
if (!exists("CHI2_RATIO")) CHI2_RATIO = 1
CHI2_RATIO = CHI2_RATIO + 0
FILTER_Q1 = (abs(CHI2_RATIO - 0.25) < 1e-6)
FILTER_MED = (abs(CHI2_RATIO - 0.5) < 1e-6)
FILTER_CHI2 = FILTER_Q1 || FILTER_MED
if (!(FILTER_CHI2 || abs(CHI2_RATIO - 1) < 1e-6)) {
    print "ERROR: CHI2_RATIO must be 1 (all points), 0.5 (chi^2 <= median), or 0.25 (chi^2 <= lower quartile)"
    exit
}

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
set tmargin 4       # room above the axes for the RMSD boxes

set output OUTBASE . ".tex"

set xtics 0, 0.2, 1.6
set xlabel "RMSD (\\AA)" offset 0,0.4
set mxtics 2

set ytics ("" 10, "" 1, "" 0.1, "$10^{-2}$" 0.01, "$10^{-3}$" 0.001, "$10^{-4}$" 0.0001, "$10^{-5}$" 0.00001, "$10^{-6}$" 0.000001)
set mytics 10 
set ylabel "$\\chi^2$" offset 0,0

if (SHOW_KEY) set key bottom right opaque nobox spacing 3.0 font ',10'
if (!SHOW_KEY) unset key

# Fixed ranges only when passed via -e (e.g. XMIN=0;XMAX=0.44;YMIN=5e-5;YMAX=5).
# Omit them to autoscale from data so every series stays visible.
if (exists("XMIN") && exists("XMAX")) set xrange [XMIN+0.0 : XMAX+0.0]
if (exists("YMIN") && exists("YMAX")) set yrange [YMIN+0.0 : YMAX+0.0]

#set label 1 'q_{max} = 4 Å^{-1}' @POS
#set logscale x 10
set logscale y 10

USING = sprintf("%d:%d", RMSD_COL, CHI2_COL)
STYLE1A = sprintf("w p pt %d ps %g lc rgb '%s' lw %g", PT1, PS1A, COL1, LW1A)
STYLE1B = sprintf("w p pt %d ps %g lc rgb '%s' lw %g", PT1B, PS1B, COL1, LW1B)
STYLE2A = sprintf("w p pt %d ps %g lc rgb '%s' lw %g", PT2, PS2A, COL2, LW2A)
STYLE2B = sprintf("w p pt %d ps %g lc rgb '%s' lw %g", PT2B, PS2B, COL2, LW2B)
clause(data, style, title) = "'" . data . "' u " . USING . " " . style . " " . title

# B first so the legend, which follows plot order, shows the thicker B markers.
PLOT_CMD = ""
if (HAS_1B) PLOT_CMD = clause(DATA_1B, STYLE1B, "t '" . NAME1 . "'")
if (HAS_2B) PLOT_CMD = PLOT_CMD . (PLOT_CMD eq "" ? "" : ", ") . clause(DATA_2B, STYLE2B, "t '" . NAME2 . "'")
PLOT_CMD = PLOT_CMD . (PLOT_CMD eq "" ? "" : ", ") . clause(DATA_1A, STYLE1A, HAS_1B ? "notitle" : ("t '" . NAME1 . "'"))
if (HAS_2A) PLOT_CMD = PLOT_CMD . ", " . clause(DATA_2A, STYLE2A, HAS_2B ? "notitle" : ("t '" . NAME2 . "'"))

# Both columns, matching the plot. A one-column "using RMSD" makes the row
# index the x value, so a tight xrange reports every point out of range and
# leaves STATS_records undefined.
# RMSD five-number summary (min, Q1, median, Q3, max).
# CHI2_RATIO 0.5 / 0.25 keeps points with chi^2 at or below stats' median
# or lower quartile. The scatter itself still includes every point.

# BOX_DATA is the series file. Sets NBOX, NTOT, BOX_CUT, and the five-number
# summary BOX_MIN BOX_Q1 BOX_MED BOX_Q3 BOX_MAX when NBOX > 0.
rmsd_box_stats = "stats BOX_DATA using RMSD_COL:CHI2_COL nooutput; " \
    . "NBOX = exists('STATS_records') ? STATS_records : 0; NTOT = NBOX; BOX_CUT = 0; " \
    . "if (NBOX > 0 && FILTER_CHI2) { " \
    . "BOX_CUT = FILTER_Q1 ? STATS_lo_quartile_y : STATS_median_y; " \
    . "stats BOX_DATA using (column(CHI2_COL) <= BOX_CUT ? column(RMSD_COL) : 1/0):(column(CHI2_COL) <= BOX_CUT ? column(CHI2_COL) : 1/0) nooutput; " \
    . "NBOX = exists('STATS_records') ? STATS_records : 0; " \
    . "}; " \
    . "if (NBOX > 0) { " \
    . "BOX_MIN = STATS_min_x; BOX_Q1 = STATS_lo_quartile_x; BOX_MED = STATS_median_x; " \
    . "BOX_Q3 = STATS_up_quartile_x; BOX_MAX = STATS_max_x; " \
    . "}"
series_note(n, ntot, cut) = FILTER_CHI2 \
    ? sprintf(" (%d of %d points, chi^2 <= %s %.4g)", n, ntot, FILTER_Q1 ? "lower quartile" : "median", cut) \
    : sprintf(" (%d points in range)", n)

if (HAS_1B) {
    BOX_DATA = DATA_1B
    eval rmsd_box_stats
    N1B = NBOX
    if (N1B > 0) {
        MIN1B = BOX_MIN; Q11B = BOX_Q1; MED1B = BOX_MED; Q31B = BOX_Q3; MAX1B = BOX_MAX
    }
    print "Series 1B: " . DATA_1B . series_note(N1B, NTOT, BOX_CUT)
} else {
    N1B = 0
}
BOX_DATA = DATA_1A
eval rmsd_box_stats
N1A = NBOX
if (N1A > 0) {
    MIN1A = BOX_MIN; Q11A = BOX_Q1; MED1A = BOX_MED; Q31A = BOX_Q3; MAX1A = BOX_MAX
}
print "Series 1A: " . DATA_1A . series_note(N1A, NTOT, BOX_CUT)
if (HAS_2B) {
    BOX_DATA = DATA_2B
    eval rmsd_box_stats
    N2B = NBOX
    if (N2B > 0) {
        MIN2B = BOX_MIN; Q12B = BOX_Q1; MED2B = BOX_MED; Q32B = BOX_Q3; MAX2B = BOX_MAX
    }
    print "Series 2B: " . DATA_2B . series_note(N2B, NTOT, BOX_CUT)
} else {
    N2B = 0
}
if (HAS_2A) {
    BOX_DATA = DATA_2A
    eval rmsd_box_stats
    N2A = NBOX
    if (N2A > 0) {
        MIN2A = BOX_MIN; Q12A = BOX_Q1; MED2A = BOX_MED; Q32A = BOX_Q3; MAX2A = BOX_MAX
    }
    print "Series 2A: " . DATA_2A . series_note(N2A, NTOT, BOX_CUT)
} else {
    N2A = 0
}

# Horizontal RMSD boxes above the axes. x is RMSD; graph y > 1 is outside
# the plot, stacked upward from the top border.
# Whisker = full range, box = interquartile range, black tick = median.
# Present series only, bottom to top: 2B, 2A, 1B, 1A. Four series keep the
# previous centers; fewer series pack together with no empty slot.
BOX_H = 0.012
BOX_STEP = 0.050
BOX_BASE = 1.045
BOX_SLOT = 0
if (N2B > 0) { YC2B = BOX_BASE + BOX_SLOT * BOX_STEP; BOX_SLOT = BOX_SLOT + 1 }
if (N2A > 0) { YC2A = BOX_BASE + BOX_SLOT * BOX_STEP; BOX_SLOT = BOX_SLOT + 1 }
if (N1B > 0) { YC1B = BOX_BASE + BOX_SLOT * BOX_STEP; BOX_SLOT = BOX_SLOT + 1 }
if (N1A > 0) { YC1A = BOX_BASE + BOX_SLOT * BOX_STEP; BOX_SLOT = BOX_SLOT + 1 }
xbox(arr, obj, xmin, q1, med, q3, xmax, yc, col, lw) = \
    sprintf("set arrow %d from first %.8g, graph %.4f to first %.8g, graph %.4f nohead lc rgb '%s' lw %.4g front", \
        arr, xmin, yc, xmax, yc, col, lw) \
    . sprintf("; set arrow %d from first %.8g, graph %.4f to first %.8g, graph %.4f nohead lc rgb '%s' lw %.4g front", \
        arr+1, xmin, yc-BOX_H, xmin, yc+BOX_H, col, lw) \
    . sprintf("; set arrow %d from first %.8g, graph %.4f to first %.8g, graph %.4f nohead lc rgb '%s' lw %.4g front", \
        arr+2, xmax, yc-BOX_H, xmax, yc+BOX_H, col, lw) \
    . sprintf("; set object %d rectangle from first %.8g, graph %.4f to first %.8g, graph %.4f fc rgb '%s' fs empty border lc rgb '%s' lw %.4g noclip front", \
        obj, q1, yc-BOX_H, q3, yc+BOX_H, col, col, lw) \
    . sprintf("; set arrow %d from first %.8g, graph %.4f to first %.8g, graph %.4f nohead lc rgb '#000000' lw %.4g front", \
        arr+3, med, yc-BOX_H, med, yc+BOX_H, 1.6)
if (N1A > 0) eval xbox(11, 11, MIN1A, Q11A, MED1A, Q31A, MAX1A, YC1A, COL1, LW1A)
if (N1B > 0) eval xbox(21, 21, MIN1B, Q11B, MED1B, Q31B, MAX1B, YC1B, COL1, LW1B)
if (N2A > 0) eval xbox(31, 31, MIN2A, Q12A, MED2A, Q32A, MAX2A, YC2A, COL2, LW2A)
if (N2B > 0) eval xbox(41, 41, MIN2B, Q12B, MED2B, Q32B, MAX2B, YC2B, COL2, LW2B)

# One frame around every RMSD summary. Object 1 is drawn before the boxes.
BOX_XMIN = 1e99
BOX_XMAX = -1e99
BOX_YLO = 1e99
BOX_YHI = -1e99
HAS_BOX = 0
if (N1A > 0) { BOX_XMIN = (MIN1A < BOX_XMIN ? MIN1A : BOX_XMIN); BOX_XMAX = (MAX1A > BOX_XMAX ? MAX1A : BOX_XMAX); BOX_YLO = (YC1A-BOX_H < BOX_YLO ? YC1A-BOX_H : BOX_YLO); BOX_YHI = (YC1A+BOX_H > BOX_YHI ? YC1A+BOX_H : BOX_YHI); HAS_BOX = 1 }
if (N1B > 0) { BOX_XMIN = (MIN1B < BOX_XMIN ? MIN1B : BOX_XMIN); BOX_XMAX = (MAX1B > BOX_XMAX ? MAX1B : BOX_XMAX); BOX_YLO = (YC1B-BOX_H < BOX_YLO ? YC1B-BOX_H : BOX_YLO); BOX_YHI = (YC1B+BOX_H > BOX_YHI ? YC1B+BOX_H : BOX_YHI); HAS_BOX = 1 }
if (N2A > 0) { BOX_XMIN = (MIN2A < BOX_XMIN ? MIN2A : BOX_XMIN); BOX_XMAX = (MAX2A > BOX_XMAX ? MAX2A : BOX_XMAX); BOX_YLO = (YC2A-BOX_H < BOX_YLO ? YC2A-BOX_H : BOX_YLO); BOX_YHI = (YC2A+BOX_H > BOX_YHI ? YC2A+BOX_H : BOX_YHI); HAS_BOX = 1 }
if (N2B > 0) { BOX_XMIN = (MIN2B < BOX_XMIN ? MIN2B : BOX_XMIN); BOX_XMAX = (MAX2B > BOX_XMAX ? MAX2B : BOX_XMAX); BOX_YLO = (YC2B-BOX_H < BOX_YLO ? YC2B-BOX_H : BOX_YLO); BOX_YHI = (YC2B+BOX_H > BOX_YHI ? YC2B+BOX_H : BOX_YHI); HAS_BOX = 1 }
BOX_PAD_X = 0.02
BOX_PAD_Y = 0.016
if (HAS_BOX) set object 1 rectangle from first (BOX_XMIN-BOX_PAD_X), graph (BOX_YLO-BOX_PAD_Y) to first (BOX_XMAX+BOX_PAD_X), graph (BOX_YHI+BOX_PAD_Y) fc rgb '#000000' fs empty border lc rgb '#000000' lw 1.2 noclip front

eval "plot ".PLOT_CMD

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
