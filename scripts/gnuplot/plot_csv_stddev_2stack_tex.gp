#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Plot 2 vertically stacked subfigures (shared x-axis) from 2 CSV files.
# Each CSV contains columns: time, data, SD  (i.e. x, y, sd)
#
# Output: standalone LaTeX file (<OUTBASE>.tex) that compiles directly to PDF.
#
# Requirements:
# - gnuplot 5.4+
# - LaTeX (pdflatex works; lualatex also works)
#
# Usage (from repo root):
#   gnuplot scripts/gnuplot/plot_csv_stddev_2stack_tex.gp
#
# Override inputs/labels/ranges without editing the file:
#   gnuplot -e "DATA1='a.csv';DATA2='b.csv';OUTBASE='myfig'" \
#          scripts/gnuplot/plot_csv_stddev_2stack_tex.gp
#
# Compare two datasets per panel (e.g. predicted vs experiment):
#   gnuplot -e "DATA1='pred1.csv';DATA1B='exp1.csv';NAME1='Pred';NAME1B='Exp';"\
#             "DATA2='pred2.csv';DATA2B='exp2.csv';NAME2='Pred';NAME2B='Exp'" \
#          scripts/gnuplot/plot_csv_stddev_2stack_tex.gp
#
# Circular dihedral helper (simple fix for -179 vs +179 discontinuity):
#   gnuplot -e "DIHEDRAL_TO3601=1;DIHEDRAL_TO3601B=1;DIHEDRAL_TO3602=1;DIHEDRAL_TO3602B=1" \
#          scripts/gnuplot/plot_csv_stddev_2stack_tex.gp
# This remaps negatives with: y<0 ? y+360 : y (applied before offset).
#
# Control relative heights of the 2 panels (any positive numbers; normalized):
#   gnuplot -e "RELH1=0.7;RELH2=0.3" scripts/gnuplot/plot_csv_stddev_2stack_tex.gp
#
# Piecewise least-squares fit on DATA1 / DATA2 (panel A curves only):
#   Rise (x <= FITn_XCUT):  y0 + A*(1 - exp(-x/tau)) with y0 FIXED at FITn_Y0_GUESS (only A,tau are fit).
#   Tail (x >= FITn_XCUT): Taylor polynomial in (x - FITn_TAIL_X0), degree FITn_POLY_DEG (0..6),
#   plus optional damped sines: sum_k A_k*exp(-(x-x0)/L_k)*sin(W_k*(x-x0)+P_k) for FITn_NSIN=1..3.
# The two segments are fit independently; the curve may jump at x = FITn_XCUT.
# Fits skip the first CSV row (header). For headerless files use FIT_DATA_SKIP=0.
# By default the fit uses the same y-transform as the plot (dihedral/offset); use FIT_RAW=1 for raw column y.
# Weights use column sd (yerror). Asymptotic stderrs from gnuplot are approximate.
# Example:
#   gnuplot -e "FIT1=1;FIT1_XCUT=1.0;FIT1_POLY_DEG=4;FIT2=1;FIT2_XCUT=0.5;FIT2_POLY_DEG=3;SHOW_KEY=1" ...
# Optional: FIT_LOG='fit.log' to record fit diagnostics; FIT1_COLOR / FIT2_COLOR for the overlay line.
# ------------------------------------------------------------------------------

# -------------------------------
# User settings (easy to customize)
# -------------------------------
if (!exists("DATA1"))   DATA1   = "data1.csv"
if (!exists("DATA2"))   DATA2   = "data2.csv"
if (!exists("OUTBASE")) OUTBASE = "figure"

# Optional second dataset per panel (e.g. experiment). Leave unset to disable.
if (!exists("DATA1B")) DATA1B = ""
if (!exists("DATA2B")) DATA2B = ""

# Optional dihedral transforms (applied to y values only; sd stays positive).
# For each dataset, you can negate and/or add an offset (degrees, etc.).
# Examples:
#   gnuplot -e "DIHEDRAL_NEGATE1=1;DIHEDRAL_OFFSET1=360" ...
#   gnuplot -e "DIHEDRAL_NEGATE1B=1;DIHEDRAL_OFFSET1B=180" ...
#
# Optional "circular" remapping (helps with -179 vs +179 discontinuity):
#   if enabled: y<0 ? y+360 : y
# Examples:
#   gnuplot -e "DIHEDRAL_TO3601=1" ...
#   gnuplot -e "DIHEDRAL_TO3601B=1;DIHEDRAL_TO3602B=1" ...
if (!exists("DIHEDRAL_OFFSET1"))  DIHEDRAL_OFFSET1  = 0
if (!exists("DIHEDRAL_OFFSET1B")) DIHEDRAL_OFFSET1B = 0
if (!exists("DIHEDRAL_OFFSET2"))  DIHEDRAL_OFFSET2  = 0
if (!exists("DIHEDRAL_OFFSET2B")) DIHEDRAL_OFFSET2B = 0
if (!exists("DIHEDRAL_NEGATE1"))  DIHEDRAL_NEGATE1  = 0
if (!exists("DIHEDRAL_NEGATE1B")) DIHEDRAL_NEGATE1B = 0
if (!exists("DIHEDRAL_NEGATE2"))  DIHEDRAL_NEGATE2  = 0
if (!exists("DIHEDRAL_NEGATE2B")) DIHEDRAL_NEGATE2B = 0
if (!exists("DIHEDRAL_TO3601"))   DIHEDRAL_TO3601   = 0
if (!exists("DIHEDRAL_TO3601B"))  DIHEDRAL_TO3601B  = 0
if (!exists("DIHEDRAL_TO3602"))   DIHEDRAL_TO3602   = 0
if (!exists("DIHEDRAL_TO3602B"))  DIHEDRAL_TO3602B  = 0

DIHEDRAL_OFFSET1  = DIHEDRAL_OFFSET1  + 0
DIHEDRAL_OFFSET1B = DIHEDRAL_OFFSET1B + 0
DIHEDRAL_OFFSET2  = DIHEDRAL_OFFSET2  + 0
DIHEDRAL_OFFSET2B = DIHEDRAL_OFFSET2B + 0
DIHEDRAL_NEGATE1  = DIHEDRAL_NEGATE1  + 0
DIHEDRAL_NEGATE1B = DIHEDRAL_NEGATE1B + 0
DIHEDRAL_NEGATE2  = DIHEDRAL_NEGATE2  + 0
DIHEDRAL_NEGATE2B = DIHEDRAL_NEGATE2B + 0
DIHEDRAL_TO3601   = DIHEDRAL_TO3601   + 0
DIHEDRAL_TO3601B  = DIHEDRAL_TO3601B  + 0
DIHEDRAL_TO3602   = DIHEDRAL_TO3602   + 0
DIHEDRAL_TO3602B  = DIHEDRAL_TO3602B  + 0

MUL1  = (DIHEDRAL_NEGATE1  != 0) ? -1 : 1
MUL1B = (DIHEDRAL_NEGATE1B != 0) ? -1 : 1
MUL2  = (DIHEDRAL_NEGATE2  != 0) ? -1 : 1
MUL2B = (DIHEDRAL_NEGATE2B != 0) ? -1 : 1

to360(x) = (x < 0 ? x + 360 : x)

# Auto-detect which panels are available by checking whether the CSV exists and is non-empty.
is_nonempty_file(f) = int(system(sprintf("bash -lc \"test -s '%s' && echo 1 || echo 0\" ", f)))
HAS1 = is_nonempty_file(DATA1)
HAS2 = is_nonempty_file(DATA2)
HAS1B = (DATA1B ne "") ? is_nonempty_file(DATA1B) : 0
HAS2B = (DATA2B ne "") ? is_nonempty_file(DATA2B) : 0

if (!HAS1) print sprintf("ERROR: DATA1 file missing or empty: %s", DATA1)
if (!HAS1) exit
if ((DATA1B ne "") && !HAS1B) print sprintf("ERROR: DATA1B file missing or empty: %s", DATA1B)
if ((DATA2B ne "") && !HAS2B) print sprintf("ERROR: DATA2B file missing or empty: %s", DATA2B)

NROWS = 1
if (HAS2) NROWS = 2

# Relative panel heights (screen-space inside MTOP/MBOTTOM). Any positive values.
# If NROWS==1, these are ignored and the single panel uses full height.
if (!exists("RELH1")) RELH1 = 0.5
if (!exists("RELH2")) RELH2 = 0.5
RELH1 = RELH1 + 0
RELH2 = RELH2 + 0
if (RELH1 <= 0 || RELH2 <= 0) print "ERROR: RELH1 and RELH2 must be > 0"
if (RELH1 <= 0 || RELH2 <= 0) exit
RELHSUM = RELH1 + RELH2

# Colors (panel curve A colors). Override via -e, e.g.:
#   gnuplot -e "COL1='#000000';COL2='#377eb8'" ...
if (!exists("COL1")) COL1 = "#1b9e77"
if (!exists("COL2")) COL2 = "#7570b3"

# Secondary dataset colors (per panel)
if (!exists("COL1B")) COL1B = "#d95f02"
if (!exists("COL2B")) COL2B = "#e7298a"

# Global color overrides (apply to both panels)
# -e "COLOR='#000000'" forces COL1 and COL2 to the same color
# -e "COLORB='#666666'" forces COL1B and COL2B to the same color
if (exists("COLOR"))  COL1 = COLOR
if (exists("COLOR"))  COL2 = COLOR
if (exists("COLORB")) COL1B = COLORB
if (exists("COLORB")) COL2B = COLORB

# Optional per-panel color aliases (equivalent to COL1/COL1B/COL2/COL2B)
if (exists("COLOR1"))  COL1  = COLOR1
if (exists("COLOR1B")) COL1B = COLOR1B
if (exists("COLOR2"))  COL2  = COLOR2
if (exists("COLOR2B")) COL2B = COLOR2B

# Optional curve B color (if you enable yBcol>0)
if (!exists("COLB")) COLB = "#666666"

# Line/point styling knobs (override via -e "LW=...;PS=...")
if (!exists("LW")) LW = 2.0
if (!exists("LWB")) LWB = LW
if (!exists("PS")) PS = 0.75
if (!exists("PSB")) PSB = PS
PSB = PSB + 0
if (!exists("PT"))  PT  = 7
if (!exists("PTB")) PTB = 5
PT  = PT  + 0
PTB = PTB + 0
LW  = LW  + 0
LWB = LWB + 0

# Per-panel/per-dataset overrides (default to global LW/LWB, PS/PSB, PT/PTB)
if (!exists("LW1"))  LW1  = LW
if (!exists("LW1B")) LW1B = LWB
if (!exists("LW2"))  LW2  = LW
if (!exists("LW2B")) LW2B = LWB
LW1  = LW1  + 0
LW1B = LW1B + 0
LW2  = LW2  + 0
LW2B = LW2B + 0

if (!exists("PS1"))  PS1  = PS
if (!exists("PS1B")) PS1B = PSB
if (!exists("PS2"))  PS2  = PS
if (!exists("PS2B")) PS2B = PSB
PS1  = PS1  + 0
PS1B = PS1B + 0
PS2  = PS2  + 0
PS2B = PS2B + 0

if (!exists("PT1"))  PT1  = PT
if (!exists("PT1B")) PT1B = PTB
if (!exists("PT2"))  PT2  = PT
if (!exists("PT2B")) PT2B = PTB
PT1  = PT1  + 0
PT1B = PT1B + 0
PT2  = PT2  + 0
PT2B = PT2B + 0

# Legend (key). Disabled by default; enable via -e "SHOW_KEY=1"
if (!exists("SHOW_KEY")) SHOW_KEY = 0

# Plot mode: choose between lines only or lines+points.
set macros
if (!exists("PLOTMODE")) PLOTMODE = 'LP'
if (!exists("PLOTMODEB")) PLOTMODEB = PLOTMODE

# Supported values: 'LINES', 'LP', 'POINTS'
PLOT_WITH  = (PLOTMODE  eq 'LINES')  ? 'lines'       : ((PLOTMODE  eq 'POINTS')  ? 'points' : 'linespoints')
PLOT_WITHB = (PLOTMODEB eq 'LINES')  ? 'lines'       : ((PLOTMODEB eq 'POINTS') ? 'points' : 'linespoints')

# Column mapping (1-indexed)
xcol = 1

# Curve A columns (mean and its standard deviation)
yAcol  = 2
sdAcol = 3
nameA  = "Series A"

# Optional curve B columns (set yBcol=0 to disable entirely)
yBcol  = 0
sdBcol = 5
nameB  = "Series B"

# Optional piecewise fit on DATA1 (panel 1) and DATA2 (panel 2); see header comment.
if (!exists("FIT1")) FIT1 = 0
if (!exists("FIT2")) FIT2 = 0
FIT1 = FIT1 + 0
FIT2 = FIT2 + 0
if (!exists("FIT_RAW")) FIT_RAW = 0
FIT_RAW = FIT_RAW + 0
if (!exists("FIT1_POLY_DEG")) FIT1_POLY_DEG = 3
if (!exists("FIT2_POLY_DEG")) FIT2_POLY_DEG = 3
FIT1_POLY_DEG = FIT1_POLY_DEG + 0
FIT2_POLY_DEG = FIT2_POLY_DEG + 0
if (!exists("FIT1_NSIN")) FIT1_NSIN = 0
if (!exists("FIT2_NSIN")) FIT2_NSIN = 0
FIT1_NSIN = FIT1_NSIN + 0
FIT2_NSIN = FIT2_NSIN + 0
# Default guesses for each damped-sine term: A*exp(-(x-x0)/L)*sin(W*(x-x0)+P)
if (!exists("FIT1_DS_A")) FIT1_DS_A = 0.01
if (!exists("FIT1_DS_L")) FIT1_DS_L = 5
if (!exists("FIT1_DS_W")) FIT1_DS_W = 1
if (!exists("FIT1_DS_P")) FIT1_DS_P = 0
if (!exists("FIT2_DS_A")) FIT2_DS_A = 0.01
if (!exists("FIT2_DS_L")) FIT2_DS_L = 5
if (!exists("FIT2_DS_W")) FIT2_DS_W = 1
if (!exists("FIT2_DS_P")) FIT2_DS_P = 0
FIT1_DS_A = FIT1_DS_A + 0
FIT1_DS_L = FIT1_DS_L + 0
FIT1_DS_W = FIT1_DS_W + 0
FIT1_DS_P = FIT1_DS_P + 0
FIT2_DS_A = FIT2_DS_A + 0
FIT2_DS_L = FIT2_DS_L + 0
FIT2_DS_W = FIT2_DS_W + 0
FIT2_DS_P = FIT2_DS_P + 0
if (!exists("FIT1_LW")) FIT1_LW = 2.0
if (!exists("FIT2_LW")) FIT2_LW = 2.0
FIT1_LW = FIT1_LW + 0
FIT2_LW = FIT2_LW + 0
if (!exists("FIT1_COLOR")) FIT1_COLOR = "#333333"
if (!exists("FIT2_COLOR")) FIT2_COLOR = "#333333"
if (!exists("FIT1_TITLE")) FIT1_TITLE = "fit"
if (!exists("FIT2_TITLE")) FIT2_TITLE = "fit"
# Rise model: y = y0 + A*(1-exp(-x/tau)). Here y0 is FIXED at FITn_Y0_GUESS (not fitted);
# only A and tau are fit (fewer DOF when the rise has few points).
if (!exists("FIT1_Y0_GUESS")) FIT1_Y0_GUESS = 0
if (!exists("FIT1_A_GUESS"))  FIT1_A_GUESS  = 1
if (!exists("FIT1_TAU_GUESS")) FIT1_TAU_GUESS = 1
if (!exists("FIT2_Y0_GUESS")) FIT2_Y0_GUESS = 0
if (!exists("FIT2_A_GUESS"))  FIT2_A_GUESS  = 1
if (!exists("FIT2_TAU_GUESS")) FIT2_TAU_GUESS = 1
FIT1_Y0_GUESS = FIT1_Y0_GUESS + 0
FIT1_A_GUESS  = FIT1_A_GUESS  + 0
FIT1_TAU_GUESS = FIT1_TAU_GUESS + 0
FIT2_Y0_GUESS = FIT2_Y0_GUESS + 0
FIT2_A_GUESS  = FIT2_A_GUESS  + 0
FIT2_TAU_GUESS = FIT2_TAU_GUESS + 0

# Legend labels for per-panel datasets
if (!exists("NAME1"))  NAME1  = "Dataset 1"
if (!exists("NAME1B")) NAME1B = "Dataset 1B"
if (!exists("NAME2"))  NAME2  = "Dataset 2"
if (!exists("NAME2B")) NAME2B = "Dataset 2B"

# Axis labels (LaTeX allowed).
if (!exists("XLABEL"))  XLABEL  = '$t$ (fs)'
if (!exists("Y1LABEL")) Y1LABEL = '$y_1$ (units)'
if (!exists("Y2LABEL")) Y2LABEL = '$y_2$ (units)'

# Add whitespace between y-label and plot (offset in character units).
# You can set per-panel x-offsets:
#   -e "Y1LABEL_OFFSET=1.5;Y2LABEL_OFFSET=1.0"
# Or use the shared fallback:
#   -e "YLABEL_OFFSETX=1.5"
if (!exists("YLABEL_OFFSETX")) YLABEL_OFFSETX = 0.75
YLABEL_OFFSETX = YLABEL_OFFSETX + 0
if (!exists("Y1LABEL_OFFSET")) Y1LABEL_OFFSET = YLABEL_OFFSETX
if (!exists("Y2LABEL_OFFSET")) Y2LABEL_OFFSET = YLABEL_OFFSETX
Y1LABEL_OFFSET = Y1LABEL_OFFSET + 0
Y2LABEL_OFFSET = Y2LABEL_OFFSET + 0

# Tick spacing (explicit and easy).
if (exists("XTIC_STEP"))  XTIC_STEP  = XTIC_STEP  + 0
if (exists("YTIC_STEP"))  YTIC_STEP  = YTIC_STEP  + 0
if (exists("YTIC_STEP1")) YTIC_STEP1 = YTIC_STEP1 + 0
if (exists("YTIC_STEP2")) YTIC_STEP2 = YTIC_STEP2 + 0

# Figure size (inches)
if (!exists("W")) W = 3.35
if (!exists("H")) H = 3.35

# Multiplot margins (screen fractions)
if (!exists("MLEFT"))   MLEFT   = 0.14
if (!exists("MRIGHT"))  MRIGHT  = 0.98
if (!exists("MBOTTOM")) MBOTTOM = 0.16
if (!exists("MTOP"))    MTOP    = 0.98

# Typography
if (!exists("FONT")) FONT = "Latin Modern Roman,10"

# Extra LaTeX preamble for the standalone .tex (inserted before \begin{document}).
if (!exists("LATEX_HEADER")) LATEX_HEADER = ""

# Mirror tics (show top x-tics and right y-tics). Disable via -e "MIRROR_TICS=0".
if (!exists("MIRROR_TICS")) MIRROR_TICS = 1
MIRROR_TICS = MIRROR_TICS + 0

# -------------------------------
# Input parsing
# -------------------------------
set datafile separator comma
set encoding utf8
set key autotitle columnhead

# -------------------------------
# Terminal: standalone LaTeX + PDF via cairo
# -------------------------------
LATEX_HDR = (LATEX_HEADER ne "") ? " header '".LATEX_HEADER."'" : ""
eval "set terminal cairolatex pdf standalone size ".sprintf("%g",W).",".sprintf("%g",H)." font '".FONT."' dashed color".LATEX_HDR
set output sprintf("%s.tex", OUTBASE)

# -------------------------------
# Styling (publication-ish defaults)
# -------------------------------
set border linewidth 1.2
set tics scale 0.75
if (MIRROR_TICS) set xtics mirror
if (MIRROR_TICS) set ytics mirror
if (!MIRROR_TICS) set xtics nomirror
if (!MIRROR_TICS) set ytics nomirror
set mxtics 2
set mytics 2
set grid back xtics ytics lw 0.6 lc rgb "#D0D0D0"

set style line 11 lw LW1  pt PT1  ps PS1
set style line 12 lw LW1B pt PT1B ps PS1B
set style line 21 lw LW2  pt PT2  ps PS2
set style line 22 lw LW2B pt PT2B ps PS2B

eblw = (LW < 1.0 ? 1.0 : 0.5*LW)
eblwB = (LWB < 1.0 ? 1.0 : 0.5*LWB)
eblw1  = (LW1  < 1.0 ? 1.0 : 0.5*LW1)
eblw1B = (LW1B < 1.0 ? 1.0 : 0.5*LW1B)
eblw2  = (LW2  < 1.0 ? 1.0 : 0.5*LW2)
eblw2B = (LW2B < 1.0 ? 1.0 : 0.5*LW2B)

unset key
if (SHOW_KEY) set key opaque box lw 0.6
unset title

unset xrange
unset yrange

#
# Note on gnuplot syntax:
# Some gnuplot builds complain about any "if (...) ..." constructs used inside
# curly-brace blocks. To stay maximally compatible, this script avoids `{}` in
# control-flow and uses only single-line `if (cond) command` statements below.
#

# Shared x-range (only set if user provided both XMIN and XMAX)
# Example:
#   gnuplot -e "XMIN=0;XMAX=10" scripts/gnuplot/plot_csv_stddev_2stack_tex.gp
if (exists("XMIN")) XMIN = XMIN + 0
if (exists("XMAX")) XMAX = XMAX + 0
if (exists("XMIN") && exists("XMAX")) set xrange [XMIN:XMAX]
if (!(exists("XMIN") && exists("XMAX"))) unset xrange

# Per-panel y-ranges (only set if user provided both bounds)
if (exists("Y1MIN")) Y1MIN = Y1MIN + 0
if (exists("Y1MAX")) Y1MAX = Y1MAX + 0
if (exists("Y2MIN")) Y2MIN = Y2MIN + 0
if (exists("Y2MAX")) Y2MAX = Y2MAX + 0

# Error rendering mode:
# - default: classic yerrorbars
# - optional: shaded band (mean ± SD) via filledcurves
if (!exists("SHADE_ALPHA")) SHADE_ALPHA = 0.20
SHADE_ALPHA = SHADE_ALPHA + 0
if (!exists("SHADEDERRORS1"))  SHADEDERRORS1  = 0
if (!exists("SHADEDERRORS1B")) SHADEDERRORS1B = 0
if (!exists("SHADEDERRORS2"))  SHADEDERRORS2  = 0
if (!exists("SHADEDERRORS2B")) SHADEDERRORS2B = 0
SHADEDERRORS1  = SHADEDERRORS1  + 0
SHADEDERRORS1B = SHADEDERRORS1B + 0
SHADEDERRORS2  = SHADEDERRORS2  + 0
SHADEDERRORS2B = SHADEDERRORS2B + 0

# Pre-build plot command strings (they include plot + line continuations)
# IMPORTANT: Quote filenames, since paths often contain '-' which gnuplot may
# otherwise parse as subtraction in expressions.
# Build per-dataset y expressions (negate -> optional to360 -> offset)
YCOL = sprintf("%d", yAcol)
SDCOL = sprintf("%d", sdAcol)

YEXPR1  = "(MUL1*$".YCOL.")"
YEXPR1B = "(MUL1B*$".YCOL.")"
YEXPR2  = "(MUL2*$".YCOL.")"
YEXPR2B = "(MUL2B*$".YCOL.")"

if (DIHEDRAL_TO3601  != 0) YEXPR1  = "to360(".YEXPR1.")"
if (DIHEDRAL_TO3601B != 0) YEXPR1B = "to360(".YEXPR1B.")"
if (DIHEDRAL_TO3602  != 0) YEXPR2  = "to360(".YEXPR2.")"
if (DIHEDRAL_TO3602B != 0) YEXPR2B = "to360(".YEXPR2B.")"

YEXPR1  = "(".YEXPR1 ."+DIHEDRAL_OFFSET1)"
YEXPR1B = "(".YEXPR1B."+DIHEDRAL_OFFSET1B)"
YEXPR2  = "(".YEXPR2 ."+DIHEDRAL_OFFSET2)"
YEXPR2B = "(".YEXPR2B."+DIHEDRAL_OFFSET2B)"

# y column used in fit (match plotted transform unless FIT_RAW)
FITY1 = (FIT_RAW != 0) ? sprintf("$%d", yAcol) : YEXPR1
FITY2 = (FIT_RAW != 0) ? sprintf("$%d", yAcol) : YEXPR2
SDCOLN = sprintf("%d", sdAcol)

if (!exists("FIT_DATA_SKIP")) FIT_DATA_SKIP = 1
FIT_DATA_SKIP = FIT_DATA_SKIP + 0
FIT_SK = (FIT_DATA_SKIP != 0) ? "skip 1 " : ""

set fit quiet
if (exists("FIT_LOG")) eval "set fit logfile '".FIT_LOG."'"

# ---- FIT1: exponential rise (x<=xc) + polynomial tail (x>=xc) on DATA1 ----
FIT1_APPEND = ""
if (FIT1 != 0 && (FIT1_POLY_DEG < 0 || FIT1_POLY_DEG > 6)) print "ERROR: FIT1_POLY_DEG must be in [0,6]"
if (FIT1 != 0 && (FIT1_POLY_DEG < 0 || FIT1_POLY_DEG > 6)) exit
if (FIT1 != 0 && (FIT1_NSIN < 0 || FIT1_NSIN > 3)) print "ERROR: FIT1_NSIN must be in [0,3]"
if (FIT1 != 0 && (FIT1_NSIN < 0 || FIT1_NSIN > 3)) exit
if (FIT1 != 0 && !exists("FIT1_XCUT")) print "ERROR: FIT1=1 requires FIT1_XCUT"
if (FIT1 != 0 && !exists("FIT1_XCUT")) exit
if (FIT1 != 0) FIT1_XCUT = FIT1_XCUT + 0
if (FIT1 != 0 && !exists("FIT1_TAIL_X0")) FIT1_TAIL_X0 = FIT1_XCUT
if (FIT1 != 0) FIT1_TAIL_X0 = FIT1_TAIL_X0 + 0
if (FIT1 != 0) XCUT1STR = sprintf("%.12g", FIT1_XCUT)
if (FIT1 != 0) XO1STR = sprintf("%.12g", FIT1_TAIL_X0)
if (FIT1 != 0) y0_r1 = FIT1_Y0_GUESS
if (FIT1 != 0) A_r1 = FIT1_A_GUESS
if (FIT1 != 0) tau_r1 = FIT1_TAU_GUESS
if (FIT1 != 0) f_rise_1(x) = y0_r1 + A_r1*(1-exp(-x/tau_r1))
if (FIT1 != 0) eval "fit f_rise_1(x) '".DATA1."' ".FIT_SK."using 1:(($1<=".XCUT1STR.")?(".FITY1."):(1/0)):".SDCOLN." yerror via A_r1,tau_r1"
if (FIT1 != 0) b0_t1 = 1e-3
if (FIT1 != 0) b1_t1 = 1e-3
if (FIT1 != 0) b2_t1 = 1e-3
if (FIT1 != 0) b3_t1 = 1e-3
if (FIT1 != 0) b4_t1 = 1e-3
if (FIT1 != 0) b5_t1 = 1e-3
if (FIT1 != 0) b6_t1 = 1e-3
if (FIT1 != 0 && FIT1_NSIN>=1) ds1a1 = FIT1_DS_A
if (FIT1 != 0 && FIT1_NSIN>=1) ds1l1 = FIT1_DS_L
if (FIT1 != 0 && FIT1_NSIN>=1) ds1w1 = FIT1_DS_W
if (FIT1 != 0 && FIT1_NSIN>=1) ds1p1 = FIT1_DS_P
if (FIT1 != 0 && FIT1_NSIN>=2) ds1a2 = FIT1_DS_A
if (FIT1 != 0 && FIT1_NSIN>=2) ds1l2 = FIT1_DS_L
if (FIT1 != 0 && FIT1_NSIN>=2) ds1w2 = FIT1_DS_W + 0.1
if (FIT1 != 0 && FIT1_NSIN>=2) ds1p2 = FIT1_DS_P
if (FIT1 != 0 && FIT1_NSIN>=3) ds1a3 = FIT1_DS_A
if (FIT1 != 0 && FIT1_NSIN>=3) ds1l3 = FIT1_DS_L
if (FIT1 != 0 && FIT1_NSIN>=3) ds1w3 = FIT1_DS_W + 0.2
if (FIT1 != 0 && FIT1_NSIN>=3) ds1p3 = FIT1_DS_P
if (FIT1 != 0) POLY1 = "b0_t1"
if (FIT1 != 0 && FIT1_POLY_DEG>=1) POLY1 = POLY1 . " + b1_t1*(x-(".XO1STR."))"
if (FIT1 != 0 && FIT1_POLY_DEG>=2) POLY1 = POLY1 . " + b2_t1*(x-(".XO1STR."))**2"
if (FIT1 != 0 && FIT1_POLY_DEG>=3) POLY1 = POLY1 . " + b3_t1*(x-(".XO1STR."))**3"
if (FIT1 != 0 && FIT1_POLY_DEG>=4) POLY1 = POLY1 . " + b4_t1*(x-(".XO1STR."))**4"
if (FIT1 != 0 && FIT1_POLY_DEG>=5) POLY1 = POLY1 . " + b5_t1*(x-(".XO1STR."))**5"
if (FIT1 != 0 && FIT1_POLY_DEG>=6) POLY1 = POLY1 . " + b6_t1*(x-(".XO1STR."))**6"
if (FIT1 != 0) DS1 = ""
if (FIT1 != 0 && FIT1_NSIN>=1) DS1 = DS1 . " + ds1a1*exp(-(x-(".XO1STR."))/ds1l1)*sin(ds1w1*(x-(".XO1STR."))+ds1p1)"
if (FIT1 != 0 && FIT1_NSIN>=2) DS1 = DS1 . " + ds1a2*exp(-(x-(".XO1STR."))/ds1l2)*sin(ds1w2*(x-(".XO1STR."))+ds1p2)"
if (FIT1 != 0 && FIT1_NSIN>=3) DS1 = DS1 . " + ds1a3*exp(-(x-(".XO1STR."))/ds1l3)*sin(ds1w3*(x-(".XO1STR."))+ds1p3)"
if (FIT1 != 0) eval "f_tail_1(x) = ".POLY1.DS1
if (FIT1 != 0) VIA1T = "b0_t1"
if (FIT1 != 0 && FIT1_POLY_DEG>=1) VIA1T = VIA1T . ",b1_t1"
if (FIT1 != 0 && FIT1_POLY_DEG>=2) VIA1T = VIA1T . ",b2_t1"
if (FIT1 != 0 && FIT1_POLY_DEG>=3) VIA1T = VIA1T . ",b3_t1"
if (FIT1 != 0 && FIT1_POLY_DEG>=4) VIA1T = VIA1T . ",b4_t1"
if (FIT1 != 0 && FIT1_POLY_DEG>=5) VIA1T = VIA1T . ",b5_t1"
if (FIT1 != 0 && FIT1_POLY_DEG>=6) VIA1T = VIA1T . ",b6_t1"
if (FIT1 != 0 && FIT1_NSIN>=1) VIA1T = VIA1T . ",ds1a1,ds1l1,ds1w1,ds1p1"
if (FIT1 != 0 && FIT1_NSIN>=2) VIA1T = VIA1T . ",ds1a2,ds1l2,ds1w2,ds1p2"
if (FIT1 != 0 && FIT1_NSIN>=3) VIA1T = VIA1T . ",ds1a3,ds1l3,ds1w3,ds1p3"
if (FIT1 != 0) eval "fit f_tail_1(x) '".DATA1."' ".FIT_SK."using 1:(($1>=".XCUT1STR.")?(".FITY1."):(1/0)):".SDCOLN." yerror via ".VIA1T
if (FIT1 != 0) f_fit_1(x) = (x<=FIT1_XCUT) ? f_rise_1(x) : f_tail_1(x)
if (FIT1 != 0) print sprintf("FIT1 DATA1 rise: y0(fixed)=%g A=%g tau=%g", y0_r1, A_r1, tau_r1)
if (FIT1 != 0) print sprintf("FIT1 DATA1 tail: poly deg %d + %d damped sine(s), x0=%g", FIT1_POLY_DEG, FIT1_NSIN, FIT1_TAIL_X0)
if (FIT1 != 0 && SHOW_KEY) FIT1_APPEND = ", f_fit_1(x) with lines lw FIT1_LW lc rgb FIT1_COLOR title '".FIT1_TITLE."'"
if (FIT1 != 0 && !SHOW_KEY) FIT1_APPEND = ", f_fit_1(x) with lines lw FIT1_LW lc rgb FIT1_COLOR notitle"

# ---- FIT2: same on DATA2 (only when second panel exists) ----
FIT2_APPEND = ""
if (FIT2 != 0 && NROWS < 2) print "WARNING: FIT2 ignored (DATA2 / second panel not used)"
if (FIT2 != 0 && NROWS >= 2 && (FIT2_POLY_DEG < 0 || FIT2_POLY_DEG > 6)) print "ERROR: FIT2_POLY_DEG must be in [0,6]"
if (FIT2 != 0 && NROWS >= 2 && (FIT2_POLY_DEG < 0 || FIT2_POLY_DEG > 6)) exit
if (FIT2 != 0 && NROWS >= 2 && (FIT2_NSIN < 0 || FIT2_NSIN > 3)) print "ERROR: FIT2_NSIN must be in [0,3]"
if (FIT2 != 0 && NROWS >= 2 && (FIT2_NSIN < 0 || FIT2_NSIN > 3)) exit
if (FIT2 != 0 && NROWS >= 2 && !exists("FIT2_XCUT")) print "ERROR: FIT2=1 requires FIT2_XCUT"
if (FIT2 != 0 && NROWS >= 2 && !exists("FIT2_XCUT")) exit
if (FIT2 != 0 && NROWS >= 2) FIT2_XCUT = FIT2_XCUT + 0
if (FIT2 != 0 && NROWS >= 2 && !exists("FIT2_TAIL_X0")) FIT2_TAIL_X0 = FIT2_XCUT
if (FIT2 != 0 && NROWS >= 2) FIT2_TAIL_X0 = FIT2_TAIL_X0 + 0
if (FIT2 != 0 && NROWS >= 2) XCUT2STR = sprintf("%.12g", FIT2_XCUT)
if (FIT2 != 0 && NROWS >= 2) XO2STR = sprintf("%.12g", FIT2_TAIL_X0)
if (FIT2 != 0 && NROWS >= 2) y0_r2 = FIT2_Y0_GUESS
if (FIT2 != 0 && NROWS >= 2) A_r2 = FIT2_A_GUESS
if (FIT2 != 0 && NROWS >= 2) tau_r2 = FIT2_TAU_GUESS
if (FIT2 != 0 && NROWS >= 2) f_rise_2(x) = y0_r2 + A_r2*(1-exp(-x/tau_r2))
if (FIT2 != 0 && NROWS >= 2) eval "fit f_rise_2(x) '".DATA2."' ".FIT_SK."using 1:(($1<=".XCUT2STR.")?(".FITY2."):(1/0)):".SDCOLN." yerror via A_r2,tau_r2"
if (FIT2 != 0 && NROWS >= 2) b0_t2 = 1e-3
if (FIT2 != 0 && NROWS >= 2) b1_t2 = 1e-3
if (FIT2 != 0 && NROWS >= 2) b2_t2 = 1e-3
if (FIT2 != 0 && NROWS >= 2) b3_t2 = 1e-3
if (FIT2 != 0 && NROWS >= 2) b4_t2 = 1e-3
if (FIT2 != 0 && NROWS >= 2) b5_t2 = 1e-3
if (FIT2 != 0 && NROWS >= 2) b6_t2 = 1e-3
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=1) ds2a1 = FIT2_DS_A
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=1) ds2l1 = FIT2_DS_L
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=1) ds2w1 = FIT2_DS_W
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=1) ds2p1 = FIT2_DS_P
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=2) ds2a2 = FIT2_DS_A
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=2) ds2l2 = FIT2_DS_L
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=2) ds2w2 = FIT2_DS_W + 0.1
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=2) ds2p2 = FIT2_DS_P
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=3) ds2a3 = FIT2_DS_A
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=3) ds2l3 = FIT2_DS_L
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=3) ds2w3 = FIT2_DS_W + 0.2
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=3) ds2p3 = FIT2_DS_P
if (FIT2 != 0 && NROWS >= 2) POLY2 = "b0_t2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=1) POLY2 = POLY2 . " + b1_t2*(x-(".XO2STR."))"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=2) POLY2 = POLY2 . " + b2_t2*(x-(".XO2STR."))**2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=3) POLY2 = POLY2 . " + b3_t2*(x-(".XO2STR."))**3"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=4) POLY2 = POLY2 . " + b4_t2*(x-(".XO2STR."))**4"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=5) POLY2 = POLY2 . " + b5_t2*(x-(".XO2STR."))**5"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=6) POLY2 = POLY2 . " + b6_t2*(x-(".XO2STR."))**6"
if (FIT2 != 0 && NROWS >= 2) DS2 = ""
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=1) DS2 = DS2 . " + ds2a1*exp(-(x-(".XO2STR."))/ds2l1)*sin(ds2w1*(x-(".XO2STR."))+ds2p1)"
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=2) DS2 = DS2 . " + ds2a2*exp(-(x-(".XO2STR."))/ds2l2)*sin(ds2w2*(x-(".XO2STR."))+ds2p2)"
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=3) DS2 = DS2 . " + ds2a3*exp(-(x-(".XO2STR."))/ds2l3)*sin(ds2w3*(x-(".XO2STR."))+ds2p3)"
if (FIT2 != 0 && NROWS >= 2) eval "f_tail_2(x) = ".POLY2.DS2
if (FIT2 != 0 && NROWS >= 2) VIA2T = "b0_t2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=1) VIA2T = VIA2T . ",b1_t2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=2) VIA2T = VIA2T . ",b2_t2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=3) VIA2T = VIA2T . ",b3_t2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=4) VIA2T = VIA2T . ",b4_t2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=5) VIA2T = VIA2T . ",b5_t2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_POLY_DEG>=6) VIA2T = VIA2T . ",b6_t2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=1) VIA2T = VIA2T . ",ds2a1,ds2l1,ds2w1,ds2p1"
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=2) VIA2T = VIA2T . ",ds2a2,ds2l2,ds2w2,ds2p2"
if (FIT2 != 0 && NROWS >= 2 && FIT2_NSIN>=3) VIA2T = VIA2T . ",ds2a3,ds2l3,ds2w3,ds2p3"
if (FIT2 != 0 && NROWS >= 2) eval "fit f_tail_2(x) '".DATA2."' ".FIT_SK."using 1:(($1>=".XCUT2STR.")?(".FITY2."):(1/0)):".SDCOLN." yerror via ".VIA2T
if (FIT2 != 0 && NROWS >= 2) f_fit_2(x) = (x<=FIT2_XCUT) ? f_rise_2(x) : f_tail_2(x)
if (FIT2 != 0 && NROWS >= 2) print sprintf("FIT2 DATA2 rise: y0(fixed)=%g A=%g tau=%g", y0_r2, A_r2, tau_r2)
if (FIT2 != 0 && NROWS >= 2) print sprintf("FIT2 DATA2 tail: poly deg %d + %d damped sine(s), x0=%g", FIT2_POLY_DEG, FIT2_NSIN, FIT2_TAIL_X0)
if (FIT2 != 0 && NROWS >= 2 && SHOW_KEY) FIT2_APPEND = ", f_fit_2(x) with lines lw FIT2_LW lc rgb FIT2_COLOR title '".FIT2_TITLE."'"
if (FIT2 != 0 && NROWS >= 2 && !SHOW_KEY) FIT2_APPEND = ", f_fit_2(x) with lines lw FIT2_LW lc rgb FIT2_COLOR notitle"

# Shaded band expressions (mean ± SD). SD is taken from $SDCOL.
YLO1  = "(".YEXPR1 ." - $".SDCOL.")"
YHI1  = "(".YEXPR1 ." + $".SDCOL.")"
YLO1B = "(".YEXPR1B." - $".SDCOL.")"
YHI1B = "(".YEXPR1B." + $".SDCOL.")"
YLO2  = "(".YEXPR2 ." - $".SDCOL.")"
YHI2  = "(".YEXPR2 ." + $".SDCOL.")"
YLO2B = "(".YEXPR2B." - $".SDCOL.")"
YHI2B = "(".YEXPR2B." + $".SDCOL.")"

# Base series (A) for a file: errorbars + curve.
E1  = "'".DATA1 ."' using xcol:(".YEXPR1 ."):sdAcol with yerrorbars lw eblw1  lc rgb COL1  pt -1 notitle"
E1B = "'".DATA1B."' using xcol:(".YEXPR1B."):sdAcol with yerrorbars lw eblw1B lc rgb COL1B pt -1 notitle"
E2  = "'".DATA2 ."' using xcol:(".YEXPR2 ."):sdAcol with yerrorbars lw eblw2  lc rgb COL2  pt -1 notitle"
E2B = "'".DATA2B."' using xcol:(".YEXPR2B."):sdAcol with yerrorbars lw eblw2B lc rgb COL2B pt -1 notitle"

if (SHADEDERRORS1  != 0) E1  = "'".DATA1 ."' using xcol:(".YLO1 ."):(".YHI1 .") with filledcurves lc rgb COL1  fs transparent solid SHADE_ALPHA noborder notitle"
if (SHADEDERRORS1B != 0) E1B = "'".DATA1B."' using xcol:(".YLO1B."):(".YHI1B.") with filledcurves lc rgb COL1B fs transparent solid SHADE_ALPHA noborder notitle"
if (SHADEDERRORS2  != 0) E2  = "'".DATA2 ."' using xcol:(".YLO2 ."):(".YHI2 .") with filledcurves lc rgb COL2  fs transparent solid SHADE_ALPHA noborder notitle"
if (SHADEDERRORS2B != 0) E2B = "'".DATA2B."' using xcol:(".YLO2B."):(".YHI2B.") with filledcurves lc rgb COL2B fs transparent solid SHADE_ALPHA noborder notitle"

P1A = E1  .", '".DATA1 ."' using xcol:(".YEXPR1 .") with @PLOT_WITH  ls 11 lc rgb COL1  title NAME1".FIT1_APPEND
P1B = E1B .", '".DATA1B."' using xcol:(".YEXPR1B.") with @PLOT_WITHB ls 12 lc rgb COL1B title NAME1B"
P2A = E2  .", '".DATA2 ."' using xcol:(".YEXPR2 .") with @PLOT_WITH  ls 21 lc rgb COL2  title NAME2".FIT2_APPEND
P2B = E2B .", '".DATA2B."' using xcol:(".YEXPR2B.") with @PLOT_WITHB ls 22 lc rgb COL2B title NAME2B"

# Compose per-panel plot commands, optionally adding the second dataset file.
P1_CMD = "plot ".P1A.(HAS1B ? ", ".P1B : "")
P2_CMD = "plot ".P2A.(HAS2B ? ", ".P2B : "")

# -------------------------------
# Multiplot layout: 2 rows with adjustable relative heights
# (started after fits and plot strings are ready)
# -------------------------------
set multiplot

set lmargin at screen MLEFT
set rmargin at screen MRIGHT

# Compute split for 2-row case
if (NROWS==2) AVAILH = MTOP - MBOTTOM
if (NROWS==2) H2 = AVAILH * (RELH2 / RELHSUM)   # bottom panel height
if (NROWS==2) YSPLIT = MBOTTOM + H2

# ---- Panel 1 ----
# NROWS==1: single panel full height; NROWS==2: top panel uses [YSPLIT..MTOP]
if (NROWS==1) set tmargin at screen MTOP
if (NROWS==1) set bmargin at screen MBOTTOM
if (NROWS==2) set tmargin at screen MTOP
if (NROWS==2) set bmargin at screen YSPLIT

set ylabel Y1LABEL offset Y1LABEL_OFFSET,0
if (NROWS==1) set xlabel XLABEL
if (NROWS==2) unset xlabel
if (NROWS==1) set format x "%g"
if (NROWS==2) set format x ""

if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP1")) set ytics YTIC_STEP1
if (!exists("YTIC_STEP1") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP1") && !exists("YTIC_STEP")) set ytics
if (SHOW_KEY) set key top right

if (exists("Y1MIN") && exists("Y1MAX")) set yrange [Y1MIN:Y1MAX]
if (!(exists("Y1MIN") && exists("Y1MAX"))) unset yrange

eval P1_CMD

# ---- Panel 2 (only when NROWS==2) ----
if (NROWS==2) set tmargin at screen YSPLIT
if (NROWS==2) set bmargin at screen MBOTTOM
if (NROWS==2) set ylabel Y2LABEL offset Y2LABEL_OFFSET,0
if (NROWS==2) set xlabel XLABEL
if (NROWS==2) set format x "%g"
if (NROWS==2 && exists("XTIC_STEP")) set xtics XTIC_STEP
if (NROWS==2 && !exists("XTIC_STEP")) set xtics
if (NROWS==2 && exists("YTIC_STEP2")) set ytics YTIC_STEP2
if (NROWS==2 && !exists("YTIC_STEP2") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (NROWS==2 && !exists("YTIC_STEP2") && !exists("YTIC_STEP")) set ytics
if (NROWS==2) unset key
if (NROWS==2 && exists("Y2MIN") && exists("Y2MAX")) set yrange [Y2MIN:Y2MAX]
if (NROWS==2 && !(exists("Y2MIN") && exists("Y2MAX"))) unset yrange
if (NROWS==2) eval P2_CMD

unset multiplot
unset output
set terminal pop

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
