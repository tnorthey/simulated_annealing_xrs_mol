#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Plot 4 vertically stacked panels (shared x-axis) comparing two .dat files per
# panel: predicted vs experimental.
#
# Output: standalone LaTeX file (<OUTBASE>.tex) that compiles directly to PDF.
#
# Requirements:
# - gnuplot 5.4+
# - LaTeX (pdflatex works; lualatex also works)
#
# Usage (from repo root):
#   gnuplot scripts/gnuplot/plot_dat_predexp_4stack_tex.gp
#
# Override inputs/labels/ranges without editing the file:
#   gnuplot -e "PRED1='p1.dat';EXP1='e1.dat';PRED2='p2.dat';EXP2='e2.dat';OUTBASE='myfig'" \
#          scripts/gnuplot/plot_dat_predexp_4stack_tex.gp
#
# Notes:
# - Assumes whitespace-separated .dat files (default gnuplot behavior).
# - Each file is plotted using columns xcol:ycol.
# ------------------------------------------------------------------------------

# -------------------------------
# User settings (easy to customize)
# -------------------------------
if (!exists("PRED1")) PRED1 = "pred1.dat"
if (!exists("EXP1"))  EXP1  = "exp1.dat"
if (!exists("PRED2")) PRED2 = "pred2.dat"
if (!exists("EXP2"))  EXP2  = "exp2.dat"
if (!exists("PRED3")) PRED3 = "pred3.dat"
if (!exists("EXP3"))  EXP3  = "exp3.dat"
if (!exists("PRED4")) PRED4 = "pred4.dat"
if (!exists("EXP4"))  EXP4  = "exp4.dat"

if (!exists("OUTBASE")) OUTBASE = "figure"

# Column mapping (1-indexed)
xcol = 1
ycol = 2

# Labels (LaTeX allowed). Defaults use single quotes.
if (!exists("XLABEL"))  XLABEL  = '$x$ (units)'
if (!exists("Y1LABEL")) Y1LABEL = '$y_1$ (units)'
if (!exists("Y2LABEL")) Y2LABEL = '$y_2$ (units)'
if (!exists("Y3LABEL")) Y3LABEL = '$y_3$ (units)'
if (!exists("Y4LABEL")) Y4LABEL = '$y_4$ (units)'

# Colors: per-panel predicted color + one global experimental color
if (!exists("COL1")) COL1 = "#1b9e77"
if (!exists("COL2")) COL2 = "#7570b3"
if (!exists("COL3")) COL3 = "#d95f02"
if (!exists("COL4")) COL4 = "#66a61e"
if (!exists("COLEXP")) COLEXP = "#000000"

# Line/point styling knobs
if (!exists("LW")) LW = 2.0
if (!exists("PS")) PS = 0.75

# Legend (key). Disabled by default; enable via -e "SHOW_KEY=1"
if (!exists("SHOW_KEY")) SHOW_KEY = 0

# Plot mode: choose between lines only or lines+points.
# - PLOTMODE='LINES'  -> curves are drawn with 'with lines'
# - PLOTMODE='LP'     -> curves are drawn with 'with linespoints' (default)
set macros
if (!exists("PLOTMODE")) PLOTMODE = 'LP'
PLOT_WITH = (PLOTMODE eq 'LINES') ? 'lines' : 'linespoints'

# X-range shared across all panels (autoscale by default).
# To force: pass BOTH bounds, e.g. -e "XMIN=0;XMAX=10"

# Per-panel y-ranges (autoscale by default). To force: pass BOTH bounds.
# Y1MIN/Y1MAX, Y2MIN/Y2MAX, Y3MIN/Y3MAX, Y4MIN/Y4MAX

# Tick spacing
# Shared x-axis: XTIC_STEP=...
# Per-panel y-axis: YTIC_STEP1..4=... (optional fallback YTIC_STEP=...)
if (exists("XTIC_STEP"))  XTIC_STEP  = XTIC_STEP  + 0
if (exists("YTIC_STEP"))  YTIC_STEP  = YTIC_STEP  + 0
if (exists("YTIC_STEP1")) YTIC_STEP1 = YTIC_STEP1 + 0
if (exists("YTIC_STEP2")) YTIC_STEP2 = YTIC_STEP2 + 0
if (exists("YTIC_STEP3")) YTIC_STEP3 = YTIC_STEP3 + 0
if (exists("YTIC_STEP4")) YTIC_STEP4 = YTIC_STEP4 + 0

# Figure size (inches)
if (!exists("W")) W = 3.35
if (!exists("H")) H = 6.30

# Multiplot margins (screen fractions)
if (!exists("MLEFT"))   MLEFT   = 0.14
if (!exists("MRIGHT"))  MRIGHT  = 0.98
if (!exists("MBOTTOM")) MBOTTOM = 0.16
if (!exists("MTOP"))    MTOP    = 0.98

# Typography
font = "Latin Modern Roman,10"

# -------------------------------
# Terminal: standalone LaTeX + PDF via cairo
# -------------------------------
set terminal cairolatex pdf standalone size W,H font font dashed color
set output sprintf("%s.tex", OUTBASE)

# -------------------------------
# Styling (publication-ish defaults)
# -------------------------------
set border linewidth 1.2
set tics scale 0.75
set xtics nomirror
set ytics nomirror
set mxtics 2
set mytics 2
set grid back xtics ytics lw 0.6 lc rgb "#D0D0D0"

# Styles (points/lines; colors set per-plot below)
# Predicted: solid. Experimental: dashed.
set style line 1 lw LW pt 7 ps PS dt 1
set style line 2 lw LW pt 5 ps PS dt 2

unset key
if (SHOW_KEY) set key opaque box lw 0.6
unset title

# Shared x-range
if (exists("XMIN") && exists("XMAX")) { set xrange [XMIN:XMAX] } else { unset xrange }

# -------------------------------
# Multiplot: 4 rows, 1 column, zero spacing
# -------------------------------
set multiplot layout 4,1 rowsfirst margins MLEFT,MRIGHT,MBOTTOM,MTOP spacing 0.0,0.0

# ---- Panel 1 ----
set ylabel Y1LABEL
unset xlabel
set format x ""
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP1")) set ytics YTIC_STEP1
if (!exists("YTIC_STEP1") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP1") && !exists("YTIC_STEP")) set ytics
if (exists("Y1MIN") && exists("Y1MAX")) { set yrange [Y1MIN:Y1MAX] } else { unset yrange }
if (SHOW_KEY) set key top right
plot \
  PRED1 using xcol:ycol with @PLOT_WITH ls 1 lc rgb COL1 title 'pred', \
  EXP1  using xcol:ycol with @PLOT_WITH ls 2 lc rgb COLEXP title 'exp'

# ---- Panel 2 ----
set ylabel Y2LABEL
unset xlabel
set format x ""
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP2")) set ytics YTIC_STEP2
if (!exists("YTIC_STEP2") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP2") && !exists("YTIC_STEP")) set ytics
if (exists("Y2MIN") && exists("Y2MAX")) { set yrange [Y2MIN:Y2MAX] } else { unset yrange }
unset key
plot \
  PRED2 using xcol:ycol with @PLOT_WITH ls 1 lc rgb COL2 notitle, \
  EXP2  using xcol:ycol with @PLOT_WITH ls 2 lc rgb COLEXP notitle

# ---- Panel 3 ----
set ylabel Y3LABEL
unset xlabel
set format x ""
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP3")) set ytics YTIC_STEP3
if (!exists("YTIC_STEP3") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP3") && !exists("YTIC_STEP")) set ytics
if (exists("Y3MIN") && exists("Y3MAX")) { set yrange [Y3MIN:Y3MAX] } else { unset yrange }
unset key
plot \
  PRED3 using xcol:ycol with @PLOT_WITH ls 1 lc rgb COL3 notitle, \
  EXP3  using xcol:ycol with @PLOT_WITH ls 2 lc rgb COLEXP notitle

# ---- Panel 4 ----
set ylabel Y4LABEL
set xlabel XLABEL
set format x "%g"
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP4")) set ytics YTIC_STEP4
if (!exists("YTIC_STEP4") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP4") && !exists("YTIC_STEP")) set ytics
if (exists("Y4MIN") && exists("Y4MAX")) { set yrange [Y4MIN:Y4MAX] } else { unset yrange }
unset key
plot \
  PRED4 using xcol:ycol with @PLOT_WITH ls 1 lc rgb COL4 notitle, \
  EXP4  using xcol:ycol with @PLOT_WITH ls 2 lc rgb COLEXP notitle

unset multiplot
unset output
set terminal pop

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)

