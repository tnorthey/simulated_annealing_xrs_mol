#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Plot 3 vertically stacked subfigures (shared x-axis) from 3 CSV files.
# Each CSV contains columns: x, y, sd  (optionally more curves: y2, sd2, ...)
#
# Output: standalone LaTeX file (<OUTBASE>.tex) that compiles directly to PDF.
#
# Requirements:
# - gnuplot 5.4+
# - LaTeX (pdflatex works; lualatex also works)
#
# Usage (from repo root):
#   gnuplot scripts/gnuplot/plot_csv_stddev_3stack_tex.gp
#
# Override inputs/labels/ranges without editing the file:
#   gnuplot -e "DATA1='a.csv';DATA2='b.csv';DATA3='c.csv';OUTBASE='myfig'" \
#          scripts/gnuplot/plot_csv_stddev_3stack_tex.gp
#
# Notes:
# - The first row is assumed to be a header and is automatically skipped.
# - Error bars assume 1-sigma in the sd column. For 95% CI use 1.96*sd.
# ------------------------------------------------------------------------------

# -------------------------------
# User settings (easy to customize)
# -------------------------------
if (!exists("DATA1"))   DATA1   = "data1.csv"
if (!exists("DATA2"))   DATA2   = "data2.csv"
if (!exists("DATA3"))   DATA3   = "data3.csv"
if (!exists("OUTBASE")) OUTBASE = "figure"

# Colors (panel curve A colors). Override via -e, e.g.:
#   gnuplot -e "COL1='#000000';COL2='#377eb8';COL3='#e41a1c'" ...
if (!exists("COL1")) COL1 = "#1b9e77"
if (!exists("COL2")) COL2 = "#7570b3"
if (!exists("COL3")) COL3 = "#d95f02"

# Optional curve B color (if you enable yBcol>0)
if (!exists("COLB")) COLB = "#666666"

# Line/point styling knobs (override via -e "LW=...;PS=...")
if (!exists("LW")) LW = 2.0
if (!exists("PS")) PS = 0.75

# Legend (key). Disabled by default; enable via -e "SHOW_KEY=1"
if (!exists("SHOW_KEY")) SHOW_KEY = 0

# Column mapping (1-indexed)
# Col 1 must be x. Each panel can plot up to 2 curves (A + optional B).
xcol = 1

# Curve A columns (mean and its standard deviation)
yAcol  = 2
sdAcol = 4
nameA  = "Series A"

# Optional curve B columns (set yBcol=0 to disable entirely)
yBcol  = 0
sdBcol = 5
nameB  = "Series B"

# Axis labels (LaTeX allowed).
# Use single quotes for defaults: some LaTeX strings behave better than with
# double quotes depending on terminal/escaping.
if (!exists("XLABEL"))  XLABEL  = '$t$ (fs)'
if (!exists("Y1LABEL")) Y1LABEL = '$y_1$ (units)'
if (!exists("Y2LABEL")) Y2LABEL = '$y_2$ (units)'
if (!exists("Y3LABEL")) Y3LABEL = '$y_3$ (units)'

# X-range shared across all panels (autoscale by default).
# To force a range, pass BOTH bounds, e.g.:
#   gnuplot -e "XMIN=0;XMAX=10" ...

# Per-panel y-ranges (autoscale by default).
# To force a panel range, pass BOTH bounds, e.g.:
#   gnuplot -e "Y2MIN=-1;Y2MAX=1" ...

# Tick spacing (explicit and easy).
# By default, gnuplot chooses tics automatically.
# Shared x-axis:
#   XTIC_STEP=50
# Per-panel y-axis (preferred):
#   YTIC_STEP1=0.2; YTIC_STEP2=5; YTIC_STEP3=1
# Optional fallback (applies to all panels if a panel-specific value is not set):
#   YTIC_STEP=0.5
if (exists("XTIC_STEP"))  XTIC_STEP  = XTIC_STEP  + 0
if (exists("YTIC_STEP"))  YTIC_STEP  = YTIC_STEP  + 0
if (exists("YTIC_STEP1")) YTIC_STEP1 = YTIC_STEP1 + 0
if (exists("YTIC_STEP2")) YTIC_STEP2 = YTIC_STEP2 + 0
if (exists("YTIC_STEP3")) YTIC_STEP3 = YTIC_STEP3 + 0

# Figure size (inches) — tune for your paper layout (override via -e "W=...;H=...")
if (!exists("W")) W = 3.35
if (!exists("H")) H = 3.35

# Multiplot margins (screen fractions; override via -e "MLEFT=...;...")
if (!exists("MLEFT"))   MLEFT   = 0.14
if (!exists("MRIGHT"))  MRIGHT  = 0.98
if (!exists("MBOTTOM")) MBOTTOM = 0.16
if (!exists("MTOP"))    MTOP    = 0.98

# Typography
font = "Latin Modern Roman,10"

# -------------------------------
# Input parsing
# -------------------------------
set datafile separator comma
set encoding utf8

# Use first row as column titles (and skip it for plotting)
set key autotitle columnhead

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

# Styles (colors are used per-panel below via COL1/COL2/COL3)
set style line 1 lw LW pt 7 ps PS
set style line 2 lw LW pt 5 ps PS

# Errorbar lines: thinner than the main curve
eblw = (LW < 1.0 ? 1.0 : 0.5*LW)

unset key
if (SHOW_KEY) set key opaque box lw 0.6
unset title

# Shared x-range (only set if user provided both XMIN and XMAX)
if (exists("XMIN") && exists("XMAX")) { set xrange [XMIN:XMAX] } else { unset xrange }

# -------------------------------
# Multiplot layout: 3 rows, 1 column, zero spacing
# -------------------------------
# spacing 0,0 ensures no gaps between panels. We also suppress redundant x-tics.
# margins left, right, bottom, top
set multiplot layout 3,1 rowsfirst margins MLEFT,MRIGHT,MBOTTOM,MTOP spacing 0.0,0.0

# Helper: apply y-range if both bounds are defined; else autoscale.
#
# (Gnuplot doesn't have user-defined functions in the same sense everywhere, so we
# repeat a small pattern per panel for clarity.)

# ---- Panel 1 ----
set ylabel Y1LABEL
unset xlabel
set format x ""          # hide x tick labels (shared x-axis shown on bottom panel)
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP1")) set ytics YTIC_STEP1
if (!exists("YTIC_STEP1") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP1") && !exists("YTIC_STEP")) set ytics
if (exists("Y1MIN") && exists("Y1MAX")) { set yrange [Y1MIN:Y1MAX] } else { unset yrange }
if (SHOW_KEY) set key top right
if (yBcol>0) {
  plot \
    DATA1 using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL1 notitle, \
    DATA1 using xcol:yAcol        with linespoints ls 1 lc rgb COL1 title nameA, \
    DATA1 using xcol:yBcol:sdBcol with yerrorbars lw eblw lc rgb COLB notitle, \
    DATA1 using xcol:yBcol        with linespoints ls 2 lc rgb COLB title nameB
} else {
  plot \
    DATA1 using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL1 notitle, \
    DATA1 using xcol:yAcol        with linespoints ls 1 lc rgb COL1 title nameA
}

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
if (yBcol>0) {
  plot \
    DATA2 using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL2 notitle, \
    DATA2 using xcol:yAcol        with linespoints ls 1 lc rgb COL2 notitle, \
    DATA2 using xcol:yBcol:sdBcol with yerrorbars lw eblw lc rgb COLB notitle, \
    DATA2 using xcol:yBcol        with linespoints ls 2 lc rgb COLB notitle
} else {
  plot \
    DATA2 using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL2 notitle, \
    DATA2 using xcol:yAcol        with linespoints ls 1 lc rgb COL2 notitle
}

# ---- Panel 3 ----
set ylabel Y3LABEL
set xlabel XLABEL
set format x "%g"        # show x tick labels only on bottom panel
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP3")) set ytics YTIC_STEP3
if (!exists("YTIC_STEP3") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP3") && !exists("YTIC_STEP")) set ytics
if (exists("Y3MIN") && exists("Y3MAX")) { set yrange [Y3MIN:Y3MAX] } else { unset yrange }
unset key
if (yBcol>0) {
  plot \
    DATA3 using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL3 notitle, \
    DATA3 using xcol:yAcol        with linespoints ls 1 lc rgb COL3 notitle, \
    DATA3 using xcol:yBcol:sdBcol with yerrorbars lw eblw lc rgb COLB notitle, \
    DATA3 using xcol:yBcol        with linespoints ls 2 lc rgb COLB notitle
} else {
  plot \
    DATA3 using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL3 notitle, \
    DATA3 using xcol:yAcol        with linespoints ls 1 lc rgb COL3 notitle
}

unset multiplot
unset output
set terminal pop

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
