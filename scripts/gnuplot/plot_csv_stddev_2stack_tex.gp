#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Plot 2 vertically stacked subfigures (shared x-axis) from 2 CSV files.
# Each CSV contains columns: x, y, sd  (optionally more curves: y2, sd2, ...)
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
# Control relative heights of the 2 panels (any positive numbers; normalized):
#   gnuplot -e "RELH1=0.7;RELH2=0.3" scripts/gnuplot/plot_csv_stddev_2stack_tex.gp
# ------------------------------------------------------------------------------

# -------------------------------
# User settings (easy to customize)
# -------------------------------
if (!exists("DATA1"))   DATA1   = "data1.csv"
if (!exists("DATA2"))   DATA2   = "data2.csv"
if (!exists("OUTBASE")) OUTBASE = "figure"

# Auto-detect which panels are available by checking whether the CSV exists and is non-empty.
is_nonempty_file(f) = int(system(sprintf("bash -lc \"test -s '%s' && echo 1 || echo 0\" ", f)))
HAS1 = is_nonempty_file(DATA1)
HAS2 = is_nonempty_file(DATA2)

if (!HAS1) print sprintf("ERROR: DATA1 file missing or empty: %s", DATA1)
if (!HAS1) exit

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

# Optional curve B color (if you enable yBcol>0)
if (!exists("COLB")) COLB = "#666666"

# Line/point styling knobs (override via -e "LW=...;PS=...")
if (!exists("LW")) LW = 2.0
if (!exists("PS")) PS = 0.75

# Legend (key). Disabled by default; enable via -e "SHOW_KEY=1"
if (!exists("SHOW_KEY")) SHOW_KEY = 0

# Plot mode: choose between lines only or lines+points.
set macros
if (!exists("PLOTMODE")) PLOTMODE = 'LP'
PLOT_WITH = (PLOTMODE eq 'LINES') ? 'lines' : 'linespoints'

# Column mapping (1-indexed)
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
if (!exists("XLABEL"))  XLABEL  = '$t$ (fs)'
if (!exists("Y1LABEL")) Y1LABEL = '$y_1$ (units)'
if (!exists("Y2LABEL")) Y2LABEL = '$y_2$ (units)'

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
set xtics nomirror
set ytics nomirror
set mxtics 2
set mytics 2
set grid back xtics ytics lw 0.6 lc rgb "#D0D0D0"

set style line 1 lw LW pt 7 ps PS
set style line 2 lw LW pt 5 ps PS

eblw = (LW < 1.0 ? 1.0 : 0.5*LW)

unset key
if (SHOW_KEY) set key opaque box lw 0.6
unset title

unset xrange
unset yrange

# -------------------------------
# Multiplot layout: 2 rows with adjustable relative heights
# We use explicit screen margins so heights can differ.
# -------------------------------
set multiplot

set lmargin at screen MLEFT
set rmargin at screen MRIGHT

#
# Note on gnuplot syntax:
# Some gnuplot builds complain about any "if (...) ..." constructs used inside
# curly-brace blocks. To stay maximally compatible, this script avoids `{}` in
# control-flow and uses only single-line `if (cond) command` statements below.
#

# Pre-build plot command strings (they include plot + line continuations)
P1_WITHB = "plot ".DATA1." using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL1 pt -1 notitle, "\
          .DATA1." using xcol:yAcol with @PLOT_WITH ls 1 lc rgb COL1 title nameA, "\
          .DATA1." using xcol:yBcol:sdBcol with yerrorbars lw eblw lc rgb COLB pt -1 notitle, "\
          .DATA1." using xcol:yBcol with @PLOT_WITH ls 2 lc rgb COLB title nameB"
P1_NO_B  = "plot ".DATA1." using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL1 pt -1 notitle, "\
          .DATA1." using xcol:yAcol with @PLOT_WITH ls 1 lc rgb COL1 title nameA"
P2_WITHB = "plot ".DATA2." using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL2 pt -1 notitle, "\
          .DATA2." using xcol:yAcol with @PLOT_WITH ls 1 lc rgb COL2 notitle, "\
          .DATA2." using xcol:yBcol:sdBcol with yerrorbars lw eblw lc rgb COLB pt -1 notitle, "\
          .DATA2." using xcol:yBcol with @PLOT_WITH ls 2 lc rgb COLB notitle"
P2_NO_B  = "plot ".DATA2." using xcol:yAcol:sdAcol with yerrorbars lw eblw lc rgb COL2 pt -1 notitle, "\
          .DATA2." using xcol:yAcol with @PLOT_WITH ls 1 lc rgb COL2 notitle"

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

set ylabel Y1LABEL
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

if (yBcol>0)  eval P1_WITHB
if (yBcol<=0) eval P1_NO_B

# ---- Panel 2 (only when NROWS==2) ----
if (NROWS==2) set tmargin at screen YSPLIT
if (NROWS==2) set bmargin at screen MBOTTOM
if (NROWS==2) set ylabel Y2LABEL
if (NROWS==2) set xlabel XLABEL
if (NROWS==2) set format x "%g"
if (NROWS==2 && exists("XTIC_STEP")) set xtics XTIC_STEP
if (NROWS==2 && !exists("XTIC_STEP")) set xtics
if (NROWS==2 && exists("YTIC_STEP2")) set ytics YTIC_STEP2
if (NROWS==2 && !exists("YTIC_STEP2") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (NROWS==2 && !exists("YTIC_STEP2") && !exists("YTIC_STEP")) set ytics
if (NROWS==2) unset key
if (NROWS==2 && yBcol>0)  eval P2_WITHB
if (NROWS==2 && yBcol<=0) eval P2_NO_B

unset multiplot
unset output
set terminal pop

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
