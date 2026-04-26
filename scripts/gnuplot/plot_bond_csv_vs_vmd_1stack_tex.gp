#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Single panel: bond length vs time — CSV (mean ± std) vs VMD trajectory (.dat).
#
# The CSV is expected as space/tab-separated **without a header** (strip the
# first line in the shell), three columns: time, mean, stddev.
# The VMD file is whitespace-separated; default columns 1:time, 2:bond (Å).
#
# To use a single-column external time file for the x-axis for both series, use
# scripts/bash/plot_bond_csv_vmd_compare.sh (4th arg or env TIME_PLOT_FILE); the
# shell merges it with the y columns by row. Row counts must match.
#
# Output: standalone LaTeX (<OUTBASE>.tex); compile with: pdflatex <OUTBASE>.tex
#
# Example:
#   gnuplot -e "DATA1='agg.dat';DATA_VMD='vmd.dat';OUTBASE='figure_bond'" \
#           scripts/gnuplot/plot_bond_csv_vs_vmd_1stack_tex.gp
# ------------------------------------------------------------------------------

if (!exists("DATA1"))     DATA1     = "bond_agg_ws.dat"
if (!exists("DATA_VMD"))  DATA_VMD  = "vmd_c1c6_traj099.dat"
if (!exists("OUTBASE"))   OUTBASE   = "figure_bond"

is_nonempty_file(f) = int(system(sprintf("bash -lc \"test -s '%s' && echo 1 || echo 0\" ", f)))
if (!is_nonempty_file(DATA1))    { print sprintf("ERROR: DATA1 missing or empty: %s", DATA1); exit }
if (!is_nonempty_file(DATA_VMD))  { print sprintf("ERROR: DATA_VMD missing or empty: %s", DATA_VMD); exit }

# Column mapping: DATA1 = t, y, sd; DATA_VMD = time, bond
xcol = 1
yAcol = 2
sdAcol = 3
if (!exists("VMD_XCOL")) VMD_XCOL = 1
if (!exists("VMD_YCOL")) VMD_YCOL = 2
VMD_XCOL = VMD_XCOL + 0
VMD_YCOL = VMD_YCOL + 0

# Optional time rescaling (e.g. align ps vs fs between the two series)
if (!exists("CSV_TSCALE")) CSV_TSCALE = 1.0
if (!exists("CSV_TOFF"))  CSV_TOFF  = 0.0
if (!exists("VMD_TSCALE")) VMD_TSCALE = 1.0
if (!exists("VMD_TOFF"))  VMD_TOFF  = 0.0
CSV_TSCALE = CSV_TSCALE + 0
CSV_TOFF = CSV_TOFF + 0
VMD_TSCALE = VMD_TSCALE + 0
VMD_TOFF = VMD_TOFF + 0

# Legend (titles are passed to cairolatex; use TeX or plain text)
if (!exists("NAME1"))  NAME1  = "mean $\\pm$ SD"
if (!exists("NAME2"))  NAME2  = "VMD"
if (!exists("SHOW_KEY")) SHOW_KEY = 1

# Colors
if (!exists("COL1")) COL1 = "#1b9e77"
if (!exists("COL2")) COL2 = "#d95f02"

if (!exists("LW1")) LW1 = 2.0
if (!exists("LW2")) LW2 = 2.0
if (!exists("PS1")) PS1 = 0.75
if (!exists("PS2")) PS2 = 0.75
if (!exists("PT1")) PT1 = 7
if (!exists("PT2")) PT2 = 5
LW1 = LW1 + 0
LW2 = LW2 + 0
PS1 = PS1 + 0
PS2 = PS2 + 0
PT1 = PT1 + 0
PT2 = PT2 + 0

set macros
if (!exists("PLOTMODE")) PLOTMODE = 'LP'
if (!exists("PLOTMODE2")) PLOTMODE2 = 'LINES'
PLOT_WITH1 = (PLOTMODE eq 'LINES') ? 'lines' : ((PLOTMODE eq 'POINTS') ? 'points' : 'linespoints')
PLOT_WITH2 = (PLOTMODE2 eq 'LINES') ? 'lines' : ((PLOTMODE2 eq 'POINTS') ? 'points' : 'linespoints')

if (!exists("SHADE_ALPHA")) SHADE_ALPHA = 0.25
SHADE_ALPHA = SHADE_ALPHA + 0
if (!exists("SHADEDERRORS1")) SHADEDERRORS1 = 1
SHADEDERRORS1 = SHADEDERRORS1 + 0

if (!exists("XLABEL"))  XLABEL  = '$t$ (fs)'
if (!exists("Y1LABEL")) Y1LABEL = 'C1--C6 bond (\\AA)'
if (!exists("Y1LABEL_OFFSET")) Y1LABEL_OFFSET = 0.75
Y1LABEL_OFFSET = Y1LABEL_OFFSET + 0

if (exists("XTIC_STEP")) XTIC_STEP = XTIC_STEP + 0
if (exists("YTIC_STEP")) YTIC_STEP = YTIC_STEP + 0
if (exists("Y1MIN")) Y1MIN = Y1MIN + 0
if (exists("Y1MAX")) Y1MAX = Y1MAX + 0
if (exists("XMIN")) XMIN = XMIN + 0
if (exists("XMAX")) XMAX = XMAX + 0
CX = int(xcol) + 0
CYm = int(yAcol) + 0
CYs = int(sdAcol) + 0
CXv = int(VMD_XCOL) + 0
CYv = int(VMD_YCOL) + 0

if (!exists("W")) W = 4.0
if (!exists("H")) H = 3.5
if (!exists("MLEFT"))   MLEFT   = 0.14
if (!exists("MRIGHT"))  MRIGHT  = 0.98
if (!exists("MBOTTOM")) MBOTTOM = 0.16
if (!exists("MTOP"))    MTOP    = 0.98
if (!exists("FONT")) FONT = "Latin Modern Roman,10"
if (!exists("LATEX_HEADER")) LATEX_HEADER = ""

LATEX_HDR = (LATEX_HEADER ne "") ? " header '".LATEX_HEADER."'" : ""
eval "set terminal cairolatex pdf standalone size ".sprintf("%g",W).",".sprintf("%g",H)." font '".FONT."' dashed color".LATEX_HDR
set output sprintf("%s.tex", OUTBASE)

unset datafile separator
set encoding utf8
unset key
if (SHOW_KEY) set key top right opaque box lw 0.6

set border linewidth 1.2
set tics scale 0.75
set xtics nomirror
set ytics nomirror
set mxtics 2
set mytics 2
set grid back xtics ytics lw 0.6 lc rgb "#D0D0D0"

set style line 11 lw LW1 pt PT1 ps PS1
set style line 12 lw LW2 pt PT2 ps PS2

eblw1 = (LW1 < 1.0 ? 1.0 : 0.5*LW1)

set lmargin at screen MLEFT
set rmargin at screen MRIGHT
set tmargin at screen MTOP
set bmargin at screen MBOTTOM

set ylabel Y1LABEL offset Y1LABEL_OFFSET,0
set xlabel XLABEL
set format x "%g"

if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP")) set ytics

if (exists("XMIN") && exists("XMAX")) set xrange [XMIN:XMAX]
if (!(exists("XMIN") && exists("XMAX"))) unset xrange
if (exists("Y1MIN") && exists("Y1MAX")) set yrange [Y1MIN:Y1MAX]
if (!(exists("Y1MIN") && exists("Y1MAX"))) unset yrange

# DATA1: columns xcol, yAcol, sdAcol  (whitespace, no header)
if (SHADEDERRORS1) \
plot \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)-column(CYs)):(column(CYm)+column(CYs)) with filledcurves lc rgb COL1 fs transparent solid SHADE_ALPHA noborder notitle, \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)) with @PLOT_WITH1 ls 11 lc rgb COL1 title NAME1, \
  DATA_VMD using (column(CXv)*VMD_TSCALE+VMD_TOFF):(column(CYv)) with @PLOT_WITH2 ls 12 lc rgb COL2 title NAME2

if (!SHADEDERRORS1) \
plot \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)):(column(CYs)) with yerrorbars lw eblw1 lc rgb COL1 pt -1 notitle, \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)) with @PLOT_WITH1 ls 11 lc rgb COL1 title NAME1, \
  DATA_VMD using (column(CXv)*VMD_TSCALE+VMD_TOFF):(column(CYv)) with @PLOT_WITH2 ls 12 lc rgb COL2 title NAME2

unset output
set terminal pop

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
