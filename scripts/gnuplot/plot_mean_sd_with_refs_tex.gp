#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Single-panel plot: mean ± SD (from aggregate_topM_geometry_csv_runs.py) plus
# optional closest_ref and target curves (2-column CSVs: time, y).
#
# Output: standalone LaTeX (<OUTBASE>.tex) -> PDF via pdflatex.
#
# Requirements: gnuplot 5.4+, LaTeX, same as plot_csv_stddev_2stack_tex.gp
#
# Example (after aggregate_topM_geometry_csv_runs.py wrote combined.csv with
#   exactly 5 columns: time,mean_runs,sd_runs,closest_ref,target (NaN if unused)):
#
#   gnuplot -e "DATA='combined.csv';OUTBASE='fig';XLABEL='time (fs)';YLABEL='dihedral (deg)'" \
#       scripts/gnuplot/plot_mean_sd_with_refs_tex.gp
#
# With reference curves in separate files (two columns each, same times as DATA):
#
#   gnuplot -e "DATA='combined.csv';DATA_CLOSEST='closest_only.csv';DATA_TARGET='target_only.csv';OUTBASE='fig'" \
#       scripts/gnuplot/plot_mean_sd_with_refs_tex.gp
#
# Optional dihedral-style transforms (applied to y for all series; SD unchanged):
#
#   gnuplot -e "DIHEDRAL_TO360=1;DIHEDRAL_OFFSET=180;DIHEDRAL_NEGATE=0" \
#       scripts/gnuplot/plot_mean_sd_with_refs_tex.gp
#
# Shaded mean ± SD band instead of yerrorbars:
#
#   gnuplot -e "SHADEDERRORS=1;SHADE_ALPHA=0.22" scripts/gnuplot/plot_mean_sd_with_refs_tex.gp
#
# Legend position (gnuplot `set key` location string; default top right):
#
#   gnuplot -e "KEY_POS='bottom right';DATA='combined.csv';OUTBASE='fig'" \
#       scripts/gnuplot/plot_mean_sd_with_refs_tex.gp
# ------------------------------------------------------------------------------

if (!exists("DATA")) DATA = "combined.csv"
if (!exists("OUTBASE")) OUTBASE = "figure_mean_sd_refs"

if (!exists("DATA_CLOSEST")) DATA_CLOSEST = ""
if (!exists("DATA_TARGET")) DATA_TARGET = ""

if (!exists("SHADEDERRORS")) SHADEDERRORS = 0
if (!exists("SHADE_ALPHA")) SHADE_ALPHA = 0.20
SHADEDERRORS = SHADEDERRORS + 0
SHADE_ALPHA = SHADE_ALPHA + 0

if (!exists("DIHEDRAL_OFFSET")) DIHEDRAL_OFFSET = 0
if (!exists("DIHEDRAL_NEGATE")) DIHEDRAL_NEGATE = 0
if (!exists("DIHEDRAL_TO360")) DIHEDRAL_TO360 = 0
DIHEDRAL_OFFSET = DIHEDRAL_OFFSET + 0
DIHEDRAL_NEGATE = DIHEDRAL_NEGATE + 0
DIHEDRAL_TO360 = DIHEDRAL_TO360 + 0

MUL = (DIHEDRAL_NEGATE != 0) ? -1 : 1
to360(x) = (x < 0 ? x + 360 : x)

# y transforms (optional): negate, optional to360, then offset (same order idea as 2-stack).
YTR(v) = ((DIHEDRAL_TO360 != 0) ? to360(MUL * v) : (MUL * v)) + DIHEDRAL_OFFSET

is_nonempty_file(f) = int(system(sprintf("bash -lc \"test -s '%s' && echo 1 || echo 0\" ", f)))
HAS_DATA = is_nonempty_file(DATA)
if (!HAS_DATA) print sprintf("ERROR: DATA missing or empty: %s", DATA)
if (!HAS_DATA) exit

HAS_CLOSEST = (DATA_CLOSEST ne "") ? is_nonempty_file(DATA_CLOSEST) : 0
if ((DATA_CLOSEST ne "") && !HAS_CLOSEST) print sprintf("ERROR: DATA_CLOSEST set but missing/empty: %s", DATA_CLOSEST)
if ((DATA_CLOSEST ne "") && !HAS_CLOSEST) exit

HAS_TARGET = (DATA_TARGET ne "") ? is_nonempty_file(DATA_TARGET) : 0
if ((DATA_TARGET ne "") && !HAS_TARGET) print sprintf("ERROR: DATA_TARGET set but missing/empty: %s", DATA_TARGET)
if ((DATA_TARGET ne "") && !HAS_TARGET) exit

# If DATA_CLOSEST / DATA_TARGET are unset, use columns 4 and 5 of DATA (NaNs skipped).
USE_INTERNAL_CLOSEST = (DATA_CLOSEST eq "") ? 1 : 0
USE_INTERNAL_TARGET = (DATA_TARGET eq "") ? 1 : 0

if (!exists("COL_MEAN")) COL_MEAN = "#1b9e77"
if (!exists("COL_SD")) COL_SD = COL_MEAN
if (!exists("COL_CLOSEST")) COL_CLOSEST = "#d95f02"
if (!exists("COL_TARGET")) COL_TARGET = "#7570b3"

if (!exists("LW")) LW = 2.0
if (!exists("LW_REF")) LW_REF = 2.0
LW = LW + 0
LW_REF = LW_REF + 0

if (!exists("SHOW_KEY")) SHOW_KEY = 1
SHOW_KEY = SHOW_KEY + 0

# gnuplot key placement (e.g. top right, bottom right, left top); see `help set key`.
if (!exists("KEY_POS")) KEY_POS = "top right"

if (!exists("XLABEL")) XLABEL = 'time'
if (!exists("YLABEL")) YLABEL = 'value'

if (!exists("NAME_MEAN")) NAME_MEAN = 'mean \\pm SD (runs)'
if (!exists("NAME_CLOSEST")) NAME_CLOSEST = 'closest ref'
if (!exists("NAME_TARGET")) NAME_TARGET = 'target'

if (!exists("W")) W = 3.35
if (!exists("H")) H = 2.8
if (!exists("MLEFT")) MLEFT = 0.14
if (!exists("MRIGHT")) MRIGHT = 0.98
if (!exists("MBOTTOM")) MBOTTOM = 0.18
if (!exists("MTOP")) MTOP = 0.95
if (!exists("FONT")) FONT = "Latin Modern Roman,10"
if (!exists("LATEX_HEADER")) LATEX_HEADER = ""

if (!exists("MIRROR_TICS")) MIRROR_TICS = 1
MIRROR_TICS = MIRROR_TICS + 0

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
if (SHOW_KEY) eval "set key ".KEY_POS." opaque box lw 0.6"
if (!SHOW_KEY) unset key

eblw = (LW < 1.0 ? 1.0 : 0.5*LW)

# Mean uses $2, sd uses $3; transforms on mean only for errorbar center; band uses mean±sd on transformed mean? 
# Match 2-stack: transform mean, keep sd as-is added to transformed mean.
YMEAN = "(YTR($2))"
YLO = "((".YMEAN.")-$3)"
YHI = "((".YMEAN.")+$3)"

if (exists("XMIN")) XMIN = XMIN + 0
if (exists("XMAX")) XMAX = XMAX + 0
if (exists("XMIN") && exists("XMAX")) set xrange [XMIN:XMAX]
if (!(exists("XMIN") && exists("XMAX"))) unset xrange

if (exists("YMIN")) YMIN = YMIN + 0
if (exists("YMAX")) YMAX = YMAX + 0
if (exists("YMIN") && exists("YMAX")) set yrange [YMIN:YMAX]
if (!(exists("YMIN") && exists("YMAX"))) unset yrange

# ---- Build plot command ----
# Primary: DATA columns 1:2:3 = time, mean, sd (row 1 = header; skip 1)
PLOT_CMD = ""
if (SHADEDERRORS != 0) PLOT_CMD = PLOT_CMD . "'".DATA."' using 1:(".YLO."):(".YHI.") skip 1 with filledcurves lc rgb COL_SD fs transparent solid SHADE_ALPHA noborder notitle, "
if (SHADEDERRORS == 0) PLOT_CMD = PLOT_CMD . "'".DATA."' using 1:(".YMEAN."):3 skip 1 with yerrorbars lw eblw lc rgb COL_SD pt -1 notitle, "
if (SHOW_KEY) PLOT_CMD = PLOT_CMD . "'".DATA."' using 1:(".YMEAN.") skip 1 with lines lw LW lc rgb COL_MEAN title '".NAME_MEAN."'"
if (!SHOW_KEY) PLOT_CMD = PLOT_CMD . "'".DATA."' using 1:(".YMEAN.") skip 1 with lines lw LW lc rgb COL_MEAN notitle"

if (HAS_CLOSEST) PLOT_CMD = PLOT_CMD . ", '".DATA_CLOSEST."' using 1:(YTR($2)) skip 1 with lines lw LW_REF dt 2 lc rgb COL_CLOSEST title '".NAME_CLOSEST."'"
if (!HAS_CLOSEST && USE_INTERNAL_CLOSEST) PLOT_CMD = PLOT_CMD . ", '".DATA."' using 1:(($4)==($4) ? YTR($4) : 1/0) skip 1 with lines lw LW_REF dt 2 lc rgb COL_CLOSEST title '".NAME_CLOSEST."'"

if (HAS_TARGET) PLOT_CMD = PLOT_CMD . ", '".DATA_TARGET."' using 1:(YTR($2)) skip 1 with lines lw LW_REF dt 4 lc rgb COL_TARGET title '".NAME_TARGET."'"
if (!HAS_TARGET && USE_INTERNAL_TARGET) PLOT_CMD = PLOT_CMD . ", '".DATA."' using 1:(($5)==($5) ? YTR($5) : 1/0) skip 1 with lines lw LW_REF dt 4 lc rgb COL_TARGET title '".NAME_TARGET."'"

set lmargin at screen MLEFT
set rmargin at screen MRIGHT
set tmargin at screen MTOP
set bmargin at screen MBOTTOM
set xlabel XLABEL
set ylabel YLABEL

eval "plot ".PLOT_CMD

unset output
set terminal pop

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
