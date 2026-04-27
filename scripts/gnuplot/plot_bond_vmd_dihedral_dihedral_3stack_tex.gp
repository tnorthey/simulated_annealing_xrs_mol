#!/usr/bin/env gnuplot
# ------------------------------------------------------------------------------
# Three stacked panels (shared x-axis):
#   Panel 1: bond — whitespace DATA1 (time, mean, std, no header) + optional VMD .dat
#   Panel 2: dihedral A — comma CSV DATA2 + optional DATA2B (same schema as plot_csv_stddev_2stack_tex.gp)
#   Panel 3: dihedral B — comma CSV DATA3 + optional DATA3B
#
# Piecewise fits (FIT1/FIT2 in 2-stack) are NOT implemented here (v1); use the
# standalone 2-stack script if you need those overlays.
#
# Requirements: gnuplot 5.4+, LaTeX (pdflatex).
#
# Bond DATA1 / DATA_VMD: same contract as plot_bond_csv_vs_vmd_1stack_tex.gp.
# Dihedral CSVs: header row; plots use skip 1 (override with DIHEDRAL_DATA_SKIP=0).
#
# Example:
#   gnuplot -e "DATA1='bond_ws.dat';DATA_VMD='vmd.dat';DATA2='d1.csv';DATA3='d2.csv';OUTBASE='fig'" \
#           scripts/gnuplot/plot_bond_vmd_dihedral_dihedral_3stack_tex.gp
# ------------------------------------------------------------------------------

if (!exists("DATA1"))      DATA1      = "bond_agg_ws.dat"
if (!exists("DATA_VMD"))   DATA_VMD   = ""
if (!exists("DATA2"))      DATA2      = "dihedral1.csv"
if (!exists("DATA2B"))     DATA2B     = ""
if (!exists("DATA3"))      DATA3      = "dihedral2.csv"
if (!exists("DATA3B"))     DATA3B     = ""
if (!exists("OUTBASE"))    OUTBASE    = "figure_bond_dihedral_3stack"

is_nonempty_file(f) = int(system(sprintf("bash -lc \"test -s '%s' && echo 1 || echo 0\" ", f)))
if (!is_nonempty_file(DATA1)) { print sprintf("ERROR: DATA1 missing or empty: %s", DATA1); exit }
HAS_VMD = (DATA_VMD ne "") ? is_nonempty_file(DATA_VMD) : 0
if ((DATA_VMD ne "") && !HAS_VMD) { print sprintf("ERROR: DATA_VMD set but missing or empty: %s", DATA_VMD); exit }

HAS2 = is_nonempty_file(DATA2)
HAS3 = is_nonempty_file(DATA3)
if (!HAS2) { print sprintf("ERROR: DATA2 missing or empty: %s", DATA2); exit }
if (!HAS3) { print sprintf("ERROR: DATA3 missing or empty: %s", DATA3); exit }
HAS2B = (DATA2B ne "") ? is_nonempty_file(DATA2B) : 0
HAS3B = (DATA3B ne "") ? is_nonempty_file(DATA3B) : 0
if ((DATA2B ne "") && !HAS2B) { print sprintf("ERROR: DATA2B missing or empty: %s", DATA2B); exit }
if ((DATA3B ne "") && !HAS3B) { print sprintf("ERROR: DATA3B missing or empty: %s", DATA3B); exit }

# ---- Panel 1 (bond): column indices ----
xcol = 1
yAcol = 2
sdAcol = 3
if (!exists("VMD_XCOL")) VMD_XCOL = 1
if (!exists("VMD_YCOL")) VMD_YCOL = 2
VMD_XCOL = VMD_XCOL + 0
VMD_YCOL = VMD_YCOL + 0
if (!exists("CSV_TSCALE")) CSV_TSCALE = 1.0
if (!exists("CSV_TOFF"))  CSV_TOFF  = 0.0
if (!exists("VMD_TSCALE")) VMD_TSCALE = 1.0
if (!exists("VMD_TOFF"))  VMD_TOFF  = 0.0
CSV_TSCALE = CSV_TSCALE + 0
CSV_TOFF = CSV_TOFF + 0
VMD_TSCALE = VMD_TSCALE + 0
VMD_TOFF = VMD_TOFF + 0
CX = int(xcol) + 0
CYm = int(yAcol) + 0
CYs = int(sdAcol) + 0
CXv = int(VMD_XCOL) + 0
CYv = int(VMD_YCOL) + 0

# ---- Dihedral transforms (panels 2 and 3 only; same knobs as 2-stack panel 2) ----
if (!exists("DIHEDRAL_OFFSET2"))  DIHEDRAL_OFFSET2  = 0
if (!exists("DIHEDRAL_OFFSET2B")) DIHEDRAL_OFFSET2B = 0
if (!exists("DIHEDRAL_OFFSET3"))  DIHEDRAL_OFFSET3  = 0
if (!exists("DIHEDRAL_OFFSET3B")) DIHEDRAL_OFFSET3B = 0
if (!exists("DIHEDRAL_NEGATE2"))  DIHEDRAL_NEGATE2  = 0
if (!exists("DIHEDRAL_NEGATE2B")) DIHEDRAL_NEGATE2B = 0
if (!exists("DIHEDRAL_NEGATE3"))  DIHEDRAL_NEGATE3  = 0
if (!exists("DIHEDRAL_NEGATE3B")) DIHEDRAL_NEGATE3B = 0
if (!exists("DIHEDRAL_TO3602"))   DIHEDRAL_TO3602   = 0
if (!exists("DIHEDRAL_TO3602B"))  DIHEDRAL_TO3602B  = 0
if (!exists("DIHEDRAL_TO3603"))   DIHEDRAL_TO3603   = 0
if (!exists("DIHEDRAL_TO3603B"))  DIHEDRAL_TO3603B  = 0
DIHEDRAL_OFFSET2  = DIHEDRAL_OFFSET2  + 0
DIHEDRAL_OFFSET2B = DIHEDRAL_OFFSET2B + 0
DIHEDRAL_OFFSET3  = DIHEDRAL_OFFSET3  + 0
DIHEDRAL_OFFSET3B = DIHEDRAL_OFFSET3B + 0
DIHEDRAL_NEGATE2  = DIHEDRAL_NEGATE2  + 0
DIHEDRAL_NEGATE2B = DIHEDRAL_NEGATE2B + 0
DIHEDRAL_NEGATE3  = DIHEDRAL_NEGATE3  + 0
DIHEDRAL_NEGATE3B = DIHEDRAL_NEGATE3B + 0
DIHEDRAL_TO3602   = DIHEDRAL_TO3602   + 0
DIHEDRAL_TO3602B  = DIHEDRAL_TO3602B  + 0
DIHEDRAL_TO3603   = DIHEDRAL_TO3603   + 0
DIHEDRAL_TO3603B  = DIHEDRAL_TO3603B  + 0
MUL2  = (DIHEDRAL_NEGATE2  != 0) ? -1 : 1
MUL2B = (DIHEDRAL_NEGATE2B != 0) ? -1 : 1
MUL3  = (DIHEDRAL_NEGATE3  != 0) ? -1 : 1
MUL3B = (DIHEDRAL_NEGATE3B != 0) ? -1 : 1
to360(x) = (x < 0 ? x + 360 : x)

if (!exists("DIHEDRAL_DATA_SKIP")) DIHEDRAL_DATA_SKIP = 1
DIHEDRAL_DATA_SKIP = DIHEDRAL_DATA_SKIP + 0
SKD = (DIHEDRAL_DATA_SKIP != 0) ? " skip 1 " : " "

# ---- Relative heights (any positive; normalized) ----
if (!exists("RELH1")) RELH1 = 0.25
if (!exists("RELH2")) RELH2 = 0.375
if (!exists("RELH3")) RELH3 = 0.375
RELH1 = RELH1 + 0
RELH2 = RELH2 + 0
RELH3 = RELH3 + 0
if (RELH1 <= 0 || RELH2 <= 0 || RELH3 <= 0) { print "ERROR: RELH1, RELH2, RELH3 must be > 0"; exit }
RELHSUM = RELH1 + RELH2 + RELH3

# ---- Names / labels ----
if (!exists("NAME1"))   NAME1   = "mean $\\pm$ SD"
if (!exists("NAME2"))   NAME2   = "VMD"
if (!exists("NAME2A")) NAME2A  = "Dataset 2"
if (!exists("NAME2B")) NAME2B  = "Dataset 2B"
if (!exists("NAME3A")) NAME3A  = "Dataset 3"
if (!exists("NAME3B")) NAME3B  = "Dataset 3B"
if (!exists("XLABEL"))  XLABEL  = '$t$ (fs)'
if (!exists("Y1LABEL")) Y1LABEL = 'Bond (\\AA)'
if (!exists("Y2LABEL")) Y2LABEL = 'Dihedral 1 (deg)'
if (!exists("Y3LABEL")) Y3LABEL = 'Dihedral 2 (deg)'
if (!exists("YLABEL_OFFSETX")) YLABEL_OFFSETX = 0.75
YLABEL_OFFSETX = YLABEL_OFFSETX + 0
if (!exists("Y1LABEL_OFFSET")) Y1LABEL_OFFSET = YLABEL_OFFSETX
if (!exists("Y2LABEL_OFFSET")) Y2LABEL_OFFSET = YLABEL_OFFSETX
if (!exists("Y3LABEL_OFFSET")) Y3LABEL_OFFSET = YLABEL_OFFSETX
Y1LABEL_OFFSET = Y1LABEL_OFFSET + 0
Y2LABEL_OFFSET = Y2LABEL_OFFSET + 0
Y3LABEL_OFFSET = Y3LABEL_OFFSET + 0

# ---- Colors ----
if (!exists("COL1")) COL1 = "#1b9e77"
if (!exists("COL2")) COL2 = "#d95f02"
if (!exists("COL2B")) COL2B = "#e7298a"
if (!exists("COL3")) COL3 = "#7570b3"
if (!exists("COL3B")) COL3B = "#666666"

# ---- Line / point (panel 1 bond) ----
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

# ---- Dihedral panels: LW/LWB (names avoid clash with bond LW1/LW2) ----
if (!exists("LW")) LW = 2.0
if (!exists("LWB")) LWB = LW
LW = LW + 0
LWB = LWB + 0
if (!exists("LWD2"))  LWD2  = LW
if (!exists("LWD2B")) LWD2B = LWB
if (!exists("LWD3"))  LWD3  = LW
if (!exists("LWD3B")) LWD3B = LWB
LWD2 = LWD2 + 0
LWD2B = LWD2B + 0
LWD3 = LWD3 + 0
LWD3B = LWD3B + 0
if (!exists("PS")) PS = 0.75
if (!exists("PSB")) PSB = PS
PS = PS + 0
PSB = PSB + 0
if (!exists("PSD2"))  PSD2  = PS
if (!exists("PSD2B")) PSD2B = PSB
if (!exists("PSD3"))  PSD3  = PS
if (!exists("PSD3B")) PSD3B = PSB
PSD2 = PSD2 + 0
PSD2B = PSD2B + 0
PSD3 = PSD3 + 0
PSD3B = PSD3B + 0
if (!exists("PT")) PT = 7
if (!exists("PTB")) PTB = 5
PT = PT + 0
PTB = PTB + 0
if (!exists("PTD2"))  PTD2  = PT
if (!exists("PTD2B")) PTD2B = PTB
if (!exists("PTD3"))  PTD3  = PT
if (!exists("PTD3B")) PTD3B = PTB
PTD2 = PTD2 + 0
PTD2B = PTD2B + 0
PTD3 = PTD3 + 0
PTD3B = PTD3B + 0

set macros
if (!exists("PLOTMODE")) PLOTMODE = 'LINES'
if (!exists("PLOTMODE2")) PLOTMODE2 = 'LINES'
PLOT_WITH1 = (PLOTMODE eq 'LINES') ? 'lines' : ((PLOTMODE eq 'POINTS') ? 'points' : 'linespoints')
PLOT_WITH2 = (PLOTMODE2 eq 'LINES') ? 'lines' : ((PLOTMODE2 eq 'POINTS') ? 'points' : 'linespoints')

if (!exists("PLOTMODE_DIH")) PLOTMODE_DIH = 'LP'
if (!exists("PLOTMODEB_DIH")) PLOTMODEB_DIH = PLOTMODE_DIH
PLOT_WITHD  = (PLOTMODE_DIH  eq 'LINES')  ? 'lines'       : ((PLOTMODE_DIH  eq 'POINTS')  ? 'points' : 'linespoints')
PLOT_WITHDB = (PLOTMODEB_DIH eq 'LINES')  ? 'lines'       : ((PLOTMODEB_DIH eq 'POINTS') ? 'points' : 'linespoints')

if (!exists("SHADE_ALPHA")) SHADE_ALPHA = 0.25
SHADE_ALPHA = SHADE_ALPHA + 0
if (!exists("SHADEDERRORS1")) SHADEDERRORS1 = 1
SHADEDERRORS1 = SHADEDERRORS1 + 0
if (!exists("SHADEDERRORS2"))  SHADEDERRORS2  = 0
if (!exists("SHADEDERRORS2B")) SHADEDERRORS2B = 0
if (!exists("SHADEDERRORS3"))  SHADEDERRORS3  = 0
if (!exists("SHADEDERRORS3B")) SHADEDERRORS3B = 0
SHADEDERRORS2  = SHADEDERRORS2  + 0
SHADEDERRORS2B = SHADEDERRORS2B + 0
SHADEDERRORS3  = SHADEDERRORS3  + 0
SHADEDERRORS3B = SHADEDERRORS3B + 0

if (!exists("SHOW_KEY")) SHOW_KEY = 1

if (exists("XTIC_STEP")) XTIC_STEP = XTIC_STEP + 0
if (exists("YTIC_STEP")) YTIC_STEP = YTIC_STEP + 0
if (exists("YTIC_STEP1")) YTIC_STEP1 = YTIC_STEP1 + 0
if (exists("YTIC_STEP2")) YTIC_STEP2 = YTIC_STEP2 + 0
if (exists("YTIC_STEP3")) YTIC_STEP3 = YTIC_STEP3 + 0
if (exists("XMIN")) XMIN = XMIN + 0
if (exists("XMAX")) XMAX = XMAX + 0
if (exists("Y1MIN")) Y1MIN = Y1MIN + 0
if (exists("Y1MAX")) Y1MAX = Y1MAX + 0
if (exists("Y2MIN")) Y2MIN = Y2MIN + 0
if (exists("Y2MAX")) Y2MAX = Y2MAX + 0
if (exists("Y3MIN")) Y3MIN = Y3MIN + 0
if (exists("Y3MAX")) Y3MAX = Y3MAX + 0

if (!exists("W")) W = 4.0
if (!exists("H")) H = 6.5
if (!exists("MLEFT"))   MLEFT   = 0.14
if (!exists("MRIGHT"))  MRIGHT  = 0.98
if (!exists("MBOTTOM")) MBOTTOM = 0.12
if (!exists("MTOP"))    MTOP    = 0.98
if (!exists("FONT")) FONT = "Latin Modern Roman,10"
if (!exists("LATEX_HEADER")) LATEX_HEADER = ""
if (!exists("MIRROR_TICS")) MIRROR_TICS = 1
MIRROR_TICS = MIRROR_TICS + 0

LATEX_HDR = (LATEX_HEADER ne "") ? " header '".LATEX_HEADER."'" : ""
eval "set terminal cairolatex pdf standalone size ".sprintf("%g",W).",".sprintf("%g",H)." font '".FONT."' dashed color".LATEX_HDR
set output sprintf("%s.tex", OUTBASE)

# ---- Y expressions for dihedral CSVs (columns yAcol / sdAcol, default 2 / 3) ----
YCOL = sprintf("%d", int(yAcol))
SDCOL = sprintf("%d", int(sdAcol))
YEXPR2  = "(MUL2*$".YCOL.")"
YEXPR2B = "(MUL2B*$".YCOL.")"
YEXPR3  = "(MUL3*$".YCOL.")"
YEXPR3B = "(MUL3B*$".YCOL.")"
if (DIHEDRAL_TO3602  != 0) YEXPR2  = "to360(".YEXPR2.")"
if (DIHEDRAL_TO3602B != 0) YEXPR2B = "to360(".YEXPR2B.")"
if (DIHEDRAL_TO3603  != 0) YEXPR3  = "to360(".YEXPR3.")"
if (DIHEDRAL_TO3603B != 0) YEXPR3B = "to360(".YEXPR3B.")"
YEXPR2  = "(".YEXPR2 ."+DIHEDRAL_OFFSET2)"
YEXPR2B = "(".YEXPR2B."+DIHEDRAL_OFFSET2B)"
YEXPR3  = "(".YEXPR3 ."+DIHEDRAL_OFFSET3)"
YEXPR3B = "(".YEXPR3B."+DIHEDRAL_OFFSET3B)"

YLO2  = "(".YEXPR2 ." - $".SDCOL.")"
YHI2  = "(".YEXPR2 ." + $".SDCOL.")"
YLO2B = "(".YEXPR2B." - $".SDCOL.")"
YHI2B = "(".YEXPR2B." + $".SDCOL.")"
YLO3  = "(".YEXPR3 ." - $".SDCOL.")"
YHI3  = "(".YEXPR3 ." + $".SDCOL.")"
YLO3B = "(".YEXPR3B." - $".SDCOL.")"
YHI3B = "(".YEXPR3B." + $".SDCOL.")"

eblw1 = (LW1 < 1.0 ? 1.0 : 0.5*LW1)
eblw2  = (LWD2  < 1.0 ? 1.0 : 0.5*LWD2)
eblw2B = (LWD2B < 1.0 ? 1.0 : 0.5*LWD2B)
eblw3  = (LWD3  < 1.0 ? 1.0 : 0.5*LWD3)
eblw3B = (LWD3B < 1.0 ? 1.0 : 0.5*LWD3B)

E2  = "'".DATA2 ."'".SKD."using xcol:(".YEXPR2 ."):sdAcol with yerrorbars lw eblw2  lc rgb COL2  pt -1 notitle"
E2B = "'".DATA2B."'".SKD."using xcol:(".YEXPR2B."):sdAcol with yerrorbars lw eblw2B lc rgb COL2B pt -1 notitle"
E3  = "'".DATA3 ."'".SKD."using xcol:(".YEXPR3 ."):sdAcol with yerrorbars lw eblw3  lc rgb COL3  pt -1 notitle"
E3B = "'".DATA3B."'".SKD."using xcol:(".YEXPR3B."):sdAcol with yerrorbars lw eblw3B lc rgb COL3B pt -1 notitle"

if (SHADEDERRORS2  != 0) E2  = "'".DATA2 ."'".SKD."using xcol:(".YLO2 ."):(".YHI2 .") with filledcurves lc rgb COL2  fs transparent solid SHADE_ALPHA noborder notitle"
if (SHADEDERRORS2B != 0) E2B = "'".DATA2B."'".SKD."using xcol:(".YLO2B."):(".YHI2B.") with filledcurves lc rgb COL2B fs transparent solid SHADE_ALPHA noborder notitle"
if (SHADEDERRORS3  != 0) E3  = "'".DATA3 ."'".SKD."using xcol:(".YLO3 ."):(".YHI3 .") with filledcurves lc rgb COL3  fs transparent solid SHADE_ALPHA noborder notitle"
if (SHADEDERRORS3B != 0) E3B = "'".DATA3B."'".SKD."using xcol:(".YLO3B."):(".YHI3B.") with filledcurves lc rgb COL3B fs transparent solid SHADE_ALPHA noborder notitle"

P2A = E2  .", '".DATA2 ."'".SKD."using xcol:(".YEXPR2 .") with @PLOT_WITHD ls 21 lc rgb COL2 title NAME2A"
P2B = E2B .", '".DATA2B."'".SKD."using xcol:(".YEXPR2B.") with @PLOT_WITHDB ls 22 lc rgb COL2B title NAME2B"
P3A = E3  .", '".DATA3 ."'".SKD."using xcol:(".YEXPR3 .") with @PLOT_WITHD ls 31 lc rgb COL3 title NAME3A"
P3B = E3B .", '".DATA3B."'".SKD."using xcol:(".YEXPR3B.") with @PLOT_WITHDB ls 32 lc rgb COL3B title NAME3B"

P2_CMD = "plot ".P2A.(HAS2B ? ", ".P2B : "")
P3_CMD = "plot ".P3A.(HAS3B ? ", ".P3B : "")

# ---- Styling (before multiplot; bond uses unset separator) ----
set border linewidth 1.2
set tics scale 0.75
if (MIRROR_TICS) set xtics mirror
if (MIRROR_TICS) set ytics mirror
if (!MIRROR_TICS) set xtics nomirror
if (!MIRROR_TICS) set ytics nomirror
set mxtics 2
set mytics 2
set grid back xtics ytics lw 0.6 lc rgb "#D0D0D0"

set style line 11 lw LW1 pt PT1 ps PS1
set style line 12 lw LW2 pt PT2 ps PS2
set style line 21 lw LWD2 pt PTD2 ps PSD2
set style line 22 lw LWD2B pt PTD2B ps PSD2B
set style line 31 lw LWD3 pt PTD3 ps PSD3
set style line 32 lw LWD3B pt PTD3B ps PSD3B

unset key
if (SHOW_KEY) set key opaque box lw 0.6
unset title

if (exists("XMIN") && exists("XMAX")) set xrange [XMIN:XMAX]
if (!(exists("XMIN") && exists("XMAX"))) unset xrange

set lmargin at screen MLEFT
set rmargin at screen MRIGHT

AVAILH = MTOP - MBOTTOM
H3 = AVAILH * (RELH3 / RELHSUM)
H2 = AVAILH * (RELH2 / RELHSUM)
YSPLIT23 = MBOTTOM + H3
YSPLIT12 = MBOTTOM + H3 + H2

set multiplot

# ---- Panel 1 (top): bond ----
set tmargin at screen MTOP
set bmargin at screen YSPLIT12
set ylabel Y1LABEL offset Y1LABEL_OFFSET,0
unset xlabel
set format x ""
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP1")) set ytics YTIC_STEP1
if (!exists("YTIC_STEP1") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP1") && !exists("YTIC_STEP")) set ytics
if (exists("Y1MIN") && exists("Y1MAX")) set yrange [Y1MIN:Y1MAX]
if (!(exists("Y1MIN") && exists("Y1MAX"))) unset yrange

unset datafile separator

if (SHADEDERRORS1 && HAS_VMD) \
plot \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)-column(CYs)):(column(CYm)+column(CYs)) with filledcurves lc rgb COL1 fs transparent solid SHADE_ALPHA noborder notitle, \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)) with @PLOT_WITH1 ls 11 lc rgb COL1 title NAME1, \
  DATA_VMD using (column(CXv)*VMD_TSCALE+VMD_TOFF):(column(CYv)) with @PLOT_WITH2 ls 12 lc rgb COL2 title NAME2

if (SHADEDERRORS1 && !HAS_VMD) \
plot \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)-column(CYs)):(column(CYm)+column(CYs)) with filledcurves lc rgb COL1 fs transparent solid SHADE_ALPHA noborder notitle, \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)) with @PLOT_WITH1 ls 11 lc rgb COL1 title NAME1

if (!SHADEDERRORS1 && HAS_VMD) \
plot \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)):(column(CYs)) with yerrorbars lw eblw1 lc rgb COL1 pt -1 notitle, \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)) with @PLOT_WITH1 ls 11 lc rgb COL1 title NAME1, \
  DATA_VMD using (column(CXv)*VMD_TSCALE+VMD_TOFF):(column(CYv)) with @PLOT_WITH2 ls 12 lc rgb COL2 title NAME2

if (!SHADEDERRORS1 && !HAS_VMD) \
plot \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)):(column(CYs)) with yerrorbars lw eblw1 lc rgb COL1 pt -1 notitle, \
  DATA1 using (column(CX)*CSV_TSCALE+CSV_TOFF):(column(CYm)) with @PLOT_WITH1 ls 11 lc rgb COL1 title NAME1

# ---- Panel 2 (middle): dihedral ----
set datafile separator comma
set tmargin at screen YSPLIT12
set bmargin at screen YSPLIT23
set ylabel Y2LABEL offset Y2LABEL_OFFSET,0
unset xlabel
set format x ""
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP2")) set ytics YTIC_STEP2
if (!exists("YTIC_STEP2") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP2") && !exists("YTIC_STEP")) set ytics
if (exists("Y2MIN") && exists("Y2MAX")) set yrange [Y2MIN:Y2MAX]
if (!(exists("Y2MIN") && exists("Y2MAX"))) unset yrange
unset key
if (SHOW_KEY) set key top right opaque box lw 0.6
eval P2_CMD

# ---- Panel 3 (bottom): dihedral ----
set tmargin at screen YSPLIT23
set bmargin at screen MBOTTOM
set ylabel Y3LABEL offset Y3LABEL_OFFSET,0
set xlabel XLABEL
set format x "%g"
if (exists("XTIC_STEP")) set xtics XTIC_STEP
if (!exists("XTIC_STEP")) set xtics
if (exists("YTIC_STEP3")) set ytics YTIC_STEP3
if (!exists("YTIC_STEP3") && exists("YTIC_STEP")) set ytics YTIC_STEP
if (!exists("YTIC_STEP3") && !exists("YTIC_STEP")) set ytics
if (exists("Y3MIN") && exists("Y3MAX")) set yrange [Y3MIN:Y3MAX]
if (!(exists("Y3MIN") && exists("Y3MAX"))) unset yrange
unset key
if (SHOW_KEY) set key top right opaque box lw 0.6
eval P3_CMD

unset multiplot
unset output
set terminal pop

print sprintf("Wrote %s.tex (compile: pdflatex %s.tex).", OUTBASE, OUTBASE)
