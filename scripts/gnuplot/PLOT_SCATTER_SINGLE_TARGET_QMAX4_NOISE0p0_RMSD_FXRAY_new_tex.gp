reset

# latex .eps output
set terminal epslatex standalone color colortext 10 font "Helvetica,12" \
    header "\\usepackage{amsmath}"

# Custom line styles

LW1= 4.0
LW2 = 0.0
PS = 1.0
PS2 = 1.2

set style line 1 lt 1 pt 7 ps PS lw LW1 lc rgb '#0072bd' # blue
set style line 2 lt 1 pt 7 ps PS lw LW1 lc rgb '#d95319' # orange
set style line 3 lt 1 pt 7 ps PS lw LW1 lc rgb '#edb120' # yellow
set style line 4 lt 1 pt 7 ps PS lw LW1 lc rgb '#7e2f8e' # purple
set style line 5 lt 1 pt 7 ps PS lw LW1 lc rgb '#77ac30' # green
set style line 6 lt 1 pt 7 ps PS lw LW1 lc rgb '#4dbeee' # light-blue
set style line 7 lt 1 pt 6 ps PS2 lw LW1 lc rgb '#a2142f' # red
set style line 8 lt 1 pt 7 ps PS lw LW1 lc rgb '#666666' # grey
set style line 9 lt 1 pt 7 ps PS lw LW1 lc rgb '#99ae52' # olive
set style line 10 lt 1 pt 7 ps PS lw LW1 lc rgb '#000000' # black

set style line 102 lc rgb '#808080' lt 0 lw 3
set grid back ls 102

set size 0.8, 0.8   # Scale up the plot instead

set output "PLOT_SCATTER_SINGLE_TARGET_QMAX4.tex"

set xtics 0, 0.2, 1.6
set xlabel "RMSD (\\AA)" offset 0,0.4
set mxtics 2

set ytics ("" 10, "" 1, "" 0.1, "$10^{-2}$" 0.01, "$10^{-3}$" 0.001, "$10^{-4}$" 0.0001, "$10^{-5}$" 0.00001, "$10^{-6}$" 0.000001)
set mytics 10 
set ylabel "$\\chi^2$" offset 1.0,-3

#set key bottom right
unset key

XMIN = 0.000
XMAX = 0.44
YMIN = 0.000050
YMAX = 5
set yrange [YMIN : YMAX]
set xrange [XMIN : XMAX]

#set label 1 'q_{max} = 4 Å^{-1}' @POS
#set logscale x 10
set logscale y 10
p "results_single_target_qmax4_c1c6open/chi2_rmsd.dat" u 5:4 w p ls 7 t "C1-C6 open", \
  "results_single_target_qmax4_c1c6closed/chi2_rmsd.dat"   u 5:4 w p ls 1 t "C1-C6 closed", \
  #"analysis_qmax4_no_constraints.dat"   u 5:4 w p ls 2, \

### End

