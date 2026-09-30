#!/bin/bash
# Overlay one x-ray fit (chi^2 closest to CHI2_TARGET) on the target curve.
# Two figures: qmax 4 and qmax 8.
#
# Edit comment / state to choose the results directories:
#   results_fig3_qmax{4,8}_${state}_${comment}
# Target file in each directory:
#   TARGET_FUNCTION_${run_id}.dat
#   e.g. TARGET_FUNCTION_fig3_qmax4_open_phi0p5.dat

comment=phi0p5_nr2
state=open
CHI2_TARGET=1e-3

set -euo pipefail

for qmax in 4 8; do
  dir="results_fig3_qmax${qmax}_${state}_${comment}"
  run_id="${dir#results_}"
  target="${dir}/TARGET_FUNCTION_${run_id}.dat"
  outbase="figure_xray_qmax${qmax}_${state}_${comment}"

  if [[ ! -f "$target" ]]; then
    echo "missing target: $target" >&2
    exit 1
  fi

  picked=$(python3 - "$dir" "$CHI2_TARGET" <<'PY'
import glob
import os
import re
import sys

directory = sys.argv[1]
chi2_target = float(sys.argv[2])
pat = re.compile(r"_(\d+\.\d+)(?:_dup\d+)?\.dat$")
best = None
for path in glob.glob(os.path.join(directory, "*.dat")):
    base = os.path.basename(path)
    if base.startswith("TARGET_FUNCTION_") or base in ("chi2_rmsd.dat", "stats.dat"):
        continue
    match = pat.search(base)
    if match is None:
        continue
    chi2 = float(match.group(1))
    key = (abs(chi2 - chi2_target), chi2, base)
    if best is None or key < best[0]:
        best = (key, path, chi2)
if best is None:
    sys.exit(f"no fit .dat in {directory}")
print(best[1])
print(f"{best[2]:.8f}")
PY
)

  fit=$(printf '%s\n' "$picked" | sed -n '1p')
  chi2=$(printf '%s\n' "$picked" | sed -n '2p')
  echo "${dir}: ${fit} (chi2=${chi2}, target=${CHI2_TARGET})"

  gnuplot -e "XMIN=0;XMAX=${qmax};TARGET='${target}';FIT='${fit}';CHI2=${chi2};OUTBASE='${outbase}'" \
    ./scripts/gnuplot/plot_dat_fit_vs_target_tex.gp
done

#pdflatex figure_xray_qmax4_${state}_${comment}.tex
#pdflatex figure_xray_qmax8_${state}_${comment}.tex
