#!/usr/bin/env bash
# Compare C1–C6 bond length: NMM (or similar) CSV with mean/std vs VMD .dat trajectory.
# Run from repo root, or pass absolute paths.
#
# Usage:
#   ./scripts/bash/plot_bond_csv_vmd_compare.sh [bond_stats.csv] [vmd_c1c6_traj099.dat] [output.pdf]
#
# Defaults (second/third args optional):
#   CSV:   bond_c1c6_mean_std.csv
#   VMD:   vmd_c1c6_traj099.dat
#   PDF:   nmm_bond_c1c6_compare.pdf
#
# Requires: gnuplot 5.4+, pdflatex

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CSV_IN="${1:-bond_c1c6_mean_std.csv}"
VMD_IN="${2:-vmd_c1c6_traj099.dat}"
OUT_PDF="${3:-nmm_bond_c1c6_compare.pdf}"
OUTBASE="figure_bond_c1c6_compare"
TMP_WS="${ROOT}/scripts/gnuplot/_tmp_bond_mean_std_ws.dat"

if [[ "$CSV_IN" != /* ]]; then CSV_IN="${ROOT}/${CSV_IN}"; fi
if [[ "$VMD_IN" != /* ]]; then VMD_IN="${ROOT}/${VMD_IN}"; fi
if [[ "$OUT_PDF" != /* ]]; then OUT_PDF="${ROOT}/${OUT_PDF}"; fi

if [[ ! -s "$CSV_IN" ]]; then
  echo "ERROR: CSV not found or empty: $CSV_IN" >&2
  exit 1
fi
if [[ ! -s "$VMD_IN" ]]; then
  echo "ERROR: VMD .dat not found or empty: $VMD_IN" >&2
  exit 1
fi

# Strip header, drop CR, convert CSV to whitespace-separated numeric rows (3+ cols -> first 3)
tail -n +2 "$CSV_IN" | tr -d '\r' | awk -F',' 'NF>=3 {gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1, $2, $3}' > "$TMP_WS"

if [[ ! -s "$TMP_WS" ]]; then
  echo "ERROR: no data rows in CSV after header (check commas / columns). File: $CSV_IN" >&2
  exit 1
fi

gnuplot -e "DATA1='${TMP_WS}';
            DATA_VMD='${VMD_IN}';
            OUTBASE='${OUTBASE}';
            XLABEL='\$t\$ (fs)';
            Y1LABEL='C1--C6 bond (\\AA)';
            PLOTMODE='LINES';
            PLOTMODE2='LINES';
            SHADEDERRORS1=1;
            SHADE_ALPHA=0.25;
            NAME2='VMD';
            SHOW_KEY=1;
            COL1='#1b9e77';
            COL2='#d95f02';
            LW1=2;
            LW2=1.5;
            " "${ROOT}/scripts/gnuplot/plot_bond_csv_vs_vmd_1stack_tex.gp"

# cairolatex: OUTBASE.tex plus *_inc*.pdf in ROOT
if ! pdflatex -interaction=nonstopmode -halt-on-error "${OUTBASE}.tex" >"pdflatex_${OUTBASE}.log" 2>&1; then
  echo "ERROR: pdflatex failed. See ${ROOT}/pdflatex_${OUTBASE}.log" >&2
  exit 1
fi
rm -f "pdflatex_${OUTBASE}.log"
cp -f "${OUTBASE}.pdf" "$OUT_PDF"
echo "Wrote: $OUT_PDF (and ${OUTBASE}.tex in ${ROOT})"
