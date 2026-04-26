#!/usr/bin/env bash
# Compare C1–C6 bond length: stats CSV (mean/std) vs VMD .dat trajectory.
# Run from repo root, or pass absolute paths.
#
# Usage (order matters: stats first, VMD second):
#   ./scripts/bash/plot_bond_csv_vmd_compare.sh <stats_mean_std.csv> <vmd_traj.dat> [output.pdf] [time.dat]
#
# stats CSV: header line, then comma-separated time_fs, mean, std (3+ columns).
# VMD .dat:  whitespace, two numbers per line (e.g. frame/ time, bond length).
# time.dat:  optional, one time value per line (replaces the first column in both stats and VMD data).
#            Must have the same number of rows as the stats and VMD files (not counting the CSV header).
#            4th argument, or set env  TIME_PLOT_FILE  (e.g. when you omit a custom [output.pdf] you can use TIME_PLOT_FILE=time.dat).
#
# If you pass a .dat and a .csv in the wrong order, the script will swap them when it can tell.
#
# Defaults (if args omitted, paths are under repo root):
#   CSV:   bond_c1c6_mean_std.csv
#   VMD:   vmd_c1c6_traj099.dat
#   PDF:   bond_c1c6_compare.pdf
#
# Requires: gnuplot 5.4+, pdflatex

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CSV_IN="${1:-bond_c1c6_mean_std.csv}"
VMD_IN="${2:-vmd_c1c6_traj099.dat}"
OUT_PDF="${3:-bond_c1c6_compare.pdf}"
# 4th arg or TIME_PLOT_FILE: one-column time series replacing x in both inputs
TIME_IN="${4:-${TIME_PLOT_FILE:-}}"
OUTBASE="figure_bond_c1c6_compare"
TMP_WS="${ROOT}/scripts/gnuplot/_tmp_bond_mean_std_ws.dat"
TMP_RETIME_STATS="${ROOT}/scripts/gnuplot/_tmp_bond_retime_stats.dat"
TMP_RETIME_VMD="${ROOT}/scripts/gnuplot/_tmp_bond_retime_vmd.dat"
TMP_TIME_NORM="${ROOT}/scripts/gnuplot/_tmp_bond_time_col.dat"

if [[ "$CSV_IN" != /* ]]; then CSV_IN="${ROOT}/${CSV_IN}"; fi
if [[ "$VMD_IN" != /* ]]; then VMD_IN="${ROOT}/${VMD_IN}"; fi
if [[ "$OUT_PDF" != /* ]]; then OUT_PDF="${ROOT}/${OUT_PDF}"; fi
if [[ -n "$TIME_IN" && "$TIME_IN" != /* ]]; then TIME_IN="${ROOT}/${TIME_IN}"; fi

if [[ ! -s "$CSV_IN" ]]; then
  echo "ERROR: CSV not found or empty: $CSV_IN" >&2
  exit 1
fi
if [[ ! -s "$VMD_IN" ]]; then
  echo "ERROR: VMD .dat not found or empty: $VMD_IN" >&2
  exit 1
fi

# Heuristic: first arg is stats CSV (commas, 3+ fields on header); second is VMD (no commas, 2 fields).
# If that pattern is reversed, swap so a common mistake still works.
looks_like_vmd() {
  local f=$1
  local line comma nf
  read -r line < "$f" || return 1
  [[ "$line" != *","* ]] || return 1
  nf=$(awk '{print NF}' <<< "$line" | tr -d '\r')
  [[ ${nf:-0} -eq 2 ]]
}
looks_like_stats_csv() {
  local f=$1
  local line nf
  read -r line < "$f" || return 1
  [[ "$line" == *","* ]] || return 1
  nf=$(awk -F, '{print NF}' <<< "$line" | tr -d '\r\n')
  [[ ${nf:-0} -ge 3 ]]
}
if looks_like_vmd "$CSV_IN" && looks_like_stats_csv "$VMD_IN"; then
  echo "NOTE: using first file as VMD and second as stats CSV (arguments were swapped). Correct order is:  script.sh  <stats.csv>  <vmd.dat>  [out.pdf]" >&2
  _tmp=$CSV_IN
  CSV_IN=$VMD_IN
  VMD_IN=$_tmp
fi

# Strip header, drop CR, convert CSV to whitespace-separated numeric rows (3+ cols -> first 3)
tail -n +2 "$CSV_IN" | tr -d '\r' | awk -F',' 'NF>=3 {gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1, $2, $3}' > "$TMP_WS"

if [[ ! -s "$TMP_WS" ]]; then
  if looks_like_vmd "$CSV_IN" && ! looks_like_stats_csv "$CSV_IN"; then
    echo "ERROR: the first file looks like a VMD trajectory (two columns, no commas), not a stats CSV." >&2
    echo "  You passed: $CSV_IN" >&2
    echo "  Use:  $0  <path/to/stats_time_mean_std.csv>  <path/to/vmd.dat>  [output.pdf]" >&2
  else
    echo "ERROR: no data rows in stats file after the header (need comma-separated time, mean, std). File: $CSV_IN" >&2
  fi
  exit 1
fi

PLOT_DATA1="$TMP_WS"
PLOT_VMD="$VMD_IN"
if [[ -n "$TIME_IN" ]]; then
  if [[ ! -s "$TIME_IN" ]]; then
    echo "ERROR: time file not found or empty: $TIME_IN" >&2
    exit 1
  fi
  tr -d '\r' < "$TIME_IN" | awk 'NF { print $1 }' > "$TMP_TIME_NORM"
  if [[ ! -s "$TMP_TIME_NORM" ]]; then
    echo "ERROR: no time values in file: $TIME_IN" >&2
    exit 1
  fi
  nt=$(wc -l < "$TMP_TIME_NORM" | tr -d ' ')
  ns=$(wc -l < "$TMP_WS" | tr -d ' ')
  nv=$(wc -l < "$VMD_IN" | tr -d ' ')
  if [[ "$nt" -ne "$ns" || "$nt" -ne "$nv" ]]; then
    echo "ERROR: time file must have the same number of lines as the stats and VMD data (after the stats header)." >&2
    echo "  Counts: time=$nt  stats=$ns  vmd=$nv  (time: $TIME_IN)" >&2
    exit 1
  fi
  paste -d' ' "$TMP_TIME_NORM" <(awk '{ print $2, $3 }' "$TMP_WS") > "$TMP_RETIME_STATS"
  paste -d' ' "$TMP_TIME_NORM" <(awk '{ print $2 }' "$VMD_IN") > "$TMP_RETIME_VMD"
  PLOT_DATA1="$TMP_RETIME_STATS"
  PLOT_VMD="$TMP_RETIME_VMD"
  echo "NOTE: x-axis values taken from: $TIME_IN" >&2
fi

gnuplot -e "DATA1='${PLOT_DATA1}';
            DATA_VMD='${PLOT_VMD}';
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
