#!/usr/bin/env bash
# Three-stack figure: bond (mean±SD vs optional VMD) + two dihedral CSV panels.
# Run from repo root, or use absolute paths.
#
# Usage:
#   ./scripts/bash/plot_bond_dihedral_3stack.sh <bond_stats.csv> <vmd_bond.dat|-> <dihedral1.csv> <dihedral2.csv> \
#       [dihedral1_expected.csv] [dihedral2_expected.csv] [output.pdf] [time.dat]
#
# bond_stats.csv: comma-separated header, then time_fs, mean, std (first three columns used).
# vmd_bond.dat:   two whitespace columns (time, bond). Use "-" or "none" to omit VMD on panel 1.
# dihedral*.csv:  comma CSV with header; same transforms as plot_csv_stddev_2stack_tex.gp (see that script).
#
# Optional second curves on dihedral panels (expected/reference):
# - positional: dihedral1_expected.csv dihedral2_expected.csv
# - or env (absolute or repo-relative): BOND_DIH_DATA2B=/path/to/ref2.csv  BOND_DIH_DATA3B=/path/to/ref3.csv
#
# Optional external time column for ALL panels:
#   6th argument, or TIME_PLOT_FILE=/path/to/time.dat
# The file must have exactly 1 column of times (first field per line is used) and
# must match the number of data rows in:
# - bond stats CSV (excluding its header)
# - VMD bond .dat (if present)
# - dihedral1.csv (excluding header) and dihedral2.csv (excluding header)
# - dihedral ref files (DATA2B/DATA3B) if provided
#
# Requires: gnuplot 5.4+, pdflatex

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

GP="${ROOT}/scripts/gnuplot/plot_bond_vmd_dihedral_dihedral_3stack_tex.gp"
OUTBASE="figure_bond_dihedral_3stack"

CSV_IN="${1:?bond stats CSV required}"
VMD_IN="${2:?VMD bond .dat or - required}"
D2_IN="${3:?dihedral CSV 1 required}"
D3_IN="${4:?dihedral CSV 2 required}"
shift 4

# Remaining args are optional and order-dependent:
#   [D2B] [D3B] [OUT_PDF] [TIME_IN]
OUT_PDF="bond_dihedral_3stack.pdf"
TIME_IN="${TIME_PLOT_FILE:-}"
D2B_POS=""
D3B_POS=""

is_pdf() { [[ "${1:-}" == *.pdf ]]; }

if [[ -n "${1:-}" && -f "$1" ]] && ! is_pdf "$1"; then
  D2B_POS="$1"
  shift
fi
if [[ -n "${1:-}" && -f "$1" ]] && ! is_pdf "$1"; then
  D3B_POS="$1"
  shift
fi
if [[ -n "${1:-}" ]]; then
  OUT_PDF="$1"
  shift
fi
if [[ -n "${1:-}" ]]; then
  TIME_IN="$1"
  shift
fi
if [[ -n "${1:-}" ]]; then
  echo "ERROR: too many arguments. See header usage." >&2
  exit 1
fi

TMP_WS="${ROOT}/scripts/gnuplot/_tmp_bond3stack_mean_std_ws.dat"
TMP_RETIME_STATS="${ROOT}/scripts/gnuplot/_tmp_bond3stack_retime_stats.dat"
TMP_RETIME_VMD="${ROOT}/scripts/gnuplot/_tmp_bond3stack_retime_vmd.dat"
TMP_TIME_NORM="${ROOT}/scripts/gnuplot/_tmp_bond3stack_time_col.dat"
TMP_RETIME_D2="${ROOT}/scripts/gnuplot/_tmp_bond3stack_retime_d2.csv"
TMP_RETIME_D3="${ROOT}/scripts/gnuplot/_tmp_bond3stack_retime_d3.csv"
TMP_RETIME_D2B="${ROOT}/scripts/gnuplot/_tmp_bond3stack_retime_d2b.csv"
TMP_RETIME_D3B="${ROOT}/scripts/gnuplot/_tmp_bond3stack_retime_d3b.csv"

if [[ "$CSV_IN" != /* ]]; then CSV_IN="${ROOT}/${CSV_IN}"; fi
if [[ "$VMD_IN" != /* && "$VMD_IN" != "-" && "$VMD_IN" != "none" ]]; then VMD_IN="${ROOT}/${VMD_IN}"; fi
if [[ "$D2_IN" != /* ]]; then D2_IN="${ROOT}/${D2_IN}"; fi
if [[ "$D3_IN" != /* ]]; then D3_IN="${ROOT}/${D3_IN}"; fi
if [[ "$OUT_PDF" != /* ]]; then OUT_PDF="${ROOT}/${OUT_PDF}"; fi
if [[ -n "$TIME_IN" && "$TIME_IN" != /* ]]; then TIME_IN="${ROOT}/${TIME_IN}"; fi

DATA2B_ARG=""
DATA3B_ARG=""
DATA2B_FILE=""
DATA3B_FILE=""
if [[ -n "${BOND_DIH_DATA2B:-}" ]]; then
  _d2b="$BOND_DIH_DATA2B"
  [[ "$_d2b" == /* ]] || _d2b="${ROOT}/${_d2b}"
  DATA2B_FILE="$_d2b"
  DATA2B_ARG="DATA2B='${_d2b}';"
fi
if [[ -n "${BOND_DIH_DATA3B:-}" ]]; then
  _d3b="$BOND_DIH_DATA3B"
  [[ "$_d3b" == /* ]] || _d3b="${ROOT}/${_d3b}"
  DATA3B_FILE="$_d3b"
  DATA3B_ARG="DATA3B='${_d3b}';"
fi

# Positional dihedral expected files override env if provided.
if [[ -n "$D2B_POS" ]]; then
  [[ "$D2B_POS" == /* ]] || D2B_POS="${ROOT}/${D2B_POS}"
  DATA2B_FILE="$D2B_POS"
  DATA2B_ARG="DATA2B='${DATA2B_FILE}';"
fi
if [[ -n "$D3B_POS" ]]; then
  [[ "$D3B_POS" == /* ]] || D3B_POS="${ROOT}/${D3B_POS}"
  DATA3B_FILE="$D3B_POS"
  DATA3B_ARG="DATA3B='${DATA3B_FILE}';"
fi

if [[ ! -s "$CSV_IN" ]]; then echo "ERROR: bond CSV not found or empty: $CSV_IN" >&2; exit 1; fi
if [[ ! -s "$D2_IN" ]]; then echo "ERROR: dihedral CSV 1 not found or empty: $D2_IN" >&2; exit 1; fi
if [[ ! -s "$D3_IN" ]]; then echo "ERROR: dihedral CSV 2 not found or empty: $D3_IN" >&2; exit 1; fi

looks_like_vmd() {
  local f=$1
  local line nf
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

csv_first_field_is_number() {
  # Returns 0 if first CSV field on first non-empty line is numeric.
  # Accepts integers / decimals / scientific notation.
  local f=$1
  tr -d "\r" < "$f" | awk -F',' '
    /^[[:space:]]*$/ { next }
    {
      x=$1
      gsub(/^[[:space:]]+|[[:space:]]+$/,"",x)
      if (x ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) found=1
      exit
    }
    END { exit (found ? 0 : 1) }
  '
}

count_csv_rows_maybe_header() {
  # Count data rows in a CSV; skip header iff first field is non-numeric.
  local f=$1
  if csv_first_field_is_number "$f"; then
    tr -d '\r' < "$f" | awk 'NF{n++} END{print n+0}'
  else
    tail -n +2 "$f" | tr -d '\r' | awk 'NF{n++} END{print n+0}'
  fi
}

retime_csv_with_time() {
  # Rewrite CSV with time column replaced by TMP_TIME_NORM (1 col).
  # Preserves header if present; otherwise writes no header.
  local in_csv=$1
  local out_csv=$2
  local header body
  if csv_first_field_is_number "$in_csv"; then
    paste -d',' "$TMP_TIME_NORM" <(tr -d '\r' < "$in_csv" | cut -d',' -f2-) > "$out_csv"
  else
    header=$(head -n 1 "$in_csv" | tr -d '\r')
    {
      echo "$header" | awk -F, 'BEGIN{OFS=","} { $1="time"; print }'
      paste -d',' "$TMP_TIME_NORM" <(tail -n +2 "$in_csv" | tr -d '\r' | cut -d',' -f2-)
    } > "$out_csv"
  fi
}

vmd_first_row_is_numeric_2col() {
  # Returns 0 if first non-empty line looks like: <num> <num>
  local f=$1
  tr -d "\r" < "$f" | awk '
    /^[[:space:]]*$/ { next }
    {
      x=$1; y=$2
      if (x ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ && y ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) ok=1
      exit
    }
    END { exit (ok ? 0 : 1) }
  '
}

count_vmd_rows_maybe_header() {
  # Count data rows in whitespace 2-col file; skip header if first row non-numeric.
  local f=$1
  if vmd_first_row_is_numeric_2col "$f"; then
    tr -d "\r" < "$f" | awk 'NF{n++} END{print n+0}'
  else
    tail -n +2 "$f" | tr -d "\r" | awk 'NF{n++} END{print n+0}'
  fi
}

retime_vmd_with_time() {
  # Replace column 1 in whitespace 2-col file by TMP_TIME_NORM, preserving column 2.
  # If file has a non-numeric first row, skip it (header).
  local in_dat=$1
  local out_dat=$2
  if vmd_first_row_is_numeric_2col "$in_dat"; then
    paste -d' ' "$TMP_TIME_NORM" <(tr -d "\r" < "$in_dat" | awk 'NF { print $2 }') > "$out_dat"
  else
    paste -d' ' "$TMP_TIME_NORM" <(tail -n +2 "$in_dat" | tr -d "\r" | awk 'NF { print $2 }') > "$out_dat"
  fi
}

if [[ "$VMD_IN" != "-" && "$VMD_IN" != "none" ]]; then
  if looks_like_vmd "$CSV_IN" && looks_like_stats_csv "$VMD_IN"; then
    echo "NOTE: treating first file as VMD and second as bond stats CSV (swapped). Correct order: bond_stats.csv vmd.dat ..." >&2
    _tmp=$CSV_IN
    CSV_IN=$VMD_IN
    VMD_IN=$_tmp
  fi
fi

tail -n +2 "$CSV_IN" | tr -d '\r' | awk -F',' 'NF>=3 {gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1, $2, $3}' > "$TMP_WS"
if [[ ! -s "$TMP_WS" ]]; then
  echo "ERROR: no bond data rows after header in: $CSV_IN" >&2
  exit 1
fi

PLOT_DATA1="$TMP_WS"
PLOT_VMD=""
PLOT_D2="$D2_IN"
PLOT_D3="$D3_IN"
PLOT_D2B=""
PLOT_D3B=""
if [[ "$VMD_IN" != "-" && "$VMD_IN" != "none" ]]; then
  if [[ ! -s "$VMD_IN" ]]; then echo "ERROR: VMD file not found or empty: $VMD_IN" >&2; exit 1; fi
  PLOT_VMD="$VMD_IN"
fi
if [[ -n "$DATA2B_FILE" ]]; then
  if [[ ! -s "$DATA2B_FILE" ]]; then echo "ERROR: dihedral DATA2B not found or empty: $DATA2B_FILE" >&2; exit 1; fi
  PLOT_D2B="$DATA2B_FILE"
fi
if [[ -n "$DATA3B_FILE" ]]; then
  if [[ ! -s "$DATA3B_FILE" ]]; then echo "ERROR: dihedral DATA3B not found or empty: $DATA3B_FILE" >&2; exit 1; fi
  PLOT_D3B="$DATA3B_FILE"
fi

if [[ -n "$TIME_IN" ]]; then
  if [[ ! -s "$TIME_IN" ]]; then echo "ERROR: time file not found or empty: $TIME_IN" >&2; exit 1; fi
  # Normalize time column:
  # - drop CRLF (\r)
  # - drop blank / whitespace-only lines
  # - take first field per line
  tr -d '\r' < "$TIME_IN" | awk '!/^[[:space:]]*$/ { print $1 }' > "$TMP_TIME_NORM"
  if [[ ! -s "$TMP_TIME_NORM" ]]; then echo "ERROR: no time values in: $TIME_IN" >&2; exit 1; fi
  nt=$(awk 'NF{n++} END{print n+0}' "$TMP_TIME_NORM")
  ns=$(awk 'NF{n++} END{print n+0}' "$TMP_WS")
  nd2=$(count_csv_rows_maybe_header "$D2_IN")
  nd3=$(count_csv_rows_maybe_header "$D3_IN")
  if [[ "$nt" -ne "$ns" || "$nt" -ne "$nd2" || "$nt" -ne "$nd3" ]]; then
    echo "ERROR: time file line count ($nt) must match bond stats ($ns) and dihedral CSV rows (d2=$nd2 d3=$nd3)." >&2
    exit 1
  fi
  if [[ -n "$PLOT_D2B" ]]; then
    nd2b=$(count_csv_rows_maybe_header "$PLOT_D2B")
    if [[ "$nt" -ne "$nd2b" ]]; then
      echo "ERROR: time file line count ($nt) must match DATA2B rows ($nd2b)." >&2
      exit 1
    fi
  fi
  if [[ -n "$PLOT_D3B" ]]; then
    nd3b=$(count_csv_rows_maybe_header "$PLOT_D3B")
    if [[ "$nt" -ne "$nd3b" ]]; then
      echo "ERROR: time file line count ($nt) must match DATA3B rows ($nd3b)." >&2
      exit 1
    fi
  fi
  if [[ -n "$PLOT_VMD" ]]; then
    nv=$(count_vmd_rows_maybe_header "$PLOT_VMD")
    if [[ "$nt" -ne "$ns" || "$nt" -ne "$nv" ]]; then
      echo "ERROR: time file line count ($nt) must match bond stats ($ns) and VMD ($nv)." >&2
      exit 1
    fi
    paste -d' ' "$TMP_TIME_NORM" <(awk '{ print $2, $3 }' "$TMP_WS") > "$TMP_RETIME_STATS"
    retime_vmd_with_time "$PLOT_VMD" "$TMP_RETIME_VMD"
    PLOT_DATA1="$TMP_RETIME_STATS"
    PLOT_VMD="$TMP_RETIME_VMD"
  else
    if [[ "$nt" -ne "$ns" ]]; then
      echo "ERROR: time file line count ($nt) must match bond stats ($ns) when VMD is omitted." >&2
      exit 1
    fi
    paste -d' ' "$TMP_TIME_NORM" <(awk '{ print $2, $3 }' "$TMP_WS") > "$TMP_RETIME_STATS"
    PLOT_DATA1="$TMP_RETIME_STATS"
  fi

  # Retimed dihedral CSVs: replace column 1 by TIME_IN; preserve header if present.
  retime_csv_with_time "$D2_IN" "$TMP_RETIME_D2"
  retime_csv_with_time "$D3_IN" "$TMP_RETIME_D3"
  PLOT_D2="$TMP_RETIME_D2"
  PLOT_D3="$TMP_RETIME_D3"
  if [[ -n "$PLOT_D2B" ]]; then
    retime_csv_with_time "$PLOT_D2B" "$TMP_RETIME_D2B"
    PLOT_D2B="$TMP_RETIME_D2B"
  fi
  if [[ -n "$PLOT_D3B" ]]; then
    retime_csv_with_time "$PLOT_D3B" "$TMP_RETIME_D3B"
    PLOT_D3B="$TMP_RETIME_D3B"
  fi

  if [[ -n "$PLOT_D2B" ]]; then DATA2B_ARG="DATA2B='${PLOT_D2B}';"; fi
  if [[ -n "$PLOT_D3B" ]]; then DATA3B_ARG="DATA3B='${PLOT_D3B}';"; fi

  echo "NOTE: x-axis values taken from: $TIME_IN" >&2
fi

VMD_GP="''"
if [[ -n "$PLOT_VMD" ]]; then VMD_GP="'${PLOT_VMD}'"; fi

gnuplot -e "DATA1='${PLOT_DATA1}';
            DATA_VMD=${VMD_GP};
            DATA2='${PLOT_D2}';
            DATA3='${PLOT_D3}';
            ${DATA2B_ARG}
            ${DATA3B_ARG}
            OUTBASE='${OUTBASE}';
            XLABEL='\$t\$ (fs)';
            Y1LABEL='Bond (\\AA)';
            PLOTMODE='LINES';
            PLOTMODE2='LINES';
            SHADEDERRORS1=1;
            SHADEDERRORS2=1;
            SHADEDERRORS3=1;
            SHADE_ALPHA=0.25;
            SHOW_KEY=1;
            NAME2='VMD';
            RELH1=0.28;
            RELH2=0.36;
            RELH3=0.36;
            NAME2A='Panel 2';
            NAME3A='Panel 3';
            " "$GP"

if ! pdflatex -interaction=nonstopmode -halt-on-error "${OUTBASE}.tex" >"pdflatex_${OUTBASE}.log" 2>&1; then
  echo "ERROR: pdflatex failed. See ${ROOT}/pdflatex_${OUTBASE}.log" >&2
  exit 1
fi
rm -f "pdflatex_${OUTBASE}.log"
cp -f "${OUTBASE}.pdf" "$OUT_PDF"
echo "Wrote: $OUT_PDF"
