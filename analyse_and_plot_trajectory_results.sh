#!/usr/bin/env bash
# Reduce a CHD trajectory results directory to C1-C6 and two dihedral time
# series, then write the three-panel figure against the VMD target.
#
# Usage:
#   ./analyse_and_plot_trajectory_results.sh <results_dir>
#   ./analyse_and_plot_trajectory_results.sh <results_dir>.npz
#   ./analyse_and_plot_trajectory_results.sh --help
#
# NPZ mode: pass the archive, or a results directory whose sibling .npz exists
# and which has no per-chain xyz files. Geometry is read from the archive.
#
# Writes into <results_dir>/analysis/:
#   topM_geometry_bond-0-5_topM-<N>.csv
#   topM_geometry_dihedral-0-1-4-5_topM-<N>.csv
#   topM_geometry_dihedral-1-2-3-4_topM-<N>.csv
#   matching matplotlib PNGs
#   figure.pdf
#   figure_trajectory.tex, .aux, .log
#
# Environment (defaults shown):
#   TOPM=200
#   TIME_FILE=chd_results/time.dat
#   TRAJ                  parsed from results_chd_traj_<digits> in the directory name
#
# Target curves (existing filenames, including the dihdral spelling):
#   chd_results/vmd_c1c6_traj${TRAJ}.csv
#   chd_results/vmd_dihdral0145_traj${TRAJ}.csv
#   chd_results/vmd_dihdral1234_traj${TRAJ}.csv
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
    sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

if [[ $# -ne 1 ]]; then
    echo "ERROR: results_dir is required" >&2
    usage >&2
    exit 1
fi

PYTHON="${PYTHON:-python3}"
ARCHIVE=""
if [[ -f "$1" && "$1" == *.npz ]]; then
    ARCHIVE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    stem="$(basename "$ARCHIVE" .npz)"
    parent="$(cd "$(dirname "$ARCHIVE")" && pwd)"
    RESULTS_DIR="${parent}/${stem}"
    mkdir -p "$RESULTS_DIR"
elif [[ -d "$1" ]]; then
    RESULTS_DIR="$(cd "$1" && pwd)"
    if [[ -f "${RESULTS_DIR}.npz" ]]; then
        chain_xyz="$("$PYTHON" "$REPO_ROOT/modules/trajectory_archive.py" count-chains --results-dir "$RESULTS_DIR")"
        if [[ "$chain_xyz" == "0" ]]; then
            ARCHIVE="${RESULTS_DIR}.npz"
        fi
    fi
else
    echo "ERROR: results directory or .npz archive not found: $1" >&2
    exit 1
fi

TOPM="${TOPM:-200}"
TIME_FILE="${TIME_FILE:-chd_results/time.dat}"

if [[ ! "$TOPM" =~ ^[0-9]+$ || "$TOPM" -lt 1 ]]; then
    echo "ERROR: TOPM must be an integer >= 1 (got $TOPM)" >&2
    exit 1
fi

if [[ "$TIME_FILE" != /* ]]; then
    TIME_FILE="${REPO_ROOT}/${TIME_FILE}"
fi

base="$(basename "$RESULTS_DIR")"
if [[ -z "${TRAJ:-}" ]]; then
    if [[ "$base" =~ results_chd_traj_([0-9]+) ]]; then
        TRAJ="${BASH_REMATCH[1]}"
    else
        echo "ERROR: cannot parse trajectory id from '${base}'." >&2
        echo "       Name the directory results_chd_traj_<digits>... or set TRAJ." >&2
        exit 1
    fi
fi

VMD_BOND="${REPO_ROOT}/chd_results/vmd_c1c6_traj${TRAJ}.csv"
VMD_DIH1="${REPO_ROOT}/chd_results/vmd_dihdral0145_traj${TRAJ}.csv"
VMD_DIH2="${REPO_ROOT}/chd_results/vmd_dihdral1234_traj${TRAJ}.csv"

missing=0
for f in "$TIME_FILE" "$VMD_BOND" "$VMD_DIH1" "$VMD_DIH2"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file: $f" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    exit 1
fi

ANALYSIS="${RESULTS_DIR}/analysis"
mkdir -p "$ANALYSIS"

BOND_CSV="${ANALYSIS}/topM_geometry_bond-0-5_topM-${TOPM}.csv"
DIH1_CSV="${ANALYSIS}/topM_geometry_dihedral-0-1-4-5_topM-${TOPM}.csv"
DIH2_CSV="${ANALYSIS}/topM_geometry_dihedral-1-2-3-4_topM-${TOPM}.csv"
PDF_OUT="${ANALYSIS}/figure.pdf"
OUTBASE="figure_trajectory"

echo "=== Trajectory analysis ==="
echo "  results:  $RESULTS_DIR"
echo "  analysis: $ANALYSIS"
echo "  traj:     $TRAJ"
echo "  topM:     $TOPM"
echo "  time:     $TIME_FILE"
if [[ -n "$ARCHIVE" ]]; then
    echo "  source:   $ARCHIVE"
else
    echo "  source:   xyz files in $RESULTS_DIR"
fi

run_geometry() {
    local archive_args=()
    if [[ -n "$ARCHIVE" ]]; then
        archive_args+=(--archive "$ARCHIVE")
    fi
    "$PYTHON" "$REPO_ROOT/scripts/python/topM_geometry_statistics.py" "$RESULTS_DIR" \
        --skip-closest --recompute --topM "$TOPM" \
        --time-units fs --time-origin 0 --time-file "$TIME_FILE" \
        --output-dir "$ANALYSIS" \
        "${archive_args[@]}" \
        "$@"
}

run_geometry --bond 0 5
run_geometry --dihedral 0 1 4 5
run_geometry --dihedral 1 2 3 4

W=4 H=4 RELH1=0.33 RELH2=0.33 RELH3=0.34 \
MLEFT=0.12 \
YLABEL_OFFSETX=1.2 Y1LABEL_OFFSET=0.4 Y2LABEL_OFFSET=-0.7 Y3LABEL_OFFSET=-0.7 \
XMIN=-1 XMAX=201 Y1MIN=1.25 Y1MAX=2.25 Y2MIN=-15 Y2MAX=55 Y3MIN=-40 Y3MAX=70 \
DIHEDRAL_TO3602=0 DIHEDRAL_TO3602B=0 DIHEDRAL_TO3603=0 DIHEDRAL_TO3603B=0 \
XTIC_STEP=50 YTIC_STEP1=0.5 YTIC_STEP2=50 YTIC_STEP3=50 \
SHADEDERRORS1=0 \
PLOTMODE='POINTS' PLOTMODE2='LINES' \
PLOTMODE_DIH='POINTS' PLOTMODEB_DIH='LINES' \
LW1=3 LW2=3 LWD2=3 LWD2B=3 LWD3=3 LWD3B=3 \
EBLW1=3 EBLW2=3 EBLW3=3 \
COL1B='#000000' COL2B='#1a4480' COL3B='#666666' \
SHOW_KEY=0 \
OUTBASE="$OUTBASE" \
XLABEL='$t$ (fs)' \
Y1LABEL='\ce{C1-C6} (\AA)' \
Y2LABEL='$\phi_{\ce{C1C2C5C6}}\\ (^\circ)$' \
Y3LABEL='$\phi_{\ce{C2C3C4C5}}\\ (^\circ)$' \
"$REPO_ROOT/scripts/bash/plot_bond_dihedral_3stack.sh" \
    "$BOND_CSV" "$VMD_BOND" \
    "$DIH1_CSV" "$DIH2_CSV" \
    "$VMD_DIH1" "$VMD_DIH2" \
    "$PDF_OUT" "$TIME_FILE"

for ext in tex aux log; do
    if [[ -f "${REPO_ROOT}/${OUTBASE}.${ext}" ]]; then
        mv -f "${REPO_ROOT}/${OUTBASE}.${ext}" "$ANALYSIS/"
    fi
done
rm -f "${REPO_ROOT}/${OUTBASE}.pdf"

echo "Wrote: $PDF_OUT"
