#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  find_outliers.sh [options] <xyz1> [<xyz2> ...]

Detect XYZ outliers where ANY carbon atom is translated too far relative to a reference,
using centroid-subtracted coordinates (no Kabsch rotation).

Options:
  --cutoff A               Outlier cutoff in Angstrom (default: 2.0)
  --ref FILE               Reference XYZ (default: first input after sorting)
  --centroid-indices EXPR  Indices for centroid subtraction (default: all)
  --one-based              Interpret --centroid-indices as 1-based

  --action list|move|delete  What to do with outliers (default: list)
  --outdir DIR               For --action move (default: outliers/)
  --print-scores              Print per-file max_C_disp scores to stdout

  --dry-run                 For move/delete: show what would happen
  --yes                     Required for --action delete (safety)
  -h, --help                Show help

Examples:
  scripts/bash/find_outliers.sh 01_*.xyz
  scripts/bash/find_outliers.sh --print-scores --cutoff 1.5 01_*.xyz
  scripts/bash/find_outliers.sh --action move --outdir bad_xyz 01_*.xyz
EOF
}

cutoff="2.0"
ref=""
centroid_indices=""
one_based=0
action="list"
outdir="outliers"
print_scores=0
dry_run=0
yes_delete=0

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cutoff)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      cutoff=$2
      shift 2
      ;;
    --ref)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      ref=$2
      shift 2
      ;;
    --centroid-indices)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      centroid_indices=$2
      shift 2
      ;;
    --one-based)
      one_based=1
      shift 1
      ;;
    --action)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      action=$2
      shift 2
      ;;
    --outdir)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      outdir=$2
      shift 2
      ;;
    --print-scores)
      print_scores=1
      shift 1
      ;;
    --dry-run)
      dry_run=1
      shift 1
      ;;
    --yes)
      yes_delete=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do args+=("$1"); shift; done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      args+=("$1")
      shift 1
      ;;
  esac
done

if [[ ${#args[@]} -lt 1 ]]; then
  usage
  exit 1
fi

case "$action" in
  list|move|delete) ;;
  *)
    echo "Invalid --action: $action (expected list|move|delete)" >&2
    exit 1
    ;;
esac

if [[ "$action" == "delete" && "$yes_delete" -ne 1 && "$dry_run" -ne 1 ]]; then
  echo "Refusing to delete without --yes" >&2
  echo "Tip: run with --dry-run first, then re-run with --yes" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py="$repo_root/scripts/python/find_xyz_outliers.py"
[[ -f "$py" ]] || { echo "Missing detector script: $py" >&2; exit 1; }

py_args=( "$py" "--cutoff" "$cutoff" "--no-header" )
if [[ -n "$ref" ]]; then
  py_args+=( "--ref" "$ref" )
fi
if [[ -n "$centroid_indices" ]]; then
  py_args+=( "--centroid-indices" "$centroid_indices" )
fi
if [[ "$one_based" -eq 1 ]]; then
  py_args+=( "--one-based" )
fi
py_args+=( "${args[@]}" )

tmp_scores="$(mktemp)"
cleanup() { rm -f "$tmp_scores"; }
trap cleanup EXIT

python3 "${py_args[@]}" > "$tmp_scores"

outliers=()
while IFS=$'\t' read -r file maxdisp is_outlier atom_idx; do
  [[ -n "${file:-}" ]] || continue
  if [[ "$print_scores" -eq 1 ]]; then
    printf "%s\t%s\t%s\n" "$file" "$maxdisp" "${atom_idx:-}"
  fi
  if [[ "${is_outlier:-0}" -eq 1 ]]; then
    outliers+=( "$file" )
  fi
done < "$tmp_scores"

if [[ "$print_scores" -eq 1 ]]; then
  :
fi

echo "cutoff_A=$cutoff ref=${ref:-<auto>} n_files=${#args[@]} n_outliers=${#outliers[@]}" >&2

if [[ ${#outliers[@]} -eq 0 ]]; then
  exit 0
fi

case "$action" in
  list)
    printf "%s\n" "${outliers[@]}"
    ;;
  move)
    if [[ "$dry_run" -eq 1 ]]; then
      for f in "${outliers[@]}"; do
        echo "would_move	$f	->	$outdir/"
      done
      printf "%s\n" "${outliers[@]}"
      exit 0
    fi
    mkdir -p "$outdir"
    for f in "${outliers[@]}"; do
      mv -n -- "$f" "$outdir/"
    done
    printf "%s\n" "${outliers[@]}"
    ;;
  delete)
    if [[ "$dry_run" -eq 1 ]]; then
      for f in "${outliers[@]}"; do
        echo "would_delete	$f"
      done
      printf "%s\n" "${outliers[@]}"
      exit 0
    fi
    for f in "${outliers[@]}"; do
      rm -f -- "$f"
    done
    printf "%s\n" "${outliers[@]}"
    ;;
esac

