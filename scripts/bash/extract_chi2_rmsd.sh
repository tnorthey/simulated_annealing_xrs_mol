#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  extract_chi2_rmsd.sh [-o OUT.dat] <directory>

Extract chi^2 (field 1) and RMSD (field 2) from line 2 of each *.xyz file in
<directory> and write a headerless space-separated .dat file.

Skips *_mean.xyz and *_target.xyz (same as aggregate_phi_sweep_stats.py).

Options:
  -o OUT.dat   Output path (default: <directory>/chi2_rmsd.dat)
  -h, --help   Show this help

Example:
  scripts/bash/extract_chi2_rmsd.sh results_phi_sweep/phi_0.5
EOF
}

out=""
dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      out=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -n "$dir" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage
        exit 1
      fi
      dir=$1
      shift
      ;;
  esac
done

if [[ -z "$dir" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$dir" ]]; then
  echo "Not a directory: $dir" >&2
  exit 1
fi

if [[ -z "$out" ]]; then
  out="${dir%/}/chi2_rmsd.dat"
fi

if [[ -x /usr/bin/find ]]; then
  FIND=/usr/bin/find
else
  FIND=find
fi

if [[ -x /usr/bin/sort ]]; then
  SORT=/usr/bin/sort
else
  SORT=sort
fi

if [[ -x /usr/bin/awk ]]; then
  AWK=/usr/bin/awk
else
  AWK=awk
fi

tmp=$(mktemp "${out}.XXXXXX")
trap 'rm -f "$tmp"' EXIT

n=0
while IFS= read -r -d '' f; do
  "$AWK" 'NR==2 {
    if (NF < 2) {
      print FILENAME ": line 2 needs >= 2 fields" > "/dev/stderr"
      exit 1
    }
    print $1, $2
    exit
  }' "$f" >>"$tmp" || exit 1
  n=$((n + 1))
done < <(
  "$FIND" -P "$dir" -maxdepth 1 -type f -name '*.xyz' -print0 \
  | LC_ALL=C "$SORT" -z \
  | while IFS= read -r -d '' f; do
      base=${f##*/}
      case "$base" in
        *_mean.xyz|*_target.xyz) ;;
        *) printf '%s\0' "$f" ;;
      esac
    done
)

if (( n == 0 )); then
  echo "No matching .xyz files in $dir" >&2
  exit 1
fi

mv -f "$tmp" "$out"
trap - EXIT

echo "Wrote $n rows -> $out" >&2
