#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <directory>" >&2
  echo "  Mean chi^2 from filenames matching ??_???.????????.xyz (digits only)." >&2
}

if [[ $# -ne 1 ]] || [[ ! -d "$1" ]]; then
  usage
  exit 1
fi

dir=$(cd "$1" && pwd)
# ??_???.????????.xyz — chi^2 is the part after the first '_', before '.xyz'
glob='[0-9][0-9]_[0-9][0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].xyz'

shopt -s nullglob
files=("$dir"/$glob)
shopt -u nullglob

if ((${#files[@]} == 0)); then
  echo "No files matching DD_DDD.DDDDDDDD.xyz in $dir" >&2
  exit 1
fi

for f in "${files[@]}"; do
  base=$(basename "$f" .xyz)
  echo "${base#*_}"
done | awk '
  { sum += $1; n++ }
  END {
    if (n == 0) exit 1
    printf "%.8f\n", sum / n
  }
'
