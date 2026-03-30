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

dir=$1
# ??_???.????????.xyz — chi^2 is the part after the first '_', before '.xyz'
name_pat='[0-9][0-9]_[0-9][0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].xyz'

# Stream basenames only: avoids storing ~50k paths in a bash array and ~50k basename(1) calls.
# Requires GNU find (-printf).
find -P "$dir" -maxdepth 1 -type f -name "$name_pat" -printf '%f\n' \
| LC_ALL=C awk -v dir="$dir" '
  {
    line = $0
    sub(/\.xyz$/, "", line)
    sub(/^[0-9][0-9]_/, "", line)
    sum += line + 0
    n++
  }
  END {
    if (n == 0) {
      printf "No files matching DD_DDD.DDDDDDDD.xyz in %s\n", dir > "/dev/stderr"
      exit 1
    }
    printf "%.8f\n", sum / n
  }
'
