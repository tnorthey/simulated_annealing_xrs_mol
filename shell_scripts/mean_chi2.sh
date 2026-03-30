#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [options] <directory>" >&2
  echo "  Mean chi^2 from filenames DD_DDD.DDDDDDDD.xyz (2-digit timestep, chi^2 after _)." >&2
  echo "Options:" >&2
  echo "  --top M          Mean of the lowest M chi^2 values (after other filters)." >&2
  echo "  --range LO,HI    Only timesteps with numeric prefix in [LO,HI] (e.g. 01,20)." >&2
}

top=""
use_range=0
rlo=0
rhi=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --top)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      top=$2
      shift 2
      ;;
    --range)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      IFS=',' read -r _rlo _rhi <<< "${2//[[:space:]]/}"
      [[ -n "${_rlo:-}" && -n "${_rhi:-}" ]] || { echo "Invalid --range (use LO,HI)" >&2; exit 1; }
      rlo=$((10#${_rlo}))
      rhi=$((10#${_rhi}))
      if (( rlo > rhi )); then
        echo "--range: LO must be <= HI" >&2
        exit 1
      fi
      use_range=1
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
      break
      ;;
  esac
done

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

if [[ ! -d "$1" ]]; then
  usage
  exit 1
fi

if [[ -n "$top" ]]; then
  if ! [[ "$top" =~ ^[1-9][0-9]*$ ]]; then
    echo "--top must be a positive integer" >&2
    exit 1
  fi
fi

dir=$1
name_pat='[0-9][0-9]_[0-9][0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].xyz'

# Stream basenames; filter by timestep; print chi^2 one per line. GNU find (-printf).
find -P "$dir" -maxdepth 1 -type f -name "$name_pat" -printf '%f\n' \
| LC_ALL=C awk -v use_range="$use_range" -v rlo="$rlo" -v rhi="$rhi" '
  {
    if (use_range) {
      ts = substr($0, 1, 2) + 0
      if (ts < rlo || ts > rhi)
        next
    }
    line = $0
    sub(/\.xyz$/, "", line)
    sub(/^[0-9][0-9]_/, "", line)
    print line + 0
  }
' \
| LC_ALL=C sort -n \
| {
    if [[ -n "$top" ]]; then
      head -n "$top"
    else
      cat
    fi
  } \
| LC_ALL=C awk -v dir="$dir" '
  { sum += $1; n++ }
  END {
    if (n == 0) {
      printf "No matching .xyz files for the given filters in %s\n", dir > "/dev/stderr"
      exit 1
    }
    printf "%.8f\n", sum / n
  }
'
