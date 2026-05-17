#!/usr/bin/env bash
# Split a multi-frame extended XYZ into 00_target.xyz, 01_target.xyz, ...
#
# Usage:  ./split_xyz_to_targets.sh <trajectory.xyz> <natoms> [pad_width]
# Example: ./split_xyz_to_targets.sh path/traj.xyz 14 2
#
# Each frame = natoms + 2 lines (count, comment, natoms coordinate lines).

set -euo pipefail

usage() {
  echo "Usage: $0 <input_multi_frame.xyz> <natoms> [pad_width]" >&2
  echo "  pad_width: zero-pad width for filenames (default: 2)" >&2
  exit 1
}

[[ $# -ge 2 ]] || usage

xyz=$1
natoms=$2
pad=${3:-2}

[[ -f "$xyz" ]] || { echo "Not a file: $xyz" >&2; exit 1; }
[[ "$natoms" =~ ^[0-9]+$ ]] && (( natoms >= 1 )) || {
  echo "natoms must be a positive integer" >&2
  exit 1
}

lines_per_frame=$((natoms + 2))
total=$(wc -l < "$xyz")
if (( total % lines_per_frame != 0 )); then
  echo "Error: $xyz has $total lines, not a multiple of $lines_per_frame (natoms+2)." >&2
  exit 1
fi

frame=0
while true; do
  IFS= read -r nline || break

  # First field only (handles leading tab/spaces on count line)
  nread=$(awk '{print $1+0; exit}' <<< "$nline")
  if (( nread != natoms )); then
    echo "Frame $frame: expected atom count $natoms, got '$nline' (parsed $nread)" >&2
    exit 1
  fi

  IFS= read -r cmt || { echo "Frame $frame: EOF after atom count" >&2; exit 1; }

  out=$(printf "target_%0${pad}d.xyz" "$frame")
  {
    printf '%s\n' "$nline"
    printf '%s\n' "$cmt"
    for ((i = 0; i < natoms; i++)); do
      IFS= read -r al || { echo "Frame $frame: EOF inside coordinates (line $i)" >&2; exit 1; }
      printf '%s\n' "$al"
    done
  } > "$out"

  ((++frame))
done < "$xyz"

echo "Wrote $frame frame(s) to %0${pad}d_target.xyz"
