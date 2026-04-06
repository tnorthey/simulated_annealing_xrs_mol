#!/usr/bin/env bash
# Create one .tar.gz per directory whose name starts with "results" or "plots".
# Archives are named <dirname>.tar.gz next to each directory (same parent dir).
set -euo pipefail

root="${1:-.}"
cd -- "$root"

shopt -s nullglob

for name in results* plots*; do
  if [[ -d "$name" ]]; then
    printf 'Archiving %s -> %s.tar.gz\n' "$name" "$name"
    tar -czf "${name}.tar.gz" -- "$name"
  fi
done

shopt -u nullglob
