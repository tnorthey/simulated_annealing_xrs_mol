#!/usr/bin/env python3
"""
prune_by_fit_percentile.py

Prune (move aside) candidate files based on the fit value encoded in filenames.

This script mirrors the filename parsing/statistics logic from plot_fit_histograms.py,
but instead of plotting, it moves candidates with "worse" fit than a chosen percentile
into a subdirectory called "pruned".

Filename convention (same as optimal_path.py / plot_fit_histograms.py):
  01_000.17533577.dat
  01_000.17533577.xyz

By default, pruning is done PER TIMESTEP:
  - Compute the per-timestep percentile cutoff (e.g. 25th percentile)
  - Keep candidates with fit <= cutoff
  - Move candidates with fit  > cutoff to <directory>/pruned/

Usage:
  python3 prune_by_fit_percentile.py results/ --percentile 25
  python3 prune_by_fit_percentile.py results/ --percentile 25 --dry-run
  python3 prune_by_fit_percentile.py results/ --percentile 50 --overwrite
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Keep in sync with plot_fit_histograms.py / optimal_path.py
NAME_RE = re.compile(r"^(?P<t>\d+?)_(?P<fit>\d+(?:\.\d+)?)(?P<rest>.*)$")


def parse_name(path: str) -> Optional[Tuple[int, float]]:
    """Parse timestep and fit value from filename."""
    base = os.path.basename(path)
    m = NAME_RE.match(base)
    if not m:
        return None
    t = int(m.group("t"))
    fit = float(m.group("fit"))
    return t, fit


@dataclass(frozen=True)
class Candidate:
    t: int
    fit: float
    dat_path: str
    xyz_path: str


def collect_candidates(directory: str, *, dat_ext: str, xyz_ext: str) -> List[Candidate]:
    """Collect matched (dat, xyz) pairs keyed by (timestep, fit)."""
    dats = glob.glob(os.path.join(directory, f"*{dat_ext}"))
    xyzs = glob.glob(os.path.join(directory, f"*{xyz_ext}"))

    dat_map: Dict[Tuple[int, float], str] = {}
    xyz_map: Dict[Tuple[int, float], str] = {}

    for p in dats:
        parsed = parse_name(p)
        if parsed:
            dat_map[parsed] = p
    for p in xyzs:
        parsed = parse_name(p)
        if parsed:
            xyz_map[parsed] = p

    keys = sorted(set(dat_map.keys()) & set(xyz_map.keys()), key=lambda x: (x[0], x[1]))
    if not keys:
        raise RuntimeError(
            "No matched (timestep,fit) pairs found with both DAT and XYZ files. "
            "Expected names like 01_0.12345678.dat and 01_0.12345678.xyz"
        )

    out: List[Candidate] = []
    for (t, fit) in keys:
        out.append(Candidate(t=t, fit=fit, dat_path=dat_map[(t, fit)], xyz_path=xyz_map[(t, fit)]))
    return out


def summarize_by_timestep(cands: Sequence[Candidate], *, percentile: float) -> Dict[int, dict]:
    by_t: Dict[int, List[float]] = defaultdict(list)
    for c in cands:
        by_t[int(c.t)].append(float(c.fit))

    summary: Dict[int, dict] = {}
    for t in sorted(by_t.keys()):
        fits = np.asarray(by_t[t], dtype=np.float64)
        summary[t] = {
            "count": int(fits.size),
            "min": float(np.min(fits)),
            "p25": float(np.percentile(fits, 25)),
            "median": float(np.median(fits)),
            "mean": float(np.mean(fits)),
            "p75": float(np.percentile(fits, 75)),
            "max": float(np.max(fits)),
            "cutoff": float(np.percentile(fits, percentile)),
        }
    return summary


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _move_file(src: str, dst_dir: str, *, overwrite: bool):
    _ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst):
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst} (use --overwrite to replace)")
        os.remove(dst)
    shutil.move(src, dst)


def main():
    p = argparse.ArgumentParser(description="Prune candidates by fit percentile into <directory>/pruned.")
    p.add_argument("directory", help="Directory containing candidate *.dat and *.xyz files.")
    p.add_argument(
        "--percentile",
        type=float,
        default=25.0,
        help="Percentile cutoff (per timestep). Candidates with fit > cutoff are moved. Default: 25.",
    )
    p.add_argument("--dat-ext", type=str, default=".dat", help="DAT extension (default: .dat)")
    p.add_argument("--xyz-ext", type=str, default=".xyz", help="XYZ extension (default: .xyz)")
    p.add_argument(
        "--pruned-dirname",
        type=str,
        default="pruned",
        help='Subdirectory name used for pruned files (default: "pruned")',
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would be moved, but do not move anything.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite files if they already exist in pruned/.")

    args = p.parse_args()

    directory = str(args.directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    perc = float(args.percentile)
    if not (0.0 <= perc <= 100.0):
        raise ValueError("--percentile must be in [0, 100]")

    cands = collect_candidates(directory, dat_ext=str(args.dat_ext), xyz_ext=str(args.xyz_ext))
    summary = summarize_by_timestep(cands, percentile=perc)

    # Print summary (similar to plot_fit_histograms.py)
    print("\n" + "=" * 72)
    print(f"Fit statistics per timestep (percentile cutoff = {perc:g})")
    print("=" * 72)
    print(f"{'Timestep':<10} {'Count':<7} {'Min':<12} {'P25':<12} {'Median':<12} {'P75':<12} {'Max':<12} {'Cutoff':<12}")
    print("-" * 72)
    for t in sorted(summary.keys()):
        s = summary[t]
        print(
            f"{t:02d}        {s['count']:<7d} "
            f"{s['min']:<12.6f} {s['p25']:<12.6f} {s['median']:<12.6f} "
            f"{s['p75']:<12.6f} {s['max']:<12.6f} {s['cutoff']:<12.6f}"
        )
    print("=" * 72 + "\n")

    pruned_dir = os.path.join(directory, str(args.pruned_dirname))

    # Build move list per timestep cutoff
    to_prune: List[Candidate] = []
    for c in cands:
        cutoff = float(summary[int(c.t)]["cutoff"])
        if float(c.fit) > cutoff:
            to_prune.append(c)

    # Report
    total = len(cands)
    n_prune = len(to_prune)
    n_keep = total - n_prune
    print(f"Total matched candidates: {total}")
    print(f"Keeping: {n_keep}")
    print(f"Pruning (moving to {pruned_dir}): {n_prune}")
    if args.dry_run:
        print("\n--dry-run enabled; no files will be moved.\n")

    # Move
    moved = 0
    for c in to_prune:
        if args.dry_run:
            print(f"PRUNE t={c.t:02d} fit={c.fit:.8f}:")
            print(f"  {c.dat_path} -> {pruned_dir}/")
            print(f"  {c.xyz_path} -> {pruned_dir}/")
            continue
        _move_file(c.dat_path, pruned_dir, overwrite=bool(args.overwrite))
        _move_file(c.xyz_path, pruned_dir, overwrite=bool(args.overwrite))
        moved += 1

    if not args.dry_run:
        print(f"\nMoved {moved} candidate pairs (2 files each) into: {pruned_dir}")


if __name__ == "__main__":
    main()

