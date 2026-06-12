#!/usr/bin/env python3
"""Aggregate phi-sweep SA outputs into plottable chi2 and RMSD statistics.

Scans results_phi_sweep/phi_*/ for structure XYZ files, parses the numeric
header line (line 2), and writes per-Phi mean/std CSVs:

  phi_chi2_stats.csv  — phi, mean_chi2, std_chi2   (chi2 = f_xray, field 0)
  phi_rmsd_stats.csv  — phi, mean_rmsd, std_rmsd     (rmsd = field 1)

Example:
  python3 scripts/python/aggregate_phi_sweep_stats.py results_phi_sweep
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

PHI_DIR_RE = re.compile(r"^phi_(\d+(?:\.\d+)?)$")
SKIP_SUFFIXES = ("_mean.xyz", "_target.xyz")


def parse_header_line(line: str) -> tuple[float, float, float | None]:
    """Return (chi2, rmsd, phi_or_none) from XYZ comment line."""
    fields = line.strip().split()
    if len(fields) < 2:
        raise ValueError(f"expected at least 2 numeric fields, got {len(fields)}")
    chi2 = float(fields[0])
    rmsd = float(fields[1])
    phi = float(fields[7]) if len(fields) >= 8 else None
    return chi2, rmsd, phi


def phi_from_dirname(dirname: str) -> float | None:
    m = PHI_DIR_RE.match(dirname)
    if not m:
        return None
    return float(m.group(1))


def should_skip_xyz(path: str) -> bool:
    base = os.path.basename(path)
    return any(base.endswith(suffix) for suffix in SKIP_SUFFIXES)


def discover_phi_dirs(results_dir: str) -> list[str]:
    """Return phi_* subdirectories, or [results_dir] if it is itself phi_*."""
    base = os.path.basename(os.path.normpath(results_dir))
    if PHI_DIR_RE.match(base):
        return [results_dir]
    pattern = os.path.join(results_dir, "phi_*")
    dirs = sorted(
        d for d in glob.glob(pattern) if os.path.isdir(d) and phi_from_dirname(os.path.basename(d)) is not None
    )
    return dirs


def collect_records(phi_dir: str, pattern: str) -> list[tuple[float, float, float]]:
    """Collect (chi2, rmsd, phi) from all valid XYZ files in one phi directory."""
    phi_default = phi_from_dirname(os.path.basename(os.path.normpath(phi_dir)))
    if phi_default is None:
        raise ValueError(f"cannot infer phi from directory name: {phi_dir!r}")

    search = os.path.join(phi_dir, pattern)
    paths = sorted(glob.glob(search))
    records: list[tuple[float, float, float]] = []

    for path in paths:
        if not os.path.isfile(path) or should_skip_xyz(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                _ = f.readline()
                comment = f.readline()
                if not comment:
                    raise ValueError("missing comment line")
            chi2, rmsd, phi_hdr = parse_header_line(comment)
            phi = phi_hdr if phi_hdr is not None else phi_default
            records.append((chi2, rmsd, phi))
        except (ValueError, OSError) as exc:
            print(f"Warning: skipping {path!r}: {exc}", file=sys.stderr)

    return records


def aggregate_by_phi(
    all_records: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group records by phi; return sorted phi, mean_chi2, std_chi2, mean_rmsd, std_rmsd."""
    buckets: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for chi2, rmsd, phi in all_records:
        buckets[phi].append((chi2, rmsd))

    if not buckets:
        raise SystemExit("ERROR: no valid structure XYZ files found")

    phis = np.array(sorted(buckets.keys()), dtype=np.float64)
    mean_chi2 = np.zeros(len(phis), dtype=np.float64)
    std_chi2 = np.zeros(len(phis), dtype=np.float64)
    mean_rmsd = np.zeros(len(phis), dtype=np.float64)
    std_rmsd = np.zeros(len(phis), dtype=np.float64)

    for i, phi in enumerate(phis):
        vals = buckets[float(phi)]
        chi2_arr = np.array([v[0] for v in vals], dtype=np.float64)
        rmsd_arr = np.array([v[1] for v in vals], dtype=np.float64)
        mean_chi2[i] = chi2_arr.mean()
        std_chi2[i] = np.std(chi2_arr, ddof=0)
        mean_rmsd[i] = rmsd_arr.mean()
        std_rmsd[i] = np.std(rmsd_arr, ddof=0)

    return phis, mean_chi2, std_chi2, mean_rmsd, std_rmsd


def write_csv(path: str, header: str, data: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10g")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate phi-sweep XYZ headers into chi2 and RMSD statistics CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        help="Parent sweep directory containing phi_* subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output CSVs (default: same as results_dir)",
    )
    parser.add_argument(
        "--pattern",
        default="*.xyz",
        help="Glob pattern for structure files within each phi_* directory",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir or results_dir)

    if not os.path.isdir(results_dir):
        raise SystemExit(f"ERROR: results directory does not exist: {results_dir}")

    phi_dirs = discover_phi_dirs(results_dir)
    if not phi_dirs:
        raise SystemExit(
            f"ERROR: no phi_* subdirectories found under {results_dir}"
        )

    all_records: list[tuple[float, float, float]] = []
    counts: dict[float, int] = defaultdict(int)

    for phi_dir in phi_dirs:
        records = collect_records(phi_dir, args.pattern)
        for chi2, rmsd, phi in records:
            all_records.append((chi2, rmsd, phi))
            counts[phi] += 1

    phis, mean_chi2, std_chi2, mean_rmsd, std_rmsd = aggregate_by_phi(all_records)

    chi2_path = os.path.join(output_dir, "phi_chi2_stats.csv")
    rmsd_path = os.path.join(output_dir, "phi_rmsd_stats.csv")

    write_csv(
        chi2_path,
        "phi,mean_chi2,std_chi2",
        np.column_stack([phis, mean_chi2, std_chi2]),
    )
    write_csv(
        rmsd_path,
        "phi,mean_rmsd,std_rmsd",
        np.column_stack([phis, mean_rmsd, std_rmsd]),
    )

    print(f"Wrote {chi2_path} ({len(phis)} Phi values)")
    print(f"Wrote {rmsd_path} ({len(phis)} Phi values)")
    for phi in phis:
        print(f"  phi={phi:g}: n_structures={counts[float(phi)]}")


if __name__ == "__main__":
    main()
