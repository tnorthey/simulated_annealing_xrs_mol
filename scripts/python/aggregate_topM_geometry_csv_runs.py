#!/usr/bin/env python3
"""
aggregate_topM_geometry_csv_runs.py

Combine several topM_geometry_statistics CSVs (same time axis, same geometry
column layout): across-run mean and population std of a chosen per-run column
(typically ``mean_*``), optional closest-to-target curve from one reference CSV,
and optional target geometry from ``<run_id>_target.xyz`` files.

Examples
--------
Aggregate three runs (outputs **always** five columns: ``time``, ``mean_runs``,
``sd_runs``, ``closest_ref``, ``target``; unused reference columns are ``nan``)::

    python3 scripts/python/aggregate_topM_geometry_csv_runs.py \\
        --out combined.csv \\
        --column-suffix Dihedral_1-2-3-4 \\
        --csv run1/topM_geometry_dihedral-0-1-2-3.csv \\
        --csv run2/topM_geometry_dihedral-0-1-2-3.csv \\
        --csv run3/topM_geometry_dihedral-0-1-2-3.csv \\
        --reference-closest run1/topM_geometry_dihedral-0-1-2-3.csv \\
        --target-dir run1 \\
        --results-dir run1 \\
        --target-run-id-pad 2 \\
        --dihedral 0 1 2 3

Plot with gnuplot (``DATA`` = combined CSV; columns 4–5 are read automatically
unless you override with ``DATA_CLOSEST`` / ``DATA_TARGET``)::

    gnuplot -e "DATA='combined.csv';OUTBASE='fig_dihedral';XLABEL='time (fs)';YLABEL='deg';KEY_POS='bottom right'" \\
        scripts/gnuplot/plot_mean_sd_with_refs_tex.gp

Use explicit timestep ids (one integer per line, same row count as CSVs)::

    python3 scripts/python/aggregate_topM_geometry_csv_runs.py \\
        --out combined.csv --column-suffix Bond_1-2 \\
        --csv a.csv --csv b.csv --timestep-ids-file steps.txt \\
        --target-dir results --bond 0 1

Precomputed target curve (column 2 = target y; column 1 = time)::

    python3 scripts/python/aggregate_topM_geometry_csv_runs.py ... \\
        --reference-target-csv target_curve.csv

``reference-target-csv`` must have the same time values in column 1 and the
target coordinate in column 2 (header optional; numeric rows only after merge).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import Any, Sequence

import numpy as np

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import modules.analysis as analysis  # noqa: E402

NAME_RE = re.compile(r"^(?P<t>\d+?)_(?P<fit>\d+(?:\.\d+)?)(?P<rest>.*)$")


def parse_name(path: str) -> tuple[int, float] | None:
    base = os.path.basename(path)
    m = NAME_RE.match(base)
    if not m:
        return None
    return int(m.group("t")), float(m.group("fit"))


def sorted_timestep_ids_from_directory(directory: str) -> list[int]:
    xyzs = glob.glob(os.path.join(directory, "*.xyz"))
    ids: set[int] = set()
    for p in xyzs:
        parsed = parse_name(p)
        if parsed:
            ids.add(parsed[0])
    if not ids:
        raise SystemExit(
            f"No xyz files matching <t>_<fit>*.xyz pattern in directory {directory!r}"
        )
    return sorted(ids)


def timestep_to_target_run_id(t: int, *, pad: int) -> str:
    p = max(1, int(pad))
    return f"{t:0{p}d}"


def read_xyz_coords(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        n = int(f.readline().strip())
        _ = f.readline()
        xyz = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            parts = f.readline().split()
            xyz[i, 0] = float(parts[1])
            xyz[i, 1] = float(parts[2])
            xyz[i, 2] = float(parts[3])
    return xyz


def compute_geometry_scalar(
    xyz: np.ndarray,
    *,
    bond: tuple[int, int] | None,
    angle: tuple[int, int, int] | None,
    dihedral: tuple[int, int, int, int] | None,
) -> float:
    if bond is not None:
        return float(analysis.calculate_bond_length(xyz, bond[0], bond[1]))
    if angle is not None:
        return float(analysis.calculate_angle(xyz, angle[0], angle[1], angle[2]))
    if dihedral is not None:
        return float(
            analysis.calculate_dihedral(
                xyz, dihedral[0], dihedral[1], dihedral[2], dihedral[3]
            )
        )
    raise RuntimeError("compute_geometry_scalar: need bond, angle, or dihedral")


def apply_dihedral_output_transforms(
    v: float,
    *,
    offset: float,
    negate: bool,
    wrap360: bool,
) -> float:
    x = v + offset
    if negate:
        x = -x
    if wrap360 and x < 0.0:
        x += 360.0
    return float(x)


def read_csv_numeric(path: str) -> tuple[list[str], np.ndarray]:
    """Return (header_fields, data) with data shape (nrows, ncols)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: list[list[float]] = []
        for parts in reader:
            if not parts or all(not c.strip() for c in parts):
                continue
            rows.append([float(c) for c in parts])
    if not rows:
        raise SystemExit(f"CSV has no data rows: {path!r}")
    return header, np.asarray(rows, dtype=np.float64)


def parse_geometry_blocks(headers: Sequence[str]) -> list[tuple[str, dict[str, int]]]:
    """Parse topM-style header into list of (short_name, {mean, std?, closest?})."""
    if len(headers) < 2:
        raise SystemExit("CSV header too short")
    blocks: list[tuple[str, dict[str, int]]] = []
    i = 1
    n = len(headers)
    while i < n:
        h = headers[i].strip()
        if not h.startswith("mean_"):
            raise SystemExit(
                f"Expected column {i+1} to start with mean_; got {h!r} in header {headers}"
            )
        short = h[len("mean_") :]
        colmap: dict[str, int] = {"mean": i}
        i += 1
        if i < n and headers[i].strip().startswith("closest_"):
            colmap["closest"] = i
            i += 1
        if i >= n or not headers[i].strip().startswith("std_"):
            raise SystemExit(
                f"Expected std_{short} after mean/closest block; header tail: {headers[i:]}"
            )
        if headers[i].strip() != f"std_{short}":
            raise SystemExit(
                f"Expected std_{short!r}, got {headers[i].strip()!r}"
            )
        colmap["std"] = i
        i += 1
        blocks.append((short, colmap))
    return blocks


def resolve_column_indices(
    headers: list[str],
    *,
    column_suffix: str | None,
    column_index: int | None,
) -> tuple[str, dict[str, int]]:
    blocks = parse_geometry_blocks(headers)
    if column_suffix is not None:
        for short, mp in blocks:
            if short == column_suffix:
                return short, mp
        raise SystemExit(
            f"No geometry block with short name {column_suffix!r}. "
            f"Available: {[b[0] for b in blocks]}"
        )
    if column_index is not None:
        if column_index < 0 or column_index >= len(blocks):
            raise SystemExit(
                f"--column-index {column_index} out of range (0..{len(blocks)-1})"
            )
        return blocks[column_index]
    raise SystemExit("Provide --column-suffix or --column-index")


def colmap_for_short(headers: list[str], short: str) -> dict[str, int]:
    blocks = parse_geometry_blocks(headers)
    for s, mp in blocks:
        if s == short:
            return mp
    raise SystemExit(
        f"No block {short!r} in {headers!r}; available: {[b[0] for b in blocks]}"
    )


def header_metric_col(
    headers: list[str],
    short: str,
    metric: str,
    colmap: dict[str, int],
) -> int:
    if metric == "mean":
        return colmap["mean"]
    if metric == "std":
        return colmap["std"]
    if metric == "closest":
        if "closest" not in colmap:
            raise SystemExit(
                "This CSV has no closest_* columns (--skip-closest was used?). "
                "Cannot use --metric closest."
            )
        return colmap["closest"]
    raise SystemExit(f"Unknown --metric {metric!r}")


def find_closest_col_index(headers: list[str], short: str) -> int:
    for i, h in enumerate(headers):
        hs = h.strip()
        if hs.startswith("closest_") and hs.endswith(short):
            return i
    raise SystemExit(f"No closest_* column for suffix {short!r} in header")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Align topM_geometry_statistics CSVs from multiple runs and write "
            "time, across-run mean/sd, optional closest_ref and target columns."
        )
    )
    parser.add_argument(
        "--csv",
        dest="csvs",
        action="append",
        default=[],
        metavar="PATH",
        help="Input CSV path (repeat for each run).",
    )
    parser.add_argument(
        "csv_positional",
        nargs="*",
        metavar="PATH",
        help="Additional CSV paths (same as --csv).",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Output combined CSV path.",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--column-suffix",
        type=str,
        default=None,
        metavar="SHORT",
        help="Geometry short name after mean_, e.g. Dihedral_1-2-3-4",
    )
    g.add_argument(
        "--column-index",
        type=int,
        default=None,
        metavar="N",
        help="0-based geometry block index (first mean_/std_/... block = 0).",
    )
    parser.add_argument(
        "--metric",
        choices=("mean", "std", "closest"),
        default="mean",
        help=(
            "Which per-run column to aggregate across runs (default: mean). "
            "Usually use mean; std aggregates within-run stds across runs (rare)."
        ),
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=1e-9,
        help="Max abs difference for time column alignment across CSVs.",
    )
    parser.add_argument(
        "--reference-closest",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "One run's CSV: copy its closest_* column for the selected geometry "
            "into output column closest_ref."
        ),
    )
    parser.add_argument(
        "--reference-target-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Two columns: time, target_y (same times as inputs; optional header row).",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory with <run_id>_target.xyz for per-timestep target geometry.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory with candidate *.xyz to infer sorted timestep ids "
            "(row i -> timestep ids[i]). Ignored if --timestep-ids-file is set."
        ),
    )
    parser.add_argument(
        "--timestep-ids-file",
        type=str,
        default=None,
        metavar="PATH",
        help="One integer timestep id per line (same length as CSV rows).",
    )
    parser.add_argument(
        "--target-run-id-pad",
        type=int,
        default=2,
        metavar="N",
        help="Zero-pad width for <run_id>_target.xyz (default 2).",
    )
    parser.add_argument(
        "--bond",
        type=int,
        nargs=2,
        default=None,
        metavar=("I", "J"),
    )
    parser.add_argument(
        "--angle",
        type=int,
        nargs=3,
        default=None,
        metavar=("I", "J", "K"),
    )
    parser.add_argument(
        "--dihedral",
        type=int,
        nargs=4,
        default=None,
        metavar=("I", "J", "K", "L"),
    )
    parser.add_argument(
        "--dihedral-offset",
        type=float,
        default=0.0,
        metavar="DEG",
        help="Applied to target geometry only (after read), like topM.",
    )
    parser.add_argument(
        "--dihedral-negate",
        action="store_true",
        help="Applied to target geometry only.",
    )
    parser.add_argument(
        "--dihedral-wrap-360",
        action="store_true",
        help="Applied to target geometry only.",
    )

    args = parser.parse_args()
    paths = list(args.csvs) + list(args.csv_positional)
    if len(paths) < 1:
        parser.error("Need at least one input CSV (--csv or positional PATH)")

    if args.bond is None and args.angle is None and args.dihedral is None:
        geom_for_target = False
    else:
        geom_for_target = True
        ngeom = sum(
            x is not None for x in (args.bond, args.angle, args.dihedral)
        )
        if ngeom != 1:
            parser.error(
                "For --target-dir geometry, specify exactly one of --bond, --angle, --dihedral"
            )

    if args.reference_target_csv and args.target_dir:
        parser.error("Use only one of --reference-target-csv or --target-dir")

    if args.target_dir and not geom_for_target:
        parser.error("--target-dir requires --bond, --angle, or --dihedral")

    if args.target_dir:
        if args.timestep_ids_file is None and args.results_dir is None:
            parser.error(
                "--target-dir requires --timestep-ids-file or --results-dir "
                "to align rows to timestep ids"
            )

    headers0, data0 = read_csv_numeric(paths[0])
    short, colmap0 = resolve_column_indices(
        headers0,
        column_suffix=args.column_suffix,
        column_index=args.column_index,
    )
    time0 = data0[:, 0]
    n = data0.shape[0]

    idx_metric0 = header_metric_col(headers0, short, args.metric, colmap0)
    series_list: list[np.ndarray] = [data0[:, idx_metric0]]

    for p in paths[1:]:
        hdr, dat = read_csv_numeric(p)
        if hdr != headers0:
            print(
                f"Warning: header string mismatch vs first file; resolving columns by block name.\n"
                f"  first: {paths[0]}\n  other: {p}",
                file=sys.stderr,
            )
        if dat.shape[0] != n:
            raise SystemExit(f"Row count mismatch: {paths[0]} has {n}, {p} has {dat.shape[0]}")
        t = dat[:, 0]
        if not np.allclose(t, time0, rtol=0.0, atol=args.time_tol):
            raise SystemExit(f"Time column mismatch between {paths[0]!r} and {p!r}")
        cmap = colmap_for_short(hdr, short)
        idx = header_metric_col(hdr, short, args.metric, cmap)
        if dat.shape[1] <= idx:
            raise SystemExit(f"{p!r}: column index {idx} out of range (ncols={dat.shape[1]})")
        series_list.append(dat[:, idx])

    mat = np.column_stack(series_list)
    mean_runs = np.mean(mat, axis=1)
    sd_runs = np.std(mat, axis=1, ddof=0)

    closest_ref: np.ndarray | None = None
    if args.reference_closest:
        hdr_c, dat_c = read_csv_numeric(args.reference_closest)
        if dat_c.shape[0] != n:
            raise SystemExit(
                f"--reference-closest rows {dat_c.shape[0]} != {n} from aggregate CSVs"
            )
        if not np.allclose(dat_c[:, 0], time0, rtol=0.0, atol=args.time_tol):
            raise SystemExit("--reference-closest time column does not match")
        ic = find_closest_col_index(hdr_c, short)
        closest_ref = dat_c[:, ic]

    target_y: np.ndarray | None = None
    if args.reference_target_csv:
        _, dat_t = read_csv_numeric(args.reference_target_csv)
        if dat_t.shape[1] < 2:
            raise SystemExit("--reference-target-csv needs at least 2 columns")
        if dat_t.shape[0] != n:
            raise SystemExit(
                f"--reference-target-csv rows {dat_t.shape[0]} != aggregate row count {n}"
            )
        if not np.allclose(dat_t[:, 0], time0, rtol=0.0, atol=args.time_tol):
            raise SystemExit("--reference-target-csv time column does not match")
        target_y = dat_t[:, 1].astype(np.float64)

    if args.target_dir:
        if args.timestep_ids_file:
            raw = np.loadtxt(args.timestep_ids_file, dtype=np.int64, comments="#")
            ids = np.atleast_1d(np.asarray(raw, dtype=np.int64).ravel())
        else:
            assert args.results_dir is not None
            ids = np.asarray(
                sorted_timestep_ids_from_directory(args.results_dir),
                dtype=np.int64,
            )
        if ids.size != n:
            raise SystemExit(
                f"Timestep id count {ids.size} != CSV row count {n}. "
                "Check --results-dir / --timestep-ids-file."
            )
        bond_t = tuple(args.bond) if args.bond is not None else None
        angle_t = tuple(args.angle) if args.angle is not None else None
        dihedral_t = tuple(args.dihedral) if args.dihedral is not None else None
        tgt_vals = np.zeros(n, dtype=np.float64)
        pad = int(args.target_run_id_pad)
        for i in range(n):
            run_id = timestep_to_target_run_id(int(ids[i]), pad=pad)
            tpath = os.path.join(args.target_dir, f"{run_id}_target.xyz")
            if not os.path.isfile(tpath):
                raise SystemExit(f"Missing target xyz: {tpath!r}")
            xyz = read_xyz_coords(tpath)
            v = compute_geometry_scalar(
                xyz, bond=bond_t, angle=angle_t, dihedral=dihedral_t
            )
            if dihedral_t is not None:
                v = apply_dihedral_output_transforms(
                    v,
                    offset=float(args.dihedral_offset),
                    negate=bool(args.dihedral_negate),
                    wrap360=bool(args.dihedral_wrap_360),
                )
            tgt_vals[i] = v
        target_y = tgt_vals

    cr = (
        closest_ref
        if closest_ref is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    ty = (
        target_y
        if target_y is not None
        else np.full(n, np.nan, dtype=np.float64)
    )
    out_arr = np.column_stack([time0, mean_runs, sd_runs, cr, ty])
    header_line = "time,mean_runs,sd_runs,closest_ref,target"
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savetxt(
        out_path,
        out_arr,
        delimiter=",",
        header=header_line,
        comments="",
        fmt="%.10g",
    )
    print(f"Wrote {out_path} ({out_arr.shape[0]} rows, 5 columns: time, mean_runs, sd_runs, closest_ref, target)")


if __name__ == "__main__":
    main()
