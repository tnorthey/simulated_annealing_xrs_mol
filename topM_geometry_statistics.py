#!/usr/bin/env python3
"""
topM_geometry_statistics.py

For each timestep, select the topM best-fitting xyz files, compute the
requested geometry (bond / angle / dihedral), and plot the per-timestep
mean and medoid.

The medoid is the actual candidate whose sum of Kabsch-RMSD to all other
candidates at the same timestep is minimal — it always corresponds to a
real frame from the dataset.

No optimal-path solving is performed; this script directly examines
the raw candidates at each timestep.

Usage:
    python3 topM_geometry_statistics.py results/ --topM 50 --dihedral 2 3 4 5
    python3 topM_geometry_statistics.py results/ --topM 100 --dihedral 2 3 4 5 --bond 0 1
    python3 topM_geometry_statistics.py results/ --dihedral 2 3 4 5 --show-individuals
"""

import argparse
import glob
import heapq
import os
import re
import sys
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

import modules.analysis as analysis

NAME_RE = re.compile(r"^(?P<t>\d+?)_(?P<fit>\d+(?:\.\d+)?)(?P<rest>.*)$")


def parse_name(path):
    base = os.path.basename(path)
    m = NAME_RE.match(base)
    if not m:
        return None
    t = int(m.group("t"))
    fit = float(m.group("fit"))
    return t, fit


def read_xyz_coords(path):
    with open(path, "r") as f:
        n = int(f.readline().strip())
        _ = f.readline()
        xyz = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            parts = f.readline().split()
            xyz[i, 0] = float(parts[1])
            xyz[i, 1] = float(parts[2])
            xyz[i, 2] = float(parts[3])
    return xyz


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch-aligned RMSD between two (N,3) coordinate arrays."""
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    Pr = Pc @ R
    d = Pr - Qc
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def select_medoid(layer: Sequence[dict]) -> int:
    """Return the index of the medoid in a layer.

    The medoid is the candidate with the smallest sum of Kabsch-RMSD
    to all other candidates. Coordinates are read on-the-fly.
    """
    K = len(layer)
    if K <= 1:
        return 0

    coords = [read_xyz_coords(c["xyz"]) for c in layer]
    sum_rmsd = np.zeros(K, dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            d = _kabsch_rmsd(coords[i], coords[j])
            sum_rmsd[i] += d
            sum_rmsd[j] += d

    return int(np.argmin(sum_rmsd))


def load_topM_candidates(directory, topM):
    """Load xyz files grouped by timestep, keeping only the topM lowest-fit per timestep."""
    xyzs = glob.glob(os.path.join(directory, "*.xyz"))

    by_t: dict[int, list[dict]] = {}
    for p in xyzs:
        parsed = parse_name(p)
        if parsed:
            t, fit = parsed
            by_t.setdefault(t, []).append({"t": t, "fit": fit, "xyz": p})

    if not by_t:
        raise RuntimeError(
            f"No xyz files with expected naming pattern found in {directory}"
        )

    timesteps = sorted(by_t.keys())
    layers = []
    for t in timesteps:
        layer = by_t[t]
        if topM is not None and len(layer) > topM:
            layer = heapq.nsmallest(topM, layer, key=lambda c: c["fit"])
        else:
            layer = sorted(layer, key=lambda c: c["fit"])
        layers.append(layer)

    return timesteps, layers


def compute_geometry_for_layer(
    layer: Sequence[dict],
    bond=None,
    angle=None,
    dihedral=None,
) -> np.ndarray:
    """Compute requested geometry for all candidates in a layer.

    Returns (n_candidates, n_cols) array.
    """
    results = []
    for c in layer:
        xyz = read_xyz_coords(c["xyz"])
        row: list[float] = []
        if bond is not None:
            row.append(analysis.calculate_bond_length(xyz, bond[0], bond[1]))
        if angle is not None:
            row.append(analysis.calculate_angle(xyz, angle[0], angle[1], angle[2]))
        if dihedral is not None:
            row.append(
                analysis.calculate_dihedral(
                    xyz, dihedral[0], dihedral[1], dihedral[2], dihedral[3]
                )
            )
        results.append(row)
    return np.array(results, dtype=np.float64)


def wrap_dihedral_column(data: np.ndarray, col_idx: int) -> np.ndarray:
    """Shift dihedrals: negative values +360 to put into [0, 360)."""
    vals = data[:, col_idx].copy()
    vals = np.where(vals < 0, vals + 360.0, vals)
    data[:, col_idx] = vals
    return data


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Circular (directional) mean of angles in degrees, returned in [0, 360)."""
    rad = np.deg2rad(angles_deg)
    mean_deg = float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))
    return mean_deg % 360.0


def circular_std_deg(angles_deg: np.ndarray) -> float:
    """Circular standard deviation of angles in degrees.

    Uses sqrt(-2 * ln(R)) where R is the mean resultant length.
    """
    rad = np.deg2rad(angles_deg)
    R = np.sqrt(np.mean(np.sin(rad)) ** 2 + np.mean(np.cos(rad)) ** 2)
    R = min(R, 1.0)
    if R < 1e-15:
        return 180.0
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))


def _dihedral_column_index(bond=None, angle=None, dihedral=None) -> Optional[int]:
    if dihedral is None:
        return None
    idx = 0
    if bond is not None:
        idx += 1
    if angle is not None:
        idx += 1
    return idx


def _column_labels(bond=None, angle=None, dihedral=None) -> List[str]:
    labels: List[str] = []
    if bond is not None:
        labels.append(f"Bond {bond[0]}-{bond[1]} (Å)")
    if angle is not None:
        labels.append(f"Angle {angle[0]}-{angle[1]}-{angle[2]} (°)")
    if dihedral is not None:
        labels.append(
            f"Dihedral {dihedral[0]}-{dihedral[1]}-{dihedral[2]}-{dihedral[3]} (°)"
        )
    return labels


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot per-timestep mean and medoid geometry "
            "for the topM best-fitting xyz files."
        ),
    )
    parser.add_argument(
        "directory",
        help="Directory containing *.xyz files (named like 01_000.12345678.xyz)",
    )
    parser.add_argument(
        "--topM",
        type=int,
        default=None,
        help="Keep only the M lowest-fit xyz files per timestep. Default: use all.",
    )
    parser.add_argument(
        "--bond",
        type=int,
        nargs=2,
        default=None,
        metavar=("I", "J"),
        help="Bond atom indices I J (0-indexed)",
    )
    parser.add_argument(
        "--angle",
        type=int,
        nargs=3,
        default=None,
        metavar=("I", "J", "K"),
        help="Angle atom indices I J K (0-indexed)",
    )
    parser.add_argument(
        "--dihedral",
        type=int,
        nargs=4,
        default=None,
        metavar=("I", "J", "K", "L"),
        help="Dihedral atom indices I J K L (0-indexed)",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Output PNG filename. Default: auto-generated from parameters.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Also write a CSV with per-timestep mean, medoid, and std.",
    )
    parser.add_argument(
        "--show-individuals",
        action="store_true",
        help="Also plot each individual candidate as a faint line.",
    )
    parser.add_argument(
        "--xmin", type=float, default=None, help="Minimum x-axis value (time in fs)."
    )
    parser.add_argument(
        "--xmax", type=float, default=None, help="Maximum x-axis value (time in fs)."
    )
    parser.add_argument(
        "--ymin", type=float, default=None, help="Minimum y-axis value."
    )
    parser.add_argument(
        "--ymax", type=float, default=None, help="Maximum y-axis value."
    )

    args = parser.parse_args()

    if args.bond is None and args.angle is None and args.dihedral is None:
        parser.error("At least one of --bond, --angle, or --dihedral must be specified")

    print(f"Loading candidates from {args.directory}...")
    timesteps, layers = load_topM_candidates(args.directory, args.topM)

    n_timesteps = len(timesteps)
    layer_sizes = [len(layer) for layer in layers]
    print(
        f"Found {n_timesteps} timesteps, candidates per timestep: "
        f"min={min(layer_sizes)}, max={max(layer_sizes)}, "
        f"topM={'all' if args.topM is None else args.topM}"
    )

    col_labels = _column_labels(
        bond=args.bond, angle=args.angle, dihedral=args.dihedral
    )
    dihedral_col = _dihedral_column_index(
        bond=args.bond, angle=args.angle, dihedral=args.dihedral
    )
    n_cols = len(col_labels)

    print("Computing geometry and selecting medoids...")
    means = np.zeros((n_timesteps, n_cols), dtype=np.float64)
    medoids = np.zeros((n_timesteps, n_cols), dtype=np.float64)
    stds = np.zeros((n_timesteps, n_cols), dtype=np.float64)

    all_geom: list[np.ndarray] = []
    for ti, layer in enumerate(layers):
        pct = (ti + 1) * 100 // n_timesteps
        print(f"\r  Timestep {ti + 1}/{n_timesteps} ({pct}%)", end="", flush=True)
        geom = compute_geometry_for_layer(
            layer, bond=args.bond, angle=args.angle, dihedral=args.dihedral
        )
        if dihedral_col is not None:
            geom = wrap_dihedral_column(geom, dihedral_col)
        all_geom.append(geom)

        for ci in range(n_cols):
            if ci == dihedral_col:
                means[ti, ci] = circular_mean_deg(geom[:, ci])
                stds[ti, ci] = circular_std_deg(geom[:, ci])
            else:
                means[ti, ci] = np.mean(geom[:, ci])
                stds[ti, ci] = np.std(geom[:, ci], ddof=0)

        medoid_idx = select_medoid(layer)
        medoids[ti, :] = geom[medoid_idx, :]

    print()  # finish progress line

    x = np.arange(n_timesteps, dtype=np.float64) * 20.0 + 10.0

    # Auto-generate output plot name
    if args.output_plot is None:
        parts = ["topM_geometry"]
        if args.bond is not None:
            parts.append(f"bond-{args.bond[0]}-{args.bond[1]}")
        if args.angle is not None:
            parts.append(f"angle-{args.angle[0]}-{args.angle[1]}-{args.angle[2]}")
        if args.dihedral is not None:
            parts.append(
                f"dihedral-{args.dihedral[0]}-{args.dihedral[1]}"
                f"-{args.dihedral[2]}-{args.dihedral[3]}"
            )
        if args.topM is not None:
            parts.append(f"topM-{args.topM}")
        args.output_plot = "_".join(parts) + ".png"

    print("Creating plot...")
    if n_cols == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_cols, 1, figsize=(10, 3.2 * n_cols), sharex=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

    for i in range(n_cols):
        ax = axes[i]

        if args.show_individuals:
            max_candidates = max(g.shape[0] for g in all_geom)
            for ci in range(max_candidates):
                ys = []
                xs = []
                for ti in range(n_timesteps):
                    if ci < all_geom[ti].shape[0]:
                        xs.append(x[ti])
                        ys.append(all_geom[ti][ci, i])
                ax.plot(xs, ys, linewidth=0.3, alpha=0.15, color="grey")

        ax.plot(x, means[:, i], linewidth=2, label="mean", color="C0")
        ax.fill_between(
            x,
            means[:, i] - stds[:, i],
            means[:, i] + stds[:, i],
            alpha=0.2,
            color="C0",
            label="±1σ",
        )
        ax.plot(
            x, medoids[:, i], linewidth=2, label="median", color="C1", linestyle="--"
        )
        ax.set_ylabel(col_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    title_parts = []
    if args.bond is not None:
        title_parts.append(f"Bond {args.bond[0]}-{args.bond[1]}")
    if args.angle is not None:
        title_parts.append(
            f"Angle {args.angle[0]}-{args.angle[1]}-{args.angle[2]}"
        )
    if args.dihedral is not None:
        title_parts.append(
            f"Dihedral {args.dihedral[0]}-{args.dihedral[1]}"
            f"-{args.dihedral[2]}-{args.dihedral[3]}"
        )
    title = " / ".join(title_parts) if title_parts else "Geometry"
    topM_str = f"topM={args.topM}" if args.topM is not None else "all candidates"
    fig.suptitle(f"{title} — mean & medoid ({topM_str})")

    axes[-1].set_xlabel("time (fs)")
    xleft = 0.0 if args.xmin is None else float(args.xmin)
    xright = None if args.xmax is None else float(args.xmax)
    for ax in axes:
        ax.set_xlim(left=xleft, right=xright)
        if args.ymin is not None or args.ymax is not None:
            ax.set_ylim(bottom=args.ymin, top=args.ymax)

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {args.output_plot}")

    if args.output_csv is not None:
        header_parts = ["time_fs"]
        for label in col_labels:
            short = label.split(" (")[0].replace(" ", "_")
            header_parts.extend([f"mean_{short}", f"medoid_{short}", f"std_{short}"])

        cols_out = [x]
        for i in range(n_cols):
            cols_out.extend([means[:, i], medoids[:, i], stds[:, i]])
        out_data = np.column_stack(cols_out)
        np.savetxt(
            args.output_csv,
            out_data,
            delimiter=",",
            header=",".join(header_parts),
            comments="",
            fmt="%.10g",
        )
        print(f"CSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()
