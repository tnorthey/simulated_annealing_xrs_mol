#!/usr/bin/env python3
"""
topM_geometry_statistics.py

For each timestep, select the topM best-fitting xyz files, compute the
requested geometry (bond / angle / dihedral), and plot the per-timestep
mean and closest-to-mean frame.

The closest-to-mean frame is the actual candidate whose RMSD to the
per-timestep mean structure is minimal.

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


def read_xyz_frame(path):
    """Read a single-frame xyz file, returning (natoms, comment, atoms, coords)."""
    with open(path, "r") as f:
        n = int(f.readline().strip())
        comment = f.readline().rstrip("\n")
        atoms: list[str] = []
        coords = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            parts = f.readline().split()
            atoms.append(parts[0])
            coords[i, 0] = float(parts[1])
            coords[i, 1] = float(parts[2])
            coords[i, 2] = float(parts[3])
    return n, comment, atoms, coords


def write_xyz_trajectory(frames: Sequence[tuple], path: str):
    """Write a multi-frame xyz trajectory.

    frames: list of (natoms, comment, atoms, coords) tuples.
    """
    with open(path, "w") as f:
        for natoms, comment, atoms, coords in frames:
            f.write(f"{natoms}\n")
            f.write(f"{comment}\n")
            for i in range(natoms):
                f.write(
                    f"{atoms[i]:>2s} {coords[i, 0]:16.10f} {coords[i, 1]:16.10f} {coords[i, 2]:16.10f}\n"
                )


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray, indices: Sequence[int] | None = None) -> float:
    """Kabsch-aligned RMSD between two (N,3) coordinate arrays."""
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int64)
        P, Q = P[idx], Q[idx]
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


def _plain_rmsd(P: np.ndarray, Q: np.ndarray, indices: Sequence[int] | None = None) -> float:
    """Centroid-aligned RMSD without Kabsch rotation (translation only)."""
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int64)
        P, Q = P[idx], Q[idx]
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    d = Pc - Qc
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def select_closest_to_mean(
    layer: Sequence[dict],
    *,
    use_kabsch: bool = True,
    rmsd_indices: Sequence[int] | None = None,
) -> int:
    """Return index of frame with minimal RMSD to layer mean structure."""
    K = len(layer)
    if K <= 1:
        return 0

    rmsd_fn = _kabsch_rmsd if use_kabsch else _plain_rmsd
    coords = [read_xyz_coords(c["xyz"]) for c in layer]
    mean_coords = np.mean(np.stack(coords, axis=0), axis=0)
    dists = np.array(
        [rmsd_fn(c, mean_coords, indices=rmsd_indices) for c in coords], dtype=np.float64
    )
    return int(np.argmin(dists))


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
        labels.append(f"Bond {bond[0] + 1}-{bond[1] + 1} (Å)")
    if angle is not None:
        labels.append(f"Angle {angle[0] + 1}-{angle[1] + 1}-{angle[2] + 1} (°)")
    if dihedral is not None:
        labels.append(
            f"Dihedral {dihedral[0] + 1}-{dihedral[1] + 1}-{dihedral[2] + 1}-{dihedral[3] + 1} (°)"
        )
    return labels


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot per-timestep mean and closest-to-mean geometry "
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
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for all outputs (plot, CSV, xyz trajectories). "
            "Default: current directory."
        ),
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Output PNG filename (placed inside --output-dir). Default: auto-generated.",
    )
    parser.add_argument(
        "--show-individuals",
        action="store_true",
        help="Also plot each individual candidate as a faint line.",
    )
    parser.add_argument(
        "--no-kabsch",
        action="store_true",
        help=(
            "Use plain centroid-aligned RMSD (no Kabsch rotation) for closest-to-mean selection. "
            "Much faster but assumes structures share a consistent orientation."
        ),
    )
    parser.add_argument(
        "--rmsd-indices",
        type=str,
        default=None,
        help=(
            "Comma-separated atom indices (0-based) for RMSD in closest-to-mean selection. "
            "Example: '0,1,2,3,4,5'. Default: all atoms."
        ),
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation even if CSV from a previous run exists.",
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
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale factor for plot text sizes (e.g. 2.0 for about double size).",
    )

    args = parser.parse_args()

    if args.bond is None and args.angle is None and args.dihedral is None:
        parser.error("At least one of --bond, --angle, or --dihedral must be specified")

    rmsd_indices: list[int] | None = None
    if args.rmsd_indices is not None:
        try:
            rmsd_indices = [int(x.strip()) for x in args.rmsd_indices.split(",") if x.strip()]
        except ValueError:
            parser.error("--rmsd-indices must be a comma-separated list of integers")
        if not rmsd_indices:
            parser.error("--rmsd-indices must contain at least one index")

    # --- Resolve output paths early so we can check for cached CSV ---
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

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        if os.path.dirname(args.output_plot) == "":
            args.output_plot = os.path.join(args.output_dir, args.output_plot)

    plot_stem = os.path.splitext(args.output_plot)[0]
    csv_path = plot_stem + ".csv"
    closest_xyz_path = plot_stem + "_closest_to_mean.xyz"
    mean_xyz_path = plot_stem + "_mean.xyz"

    col_labels = _column_labels(
        bond=args.bond, angle=args.angle, dihedral=args.dihedral
    )
    n_cols = len(col_labels)
    dihedral_col = _dihedral_column_index(
        bond=args.bond, angle=args.angle, dihedral=args.dihedral
    )

    # --- Try to reuse cached CSV to skip expensive computation ---
    loaded_from_csv = False
    if not args.recompute and os.path.exists(csv_path):
        try:
            raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            expected_cols = 1 + 3 * n_cols  # time + (mean, closest, std) per column
            if raw.shape[1] == expected_cols:
                x = raw[:, 0]
                means = np.zeros((raw.shape[0], n_cols), dtype=np.float64)
                closest = np.zeros((raw.shape[0], n_cols), dtype=np.float64)
                stds = np.zeros((raw.shape[0], n_cols), dtype=np.float64)
                for ci in range(n_cols):
                    base = 1 + ci * 3
                    means[:, ci] = raw[:, base]
                    closest[:, ci] = raw[:, base + 1]
                    stds[:, ci] = raw[:, base + 2]
                n_timesteps = raw.shape[0]
                try:
                    with open(csv_path, "r") as f:
                        header_line = f.readline().strip()
                    loaded_from_csv = "closest_" in header_line
                except OSError:
                    loaded_from_csv = False
                if loaded_from_csv:
                    print(
                        f"Loaded cached data from {csv_path} ({n_timesteps} timesteps). "
                        f"Use --recompute to force recalculation."
                    )
                else:
                    print("Cached CSV uses legacy median columns, recomputing...")
            else:
                print(f"CSV column count mismatch ({raw.shape[1]} vs expected {expected_cols}), recomputing...")
        except Exception as e:
            print(f"Could not load cached CSV ({e}), recomputing...")

    layers = None
    closest_indices: list[int] = []
    all_geom: list[np.ndarray] = []
    timesteps: list[int] = []

    if not loaded_from_csv:
        print(f"Loading candidates from {args.directory}...")
        timesteps, layers = load_topM_candidates(args.directory, args.topM)

        n_timesteps = len(timesteps)
        layer_sizes = [len(layer) for layer in layers]
        print(
            f"Found {n_timesteps} timesteps, candidates per timestep: "
            f"min={min(layer_sizes)}, max={max(layer_sizes)}, "
            f"topM={'all' if args.topM is None else args.topM}"
        )

        print("Computing geometry and selecting closest-to-mean frames...")
        means = np.zeros((n_timesteps, n_cols), dtype=np.float64)
        closest = np.zeros((n_timesteps, n_cols), dtype=np.float64)
        stds = np.zeros((n_timesteps, n_cols), dtype=np.float64)

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

            closest_idx = select_closest_to_mean(
                layer, use_kabsch=not args.no_kabsch, rmsd_indices=rmsd_indices
            )
            closest_indices.append(closest_idx)
            closest[ti, :] = geom[closest_idx, :]

        print()  # finish progress line

        x = np.arange(n_timesteps, dtype=np.float64) * 20.0 + 10.0

        # Write CSV
        header_parts = ["time_fs"]
        for label in col_labels:
            short = label.split(" (")[0].replace(" ", "_")
            header_parts.extend([f"mean_{short}", f"closest_{short}", f"std_{short}"])

        cols_out = [x]
        for i in range(n_cols):
            cols_out.extend([means[:, i], closest[:, i], stds[:, i]])
        out_data = np.column_stack(cols_out)
        np.savetxt(
            csv_path,
            out_data,
            delimiter=",",
            header=",".join(header_parts),
            comments="",
            fmt="%.10g",
        )
        print(f"CSV written to: {csv_path}")

        # Write closest-to-mean and mean xyz trajectories
        print("Writing closest-to-mean trajectory...")
        closest_frames = []
        for ti, layer_i in enumerate(layers):
            mi = closest_indices[ti]
            n, comment, atoms, coords = read_xyz_frame(layer_i[mi]["xyz"])
            comment = f"{comment} | closest-to-mean | timestep={timesteps[ti]}"
            closest_frames.append((n, comment, atoms, coords))
        write_xyz_trajectory(closest_frames, closest_xyz_path)
        print(f"Closest-to-mean trajectory written to: {closest_xyz_path}")

        print("Writing mean trajectory...")
        mean_frames = []
        for ti, layer_i in enumerate(layers):
            all_coords = []
            atoms_ref = None
            natoms_ref = None
            for c in layer_i:
                n, _comment, atoms, coords = read_xyz_frame(c["xyz"])
                all_coords.append(coords)
                if atoms_ref is None:
                    atoms_ref = atoms
                    natoms_ref = n
            mean_coords = np.mean(np.stack(all_coords, axis=0), axis=0)
            comment = f"mean over {len(layer_i)} candidates | timestep={timesteps[ti]}"
            mean_frames.append((natoms_ref, comment, atoms_ref, mean_coords))
        write_xyz_trajectory(mean_frames, mean_xyz_path)
        print(f"Mean trajectory written to: {mean_xyz_path}")

    # --- Plot (always runs, using either fresh or cached data) ---
    print("Creating plot...")
    font_scale = float(args.font_scale)
    if font_scale <= 0:
        parser.error("--font-scale must be > 0")
    title_fs = 16.0 * font_scale
    label_fs = 12.0 * font_scale
    tick_fs = 10.0 * font_scale
    legend_fs = 10.0 * font_scale

    if n_cols == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_cols, 1, figsize=(10, 3.2 * n_cols), sharex=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

    for i in range(n_cols):
        ax = axes[i]

        if args.show_individuals and all_geom:
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
            x,
            closest[:, i],
            linewidth=2,
            label="closest-to-mean",
            color="C1",
            linestyle="--",
        )
        ax.set_ylabel(col_labels[i], fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=legend_fs)

    axes[-1].set_xlabel("time (fs)", fontsize=label_fs)
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


if __name__ == "__main__":
    main()
