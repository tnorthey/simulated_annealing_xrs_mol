#!/usr/bin/env python3
"""
Compute the PCD signal of a (median) optimal trajectory XYZ relative to a reference DAT file,
scale by an excitation factor, and make an overhead-view 3D plot with a colorbar.

PCD definition (per q, per frame):
    PCD(q,t) = excitation_factor * 100 * ( I(q,t) / I_ref(q) - 1 )

The reference DAT is expected to have either:
  - 2 columns: q, I_ref
  - 1 column: I_ref (q is treated as integer index)

Example:
    python3 plot_pcd_median_trajectory.py results/dihedral_mean_median_median.xyz \
        --ref-dat data/chd_reference.dat \
        --excitation-factor 0.628 \
        --output-plot pcd_heatmap.png \
        --output-dat pcd_scaled.dat
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np

import modules.mol as mol
import modules.x as xray


def read_xyz_trajectory(filename: str):
    """Read XYZ file(s) - handles both single structure and trajectory."""
    structures = []
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except (ValueError, AttributeError):
                break

            comment = f.readline().strip()
            atomlist = []
            xyzmatrix = []
            for _ in range(natoms):
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split()
                if len(parts) < 4:
                    break
                atomlist.append(parts[0])
                xyzmatrix.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if len(atomlist) != natoms:
                break

            structures.append(
                (
                    natoms,
                    comment,
                    np.array(atomlist, dtype=str),
                    np.array(xyzmatrix, dtype=float),
                )
            )

    if not structures:
        raise ValueError(f"No valid structures found in {filename}")
    return structures


def load_reference_dat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        ref_I = np.asarray(data, dtype=float)
        ref_q = np.arange(ref_I.size, dtype=float)
        return ref_q, ref_I
    if data.shape[1] < 2:
        raise ValueError(f"Reference DAT must have 1 or 2 columns; got shape {data.shape} from {path}")
    ref_q = np.asarray(data[:, 0], dtype=float)
    ref_I = np.asarray(data[:, 1], dtype=float)
    return ref_q, ref_I


def compute_iam_trajectory(structures, qvector: np.ndarray, include_inelastic: bool) -> np.ndarray:
    """Return IAM(q,t) with shape (n_frames, len(qvector))."""
    m = mol.Xyz()
    x = xray.Xray()

    natoms0, _, atomlist0, _ = structures[0]
    atomic_numbers = [m.periodic_table(sym) for sym in atomlist0]
    if any(z is None for z in atomic_numbers):
        bad = [sym for sym, z in zip(atomlist0, atomic_numbers) if z is None]
        raise ValueError(f"Unknown element symbols in XYZ: {sorted(set(bad))}")

    for i, (natoms, _, atomlist, _) in enumerate(structures):
        if natoms != natoms0:
            raise ValueError(f"Frame {i} natoms={natoms} differs from first frame natoms={natoms0}")
        if not np.array_equal(atomlist, atomlist0):
            raise ValueError(f"Frame {i} atom ordering/types differ from first frame; cannot compute consistent IAM")

    compton_array = None
    if include_inelastic:
        try:
            compton_array = x.compton_spline(atomic_numbers, qvector)
        except FileNotFoundError:
            print("Warning: Compton data file not found; continuing with elastic scattering only.", file=sys.stderr)
            include_inelastic = False

    out = np.zeros((len(structures), len(qvector)), dtype=float)
    for i, (_, _, _, xyz) in enumerate(structures):
        iam, _, _, _, _ = x.iam_calc(
            atomic_numbers,
            xyz,
            qvector,
            electron_mode=False,
            inelastic=include_inelastic,
            compton_array=compton_array if compton_array is not None else np.zeros(0),
        )
        out[i, :] = iam
    return out


def compute_pcd_scaled(iam_tq: np.ndarray, ref_I: np.ndarray, excitation_factor: float) -> np.ndarray:
    """Compute scaled PCD with shape (n_frames, n_q)."""
    ref_I = np.asarray(ref_I, dtype=float)
    if ref_I.ndim != 1:
        raise ValueError("ref_I must be 1D")
    if iam_tq.shape[1] != ref_I.size:
        raise ValueError(f"IAM q-grid ({iam_tq.shape[1]}) does not match reference I size ({ref_I.size})")

    with np.errstate(divide="ignore", invalid="ignore"):
        pcd = 100.0 * (iam_tq / ref_I[None, :] - 1.0)
    pcd = np.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)
    return float(excitation_factor) * pcd


def plot_pcd_surface(
    time_fs: np.ndarray,
    qvector: np.ndarray,
    pcd_tq: np.ndarray,
    output_plot: str,
    view_elev: float,
    view_azim: float,
    zlabel: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    zmin: float | None,
    zmax: float | None,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    title: str | None,
    figsize: tuple[float, float],
    dpi: int,
    invert_q_axis: bool,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

    T, Q = np.meshgrid(time_fs, qvector, indexing="ij")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        T,
        Q,
        pcd_tq,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        antialiased=False,
        rcount=min(200, pcd_tq.shape[0]),
        ccount=min(200, pcd_tq.shape[1]),
    )
    ax.view_init(elev=view_elev, azim=view_azim)

    if title:
        ax.set_title(title)

    ax.set_xlabel("time (fs)")
    ax.set_ylabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax.set_zlabel(zlabel)

    if invert_q_axis:
        ax.invert_yaxis()

    # Start at 0 fs to show blank space before the first frame (matches prior plotting convention),
    # but allow user overrides.
    if xmin is None:
        xmin_ = 0.0
    else:
        xmin_ = float(xmin)
    xmax_ = float(np.max(time_fs)) if xmax is None else float(xmax)
    ax.set_xlim(left=xmin_, right=xmax_)

    if ymin is not None or ymax is not None:
        ymin_ = float(np.min(qvector)) if ymin is None else float(ymin)
        ymax_ = float(np.max(qvector)) if ymax is None else float(ymax)
        ax.set_ylim(bottom=ymin_, top=ymax_)

    if zmin is not None or zmax is not None:
        zmin_ = float(np.min(pcd_tq)) if zmin is None else float(zmin)
        zmax_ = float(np.max(pcd_tq)) if zmax is None else float(zmax)
        ax.set_zlim(bottom=zmin_, top=zmax_)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.85, pad=0.08)
    cbar.set_label(zlabel)

    fig.tight_layout()
    fig.savefig(output_plot, bbox_inches="tight")
    plt.close(fig)


def plot_pcd_heatmap(
    time_fs: np.ndarray,
    qvector: np.ndarray,
    pcd_tq: np.ndarray,
    output_plot: str,
    zlabel: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    title: str | None,
    figsize: tuple[float, float],
    dpi: int,
    invert_q_axis: bool,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Heatmap with x=time, y=q to match 3D plot semantics and existing CLI args.
    extent = [
        float(np.min(time_fs)),
        float(np.max(time_fs)),
        float(np.min(qvector)),
        float(np.max(qvector)),
    ]

    im = ax.imshow(
        pcd_tq.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlabel("time (fs)")
    ax.set_ylabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    if title:
        ax.set_title(title)

    # Start at 0 fs to show blank space before the first frame (matches prior plotting convention),
    # but allow user overrides.
    if xmin is None:
        xmin_ = 0.0
    else:
        xmin_ = float(xmin)
    xmax_ = float(np.max(time_fs)) if xmax is None else float(xmax)
    ax.set_xlim(left=xmin_, right=xmax_)

    if ymin is not None or ymax is not None:
        ymin_ = float(np.min(qvector)) if ymin is None else float(ymin)
        ymax_ = float(np.max(qvector)) if ymax is None else float(ymax)
        ax.set_ylim(bottom=ymin_, top=ymax_)

    if invert_q_axis:
        ax.invert_yaxis()

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(zlabel)

    fig.tight_layout()
    fig.savefig(output_plot, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute scaled PCD for a (median) optimal trajectory XYZ and plot a heatmap or 3D surface."
    )
    parser.add_argument("input_xyz", help="Median optimal trajectory XYZ (can be a trajectory with many frames).")
    parser.add_argument(
        "--ref-dat",
        default="data/chd_reference.dat",
        help="Reference DAT file (q, I_ref). Default: data/chd_reference.dat",
    )
    parser.add_argument(
        "--excitation-factor",
        type=float,
        default=0.628,
        help="Excitation factor scaling applied to PCD (multiply). Default: 0.628",
    )
    parser.add_argument(
        "--inelastic",
        action="store_true",
        help="Include inelastic (Compton) scattering in IAM before forming PCD (requires Compton NPZ).",
    )
    parser.add_argument(
        "--dt-fs",
        type=float,
        default=20.0,
        help="Time per frame in fs. Default: 20",
    )
    parser.add_argument(
        "--t0-fs",
        type=float,
        default=10.0,
        help="Time of the first frame in fs. Default: 10",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Output PNG. Default: <input_basename>_pcd_heatmap.png (or _pcd_3d.png for --plot-type surface)",
    )
    parser.add_argument(
        "--output-dat",
        default=None,
        help="Optional output DAT with columns: q, PCD_frame1, PCD_frame2, ... (scaled).",
    )
    parser.add_argument(
        "--output-npz",
        default=None,
        help="Optional output NPZ containing q, time_fs, pcd_scaled (t,q).",
    )
    parser.add_argument(
        "--plot-type",
        choices=["heatmap", "surface"],
        default="heatmap",
        help='Plot type. "heatmap" is a 2D image with colorbar; "surface" is a 3D surface. Default: heatmap',
    )
    parser.add_argument("--view-elev", type=float, default=90.0, help="3D view elevation (deg). Default: 90 (top-down).")
    parser.add_argument("--view-azim", type=float, default=-90.0, help="3D view azimuth (deg). Default: -90.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name. Default: viridis")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale min (PCD). Default: auto")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale max (PCD). Default: auto")
    parser.add_argument("--zmin", type=float, default=None, help="Z-axis min (PCD). Default: auto")
    parser.add_argument("--zmax", type=float, default=None, help="Z-axis max (PCD). Default: auto")
    parser.add_argument("--xmin", type=float, default=None, help="X-axis min time (fs). Default: 0")
    parser.add_argument("--xmax", type=float, default=None, help="X-axis max time (fs). Default: last frame time")
    parser.add_argument("--ymin", type=float, default=None, help="Y-axis min q (A^-1). Default: min q")
    parser.add_argument("--ymax", type=float, default=None, help="Y-axis max q (A^-1). Default: max q")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument(
        "--figsize",
        type=str,
        default="10,6",
        help='Figure size as "W,H" in inches. Default: "10,6"',
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI. Default: 150")
    invert_group = parser.add_mutually_exclusive_group()
    invert_group.add_argument(
        "--invert-q-axis",
        dest="invert_q_axis",
        action="store_true",
        help="Invert the q axis direction in the plot (visual only).",
    )
    invert_group.add_argument(
        "--no-invert-q-axis",
        dest="invert_q_axis",
        action="store_false",
        help="Do not invert the q axis direction in the plot.",
    )
    parser.set_defaults(invert_q_axis=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_xyz):
        raise FileNotFoundError(f"Input XYZ not found: {args.input_xyz}")
    if not os.path.exists(args.ref_dat):
        raise FileNotFoundError(f"Reference DAT not found: {args.ref_dat}")
    if args.excitation_factor < 0:
        raise ValueError("excitation-factor must be >= 0")
    if args.dt_fs <= 0:
        raise ValueError("dt-fs must be > 0")

    ref_q, ref_I = load_reference_dat(args.ref_dat)
    qvector = ref_q  # compute IAM on the same q-grid as the reference

    structures = read_xyz_trajectory(args.input_xyz)
    n_frames = len(structures)
    time_fs = np.arange(n_frames, dtype=float) * float(args.dt_fs) + float(args.t0_fs)

    iam_tq = compute_iam_trajectory(structures, qvector, include_inelastic=args.inelastic)
    pcd_scaled_tq = compute_pcd_scaled(iam_tq, ref_I, excitation_factor=args.excitation_factor)

    base = os.path.splitext(os.path.basename(args.input_xyz))[0]
    if args.plot_type == "surface":
        default_plot = f"{base}_pcd_3d.png"
    else:
        default_plot = f"{base}_pcd_heatmap.png"
    output_plot = args.output_plot or default_plot

    if args.output_dat:
        out = np.column_stack([qvector] + [pcd_scaled_tq[i, :] for i in range(n_frames)])
        header = "q(A^-1)  " + "  ".join([f"PCD_scaled_{i+1}" for i in range(n_frames)])
        np.savetxt(args.output_dat, out, fmt="%.6e", header=header)

    if args.output_npz:
        np.savez_compressed(args.output_npz, q=qvector, time_fs=time_fs, pcd_scaled=pcd_scaled_tq)

    try:
        w_str, h_str = (x.strip() for x in str(args.figsize).split(","))
        figsize = (float(w_str), float(h_str))
    except Exception as e:
        raise ValueError(f'Invalid --figsize "{args.figsize}". Expected "W,H" (e.g. "10,6").') from e
    if figsize[0] <= 0 or figsize[1] <= 0:
        raise ValueError("--figsize values must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    zlabel = f"PCD × {args.excitation_factor:g} (%)"
    if args.plot_type == "surface":
        plot_pcd_surface(
            time_fs=time_fs,
            qvector=qvector,
            pcd_tq=pcd_scaled_tq,
            output_plot=output_plot,
            view_elev=args.view_elev,
            view_azim=args.view_azim,
            zlabel=zlabel,
            cmap=str(args.cmap),
            vmin=args.vmin,
            vmax=args.vmax,
            zmin=args.zmin,
            zmax=args.zmax,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
            title=args.title,
            figsize=figsize,
            dpi=int(args.dpi),
            invert_q_axis=bool(args.invert_q_axis),
        )
    else:
        plot_pcd_heatmap(
            time_fs=time_fs,
            qvector=qvector,
            pcd_tq=pcd_scaled_tq,
            output_plot=output_plot,
            zlabel=zlabel,
            cmap=str(args.cmap),
            vmin=args.vmin,
            vmax=args.vmax,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
            title=args.title,
            figsize=figsize,
            dpi=int(args.dpi),
            invert_q_axis=bool(args.invert_q_axis),
        )

    print(f"Wrote plot: {output_plot}")
    if args.output_dat:
        print(f"Wrote DAT:  {args.output_dat}")
    if args.output_npz:
        print(f"Wrote NPZ:  {args.output_npz}")


if __name__ == "__main__":
    main()

