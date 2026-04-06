#!/usr/bin/env python3
"""
average_xyz.py

Compute an average structure from many XYZ files (and/or XYZ trajectories).

Common use cases:
  - Average many single-frame XYZs into one mean XYZ
  - Average all frames across many trajectories into one mean XYZ
  - Average trajectories frame-by-frame across many runs (same number of frames)

Examples:
  # Average the first frame from many XYZ files (default)
  python3 average_xyz.py a.xyz b.xyz c.xyz -o mean.xyz

  # Average ALL frames from each file (treat every frame as a sample)
  python3 average_xyz.py traj1.xyz traj2.xyz --use-all-frames -o mean.xyz

  # Frame-by-frame average across trajectories (requires same #frames in each input)
  python3 average_xyz.py traj1.xyz traj2.xyz traj3.xyz --per-frame -o mean_traj.xyz

  # Kabsch-align each sample to the first sample before averaging (recommended)
  python3 average_xyz.py *.xyz --align kabsch -o mean_aligned.xyz

  # Align using only a subset of atoms (0-indexed)
  python3 average_xyz.py *.xyz --align kabsch --align-indices 0 1 2 3 -o mean_aligned.xyz
"""

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np


def read_xyz_trajectory(filename: str) -> List[Tuple[int, str, np.ndarray, np.ndarray]]:
    """
    Read an XYZ file that may contain multiple frames.
    Returns a list of (natoms, comment, atoms(str array), coords(N,3) float array).
    """
    structures = []
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                natoms = int(line)
            except ValueError:
                raise ValueError(f"Invalid XYZ format in {filename}: expected natoms, got {line!r}")

            comment = f.readline()
            if comment == "":
                raise ValueError(f"Invalid XYZ format in {filename}: missing comment line")
            comment = comment.rstrip("\n")

            atoms = []
            xyz = np.zeros((natoms, 3), dtype=np.float64)
            for i in range(natoms):
                row = f.readline()
                if row == "":
                    raise ValueError(f"Invalid XYZ format in {filename}: truncated atom block")
                parts = row.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid XYZ format in {filename}: bad atom line {row!r}")
                atoms.append(parts[0])
                xyz[i, 0] = float(parts[1])
                xyz[i, 1] = float(parts[2])
                xyz[i, 2] = float(parts[3])

            structures.append((natoms, comment, np.array(atoms, dtype=str), xyz))

    if not structures:
        raise ValueError(f"No valid structures found in {filename}")
    return structures


def write_xyz_trajectory(
    out_path: str,
    atoms: np.ndarray,
    frames: Sequence[np.ndarray],
    comment_prefix: str = "mean",
) -> None:
    natoms = int(len(atoms))
    with open(out_path, "w") as f:
        for i, xyz in enumerate(frames):
            f.write(f"{natoms}\n")
            f.write(f"{comment_prefix} frame={i}\n")
            for a, (x, y, z) in zip(atoms, xyz):
                f.write(f"{a} {x:.10f} {y:.10f} {z:.10f}\n")


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Returns rotation matrix R such that (P @ R) best matches Q in least squares sense.
    P and Q should already be centered.
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


def align_to_reference_kabsch(
    xyz: np.ndarray,
    ref: np.ndarray,
    indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Rigidly aligns xyz onto ref using the Kabsch algorithm.
    If indices is provided, the transform is fit using only those atoms,
    but applied to all atoms.
    """
    if indices is None:
        idx = np.arange(xyz.shape[0])
    else:
        idx = np.array(indices, dtype=int)

    P_sub = xyz[idx, :]
    Q_sub = ref[idx, :]

    P_mean = P_sub.mean(axis=0)
    Q_mean = Q_sub.mean(axis=0)

    Pc = P_sub - P_mean
    Qc = Q_sub - Q_mean

    R = _kabsch_rotation(Pc, Qc)
    aligned = (xyz - P_mean) @ R + Q_mean
    return aligned


def _validate_consistency(
    atoms_ref: np.ndarray,
    natoms_ref: int,
    natoms: int,
    atoms: np.ndarray,
    path: str,
) -> None:
    if natoms != natoms_ref:
        raise ValueError(
            f"Atom count mismatch for {path}: expected {natoms_ref}, got {natoms}. "
            "All inputs must have the same atom count."
        )
    if atoms.shape != atoms_ref.shape or not np.array_equal(atoms, atoms_ref):
        raise ValueError(
            f"Atom ordering/symbol mismatch for {path}. "
            "All inputs must have identical atom symbols in the same order."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average many XYZ files into a mean XYZ (optionally with alignment).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input XYZ files (single-frame or trajectory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mean.xyz",
        help="Output XYZ path (default: mean.xyz).",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="For trajectory inputs, which frame to use (default: 0). Ignored if --use-all-frames or --per-frame is set.",
    )
    parser.add_argument(
        "--use-all-frames",
        action="store_true",
        help="Use all frames from each input as samples (outputs a single averaged frame).",
    )
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Average trajectories frame-by-frame across inputs (outputs a trajectory). Requires each input to have the same number of frames.",
    )
    parser.add_argument(
        "--align",
        choices=["none", "kabsch"],
        default="none",
        help="Alignment to apply before averaging (default: none).",
    )
    parser.add_argument(
        "--reference-xyz",
        default=None,
        help="Optional reference XYZ to align to. If omitted and --align kabsch is used, aligns to the first sample.",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Which frame of --reference-xyz to use (default: 0).",
    )
    parser.add_argument(
        "--align-indices",
        type=int,
        nargs="+",
        default=None,
        help="Atom indices (0-indexed) to use for alignment fit (default: all atoms).",
    )
    parser.add_argument(
        "--stat",
        choices=["mean", "median"],
        default="mean",
        help="Statistic to compute across samples (default: mean).",
    )

    args = parser.parse_args()

    if args.use_all_frames and args.per_frame:
        parser.error("Choose only one of --use-all-frames or --per-frame.")

    for p in args.inputs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input not found: {p}")

    # Load reference (if provided)
    ref_atoms = None
    ref_xyz = None
    if args.reference_xyz is not None:
        ref_structs = read_xyz_trajectory(args.reference_xyz)
        if args.reference_frame < 0 or args.reference_frame >= len(ref_structs):
            raise ValueError(
                f"--reference-frame out of range: {args.reference_frame} (file has {len(ref_structs)} frame(s))"
            )
        natoms, comment, atoms, xyz = ref_structs[args.reference_frame]
        ref_atoms = atoms
        ref_xyz = xyz

    # Load inputs
    input_structs: List[List[Tuple[int, str, np.ndarray, np.ndarray]]] = []
    for path in args.inputs:
        input_structs.append(read_xyz_trajectory(path))

    # Determine atom reference from first available frame
    natoms0, _c0, atoms0, _xyz0 = input_structs[0][0]

    # Validate all files are consistent (at least for the frames we'll use)
    for path, structs in zip(args.inputs, input_structs):
        n, c, a, x = structs[0]
        _validate_consistency(atoms0, natoms0, n, a, path)

    # If reference xyz provided, validate it matches too
    if ref_xyz is not None and ref_atoms is not None:
        _validate_consistency(atoms0, natoms0, int(len(ref_atoms)), ref_atoms, args.reference_xyz)

    # Build reference sample for alignment if needed
    if args.align == "kabsch":
        if ref_xyz is None:
            ref_xyz = input_structs[0][0][3]

    if args.per_frame:
        # Ensure all have same number of frames
        nframes = len(input_structs[0])
        for path, structs in zip(args.inputs, input_structs):
            if len(structs) != nframes:
                raise ValueError(
                    f"--per-frame requires same number of frames for all inputs. "
                    f"{args.inputs[0]} has {nframes} frame(s), {path} has {len(structs)} frame(s)."
                )

        out_frames = []
        for fi in range(nframes):
            samples = []
            for structs in input_structs:
                xyz = structs[fi][3]
                if args.align == "kabsch":
                    xyz = align_to_reference_kabsch(xyz, ref_xyz, indices=args.align_indices)
                samples.append(xyz)
            stack = np.stack(samples, axis=0)
            if args.stat == "mean":
                out_frames.append(np.mean(stack, axis=0))
            else:
                out_frames.append(np.median(stack, axis=0))

        write_xyz_trajectory(
            args.output,
            atoms0,
            out_frames,
            comment_prefix=f"{args.stat} nfiles={len(args.inputs)}",
        )
        return

    # Otherwise: output a single averaged frame
    samples = []
    for structs in input_structs:
        if args.use_all_frames:
            for _n, _c, _a, xyz in structs:
                if args.align == "kabsch":
                    xyz = align_to_reference_kabsch(xyz, ref_xyz, indices=args.align_indices)
                samples.append(xyz)
        else:
            fi = int(args.frame_index)
            if fi < 0 or fi >= len(structs):
                raise ValueError(
                    f"--frame-index out of range: {fi} (one input has only {len(structs)} frame(s))"
                )
            xyz = structs[fi][3]
            if args.align == "kabsch":
                xyz = align_to_reference_kabsch(xyz, ref_xyz, indices=args.align_indices)
            samples.append(xyz)

    stack = np.stack(samples, axis=0)
    if args.stat == "mean":
        out_xyz = np.mean(stack, axis=0)
    else:
        out_xyz = np.median(stack, axis=0)
    write_xyz_trajectory(args.output, atoms0, [out_xyz], comment_prefix=f"{args.stat} nsamples={len(samples)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

