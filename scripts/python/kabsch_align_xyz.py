#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class XyzFrame:
    comment: str
    atoms: List[str]
    xyz: np.ndarray  # shape (natom, 3), float


def _parse_indices(expr: str, natom: int, one_based: bool) -> List[int]:
    """
    Parse indices like:
      - "0,1,2"
      - "0-5"
      - "0-5,8,10-12"
    """
    expr = expr.strip()
    if not expr:
        return list(range(natom))

    out: List[int] = []
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    for p in parts:
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", p)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            if one_based:
                a -= 1
                b -= 1
            if a > b:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            idx = int(p)
            if one_based:
                idx -= 1
            out.append(idx)

    uniq = sorted(set(out))
    for i in uniq:
        if i < 0 or i >= natom:
            raise ValueError(f"Index {i} out of range for natom={natom}")
    return uniq


def read_xyz_trajectory(path: str) -> List[XyzFrame]:
    frames: List[XyzFrame] = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                natom = int(line.strip())
            except ValueError as e:
                raise ValueError(f"Failed to parse natom line: {line!r}") from e

            comment = f.readline()
            if comment == "":
                raise ValueError("Unexpected EOF while reading comment line")
            comment = comment.rstrip("\n")

            atoms: List[str] = []
            xyz = np.zeros((natom, 3), dtype=float)
            for i in range(natom):
                row = f.readline()
                if row == "":
                    raise ValueError("Unexpected EOF while reading atom lines")
                fields = row.split()
                if len(fields) < 4:
                    raise ValueError(f"Bad XYZ atom line (need >=4 fields): {row!r}")
                atoms.append(fields[0])
                xyz[i, 0] = float(fields[1])
                xyz[i, 1] = float(fields[2])
                xyz[i, 2] = float(fields[3])

            frames.append(XyzFrame(comment=comment, atoms=atoms, xyz=xyz))

    if not frames:
        raise ValueError(f"No frames found in {path!r}")

    nat0 = len(frames[0].atoms)
    at0 = frames[0].atoms
    for fi, fr in enumerate(frames):
        if len(fr.atoms) != nat0:
            raise ValueError(f"Inconsistent natom at frame {fi}: {len(fr.atoms)} vs {nat0}")
        if fr.atoms != at0:
            raise ValueError(
                f"Atom labels differ at frame {fi}. "
                "This script currently requires identical ordering across frames."
            )
        if fr.xyz.shape != (nat0, 3):
            raise ValueError(f"Bad xyz shape at frame {fi}: {fr.xyz.shape}")

    return frames


def write_xyz_trajectory(path: str, frames: Sequence[XyzFrame]) -> None:
    if not frames:
        raise ValueError("No frames to write")
    natom = len(frames[0].atoms)
    with open(path, "w") as f:
        for fr in frames:
            f.write(f"{natom}\n")
            f.write(f"{fr.comment}\n")
            for a, (x, y, z) in zip(fr.atoms, fr.xyz):
                f.write(f"{a:2s} {x: .10f} {y: .10f} {z: .10f}\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Kabsch-align each frame of an XYZ trajectory to a reference frame."
    )
    p.add_argument("input_xyz", help="Input XYZ trajectory (multi-frame XYZ).")
    p.add_argument("output_xyz", help="Output aligned XYZ trajectory.")
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Align frame t to the (aligned) frame t-1 (overrides --ref-frame).",
    )
    p.add_argument(
        "--ref-frame",
        type=int,
        default=0,
        help="Reference frame index to align to (default: 0).",
    )
    p.add_argument(
        "--indices",
        default="",
        help='Atom indices to use for alignment (default: all). Example: "0-5,8,10-12".',
    )
    p.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret --indices as 1-based (e.g. '1-3' means atoms 0..2).",
    )
    p.add_argument(
        "--annotate-rmsd",
        action="store_true",
        help="Append per-frame RMSD (Å) to the comment line.",
    )
    args = p.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from modules.mol import Xyz  # noqa: E402

    frames = read_xyz_trajectory(args.input_xyz)
    nframe = len(frames)
    natom = len(frames[0].atoms)

    indices = _parse_indices(args.indices, natom=natom, one_based=args.one_based)
    if len(indices) == 0:
        raise ValueError("No indices selected for alignment")

    xyz_util = Xyz()

    aligned: List[XyzFrame] = []
    if args.sequential:
        prev_aligned = frames[0]
        aligned.append(prev_aligned)
        for t in range(1, nframe):
            fr = frames[t]

            ref_sel = prev_aligned.xyz[indices, :]
            ref_centroid = ref_sel.mean(axis=0)
            mov_sel = fr.xyz[indices, :]
            mov_centroid = mov_sel.mean(axis=0)

            rmsd, R = xyz_util.rmsd_kabsch(fr.xyz, prev_aligned.xyz, indices)
            xyz_aligned = (fr.xyz - mov_centroid) @ R + ref_centroid

            comment = fr.comment
            if args.annotate_rmsd:
                suffix = f" | rmsd_kabsch={rmsd:.6f}A ref=t-1 idx={len(indices)}"
                comment = (comment + suffix).strip()

            prev_aligned = XyzFrame(comment=comment, atoms=fr.atoms, xyz=xyz_aligned)
            aligned.append(prev_aligned)
    else:
        if args.ref_frame < 0 or args.ref_frame >= nframe:
            raise ValueError(
                f"--ref-frame {args.ref_frame} out of range [0, {nframe-1}]"
            )

        ref = frames[args.ref_frame]
        ref_sel = ref.xyz[indices, :]
        ref_centroid = ref_sel.mean(axis=0)

        for fr in frames:
            mov_sel = fr.xyz[indices, :]
            mov_centroid = mov_sel.mean(axis=0)

            rmsd, R = xyz_util.rmsd_kabsch(fr.xyz, ref.xyz, indices)
            xyz_aligned = (fr.xyz - mov_centroid) @ R + ref_centroid

            comment = fr.comment
            if args.annotate_rmsd:
                suffix = (
                    f" | rmsd_kabsch={rmsd:.6f}A ref={args.ref_frame} idx={len(indices)}"
                )
                comment = (comment + suffix).strip()

            aligned.append(XyzFrame(comment=comment, atoms=fr.atoms, xyz=xyz_aligned))

    write_xyz_trajectory(args.output_xyz, aligned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

