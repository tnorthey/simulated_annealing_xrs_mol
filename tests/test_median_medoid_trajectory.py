import os
import tempfile

import numpy as np


def _write_xyz(path: str, frames: list[tuple[str, np.ndarray]]):
    # frames: [(comment, coords (N,3))]
    atoms = ["H", "H", "H"]
    with open(path, "w") as f:
        for comment, xyz in frames:
            f.write(f"{len(atoms)}\n")
            f.write(f"{comment}\n")
            for sym, (x, y, z) in zip(atoms, xyz):
                f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")


def test_write_median_trajectory_as_medoid_picks_existing_frame():
    # Import inside test to avoid import-time side effects
    from compare_random_subsets import write_median_trajectory_as_medoid

    # Build 3 subset trajectories with 2 frames each.
    # Important: Kabsch RMSD removes translation/rotation, so frames must differ in *internal geometry*,
    # not just in absolute position. We use different bond lengths.
    #
    # For both frames, the medoid should be subset1 (middle geometry).
    def chain(scale: float):
        return np.array(
            [[0.0, 0.0, 0.0], [1.0 * scale, 0.0, 0.0], [2.0 * scale, 0.0, 0.0]],
            dtype=float,
        )

    with tempfile.TemporaryDirectory() as td:
        f0 = os.path.join(td, "s0.xyz")
        f1 = os.path.join(td, "s1.xyz")
        f2 = os.path.join(td, "s2.xyz")
        out = os.path.join(td, "median.xyz")

        _write_xyz(f0, [("frame0 s0", chain(1.0)), ("frame1 s0", chain(1.0))])
        _write_xyz(f1, [("frame0 s1", chain(1.1)), ("frame1 s1", chain(1.2))])
        _write_xyz(f2, [("frame0 s2", chain(2.0)), ("frame1 s2", chain(2.5))])

        ok = write_median_trajectory_as_medoid([f0, f1, f2], out)
        assert ok is True

        # Read output and assert the selected frames match subset1 exactly (by coordinates)
        coords = []
        with open(out, "r") as f:
            for _ in range(2):
                n = int(f.readline().strip())
                assert n == 3
                _comment = f.readline()
                xyz = np.loadtxt([f.readline() for _ in range(n)], usecols=[1, 2, 3])
                coords.append(xyz)

        np.testing.assert_allclose(coords[0], chain(1.1), atol=1e-8)
        np.testing.assert_allclose(coords[1], chain(1.2), atol=1e-8)


def test_write_median_trajectory_as_medoid_can_smooth_flicker():
    from compare_random_subsets import write_median_trajectory_as_medoid

    # Construct a 3-frame scenario where per-frame medoid would flicker:
    # - subset1 is stable (scale=2.0) and should be chosen by the smoothed DP
    # - subset0 has a single-frame excursion (scale=3.0 at frame1) which wins the per-frame medoid
    #   at that frame, causing a switch without smoothing.
    # - subset2 is an outlier (scale=10.0)
    def chain(scale: float):
        return np.array(
            [[0.0, 0.0, 0.0], [1.0 * scale, 0.0, 0.0], [2.0 * scale, 0.0, 0.0]],
            dtype=float,
        )

    with tempfile.TemporaryDirectory() as td:
        f0 = os.path.join(td, "s0.xyz")
        f1 = os.path.join(td, "s1.xyz")
        f2 = os.path.join(td, "s2.xyz")
        out0 = os.path.join(td, "median0.xyz")
        outS = os.path.join(td, "medianS.xyz")

        frames0 = [("f0 s0", chain(1.0)), ("f1 s0", chain(3.0)), ("f2 s0", chain(1.0))]
        frames1 = [("f0 s1", chain(2.0)), ("f1 s1", chain(2.0)), ("f2 s1", chain(2.0))]
        frames2 = [("f0 s2", chain(10.0)), ("f1 s2", chain(10.0)), ("f2 s2", chain(10.0))]

        _write_xyz(f0, frames0)
        _write_xyz(f1, frames1)
        _write_xyz(f2, frames2)

        # Without smoothing (per-frame medoid)
        write_median_trajectory_as_medoid([f0, f1, f2], out0, smooth_weight=0.0)
        # With smoothing: should avoid switching (high transition penalty)
        write_median_trajectory_as_medoid([f0, f1, f2], outS, smooth_weight=10.0)

        def read_scales(path: str):
            scales = []
            with open(path, "r") as f:
                for _ in range(3):
                    n = int(f.readline().strip())
                    assert n == 3
                    _ = f.readline()
                    xyz = np.loadtxt([f.readline() for _ in range(n)], usecols=[1, 2, 3])
                    # infer scale from bond length between atom0 and atom1
                    scales.append(float(np.linalg.norm(xyz[1] - xyz[0])))
            return scales

        s0 = read_scales(out0)
        sS = read_scales(outS)

        # Smoothed sequence should be less "flickery": fewer changes in scale
        changes0 = sum(1 for a, b in zip(s0, s0[1:]) if abs(a - b) > 1e-9)
        changesS = sum(1 for a, b in zip(sS, sS[1:]) if abs(a - b) > 1e-9)
        assert changesS <= changes0

