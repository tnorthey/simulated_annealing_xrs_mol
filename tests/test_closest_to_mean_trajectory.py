import os
import tempfile

import numpy as np


def _write_xyz(path: str, frames: list[tuple[str, np.ndarray]]):
    atoms = ["H", "H", "H"]
    with open(path, "w") as f:
        for comment, xyz in frames:
            f.write(f"{len(atoms)}\n")
            f.write(f"{comment}\n")
            for sym, (x, y, z) in zip(atoms, xyz):
                f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")


def test_write_closest_to_mean_trajectory_per_frame_selects_existing_frames():
    from compare_random_subsets import write_closest_to_mean_trajectory_per_frame

    def chain(scale: float):
        return np.array(
            [[0.0, 0.0, 0.0], [1.0 * scale, 0.0, 0.0], [2.0 * scale, 0.0, 0.0]],
            dtype=float,
        )

    with tempfile.TemporaryDirectory() as td:
        s0 = os.path.join(td, "s0.xyz")
        s1 = os.path.join(td, "s1.xyz")
        mean = os.path.join(td, "mean.xyz")
        out = os.path.join(td, "closest.xyz")

        # Two frames. Mean is closer to s0 at frame0 and closer to s1 at frame1.
        _write_xyz(s0, [("f0 s0", chain(1.0)), ("f1 s0", chain(1.0))])
        _write_xyz(s1, [("f0 s1", chain(2.0)), ("f1 s1", chain(3.0))])
        _write_xyz(mean, [("f0 mean", chain(1.1)), ("f1 mean", chain(2.9))])

        ok = write_closest_to_mean_trajectory_per_frame([s0, s1], mean, out)
        assert ok is True

        # Read output and confirm it picked s0 then s1.
        coords = []
        with open(out, "r") as f:
            for _ in range(2):
                n = int(f.readline().strip())
                assert n == 3
                _ = f.readline()
                xyz = np.loadtxt([f.readline() for _ in range(n)], usecols=[1, 2, 3])
                coords.append(xyz)

        np.testing.assert_allclose(coords[0], chain(1.0), atol=1e-8)
        np.testing.assert_allclose(coords[1], chain(3.0), atol=1e-8)

