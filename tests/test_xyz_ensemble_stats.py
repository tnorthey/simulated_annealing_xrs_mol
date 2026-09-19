"""Tests for scripts/python/xyz_ensemble_stats.py"""
import os
import numpy as np

import xyz_ensemble_stats as ens


def _write_xyz(path: str, coords: np.ndarray, comment: str = "test") -> None:
    atoms = ["C"] * coords.shape[0]
    lines = [str(len(atoms)), comment]
    for atom, (x, y, z) in zip(atoms, coords):
        lines.append(f"{atom} {x:.10f} {y:.10f} {z:.10f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_circular_stats_clustered():
    angles = np.array([-10.0, 0.0, 10.0])
    mean = ens.circular_mean_deg(angles)
    median = ens.circular_median_deg(angles)
    amin, amax, crange = ens.circular_min_max_range_deg(angles)
    np.testing.assert_allclose(mean, 0.0, atol=1e-8)
    np.testing.assert_allclose(median, 0.0, atol=1e-8)
    np.testing.assert_allclose(amin, -10.0, atol=1e-8)
    np.testing.assert_allclose(amax, 10.0, atol=1e-8)
    np.testing.assert_allclose(crange, 20.0, atol=1e-8)


def test_circular_stats_across_cut():
    angles = np.array([170.0, 180.0, -170.0])
    mean = ens.circular_mean_deg(angles)
    amin, amax, crange = ens.circular_min_max_range_deg(angles)
    np.testing.assert_allclose(mean, 180.0, atol=1e-6)
    np.testing.assert_allclose(crange, 20.0, atol=1e-6)
    np.testing.assert_allclose(amin, 170.0, atol=1e-6)
    np.testing.assert_allclose(amax, -170.0, atol=1e-6)


def test_ensemble_stats_on_shifted_structures(tmp_path):
    # 6-atom chain: bond 0-5 = 5 Å, dihedral 0-1-4-5 is planar (0° or 180°).
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    a = target.copy()
    a[5, 0] = 5.2
    b = target.copy()
    b[5, 0] = 4.8

    target_path = str(tmp_path / "mol_target.xyz")
    a_path = str(tmp_path / "mol_a.xyz")
    b_path = str(tmp_path / "mol_b.xyz")
    _write_xyz(target_path, target)
    _write_xyz(a_path, a)
    _write_xyz(b_path, b)

    out = str(tmp_path / "stats.dat")
    rc = ens.main(
        [
            a_path,
            b_path,
            target_path,
            "--target",
            target_path,
            "--bond",
            "0",
            "5",
            "--dihedral",
            "0",
            "1",
            "4",
            "5",
            "-o",
            out,
        ]
    )
    assert rc == 0
    assert os.path.isfile(out)

    text = open(out).read()
    assert "bond_0-5" in text
    assert "dihedral_0-1-4-5" in text
    assert "rmsd_kabsch" in text
    assert "# SUMMARY" in text
    data_block = text.split("# SUMMARY")[0]
    assert "mol_a.xyz" in data_block
    assert "mol_b.xyz" in data_block
    data_lines = [line for line in data_block.splitlines() if line and not line.startswith("#")]
    assert len(data_lines) == 2
    assert all("mol_target.xyz" not in line for line in data_lines)

    n_target, _atoms, coords_t = ens.read_xyz(target_path)
    n_a, _atoms_a, coords_a = ens.read_xyz(a_path)
    assert n_target == 6 and n_a == 6
    bond_a = ens.compute_geometry(coords_a, [[0, 5]], [])[0]
    bond_b = ens.compute_geometry(b, [[0, 5]], [])[0]
    np.testing.assert_allclose(bond_a, 5.2, atol=1e-8)
    np.testing.assert_allclose(bond_b, 4.8, atol=1e-8)
    rmsd_self = ens.kabsch_rmsd(coords_t, coords_t, list(range(6)))
    np.testing.assert_allclose(rmsd_self, 0.0, atol=1e-10)
