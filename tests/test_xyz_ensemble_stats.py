"""Tests for scripts/python/xyz_ensemble_stats.py"""
import os
import numpy as np
import pytest

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
            "--qmax",
            "8",
            "--open-bonds",
            "c1c6",
        ]
    )
    assert rc == 0
    assert os.path.isfile(out)
    tex_path = os.path.join(os.path.dirname(out), f"{os.path.basename(str(tmp_path))}_table.tex")
    assert os.path.isfile(tex_path)
    tex = open(tex_path).read()
    assert r"\begin{table}[htbp]" in tex
    assert r"\ce{C1-C6}" in tex
    assert "8 &" in tex
    assert r"\footnotesize(" in tex
    assert tex.count(r"\\") >= 4

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


def test_infer_qmax_and_open_bonds():
    assert ens.infer_qmax("results_fig3_qmax8_open_phi0p5") == 8
    assert ens.infer_qmax("results_single_target_qmax4_c1c6_closed") == 4
    assert ens.infer_open_bonds_tex("results_fig3_qmax8_open_phi0p5") == r"\ce{C1-C6}"
    assert ens.infer_open_bonds_tex("results_single_target_qmax4_c1c6_closed") == "None"


def test_write_table_tex_one_row(tmp_path):
    path = str(tmp_path / "results_fig3_qmax8_open_phi0p5_table.tex")
    ens.write_table_tex(
        path,
        qmax=8,
        open_bonds=r"\ce{C1-C6}",
        rmsd_median=0.38,
        rmsd_min=0.11,
        rmsd_max=0.48,
        bond_median=2.23,
        bond_min=2.18,
        bond_max=2.31,
        dih0145_median=47.49,
        dih0145_min=40.0,
        dih0145_max=57.0,
        dih1234_median=42.6,
        dih1234_min=-17.0,
        dih1234_max=58.0,
        target_bond=2.22,
        target_dih0145=46.7,
        target_dih1234=-3.1,
    )
    tex = open(path).read()
    assert r"\begin{table}[htbp]" in tex
    assert r"\label{tab:isotropic_median_rmsd_c1c6_dihedral}" in tex
    assert r"8 & \ce{C1-C6} & 0.38 & 2.23 & 47.49 & 42.60\\" in tex
    assert (
        r"\footnotesize(0.11, 0.48) & \footnotesize(2.18, 2.31) & "
        r"\footnotesize(40, 57) & \footnotesize(-17, 58)\\" in tex
    )
    assert r"\SI{46.7}{\degree}" in tex
    assert r"\SI{-3.1}{\degree}" in tex
    assert tex.count(r"\midrule") == 1
    # One experimental row pair only (median + range), not the 4-condition table.
    assert tex.split(r"\midrule", 1)[1].count("&") == 10


def _chain(bond_0_5: float) -> np.ndarray:
    coords = np.zeros((6, 3))
    coords[:, 0] = np.arange(6, dtype=float)
    coords[5, 0] = bond_0_5
    return coords


def test_chi2_ratio_lower_quartile_prints_tex_rows(tmp_path, capsys):
    target = _chain(5.0)
    structures = [
        ("low.xyz", 1.0, 5.2),
        ("mid_a.xyz", 2.0, 6.0),
        ("mid_b.xyz", 3.0, 7.0),
        ("high.xyz", 4.0, 8.0),
    ]
    target_path = str(tmp_path / "mol_target.xyz")
    _write_xyz(target_path, target, comment="0.0 0.0")
    paths = []
    for name, chi2, bond in structures:
        path = str(tmp_path / name)
        _write_xyz(path, _chain(bond), comment=f"{chi2:.1f} 0.1")
        paths.append(path)

    out = str(tmp_path / "stats.dat")
    rc = ens.main(
        paths
        + [
            target_path,
            "--target",
            target_path,
            "--rmsd-indices",
            "0,1,2,3,4,5",
            "--chi2-ratio",
            "0.25",
            "--print-tex-rows",
            "--qmax",
            "4",
            "--open-bonds",
            "none",
            "-o",
            out,
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 2
    assert printed[0].startswith(r"4 & None &")
    assert "5.20" in printed[0]
    assert r"\footnotesize" in printed[1]
    assert printed[0].endswith(r"\\")
    assert printed[1].endswith(r"\\")

    stats = open(out).read()
    assert "low.xyz" in stats
    assert "mid_a.xyz" not in stats.split("# SUMMARY")[0]
    assert "high.xyz" not in stats.split("# SUMMARY")[0]
    assert "chi2_ratio = 0.25" in stats

    tex_path = os.path.join(str(tmp_path), f"{os.path.basename(str(tmp_path))}_table.tex")
    tex = open(tex_path).read()
    assert r"lower quartile" in tex
    assert r"\chi^2 < 10^{-3}" not in tex

    rc = ens.main(
        paths
        + [
            target_path,
            "--target",
            target_path,
            "--rmsd-indices",
            "0,1,2,3,4,5",
            "--chi2-ratio",
            "0.25",
            "--print-tex-rows",
            "--qmax",
            "8",
            "--open-bonds",
            "c1c6",
            "-o",
            out,
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out.strip().splitlines()
    assert printed[0].startswith(r"8 & \ce{C1-C6} &")
    assert r"\footnotesize" in printed[1]


def test_chi2_ratio_rejects_other_values():
    with pytest.raises(SystemExit, match="chi2-ratio"):
        ens.main(["missing.xyz", "--chi2-ratio", "0.1"])
