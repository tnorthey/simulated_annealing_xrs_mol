"""NPZ trajectory archive: prune ranking and geometry parity with xyz files."""

import os

import numpy as np
import pytest

import topM_geometry_statistics as geom
from modules import trajectory_archive as arch


Q = np.linspace(0.1, 1.0, 5)
ATOMS = ["C", "C", "C", "C"]


def _frame(bond: float, twist_deg: float) -> np.ndarray:
    xyz = np.zeros((4, 3), dtype=np.float64)
    xyz[1, 0] = bond
    xyz[2, 0] = bond
    xyz[2, 1] = 1.5
    ang = np.deg2rad(twist_deg)
    xyz[3, 0] = bond
    xyz[3, 1] = 1.5 + np.cos(ang)
    xyz[3, 2] = np.sin(ang)
    return xyz


def _batch(frames):
    """frames: list of (fit, bond, twist)."""
    fits = np.array([fit for fit, _bond, _twist in frames], dtype=np.float64)
    xyz = np.stack([_frame(bond, twist) for _fit, bond, twist in frames], axis=0)
    xyz[:, 0, 2] = fits
    signal = fits[:, None] * Q[None, :]
    return fits, xyz, signal


def _write_chain_xyz(directory, timestep: int, fit: float, coords: np.ndarray) -> str:
    prefix = f"{timestep:02d}"
    path = os.path.join(directory, f"{prefix}_{arch.format_fit_stub(fit)}.xyz")
    arch._write_xyz(path, ATOMS, coords, f"{fit:.8f}")
    return path


def test_merge_keeps_lowest_fit_and_other_timesteps(tmp_path):
    results = tmp_path / "results_run"
    results.mkdir()
    fits, xyz, signal = _batch(
        [(0.30, 1.4, 10.0), (0.10, 1.5, 20.0), (0.20, 1.6, 30.0)]
    )
    arch.merge_and_prune(
        str(results),
        timestep=0,
        f_xray=fits,
        xyz=xyz,
        signal=signal,
        q=Q,
        atoms=ATOMS,
        top_m=2,
    )
    fits1, xyz1, signal1 = _batch([(0.50, 1.7, 40.0), (0.40, 1.8, 50.0)])
    arch.merge_and_prune(
        str(results),
        timestep=1,
        f_xray=fits1,
        xyz=xyz1,
        signal=signal1,
        q=Q,
        atoms=ATOMS,
        top_m=2,
    )
    better, xyz_b, signal_b = _batch([(0.05, 1.1, 5.0)])
    arch.merge_and_prune(
        str(results),
        timestep=0,
        f_xray=better,
        xyz=xyz_b,
        signal=signal_b,
        q=Q,
        atoms=ATOMS,
        top_m=2,
    )

    archive = arch.archive_path(str(results))
    assert archive == os.path.join(str(results), "trajectory.npz")
    assert os.path.isfile(archive)
    data = arch._load(archive)
    t0 = np.sort(data["f_xray"][data["t"] == 0])
    t1 = np.sort(data["f_xray"][data["t"] == 1])
    assert t0.tolist() == pytest.approx([0.05, 0.10])
    assert t1.tolist() == pytest.approx([0.40, 0.50])
    for i in range(data["t"].shape[0]):
        assert data["xyz"][i, 0, 2] == pytest.approx(float(data["f_xray"][i]))
        assert np.allclose(data["signal"][i], data["f_xray"][i] * data["q"])

    with pytest.raises(ValueError, match="q grid"):
        arch.merge_and_prune(
            str(results),
            timestep=1,
            f_xray=fits1[:1],
            xyz=xyz1[:1],
            signal=signal1[:1],
            q=Q * 2,
            atoms=ATOMS,
            top_m=2,
        )


def test_archive_geometry_matches_xyz_files(tmp_path):
    results = tmp_path / "results_run"
    results.mkdir()
    frames = [
        (0, [(0.30, 1.40, 10.0), (0.10, 1.50, 20.0), (0.20, 1.60, 30.0)]),
        (1, [(0.50, 1.70, 40.0), (0.40, 1.80, 50.0)]),
    ]
    for timestep, rows in frames:
        fits, xyz, signal = _batch(rows)
        for fit, coords in zip(fits, xyz):
            _write_chain_xyz(str(results), timestep, float(fit), coords)
        arch.merge_and_prune(
            str(results),
            timestep=timestep,
            f_xray=fits,
            xyz=xyz,
            signal=signal,
            q=Q,
            atoms=ATOMS,
            top_m=3,
        )

    file_t, file_layers = geom.load_topM_candidates(str(results), 2)
    arch_t, arch_layers = geom.load_topM_candidates_from_archive(
        arch.archive_path(str(results)), 2
    )
    assert file_t == arch_t == [0, 1]
    for file_layer, arch_layer in zip(file_layers, arch_layers):
        assert [c["fit"] for c in file_layer] == pytest.approx(
            [c["fit"] for c in arch_layer]
        )
        file_geom = geom.compute_geometry_for_layer(
            file_layer, bond=(0, 1), dihedral=(0, 1, 2, 3)
        )
        arch_geom = geom.compute_geometry_for_layer(
            arch_layer, bond=(0, 1), dihedral=(0, 1, 2, 3)
        )
        assert np.allclose(file_geom, arch_geom)


def test_delete_chain_files_keeps_mean_target_and_start(tmp_path):
    results = tmp_path / "results_run"
    results.mkdir()
    kept = [
        "00_mean.xyz",
        "00_target.xyz",
        "00_start_sampled.xyz",
        "01_000.20000000.xyz",
    ]
    removed_names = ["00_000.10000000.xyz", "00_000.10000000.dat", "00_000.30000000_dup1.xyz"]
    for name in kept + removed_names:
        (results / name).write_text("x", encoding="utf-8")
    n_removed = arch.delete_chain_files(str(results), 0)
    assert n_removed == 3
    for name in kept:
        assert (results / name).is_file()
    for name in removed_names:
        assert not (results / name).exists()
    assert arch.count_chain_xyz(str(results)) == 1


def test_materialize_writes_lowest_fit_first(tmp_path):
    results = tmp_path / "results_run"
    results.mkdir()
    fits, xyz, signal = _batch(
        [(0.30, 1.40, 10.0), (0.10, 1.50, 20.0), (0.20, 1.60, 30.0)]
    )
    arch.merge_and_prune(
        str(results),
        timestep=0,
        f_xray=fits,
        xyz=xyz,
        signal=signal,
        q=Q,
        atoms=ATOMS,
        top_m=3,
    )
    dest = tmp_path / "_npz_pool"
    paths = arch.materialize_timestep(str(results), "00", str(dest), top_n=2)
    assert len(paths) == 2
    first = geom.read_xyz_coords(paths[0])
    second = geom.read_xyz_coords(paths[1])
    assert first[1, 0] == pytest.approx(1.50)
    assert second[1, 0] == pytest.approx(1.60)


def test_coords_table_summarizes_topm_c1c6(tmp_path):
    results = tmp_path / "results_run"
    results.mkdir()
    atoms = ["C", "C", "C", "C", "C", "C"]
    q = np.linspace(0.1, 1.0, 4)

    def structure(bond: float, twist: float) -> np.ndarray:
        xyz = np.zeros((6, 3), dtype=np.float64)
        xyz[1] = [1.0, 0.0, 0.0]
        xyz[2] = [1.0, 1.0, 0.0]
        xyz[3] = [0.0, 1.0, 0.0]
        ang = np.deg2rad(twist)
        xyz[4] = [0.0, 1.0 + np.cos(ang), np.sin(ang)]
        xyz[5, 0] = bond
        return xyz

    frames = [(0.30, 1.50, 0.0), (0.10, 2.50, 90.0), (0.90, 9.00, 0.0)]
    fits = np.array([fit for fit, _bond, _twist in frames], dtype=np.float64)
    xyz = np.stack([structure(bond, twist) for _fit, bond, twist in frames], axis=0)
    signal = fits[:, None] * q[None, :]
    arch.merge_and_prune(
        str(results),
        timestep=3,
        f_xray=fits,
        xyz=xyz,
        signal=signal,
        q=q,
        atoms=atoms,
        top_m=3,
    )
    time_file = tmp_path / "time.dat"
    time_file.write_text("12.5\n", encoding="utf-8")
    out = tmp_path / "analysis"
    written = geom.write_coords_tables(
        directory=str(results),
        archive=arch.archive_path(str(results)),
        top_m=2,
        time_file=str(time_file),
        output_dir=str(out),
    )
    assert [os.path.basename(path) for path in written] == ["coords_table_03.dat"]
    text = (out / "coords_table_03.dat").read_text(encoding="utf-8")
    assert "# time_fs 12.5" in text
    assert "# topM 2" in text
    assert "# n 2" in text
    assert "dihedral_0-1-4-5" in text
    assert "dihedral_1-2-3-4" in text
    bond_row = next(line for line in text.splitlines() if line.startswith("C1-C6 "))
    median, mean, lo, hi, span = [float(value) for value in bond_row.split()[1:]]
    assert median == pytest.approx(2.0)
    assert mean == pytest.approx(2.0)
    assert lo == pytest.approx(1.5)
    assert hi == pytest.approx(2.5)
    assert span == pytest.approx(1.0)


def test_configured_archive_top_m(tmp_path):
    cfg = tmp_path / "input.toml"
    cfg.write_text(
        "[options]\nnpz_mode_bool = true\narchive_top_m = 15\n",
        encoding="utf-8",
    )
    assert arch.configured_archive_top_m(str(cfg)) == 15
    assert arch.npz_mode_enabled(str(cfg), str(tmp_path / "missing")) is True
    cfg.write_text(
        "[options]\nnpz_mode_bool = false\narchive_top_m = 15\n",
        encoding="utf-8",
    )
    assert arch.configured_archive_top_m(str(cfg)) is None
    assert arch.npz_mode_enabled(str(cfg), str(tmp_path / "missing")) is False
