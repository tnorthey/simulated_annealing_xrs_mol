"""Tests for per-chain GPU random pool sampling."""

import os
import tempfile

import numpy as np
import pytest

import modules.mol as mol
from modules.wrap import build_gpu_per_chain_start_batch


def _write_xyz(path: str, coords: np.ndarray) -> None:
    natoms = coords.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{natoms}\n")
        f.write("test\n")
        for i, (x, y, z) in enumerate(coords):
            f.write(f"C {x:.6f} {y:.6f} {z:.6f}\n")


def test_build_gpu_per_chain_start_batch_shape_and_variation():
    m = mol.Xyz()
    with tempfile.TemporaryDirectory() as tmp:
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        b = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        path_a = os.path.join(tmp, "a.xyz")
        path_b = os.path.join(tmp, "b.xyz")
        _write_xyz(path_a, a)
        _write_xyz(path_b, b)
        manifest = os.path.join(tmp, "pool.lst")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(f"{path_a}\n{path_b}\n")

        batch, picks = build_gpu_per_chain_start_batch(
            manifest, n_chains=8, mol_reader=m, seed=42
        )
        assert batch.shape == (8, 2, 3)
        assert len(picks) == 8
        assert all(p in (path_a, path_b) for p in picks)
        # With two distinct structures and 8 draws, both should appear
        assert path_a in picks and path_b in picks
        # Coordinates must match picked files
        for i, path in enumerate(picks):
            expected = a if path == path_a else b
            np.testing.assert_allclose(batch[i], expected)


def test_build_gpu_per_chain_start_batch_replacement_when_pool_smaller_than_chains():
    m = mol.Xyz()
    with tempfile.TemporaryDirectory() as tmp:
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        path = os.path.join(tmp, "only.xyz")
        _write_xyz(path, coords)
        manifest = os.path.join(tmp, "pool.lst")
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(f"{path}\n")

        batch, picks = build_gpu_per_chain_start_batch(
            manifest, n_chains=5, mol_reader=m, seed=0
        )
        assert batch.shape == (5, 1, 3)
        assert len(picks) == 5
        assert all(p == path for p in picks)


def test_build_gpu_per_chain_start_batch_empty_manifest_raises():
    m = mol.Xyz()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lst", delete=False) as f:
        manifest = f.name
    try:
        with pytest.raises(ValueError, match="empty"):
            build_gpu_per_chain_start_batch(manifest, n_chains=2, mol_reader=m)
    finally:
        os.unlink(manifest)
