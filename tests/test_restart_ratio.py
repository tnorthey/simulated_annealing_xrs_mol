"""Unit tests for select_restart_batch (restart_ratio tiling + carried scores)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.wrap import select_restart_batch


def _fake_phase(n_chains=100, natoms=3, qlen=8):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(n_chains, natoms, 3))
    # Distinct, sortable scores: chain i has f = i (so lowest-f are 0..K-1)
    f = np.arange(n_chains, dtype=np.float64)
    fx = f * 0.5
    pred = np.arange(n_chains, dtype=np.float64)[:, None] + np.linspace(
        0.0, 1.0, qlen
    )
    return xyz, f, fx, pred


@pytest.mark.unit
def test_restart_ratio_pool_and_tile():
    n = 100
    xyz, f, fx, pred = _fake_phase(n_chains=n)
    xyz_b, f_b, fx_b, pred_b, k = select_restart_batch(
        xyz, f, fx, pred, 0.1
    )
    assert k == 10
    assert xyz_b.shape == (n, 3, 3)
    assert f_b.shape == (n,)
    # Only geometries from chains 0..9 (lowest f)
    for i in range(n):
        src = i % 10
        np.testing.assert_allclose(xyz_b[i], xyz[src])
        assert f_b[i] == pytest.approx(f[src])
        assert fx_b[i] == pytest.approx(fx[src])
        np.testing.assert_allclose(pred_b[i], pred[src])


@pytest.mark.unit
def test_restart_ratio_one_uses_full_set_sorted():
    n = 8
    xyz, f, fx, pred = _fake_phase(n_chains=n)
    # Shuffle scores so sort order != identity
    f = np.array([5.0, 1.0, 7.0, 0.0, 3.0, 6.0, 2.0, 4.0])
    fx = f * 0.5
    xyz_b, f_b, fx_b, pred_b, k = select_restart_batch(xyz, f, fx, pred, 1.0)
    assert k == n
    order = np.argsort(f, kind="mergesort")
    for i in range(n):
        np.testing.assert_allclose(xyz_b[i], xyz[order[i]])
        assert f_b[i] == pytest.approx(f[order[i]])
    # Set of XYZs equals full previous set
    assert {tuple(x.ravel()) for x in xyz_b} == {tuple(x.ravel()) for x in xyz}


@pytest.mark.unit
def test_restart_force_k_one_global_best():
    n = 16
    xyz, f, fx, pred = _fake_phase(n_chains=n)
    f = np.linspace(10.0, 0.0, n)  # best is last index
    fx = f * 0.5
    best = int(np.argmin(f))
    xyz_b, f_b, fx_b, pred_b, k = select_restart_batch(
        xyz, f, fx, pred, 1.0, force_k=1
    )
    assert k == 1
    for i in range(n):
        np.testing.assert_allclose(xyz_b[i], xyz[best])
        assert f_b[i] == pytest.approx(f[best])
        assert fx_b[i] == pytest.approx(fx[best])
        np.testing.assert_allclose(pred_b[i], pred[best])


@pytest.mark.unit
def test_tiled_clones_share_score_bar():
    n = 20
    xyz, f, fx, pred = _fake_phase(n_chains=n)
    xyz_b, f_b, fx_b, pred_b, k = select_restart_batch(xyz, f, fx, pred, 0.2)
    assert k == 4
    # Threads 0 and 4 both map to pool[0]
    assert f_b[0] == f_b[4]
    assert fx_b[0] == fx_b[4]
    np.testing.assert_allclose(xyz_b[0], xyz_b[4])
    np.testing.assert_allclose(pred_b[0], pred_b[4])


@pytest.mark.unit
def test_restart_ranks_by_f_xray_not_total_f():
    """A low-MM / high-χ² chain must not crowd out the best χ² chain."""
    n = 4
    natoms = 3
    qlen = 8
    xyz = np.zeros((n, natoms, 3), dtype=np.float64)
    for i in range(n):
        xyz[i, 0, 0] = float(i)
    # Chain 2 has the best χ² but the worst total f (high MM).
    f = np.array([1.0, 2.0, 50.0, 3.0])
    fx = np.array([8.0, 7.0, 0.001, 6.0])
    pred = np.arange(n, dtype=np.float64)[:, None] + np.linspace(0.0, 1.0, qlen)
    xyz_b, f_b, fx_b, pred_b, k = select_restart_batch(xyz, f, fx, pred, 0.25)
    assert k == 1
    np.testing.assert_allclose(xyz_b[0], xyz[2])
    assert fx_b[0] == pytest.approx(0.001)
    assert f_b[0] == pytest.approx(50.0)
    for i in range(n):
        np.testing.assert_allclose(xyz_b[i], xyz[2])


@pytest.mark.unit
def test_restart_ratio_rejects_invalid():
    xyz, f, fx, pred = _fake_phase(n_chains=4)
    with pytest.raises(ValueError, match="restart_ratio"):
        select_restart_batch(xyz, f, fx, pred, 0.0)
    with pytest.raises(ValueError, match="restart_ratio"):
        select_restart_batch(xyz, f, fx, pred, 1.5)
