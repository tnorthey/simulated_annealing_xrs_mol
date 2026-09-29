"""Unit tests for select_restart_batch (restart_ratio tiling + carried scores)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.wrap import select_adaptive_restart_batch, select_restart_batch


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


def _adaptive(xyz, fx, pred, ratio):
    return select_adaptive_restart_batch(xyz, fx, pred, ratio)


@pytest.mark.unit
def test_adaptive_flat_chi2_keeps_every_chain():
    """A flat bad population must not be collapsed onto its least-bad members."""
    n = 32
    xyz, _f, fx, pred = _fake_phase(n_chains=n)
    fx = np.full(n, 5.0, dtype=np.float64)
    xyz_b, fx_b, pred_b, k, n_pass, median_fx, cutoff, mode = _adaptive(
        xyz, fx, pred, 0.2
    )
    assert mode == "keep_all_no_separation"
    assert n_pass == 0
    assert k == n
    assert median_fx == pytest.approx(5.0)
    assert cutoff == pytest.approx(2.5)
    np.testing.assert_allclose(xyz_b, xyz)
    np.testing.assert_allclose(fx_b, fx)
    np.testing.assert_allclose(pred_b, pred)


@pytest.mark.unit
def test_adaptive_near_miss_does_not_use_absolute_cutoff():
    """χ² of 5.3 with a median near 6 is not a separated elite."""
    n = 20
    xyz, _f, _fx, pred = _fake_phase(n_chains=n)
    fx = np.linspace(5.3, 8.0, n)
    xyz_b, fx_b, _pred_b, k, n_pass, _median, cutoff, mode = _adaptive(
        xyz, fx, pred, 0.2
    )
    assert mode == "keep_all_no_separation"
    assert n_pass == 0
    assert k == n
    assert float(np.min(fx)) > cutoff
    np.testing.assert_allclose(xyz_b, xyz)
    np.testing.assert_allclose(fx_b, fx)


@pytest.mark.unit
def test_adaptive_low_minority_is_capped_and_passers_keep_geometry():
    """0.2 passes against a bulk near 5, and extra passers are not overwritten."""
    n = 10
    natoms = 3
    qlen = 8
    xyz = np.zeros((n, natoms, 3), dtype=np.float64)
    for i in range(n):
        xyz[i, 0, 0] = float(i + 1)
    fx = np.array([0.1, 0.2, 0.3, 5, 5, 5, 5, 5, 5, 5], dtype=np.float64)
    pred = np.arange(n, dtype=np.float64)[:, None] + np.linspace(0.0, 1.0, qlen)
    xyz_b, fx_b, pred_b, k, n_pass, median_fx, cutoff, mode = _adaptive(
        xyz, fx, pred, 0.2
    )
    assert mode == "adaptive"
    assert median_fx == pytest.approx(5.0)
    assert cutoff == pytest.approx(2.5)
    assert n_pass == 3
    assert k == 2  # ceil(0.2 * 10), not all three passers
    # Passers keep themselves, including the one outside the tiling pool.
    for i in (0, 1, 2):
        np.testing.assert_allclose(xyz_b[i], xyz[i])
        assert fx_b[i] == pytest.approx(fx[i])
        np.testing.assert_allclose(pred_b[i], pred[i])
    # Failures are tiled from the best two passers: pool = chains 0, 1.
    pool = (0, 1)
    for i in range(3, n):
        src = pool[i % k]
        np.testing.assert_allclose(xyz_b[i], xyz[src])
        assert fx_b[i] == pytest.approx(fx[src])
        np.testing.assert_allclose(pred_b[i], pred[src])


@pytest.mark.unit
def test_adaptive_restart_ratio_rejects_invalid():
    xyz, _f, fx, pred = _fake_phase(n_chains=4)
    with pytest.raises(ValueError, match="restart_ratio"):
        select_adaptive_restart_batch(xyz, fx, pred, 0.0)
    with pytest.raises(ValueError, match="restart_ratio"):
        select_adaptive_restart_batch(xyz, fx, pred, 1.5)
