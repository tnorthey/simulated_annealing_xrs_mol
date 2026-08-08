"""
Tests for multi-chain GPU conformational diversity plumbing.

Uses CUDA backend with CPU emulation (no discrete GPU required).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.sa import Annealing
from modules.wrap import (
    apply_per_chain_boltzmann_displacement,
    print_chain_diversity_summary,
    summarize_chain_diversity,
)
from modules.x import Xray


def _minimal_sa_inputs(natoms: int = 3, qlen: int = 16):
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.6, 0.9, 0.0],
        ],
        dtype=np.float64,
    )[:natoms]
    qvector = np.linspace(0.2, 4.0, qlen, dtype=np.float64)
    x = Xray()
    atomic_numbers = [6, 6, 1][:natoms]
    _iam, atomic_total, _mol, compton, pre_molecular = x.iam_calc(
        atomic_numbers,
        xyz,
        qvector,
        electron_mode=False,
        inelastic=False,
        compton_array=np.zeros((0,)),
    )
    return {
        "starting_xyz": xyz,
        "qvector": qvector,
        "atomic_total": atomic_total,
        "compton": compton,
        "pre_molecular": pre_molecular,
        "displacements": np.zeros((1, natoms, 3), dtype=np.float64),
        "mode_indices": np.array([0], dtype=np.int64),
        "step_size_array": np.array([0.0], dtype=np.float64),
        "bond_param_array": np.zeros((0, 4), dtype=np.float64),
        "angle_param_array": np.zeros((0, 5), dtype=np.float64),
        "torsion_param_array": np.zeros((0, 6), dtype=np.float64),
        "target_function": np.ones(qlen, dtype=np.float64),
        "reference_iam": np.ones(qlen, dtype=np.float64),
        "th": np.array([0.0, np.pi], dtype=np.float64),
        "ph": np.array([0.0, np.pi], dtype=np.float64),
    }


def _run_batched_sa(
    *,
    n_chains: int,
    starting_xyz: np.ndarray,
    gpu_starting_xyz_batch: np.ndarray | None,
    f_start,
    f_xray_start,
    predicted_start,
    nsteps: int = 1,
    step_size: float = 0.0,
    inp: dict | None = None,
):
    if inp is None:
        inp = _minimal_sa_inputs(natoms=starting_xyz.shape[0])
    a = Annealing()
    step_size_array = np.array([step_size], dtype=np.float64)
    (
        f_best,
        f_xray_best,
        predicted_best,
        xyz_best,
        _c_tuning_adjusted,
    ) = a.simulated_annealing_modes_ho(
        starting_xyz=starting_xyz.astype(np.float64),
        displacements=inp["displacements"],
        mode_indices=inp["mode_indices"],
        target_function=inp["target_function"],
        reference_iam=inp["reference_iam"],
        qvector=inp["qvector"],
        th=inp["th"],
        ph=inp["ph"],
        compton=inp["compton"].astype(np.float64),
        atomic_total=inp["atomic_total"].astype(np.float64),
        pre_molecular=inp["pre_molecular"].astype(np.float64),
        step_size_array=step_size_array,
        bond_param_array=inp["bond_param_array"],
        angle_param_array=inp["angle_param_array"],
        torsion_param_array=inp["torsion_param_array"],
        starting_temp=0.0,
        nsteps=nsteps,
        inelastic=False,
        pcd_mode=False,
        ewald_mode=False,
        bonds_bool=False,
        angles_bool=False,
        torsions_bool=False,
        f_start=f_start,
        f_xray_start=f_xray_start,
        predicted_start=predicted_start,
        verbose=False,
        backend="cuda",
        gpu_emulation=True,
        gpu_chains=n_chains,
        gpu_starting_xyz_batch=gpu_starting_xyz_batch,
    )
    return a, f_best, f_xray_best, predicted_best, xyz_best


@pytest.mark.unit
def test_multi_chain_preserves_distinct_starts_across_restarts():
    """Second phase seeded with xyz_best_all must not collapse all chains to best[0]."""
    inp = _minimal_sa_inputs()
    base = inp["starting_xyz"].copy()
    n_chains = 4
    batch = np.stack(
        [base + np.array([0.1 * k, 0.0, 0.0], dtype=np.float64) for k in range(n_chains)],
        axis=0,
    )

    a1, *_ = _run_batched_sa(
        n_chains=n_chains,
        starting_xyz=base,
        gpu_starting_xyz_batch=batch,
        f_start=1e10,
        f_xray_start=1e10,
        predicted_start=0,
        nsteps=1,
        step_size=0.0,
        inp=inp,
    )
    assert a1.last_chain_results is not None
    xyz_phase1 = np.asarray(a1.last_chain_results["xyz_best_all"], dtype=np.float64)
    f_phase1 = np.asarray(a1.last_chain_results["f_best_all"], dtype=np.float64)
    fx_phase1 = np.asarray(a1.last_chain_results["f_xray_best_all"], dtype=np.float64)
    pred_phase1 = np.asarray(
        a1.last_chain_results["predicted_best_all"], dtype=np.float64
    )

    # Distinct chain geometries after phase 1 (zero move size).
    assert not np.allclose(xyz_phase1[0], xyz_phase1[1])
    assert not np.allclose(xyz_phase1[0], xyz_phase1[-1])
    np.testing.assert_allclose(xyz_phase1, batch, atol=1e-12)

    # Phase 2: continue from per-chain bests (wrap restart semantics).
    a2, *_ = _run_batched_sa(
        n_chains=n_chains,
        starting_xyz=xyz_phase1[0],  # would be wrong to use for all if batch missing
        gpu_starting_xyz_batch=xyz_phase1,
        f_start=f_phase1,
        f_xray_start=fx_phase1,
        predicted_start=pred_phase1,
        nsteps=1,
        step_size=0.0,
        inp=inp,
    )
    xyz_phase2 = np.asarray(a2.last_chain_results["xyz_best_all"], dtype=np.float64)
    assert not np.allclose(xyz_phase2[0], xyz_phase2[1])
    assert not np.allclose(xyz_phase2[0], xyz_phase2[-1])
    np.testing.assert_allclose(xyz_phase2, xyz_phase1, atol=1e-12)

    # Without batch, all chains would be cloned from a single start.
    a_collapse, *_ = _run_batched_sa(
        n_chains=n_chains,
        starting_xyz=xyz_phase1[0],
        gpu_starting_xyz_batch=None,
        f_start=float(f_phase1[0]),
        f_xray_start=float(fx_phase1[0]),
        predicted_start=pred_phase1[0],
        nsteps=1,
        step_size=0.0,
        inp=inp,
    )
    xyz_collapse = np.asarray(
        a_collapse.last_chain_results["xyz_best_all"], dtype=np.float64
    )
    for k in range(1, n_chains):
        np.testing.assert_allclose(xyz_collapse[k], xyz_collapse[0], atol=1e-12)


@pytest.mark.unit
def test_per_chain_f_start_accepts_vector():
    inp = _minimal_sa_inputs()
    base = inp["starting_xyz"].copy()
    n_chains = 3
    batch = np.stack([base + 0.05 * k for k in range(n_chains)], axis=0)
    f_vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    a, f_best, *_ = _run_batched_sa(
        n_chains=n_chains,
        starting_xyz=base,
        gpu_starting_xyz_batch=batch,
        f_start=f_vec,
        f_xray_start=f_vec * 0.5,
        predicted_start=0,
        nsteps=1,
        step_size=0.0,
        inp=inp,
    )
    # With zero steps essentially evaluating same IAM, f_best may update;
    # start vectors must not raise and chains stay distinct.
    assert a.last_chain_results is not None
    xyz_all = np.asarray(a.last_chain_results["xyz_best_all"])
    assert not np.allclose(xyz_all[0], xyz_all[1])
    assert np.isfinite(f_best)


@pytest.mark.unit
def test_auto_diverse_boltzmann_starts_not_identical():
    """Multi-chain Boltzmann starts (as used when sampling off / no pool) differ."""
    np.random.seed(99)
    base = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.8, 0.0]], dtype=np.float64
    )
    n_chains = 5
    batch = np.repeat(base[np.newaxis, :, :], n_chains, axis=0)
    modes = np.random.randn(4, 3, 3)
    freqs = np.array([100.0, 150.0, 200.0, 250.0])
    mu = np.array([1.0, 1.2, 1.4, 1.6])
    out = apply_per_chain_boltzmann_displacement(
        batch, modes, freqs, 300.0, reduced_mass_amu=mu
    )
    assert not np.allclose(out[0], out[1])
    assert not np.allclose(out[0], out[-1])
    # Chains differ from the shared seed geometry.
    assert not np.allclose(out[0], base)


@pytest.mark.unit
def test_summarize_chain_diversity_stats():
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.8, 0.0],
        ],
        dtype=np.float64,
    )
    # Two identical + one shifted structure
    xyz_all = np.stack(
        [
            base,
            base.copy(),
            base + np.array([0.0, 0.5, 0.0]),
        ],
        axis=0,
    )
    stats = summarize_chain_diversity(
        xyz_all,
        rmsd_indices=[0, 1, 2],
        bond_indices=[0, 1],
        max_pairs=64,
    )
    assert stats["n_chains"] == 3
    assert stats["n_pairs"] == 3
    assert stats["rmsd_min"] == pytest.approx(0.0, abs=1e-12)
    assert stats["rmsd_max"] > 0.0
    assert stats["bond_min"] > 0.0
    assert stats["bond_std"] >= 0.0


@pytest.mark.unit
def test_print_chain_diversity_summary_smoke(capsys):
    stats = {
        "n_chains": 4,
        "n_pairs": 6,
        "rmsd_min": 0.0,
        "rmsd_median": 0.1,
        "rmsd_max": 0.3,
        "bond_n": 4,
        "bond_min": 1.5,
        "bond_median": 1.55,
        "bond_max": 1.6,
        "bond_std": 0.02,
    }
    print_chain_diversity_summary(stats, bond_label="bond 0-5")
    out = capsys.readouterr().out
    assert "pairwise centroid-aligned RMSD" in out
    assert "bond 0-5" in out
