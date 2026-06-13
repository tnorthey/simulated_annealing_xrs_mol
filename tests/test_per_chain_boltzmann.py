"""Tests for per-chain GPU Boltzmann displacement."""

import numpy as np

from modules.wrap import apply_per_chain_boltzmann_displacement


def test_apply_per_chain_boltzmann_gives_independent_displacements():
    np.random.seed(12345)
    natoms = 3
    nmodes = 4
    n_chains = 6
    batch = np.zeros((n_chains, natoms, 3), dtype=np.float64)
    modes = np.random.randn(nmodes, natoms, 3)
    freqs_cm1 = np.array([120.0, 180.0, 240.0, 300.0])
    reduced_mass_amu = np.array([1.0, 1.5, 2.0, 2.5])

    out = apply_per_chain_boltzmann_displacement(
        batch,
        modes,
        freqs_cm1,
        300.0,
        reduced_mass_amu=reduced_mass_amu,
    )

    assert out.shape == (n_chains, natoms, 3)
    assert np.isfinite(out).all()
    # Independent draws should not produce identical chain displacements.
    assert not np.allclose(out[0], out[1])
    assert not np.allclose(out[0], out[-1])


def test_apply_per_chain_boltzmann_does_not_mutate_input():
    np.random.seed(0)
    batch = np.ones((2, 2, 3), dtype=np.float64)
    original = batch.copy()
    modes = np.eye(2)[:, :, np.newaxis] * np.ones((2, 1, 3))
    freqs_cm1 = np.array([100.0, 200.0])
    reduced_mass_amu = np.array([1.0, 2.0])

    out = apply_per_chain_boltzmann_displacement(
        batch,
        modes,
        freqs_cm1,
        300.0,
        reduced_mass_amu=reduced_mass_amu,
    )

    np.testing.assert_allclose(batch, original)
    assert not np.allclose(out, original)
