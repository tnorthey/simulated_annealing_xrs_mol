import numpy as np


def test_boltzmann_sampling_accepts_reduced_mass_without_atomic_numbers():
    from modules.sample import Sample

    s = Sample()
    nmodes, natoms = 3, 2
    modes = np.random.randn(nmodes, natoms, 3)
    freqs_cm1 = np.array([100.0, 200.0, 300.0])
    reduced_mass_amu = np.array([1.0, 2.0, 3.0])

    disp = s.generate_boltzmann_displacement(
        modes, freqs_cm1, T=300.0, atomic_numbers=None, reduced_mass_amu=reduced_mass_amu
    )
    assert disp.shape == (natoms, 3)
    assert np.isfinite(disp).all()

