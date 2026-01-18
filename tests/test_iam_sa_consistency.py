"""
Ensure IAM computed in the SA njit loop matches the standalone IAM calculation.

This test is important because the SA loop uses a hand-written (numba) IAM kernel
based on precomputed `atomic_total` and `pre_molecular`. The standalone path uses
`modules.x.Xray().iam_calc()`. They must produce the same IAM(q) for the same inputs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.sa import Annealing
from modules.x import Xray


def _run_sa_one_step_get_predicted(
    starting_xyz: np.ndarray,
    qvector: np.ndarray,
    atomic_total: np.ndarray,
    pre_molecular: np.ndarray,
    compton_total: np.ndarray,
    *,
    inelastic: bool,
) -> np.ndarray:
    """
    Run SA for 1 step with zero displacement and return `predicted_best`,
    which (for this setup) is exactly the IAM(q) computed inside the njit loop.
    """
    natoms = starting_xyz.shape[0]
    qlen = len(qvector)

    # Deterministic "do nothing" displacement setup
    displacements = np.zeros((1, natoms, 3), dtype=np.float64)
    mode_indices = np.array([0], dtype=np.int64)
    step_size_array = np.array([0.0], dtype=np.float64)

    # Dummy arrays not used when bonds/angles/torsions are disabled
    bond_param_array = np.zeros((0, 4), dtype=np.float64)
    angle_param_array = np.zeros((0, 5), dtype=np.float64)
    torsion_param_array = np.zeros((0, 6), dtype=np.float64)

    # Target/reference arrays just need to be nonzero to avoid division by zero
    target_function = np.ones(qlen, dtype=np.float64)
    reference_iam = np.ones(qlen, dtype=np.float64)

    # th/ph are required by the SA signature even in non-ewald mode
    th = np.array([0.0, np.pi], dtype=np.float64)
    ph = np.array([0.0, np.pi], dtype=np.float64)

    a = Annealing()
    (
        _f_best,
        _f_xray_best,
        predicted_best,
        _xyz_best,
        _c_tuning_adjusted,
    ) = a.simulated_annealing_modes_ho(
        starting_xyz=starting_xyz.astype(np.float64),
        displacements=displacements,
        mode_indices=mode_indices,
        target_function=target_function,
        reference_iam=reference_iam,
        qvector=qvector.astype(np.float64),
        th=th,
        ph=ph,
        compton=compton_total.astype(np.float64),
        atomic_total=atomic_total.astype(np.float64),
        pre_molecular=pre_molecular.astype(np.float64),
        step_size_array=step_size_array,
        bond_param_array=bond_param_array,
        angle_param_array=angle_param_array,
        torsion_param_array=torsion_param_array,
        starting_temp=0.0,  # disable temperature early-accept branch
        nsteps=1,
        inelastic=inelastic,
        pcd_mode=False,
        ewald_mode=False,
        bonds_bool=False,
        angles_bool=False,
        torsions_bool=False,
        verbose=False,
    )

    return np.asarray(predicted_best, dtype=np.float64)


@pytest.mark.unit
def test_iam_matches_sa_njit_loop_elastic():
    x = Xray()

    # Small but nontrivial geometry including carbon (exercises dd/ee term path)
    atomic_numbers = [1, 6, 1]  # H-C-H
    xyz = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [1.0900, 0.0000, 0.0000],
            [2.1800, 0.1000, 0.0000],
        ],
        dtype=np.float64,
    )
    qvector = np.linspace(0.1, 6.0, 64, dtype=np.float64)  # q>0 avoids sin(qr)/(qr) singularity

    iam_py, atomic_total, _molecular, compton_total, pre_molecular = x.iam_calc(
        atomic_numbers,
        xyz,
        qvector,
        electron_mode=False,
        inelastic=False,
        compton_array=np.zeros((0,)),
    )

    predicted_sa = _run_sa_one_step_get_predicted(
        starting_xyz=xyz,
        qvector=qvector,
        atomic_total=atomic_total,
        pre_molecular=pre_molecular,
        compton_total=compton_total,
        inelastic=False,
    )

    np.testing.assert_allclose(predicted_sa, iam_py, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
@pytest.mark.skipif(
    not os.path.exists("data/Compton_Scattering_Intensities.npz"),
    reason="Compton data file not found",
)
def test_iam_matches_sa_njit_loop_inelastic():
    x = Xray()

    atomic_numbers = [1, 6, 1]
    xyz = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [1.0900, 0.0000, 0.0000],
            [2.1800, 0.1000, 0.0000],
        ],
        dtype=np.float64,
    )
    qvector = np.linspace(0.1, 6.0, 64, dtype=np.float64)

    compton_array = x.compton_spline(atomic_numbers, qvector)
    iam_py, atomic_total, _molecular, compton_total, pre_molecular = x.iam_calc(
        atomic_numbers,
        xyz,
        qvector,
        electron_mode=False,
        inelastic=True,
        compton_array=compton_array,
    )

    predicted_sa = _run_sa_one_step_get_predicted(
        starting_xyz=xyz,
        qvector=qvector,
        atomic_total=atomic_total,
        pre_molecular=pre_molecular,
        compton_total=compton_total,
        inelastic=True,
    )

    np.testing.assert_allclose(predicted_sa, iam_py, rtol=1e-12, atol=1e-12)

