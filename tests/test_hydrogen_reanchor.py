"""Tests for hydrogen re-anchoring helpers in modules.wrap."""

import numpy as np
import pytest

from modules.wrap import (
    _build_hydrogen_parent_map,
    _mm_harmonic_contribs,
    reanchor_hydrogens_reference_relative,
)


def test_build_hydrogen_parent_map_simple():
    atomlist = ["C", "H", "H"]
    bond_param_array = np.array(
        [
            [0, 1, 1.09, 100.0],
            [0, 2, 1.09, 100.0],
        ],
        dtype=np.float64,
    )
    parents = _build_hydrogen_parent_map(atomlist, bond_param_array)
    assert parents == {1: 0, 2: 0}


def test_reanchor_places_h_on_reference_ray():
    """C + 2 H: displaced H should lie on reference direction from current C."""
    atomlist = ["C", "H", "H"]
    reference_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.866, 0.0],
        ],
        dtype=np.float64,
    )
    xyz = reference_xyz.copy()
    # Move heavy atom and scramble H positions
    xyz[0] = [0.1, -0.2, 0.3]
    xyz[1] = [5.0, 5.0, 5.0]
    xyz[2] = [-4.0, 1.0, 2.0]
    bond_param_array = np.array(
        [
            [0, 1, 1.0, 100.0],
            [0, 2, 1.0, 100.0],
        ],
        dtype=np.float64,
    )

    heavy_before = xyz[0].copy()
    out, max_disp = reanchor_hydrogens_reference_relative(
        xyz, atomlist, reference_xyz, bond_param_array
    )
    assert max_disp > 0.0
    np.testing.assert_allclose(out[0], heavy_before)

    for h_idx in (1, 2):
        parent = 0
        vec_ref = reference_xyz[h_idx] - reference_xyz[parent]
        r0_ref = np.linalg.norm(vec_ref)
        u_ref = vec_ref / r0_ref
        expected = out[parent] + r0_ref * u_ref
        np.testing.assert_allclose(out[h_idx], expected, rtol=1e-10, atol=1e-10)


def test_reanchor_skips_degenerate_reference_bond():
    atomlist = ["C", "H"]
    reference_xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    xyz = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    bond_param_array = np.array([[0, 1, 1.09, 100.0]], dtype=np.float64)
    out, max_disp = reanchor_hydrogens_reference_relative(
        xyz, atomlist, reference_xyz, bond_param_array
    )
    np.testing.assert_allclose(out[1], xyz[1])
    assert max_disp == 0.0


def test_mm_harmonic_contribs_bond_stretch():
    xyz = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    bond_param_array = np.array([[0, 1, 1.0, 100.0]], dtype=np.float64)
    angle_param_array = np.zeros((0, 5))
    torsion_param_array = np.zeros((0, 6))
    b, a, t = _mm_harmonic_contribs(
        xyz,
        bond_param_array,
        angle_param_array,
        torsion_param_array,
        bonds_bool=True,
        angles_bool=False,
        torsions_bool=False,
    )
    assert a == 0.0 and t == 0.0
    expected = 100.0 * 0.5 * (0.5 ** 2)
    assert b == pytest.approx(expected)


def test_multiple_heavy_parents_raises():
    atomlist = ["C", "O", "H"]
    bond_param_array = np.array(
        [
            [0, 2, 1.0, 100.0],
            [1, 2, 1.0, 100.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="multiple heavy-atom parents"):
        _build_hydrogen_parent_map(atomlist, bond_param_array)
