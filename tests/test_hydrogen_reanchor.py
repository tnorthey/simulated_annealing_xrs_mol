"""Tests for hydrogen re-anchoring helpers in modules.wrap."""

import os

import numpy as np
import pytest

import modules.mol as mol_module
from modules.wrap import (
    _build_hydrogen_parent_map_from_reference,
    _mm_harmonic_contribs,
    reanchor_hydrogens_reference_relative,
)

m = mol_module.Xyz()


def test_reference_map_picks_closest_parent_chd_like():
    """CHD-like: H13 is closest to C5; spurious bond_param C4-H must not matter."""
    atomlist = ["C"] * 6 + ["H"] * 8
    reference_xyz = np.array(
        [
            [0.729, -1.239, -0.116],
            [1.406, 0.070, 0.217],
            [0.734, 1.209, 0.172],
            [-0.701, 1.220, -0.173],
            [-1.404, 0.099, -0.165],
            [-0.763, -1.210, 0.230],
            [1.213, -2.060, 0.404],
            [0.856, -1.430, -1.182],
            [2.459, 0.063, 0.442],
            [1.223, 2.147, 0.370],
            [-1.164, 2.161, -0.417],
            [-2.457, 0.111, -0.391],
            [-1.270, -2.042, -0.250],
            [-0.894, -1.346, 1.305],
        ],
        dtype=np.float64,
    )
    parents = _build_hydrogen_parent_map_from_reference(atomlist, reference_xyz)
    assert parents[13] == 5
    assert len(parents) == 8


def test_reference_map_chd_reference_xyz_file():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = os.path.join(repo_root, "xyz", "chd_reference.xyz")
    if not os.path.isfile(ref_path):
        pytest.skip(f"missing {ref_path}")
    _, _, atomlist, reference_xyz = m.read_xyz(ref_path)
    parents = _build_hydrogen_parent_map_from_reference(atomlist, reference_xyz)
    h_indices = [i for i, s in enumerate(atomlist) if s.upper() in ("H", "D")]
    assert len(parents) == len(h_indices)
    for h_idx in h_indices:
        assert h_idx in parents
        assert not atomlist[parents[h_idx]].upper() in ("H", "D")


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
    xyz[0] = [0.1, -0.2, 0.3]
    xyz[1] = [5.0, 5.0, 5.0]
    xyz[2] = [-4.0, 1.0, 2.0]

    heavy_before = xyz[0].copy()
    out, max_disp = reanchor_hydrogens_reference_relative(
        xyz, atomlist, reference_xyz
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
    out, max_disp = reanchor_hydrogens_reference_relative(
        xyz, atomlist, reference_xyz
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


def test_reanchor_with_ambiguous_bond_param_still_succeeds():
    """Spurious second X-H in bond_param must not break re-anchoring."""
    atomlist = ["C"] * 6 + ["H"] * 8
    reference_xyz = np.array(
        [
            [0.729, -1.239, -0.116],
            [1.406, 0.070, 0.217],
            [0.734, 1.209, 0.172],
            [-0.701, 1.220, -0.173],
            [-1.404, 0.099, -0.165],
            [-0.763, -1.210, 0.230],
            [1.213, -2.060, 0.404],
            [0.856, -1.430, -1.182],
            [2.459, 0.063, 0.442],
            [1.223, 2.147, 0.370],
            [-1.164, 2.161, -0.417],
            [-2.457, 0.111, -0.391],
            [-1.270, -2.042, -0.250],
            [-0.894, -1.346, 1.305],
        ],
        dtype=np.float64,
    )
    xyz = reference_xyz.copy()
    xyz[13] = reference_xyz[4] + 0.5 * (reference_xyz[13] - reference_xyz[5])
    out, max_disp = reanchor_hydrogens_reference_relative(
        xyz, atomlist, reference_xyz
    )
    parent = 5
    vec_ref = reference_xyz[13] - reference_xyz[parent]
    r0_ref = np.linalg.norm(vec_ref)
    expected = out[parent] + r0_ref * (vec_ref / r0_ref)
    np.testing.assert_allclose(out[13], expected, rtol=1e-10, atol=1e-10)
    assert max_disp > 0.0
