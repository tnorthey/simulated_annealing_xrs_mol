"""
Ensure Wrapper basic geometry parameter extraction works without OpenFF.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_basic_mm_params_fallback_without_openff(monkeypatch):
    # Import inside test so we can monkeypatch module globals
    import modules.wrap as wrap

    monkeypatch.setattr(wrap, "HAVE_OPENFF", False, raising=True)
    monkeypatch.setattr(wrap, "openff_retreive_mm_params", None, raising=True)

    # 4-atom chain: should yield 3 bonds, 2 angles, 1 torsion with our heuristic
    xyz_content = """4
test chain
C  0.000  0.000  0.000
C  1.540  0.000  0.000
C  3.080  0.000  0.000
C  4.620  0.100  0.000
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        xyz_file = f.name

    class P:
        # minimal params object used by Wrapper.run_xyz_openff_mm_params
        start_xyz_file = xyz_file
        start_sdf_file = "unused.sdf"
        forcefield_file = "unused.offxml"
        mm_param_method = "basic"
        bond_ignore_array = np.array([], dtype=np.int64)
        angle_ignore_array = np.array([], dtype=np.int64)
        torsion_ignore_array = np.array([], dtype=np.int64)

    try:
        w = wrap.Wrapper()
        p = w.run_xyz_openff_mm_params(P(), xyz_file)
        assert hasattr(p, "bond_param_array")
        assert hasattr(p, "angle_param_array")
        assert hasattr(p, "torsion_param_array")

        assert p.bond_param_array.shape[1] == 4
        assert p.angle_param_array.shape[1] == 5
        assert p.torsion_param_array.shape[1] == 6

        assert p.bond_param_array.shape[0] >= 1
    finally:
        if os.path.exists(xyz_file):
            os.remove(xyz_file)


def test_bond_ignore_also_drops_angles_and_torsions_using_that_bond(monkeypatch):
    """Ignoring bond 0-1 must also drop angle 0-1-2 and torsion 0-1-2-3."""
    import modules.wrap as wrap

    monkeypatch.setattr(wrap, "HAVE_OPENFF", False, raising=True)
    monkeypatch.setattr(wrap, "openff_retreive_mm_params", None, raising=True)

    xyz_content = """4
test chain
C  0.000  0.000  0.000
C  1.540  0.000  0.000
C  3.080  0.000  0.000
C  4.620  0.100  0.000
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        xyz_file = f.name

    class P:
        start_xyz_file = xyz_file
        start_sdf_file = "unused.sdf"
        forcefield_file = "unused.offxml"
        mm_param_method = "basic"
        bond_ignore_array = np.array([[0, 1]], dtype=np.int64)
        angle_ignore_array = np.array([], dtype=np.int64)
        torsion_ignore_array = np.array([], dtype=np.int64)

    try:
        w = wrap.Wrapper()
        p = w.run_xyz_openff_mm_params(P(), xyz_file)
        bonds = {(int(min(r[0], r[1])), int(max(r[0], r[1]))) for r in p.bond_param_array}
        assert (0, 1) not in bonds
        assert (1, 2) in bonds
        for row in p.angle_param_array:
            a1, a2, a3 = int(row[0]), int(row[1]), int(row[2])
            legs = {(min(a1, a2), max(a1, a2)), (min(a2, a3), max(a2, a3))}
            assert (0, 1) not in legs
        for row in p.torsion_param_array:
            atoms = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
            pairs = {
                (min(atoms[i], atoms[i + 1]), max(atoms[i], atoms[i + 1]))
                for i in range(3)
            }
            assert (0, 1) not in pairs
    finally:
        if os.path.exists(xyz_file):
            os.remove(xyz_file)

