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

