"""
Tests for hydrogen force constant scaling in modules/wrap.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.wrap import scale_hydrogen_force_constants


class TestScaleHydrogenForceConstants:
    """Unit tests for scale_hydrogen_force_constants."""

    @pytest.fixture
    def atomic_numbers_ch2(self):
        # 0=C, 1=H, 2=C, 3=H
        return np.array([6, 1, 6, 1], dtype=np.int64)

    @pytest.fixture
    def atomic_numbers_ccc_h(self):
        # 0=C, 1=C, 2=C, 3=H
        return np.array([6, 6, 6, 1], dtype=np.int64)

    def test_scale_one_is_identity(self, atomic_numbers_ch2):
        bonds = np.array([[0, 1, 1.09, 100.0], [0, 2, 1.54, 200.0]], dtype=np.float64)
        angles = np.array([[1, 0, 2, 1.9, 60.0]], dtype=np.float64)
        torsions = np.array([[3, 0, 2, 1, 0.0, 2.0]], dtype=np.float64)

        b2, a2, t2 = scale_hydrogen_force_constants(
            bonds, angles, torsions, atomic_numbers_ch2, 1.0
        )
        assert b2 is bonds
        assert a2 is angles
        assert t2 is torsions

    def test_scales_h_bonds_only(self, atomic_numbers_ch2):
        bonds = np.array(
            [
                [0, 1, 1.09, 100.0],  # C-H
                [0, 2, 1.54, 200.0],  # C-C
            ],
            dtype=np.float64,
        )
        angles = np.empty((0, 5), dtype=np.float64)
        torsions = np.empty((0, 6), dtype=np.float64)

        b2, _, _ = scale_hydrogen_force_constants(
            bonds, angles, torsions, atomic_numbers_ch2, 3.0
        )
        assert b2[0, 3] == pytest.approx(300.0)
        assert b2[1, 3] == pytest.approx(200.0)

    def test_scales_h_angles_only(self, atomic_numbers_ccc_h):
        bonds = np.empty((0, 4), dtype=np.float64)
        angles = np.array(
            [
                [3, 0, 1, 1.9, 60.0],  # H-C-C
                [0, 1, 2, 2.0, 80.0],  # C-C-C
            ],
            dtype=np.float64,
        )
        torsions = np.empty((0, 6), dtype=np.float64)

        _, a2, _ = scale_hydrogen_force_constants(
            bonds, angles, torsions, atomic_numbers_ccc_h, 3.0
        )
        assert a2[0, 4] == pytest.approx(180.0)
        assert a2[1, 4] == pytest.approx(80.0)

    def test_scales_torsion_with_hydrogen(self, atomic_numbers_ch2):
        bonds = np.empty((0, 4), dtype=np.float64)
        angles = np.empty((0, 5), dtype=np.float64)
        torsions = np.array([[3, 0, 2, 1, 0.5, 2.0]], dtype=np.float64)

        _, _, t2 = scale_hydrogen_force_constants(
            bonds, angles, torsions, atomic_numbers_ch2, 2.0
        )
        assert t2[0, 5] == pytest.approx(4.0)
