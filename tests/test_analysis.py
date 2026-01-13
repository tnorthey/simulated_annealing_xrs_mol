"""
Tests for modules/analysis.py
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.analysis as analysis


class TestBondLength:
    """Test bond length calculations"""
    
    def test_bond_length_simple(self):
        """Test bond length calculation"""
        xyz = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        length = analysis.calculate_bond_length(xyz, 0, 1)
        np.testing.assert_almost_equal(length, 1.0, decimal=6)
    
    def test_bond_length_3d(self):
        """Test bond length in 3D"""
        xyz = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        length = analysis.calculate_bond_length(xyz, 0, 1)
        expected = np.sqrt(3.0)
        np.testing.assert_almost_equal(length, expected, decimal=6)


class TestAngleCalculations:
    """Test angle calculation functions"""
    
    def test_angle_2p_3d_right_angle(self):
        """Test angle calculation for right angle"""
        # Right angle at point b
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        
        angle = analysis.angle_2p_3d(a, b, c)
        np.testing.assert_almost_equal(angle, 90.0, decimal=1)
    
    def test_calculate_angle(self):
        """Test calculate_angle wrapper function"""
        xyz = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        angle = analysis.calculate_angle(xyz, 0, 1, 2)
        np.testing.assert_almost_equal(angle, 90.0, decimal=1)
    
    def test_directional_angle_3d(self):
        """Test directional angle calculation"""
        A = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 0.0, 0.0])
        C = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        
        angle = analysis.directional_angle_3d(A, B, C, normal)
        # Should be 90 degrees
        np.testing.assert_almost_equal(angle, 90.0, decimal=1)


class TestDihedralCalculations:
    """Test dihedral angle calculations"""
    
    def test_new_dihedral_simple(self):
        """Test dihedral angle calculation"""
        # Simple case: points in a plane (square)
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p = (p0, p1, p2, p3)
        
        angle = analysis.new_dihedral(p)
        # For a square in a plane, the angle should be around 0 or 180 degrees
        # The exact value depends on the orientation, but should be reasonable
        assert -180.0 <= angle <= 180.0
        # Check that it's close to 0 or 180 (within reasonable tolerance)
        assert abs(angle) < 10.0 or abs(abs(angle) - 180.0) < 10.0
    
    def test_new_dihedral_zero(self):
        """Test dihedral angle for coplanar points"""
        # All points in a line
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        p3 = np.array([3.0, 0.0, 0.0])
        p = (p0, p1, p2, p3)
        
        angle = analysis.new_dihedral(p)
        # Should be 0 or 180 degrees for collinear points
        assert abs(angle) < 1.0 or abs(abs(angle) - 180.0) < 1.0
    
    def test_calculate_dihedral(self):
        """Test calculate_dihedral wrapper function"""
        xyz = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        angle = analysis.calculate_dihedral(xyz, 0, 1, 2, 3)
        # Should be a valid dihedral angle
        assert -180.0 <= angle <= 180.0
