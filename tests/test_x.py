"""
Tests for modules/x.py
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.x import Xray


class TestXrayInitialization:
    """Test Xray class initialization"""
    
    def test_init(self):
        """Test Xray class can be initialized"""
        x = Xray()
        assert hasattr(x, 'aa')
        assert hasattr(x, 'bb')
        assert hasattr(x, 'cc')
        assert x.aa.shape[0] == 17  # Number of elements in coefficients
        assert x.bb.shape[0] == 17
        assert len(x.cc) == 17


class TestAtomicFactor:
    """Test atomic_factor function"""
    
    def setup_method(self):
        self.x = Xray()
    
    def test_atomic_factor_hydrogen(self):
        """Test atomic factor for hydrogen"""
        qvector = np.array([0.0, 1.0, 5.0])
        factor = self.x.atomic_factor(1, qvector)  # H
        
        assert len(factor) == len(qvector)
        # At q=0, factor should be approximately Z (1 for H)
        assert abs(factor[0] - 1.0) < 0.1
    
    def test_atomic_factor_carbon(self):
        """Test atomic factor for carbon"""
        qvector = np.array([0.0, 1.0, 5.0])
        factor = self.x.atomic_factor(6, qvector)  # C
        
        assert len(factor) == len(qvector)
        # At q=0, factor should be approximately Z (6 for C)
        assert abs(factor[0] - 6.0) < 0.5
    
    def test_atomic_factor_decreases_with_q(self):
        """Test that atomic factor decreases with increasing q"""
        qvector = np.linspace(0.0, 10.0, 50)
        factor = self.x.atomic_factor(6, qvector)  # C
        
        # Factor should generally decrease (not strictly, but trend)
        # Check that q=0 value is larger than q=10 value
        assert factor[0] > factor[-1]


class TestComptonSpline:
    """Test compton_spline function"""
    
    def setup_method(self):
        self.x = Xray()
    
    @pytest.mark.skipif(
        not os.path.exists("data/Compton_Scattering_Intensities.npz"),
        reason="Compton data file not found"
    )
    def test_compton_spline_basic(self, sample_qvector):
        """Test basic compton spline calculation"""
        atomic_numbers = [1, 6]  # H, C
        compton_array = self.x.compton_spline(atomic_numbers, sample_qvector)
        
        assert compton_array.shape == (2, len(sample_qvector))
        # Values should be non-negative
        assert np.all(compton_array >= 0)
    
    @pytest.mark.skipif(
        not os.path.exists("data/Compton_Scattering_Intensities.npz"),
        reason="Compton data file not found"
    )
    def test_compton_spline_calc(self, sample_qvector):
        """Test compton_spline_calc function"""
        atomic_numbers = [1, 6]  # H, C
        compton_total, compton_array = self.x.compton_spline_calc(
            atomic_numbers, sample_qvector
        )
        
        assert compton_array.shape == (2, len(sample_qvector))
        assert len(compton_total) == len(sample_qvector)
        # Total should be sum of individual components
        np.testing.assert_array_almost_equal(
            compton_total, np.sum(compton_array, axis=0), decimal=6
        )


class TestIAMCalc:
    """Test IAM calculation functions"""
    
    def setup_method(self):
        self.x = Xray()
    
    def test_iam_calc_basic(self):
        """Test basic IAM calculation"""
        atomic_numbers = [1, 6]  # H, C
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # 1 Å apart
        qvector = np.linspace(0.1, 5.0, 10)
        
        iam, atomic, molecular, compton, pre_molecular = self.x.iam_calc(
            atomic_numbers, xyz, qvector, electron_mode=False, inelastic=False
        )
        
        assert len(iam) == len(qvector)
        assert len(atomic) == len(qvector)
        assert len(molecular) == len(qvector)
        assert pre_molecular.shape == (1, len(qvector))  # 1 pair for 2 atoms
    
    def test_iam_calc_with_compton(self):
        """Test IAM calculation with Compton scattering"""
        atomic_numbers = [1, 6]
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        qvector = np.linspace(0.1, 5.0, 10)
        compton_array = np.ones((2, len(qvector))) * 0.1
        
        iam, atomic, molecular, compton, _ = self.x.iam_calc(
            atomic_numbers, xyz, qvector, 
            electron_mode=False, inelastic=True, compton_array=compton_array
        )
        
        assert len(compton) == len(qvector)
        # IAM should include compton contribution
        np.testing.assert_array_almost_equal(
            iam, atomic + molecular + compton, decimal=6
        )
    
    def test_jq_atomic_factors_calc(self):
        """Test jq_atomic_factors_calc function"""
        atomic_numbers = [1, 6]
        qvector = np.linspace(0.1, 5.0, 10)
        
        jq, atomic_factor_arr = self.x.jq_atomic_factors_calc(
            atomic_numbers, qvector
        )
        
        assert len(jq) == len(qvector)
        assert atomic_factor_arr.shape == (2, len(qvector))
        # J(q) should be sum of squares of atomic factors
        expected_jq = np.sum(atomic_factor_arr**2, axis=0)
        np.testing.assert_array_almost_equal(jq, expected_jq, decimal=6)


class TestImolCalc:
    """Test Imol_calc function"""
    
    def setup_method(self):
        self.x = Xray()
    
    def test_imol_calc(self):
        """Test molecular term calculation"""
        # Create atomic factors
        atomic_factor_arr = np.array([
            [1.0, 1.0, 1.0],  # Atom 1
            [2.0, 2.0, 2.0]   # Atom 2
        ])
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        qvector = np.array([0.5, 1.0, 2.0])
        
        imol = self.x.Imol_calc(atomic_factor_arr, xyz, qvector)
        
        assert len(imol) == len(qvector)
        # Values should be reasonable (positive for this case)
        assert np.all(imol >= 0)


class TestSphericalRotAvg:
    """Test spherical_rotavg function"""
    
    def setup_method(self):
        self.x = Xray()
    
    def test_spherical_rotavg(self):
        """Test spherical rotational average"""
        # Create a simple 3D array
        qlen, tlen, plen = 5, 10, 20
        f = np.ones((qlen, tlen, plen))
        th = np.linspace(0, np.pi, tlen)
        ph = np.linspace(0, 2*np.pi, plen)
        
        f_rotavg = self.x.spherical_rotavg(f, th, ph)
        
        assert len(f_rotavg) == qlen
        # For constant array, result should be constant
        np.testing.assert_allclose(f_rotavg, f_rotavg[0], rtol=1e-6)
