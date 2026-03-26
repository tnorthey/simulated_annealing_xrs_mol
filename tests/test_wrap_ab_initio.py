"""Tests for ab-initio-derived correction ratio helper."""
import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.wrap import _safe_ab_initio_correction_ratio


def test_safe_ratio_basic():
    I = np.array([2.0, 4.0, 6.0])
    iam = np.array([1.0, 2.0, 3.0])
    r = _safe_ab_initio_correction_ratio(I, iam)
    np.testing.assert_allclose(r, [2.0, 2.0, 2.0])


def test_safe_ratio_tiny_iam_defaults_to_one():
    I = np.array([1.0, 2.0])
    iam = np.array([1e-40, 2.0])
    r = _safe_ab_initio_correction_ratio(I, iam, eps=1e-30)
    assert r[0] == 1.0
    assert r[1] == 1.0


def test_safe_ratio_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        _safe_ab_initio_correction_ratio(np.ones(3), np.ones(2))


def test_interp_matches_direct_when_q_grids_align():
    """Same logic as wrap: ratio on abi grid, interp to qvector."""
    q_abi = np.linspace(0.1, 4.0, 20)
    corr_abi = np.full(20, 1.25)
    qvector = q_abi.copy()
    correction_factor_q = np.interp(
        qvector, q_abi, corr_abi, left=corr_abi[0], right=corr_abi[-1]
    )
    np.testing.assert_allclose(correction_factor_q, 1.25)
