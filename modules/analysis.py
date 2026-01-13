"""
Geometry analysis functions for molecular structures.

This module contains functions for calculating geometric properties
such as bond lengths, angles, and dihedral angles.
"""

import math
import numpy as np
from numpy.typing import NDArray


def calculate_bond_length(xyz: NDArray, i: int, j: int) -> float:
    """
    Calculate bond length between atoms i and j (0-indexed).
    
    Parameters:
    -----------
    xyz : NDArray
        Array of atomic coordinates, shape (n_atoms, 3)
    i : int
        Index of first atom (0-indexed)
    j : int
        Index of second atom (0-indexed)
    
    Returns:
    --------
    float
        Bond length in Angstroms
    """
    return np.linalg.norm(xyz[i] - xyz[j])


def angle_2p_3d(a, b, c):
    """
    Calculate angle between three points in 3D space.
    
    Parameters:
    -----------
    a : array-like
        First point (3D coordinates)
    b : array-like
        Central point (3D coordinates)
    c : array-like
        Third point (3D coordinates)
    
    Returns:
    --------
    float
        Angle in degrees (a-b-c)
    """
    v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])

    v1mag = np.sqrt([v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]])
    v1norm = np.array([v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag])

    v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    v2norm = np.array([v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag])
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
    angle_rad = np.arccos(res)
    return math.degrees(angle_rad[0])


def calculate_angle(xyz: NDArray, i: int, j: int, k: int) -> float:
    """
    Calculate angle i-j-k in degrees (0-indexed).
    
    Parameters:
    -----------
    xyz : NDArray
        Array of atomic coordinates, shape (n_atoms, 3)
    i : int
        Index of first atom (0-indexed)
    j : int
        Index of central atom (0-indexed)
    k : int
        Index of third atom (0-indexed)
    
    Returns:
    --------
    float
        Angle in degrees
    """
    return angle_2p_3d(xyz[i], xyz[j], xyz[k])


def new_dihedral(p: tuple):
    """
    Calculate dihedral angle using Praxeolitic formula.
    
    Uses 1 sqrt and 1 cross product for efficiency.
    
    Parameters:
    -----------
    p : tuple
        Tuple of 4 points (p0, p1, p2, p3) representing the dihedral angle
        p0-p1-p2-p3
    
    Returns:
    --------
    float
        Dihedral angle in degrees
    """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def calculate_dihedral(xyz: NDArray, i: int, j: int, k: int, l: int) -> float:
    """
    Calculate dihedral angle i-j-k-l in degrees (0-indexed).
    
    Parameters:
    -----------
    xyz : NDArray
        Array of atomic coordinates, shape (n_atoms, 3)
    i : int
        Index of first atom (0-indexed)
    j : int
        Index of second atom (0-indexed)
    k : int
        Index of third atom (0-indexed)
    l : int
        Index of fourth atom (0-indexed)
    
    Returns:
    --------
    float
        Dihedral angle in degrees
    """
    return new_dihedral((xyz[i], xyz[j], xyz[k], xyz[l]))


def directional_angle_3d(A, B, C, normal):
    """
    Calculate directional angle between three points in 3D space.
    
    The angle is signed based on the direction of the cross product
    relative to the normal vector.
    
    Parameters:
    -----------
    A : array-like
        First point (3D coordinates)
    B : array-like
        Central point (3D coordinates)
    C : array-like
        Third point (3D coordinates)
    normal : array-like
        Normal vector for determining sign of angle
    
    Returns:
    --------
    float
        Signed angle in degrees (A-B-C)
    """
    A, B, C, normal = map(np.array, (A, B, C, normal))
    v1 = A - B
    v2 = C - B

    # Normalize input vectors
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # Angle (unsigned)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot)

    # Sign of angle via direction of cross product
    cross = np.cross(v1, v2)
    sign = np.sign(np.dot(cross, normal))

    # Apply sign
    signed_angle = np.degrees(angle) * sign % 360
    return signed_angle
