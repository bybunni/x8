"""Tests for utility functions."""

import numpy as np
import jax.numpy as jnp
import pytest

from jax8.utils.matrices import smtrx, rzyx, transformation_matrix


def test_smtrx():
    """Test the skew-symmetric matrix function."""
    # Test case
    v = np.array([1.0, 2.0, 3.0])
    
    # Expected result
    expected = np.array([
        [0.0, -3.0, 2.0],
        [3.0, 0.0, -1.0],
        [-2.0, 1.0, 0.0]
    ])
    
    # Compute result
    result = np.array(smtrx(v))
    
    # Check result
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    # Check skew-symmetry property
    np.testing.assert_allclose(result + result.T, np.zeros((3, 3)), rtol=1e-5)


def test_rzyx():
    """Test the rotation matrix function."""
    # Test cases
    test_cases = [
        # phi, theta, psi
        (0.0, 0.0, 0.0),  # Identity
        (np.pi/4, 0.0, 0.0),  # Pure roll
        (0.0, np.pi/4, 0.0),  # Pure pitch
        (0.0, 0.0, np.pi/4),  # Pure yaw
        (np.pi/4, np.pi/4, np.pi/4),  # Combined rotation
    ]
    
    for phi, theta, psi in test_cases:
        # Compute rotation matrix
        R = np.array(rzyx(phi, theta, psi))
        
        # Check orthogonality property: R * R.T = I
        np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=1e-5)
        
        # Check determinant = 1 (proper rotation)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=1e-5)
        
        # For identity case, check that R = I
        if phi == 0.0 and theta == 0.0 and psi == 0.0:
            np.testing.assert_allclose(R, np.eye(3), rtol=1e-5)


def test_transformation_matrix():
    """Test the transformation matrix function."""
    # Test cases
    test_cases = [
        # phi, theta, psi
        (0.0, 0.1, 0.0),  # Small angles
        (np.pi/4, np.pi/6, 0.0),  # Larger angles
    ]
    
    for phi, theta, psi in test_cases:
        # Create Euler angles vector
        euler_ang = jnp.array([phi, theta, psi])
        
        # Compute transformation matrix
        T = np.array(transformation_matrix(euler_ang))
        
        # For the test, apply the transformation to a sample body rate
        omega_body = np.array([0.1, 0.2, 0.3])
        euler_rates = T @ omega_body
        
        # Check specific properties (these would need to be computed analytically)
        # For now, just check that the transformation matrix has the correct shape
        assert T.shape == (3, 3)