"""
Coordinate frame transformation utilities for x8 dynamics.

Includes rotation matrices and kinematic transformation matrices
for converting between NED (North-East-Down) and body frames.
"""

import numpy as np


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix of a 3D vector.

    This implements the cross-product operator: S(a) @ b = a × b

    Args:
        v: 3D vector [v1, v2, v3]

    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rotation_matrix_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Compute rotation matrix from NED frame to body frame using ZYX Euler angles.

    This transforms vectors from body frame to NED frame: v_ned = R @ v_body
    To transform from NED to body: v_body = R.T @ v_ned

    Args:
        phi: Roll angle (rad)
        theta: Pitch angle (rad)
        psi: Yaw angle (rad)

    Returns:
        3x3 rotation matrix
    """
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    return np.array([
        [c_psi * c_theta, -s_psi * c_phi + c_psi * s_theta * s_phi, s_psi * s_phi + c_psi * c_phi * s_theta],
        [s_psi * c_theta, c_psi * c_phi + s_phi * s_theta * s_psi, -c_psi * s_phi + s_theta * s_psi * c_phi],
        [-s_theta, c_theta * s_phi, c_theta * c_phi]
    ])


def euler_kinematic_matrix(phi: float, theta: float, psi: float = None) -> np.ndarray:
    """
    Compute the kinematic transformation matrix from body angular rates to Euler angle rates.

    Theta_dot = T @ Omega, where Omega = [p, q, r]' and Theta = [phi, theta, psi]'

    Args:
        phi: Roll angle (rad)
        theta: Pitch angle (rad)
        psi: Yaw angle (rad) - not used in computation but included for API consistency

    Returns:
        3x3 transformation matrix
    """
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta = np.cos(theta)
    t_theta = np.tan(theta)

    # Avoid division by zero near singularity at theta = ±90°
    if abs(c_theta) < 1e-10:
        c_theta = 1e-10 if c_theta >= 0 else -1e-10

    return np.array([
        [1, s_phi * t_theta, c_phi * t_theta],
        [0, c_phi, -s_phi],
        [0, s_phi / c_theta, c_phi / c_theta]
    ])
