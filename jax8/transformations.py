"""Coordinate transformations and rotation matrices for UAV dynamics."""

import jax
import jax.numpy as jnp

from jax8.utils.matrices import rzyx, transformation_matrix


@jax.jit
def body_to_inertial(euler_angles: jnp.ndarray, vel_body: jnp.ndarray) -> jnp.ndarray:
    """
    Transform velocity from body to inertial (NED) frame.
    
    Args:
        euler_angles: Euler angles [phi, theta, psi] in radians
        vel_body: Velocity in body frame [u, v, w]
        
    Returns:
        Velocity in inertial frame [vn, ve, vd]
    """
    phi, theta, psi = euler_angles
    R = rzyx(phi, theta, psi)
    vel_inertial = R @ vel_body
    return vel_inertial


@jax.jit
def body_rate_to_euler_rate(euler_angles: jnp.ndarray, omega_body: jnp.ndarray) -> jnp.ndarray:
    """
    Transform angular rates from body to Euler rates.
    
    Args:
        euler_angles: Euler angles [phi, theta, psi] in radians
        omega_body: Angular rates in body frame [p, q, r]
        
    Returns:
        Euler angle rates [phi_dot, theta_dot, psi_dot]
    """
    T = transformation_matrix(euler_angles)
    euler_rates = T @ omega_body
    return euler_rates


@jax.jit
def inertial_to_body(euler_angles: jnp.ndarray, vec_inertial: jnp.ndarray) -> jnp.ndarray:
    """
    Transform a vector from inertial (NED) to body frame.
    
    Args:
        euler_angles: Euler angles [phi, theta, psi] in radians
        vec_inertial: Vector in inertial frame
        
    Returns:
        Vector in body frame
    """
    phi, theta, psi = euler_angles
    R = rzyx(phi, theta, psi)
    vec_body = R.T @ vec_inertial  # Transpose of rotation matrix for inverse transform
    return vec_body


@jax.jit
def wind_to_body(alpha: float, beta: float, vec_wind: jnp.ndarray) -> jnp.ndarray:
    """
    Transform a vector from wind frame to body frame.
    
    Args:
        alpha: Angle of attack (radians)
        beta: Sideslip angle (radians)
        vec_wind: Vector in wind frame
        
    Returns:
        Vector in body frame
    """
    # Rotation from wind to body frame
    R = rzyx(0, alpha, beta).T
    vec_body = R @ vec_wind
    return vec_body