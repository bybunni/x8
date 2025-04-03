"""Matrix utility functions for the JAX8 package."""

from typing import Any

import jax
import jax.numpy as jnp


def smtrx(v: jnp.ndarray) -> jnp.ndarray:
    """
    Create a skew-symmetric matrix from a 3-element vector.
    
    This is the matrix equivalent of the cross product operator.
    
    Args:
        v: A 3-element vector [v1, v2, v3]
        
    Returns:
        A 3x3 skew-symmetric matrix:
        [  0  -v3  v2]
        [ v3    0 -v1]
        [-v2   v1   0]
    """
    v = jnp.asarray(v, dtype=jnp.float32)
    return jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=jnp.float32)


@jax.jit
def rzyx(phi: float, theta: float, psi: float) -> jnp.ndarray:
    """
    Compute the rotation matrix from Euler angles using ZYX convention.
    
    Args:
        phi: Roll angle (radians)
        theta: Pitch angle (radians)
        psi: Yaw angle (radians)
        
    Returns:
        A 3x3 rotation matrix
    """
    c_psi = jnp.cos(psi)
    s_psi = jnp.sin(psi)
    c_theta = jnp.cos(theta)
    s_theta = jnp.sin(theta)
    c_phi = jnp.cos(phi)
    s_phi = jnp.sin(phi)
    
    R = jnp.array([
        [c_psi * c_theta, -s_psi * c_phi + c_psi * s_theta * s_phi, s_psi * s_phi + c_psi * c_phi * s_theta],
        [s_psi * c_theta, c_psi * c_phi + s_phi * s_theta * s_psi, -c_psi * s_phi + s_theta * s_psi * c_phi],
        [-s_theta, c_theta * s_phi, c_theta * c_phi]
    ], dtype=jnp.float32)
    
    return R


@jax.jit
def transformation_matrix(euler_ang: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the transformation matrix from body rates to Euler rates.
    
    Args:
        euler_ang: Euler angles [phi, theta, psi] in radians
        
    Returns:
        A 3x3 transformation matrix
    """
    phi = euler_ang[0]
    theta = euler_ang[1]
    psi = euler_ang[2]  # Not used in the transformation matrix
    
    T = jnp.array([
        [1, jnp.sin(phi) * jnp.sin(theta) / jnp.cos(theta), jnp.cos(phi) * jnp.sin(theta) / jnp.cos(theta)],
        [0, jnp.cos(phi), -jnp.sin(phi)],
        [0, jnp.sin(phi) / jnp.cos(theta), jnp.cos(phi) / jnp.cos(theta)]
    ], dtype=jnp.float32)
    
    return T