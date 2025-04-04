"""Rigid body dynamics for the X8 UAV model."""

from typing import Dict, Tuple, Callable, Any

import jax
import jax.numpy as jnp

from jax8.forces import compute_forces_moments
from jax8.transformations import body_to_inertial, body_rate_to_euler_rate
from jax8.utils.matrices import smtrx


@jax.jit
def rigid_body_dynamics(
    t: float,
    state: jnp.ndarray,
    control_inputs: jnp.ndarray,
    wind: jnp.ndarray,
    params: Dict
) -> jnp.ndarray:
    """
    Compute state derivatives for the X8 UAV rigid body dynamics.
    
    Args:
        t: Current time (seconds)
        state: State vector [pos, Theta, vel, rates]
            pos: Position in NED frame [north, east, down]
            Theta: Euler angles [phi, theta, psi]
            vel: Body-frame velocity [u, v, w]
            rates: Body-frame angular rates [p, q, r]
        control_inputs: Control inputs [elevator, aileron, rudder, throttle]
        wind: Wind components [uw, vw, ww, wp, wq, wr] in body frame
        params: Dictionary of X8 parameters
        
    Returns:
        State derivative vector [pos_dot, Theta_dot, vel_dot, rates_dot]
    """
    # Extract state components
    pos = state[0:3]
    euler_angles = state[3:6]
    vel = state[6:9]
    omega = state[9:12]
    
    # Compute forces and moments
    tau = compute_forces_moments(t, state, control_inputs, wind, params)
    forces = tau[0:3]
    moments = tau[3:6]
    
    # Extract parameters
    M_rb = params['M_rb']
    r_cg = params['r_cg']
    mass = params['mass']
    I_cg = params['I_cg']
    
    # Compute Coriolis matrix
    C_rb = jnp.block([
        [jnp.zeros((3, 3)), -mass * smtrx(vel) - mass * smtrx(omega) @ smtrx(r_cg)],
        [-mass * smtrx(vel) + mass * smtrx(r_cg) @ smtrx(omega), -smtrx(I_cg @ omega)]
    ])
    
    # Compute state derivatives
    ny_dot = jnp.linalg.solve(M_rb, tau - C_rb @ jnp.concatenate([vel, omega]))
    vel_dot = ny_dot[0:3]
    omega_dot = ny_dot[3:6]
    
    # Transform body-frame velocity to inertial-frame
    pos_dot = body_to_inertial(euler_angles, vel)
    
    # Transform body-frame angular rates to Euler rates
    euler_dot = body_rate_to_euler_rate(euler_angles, omega)
    
    # Combine all state derivatives
    x_dot = jnp.concatenate([pos_dot, euler_dot, vel_dot, omega_dot])
    
    return x_dot


def create_dynamics_function(
    params: Dict,
    control_function: Callable[[float, jnp.ndarray, Dict], jnp.ndarray],
    wind_function: Callable[[float, jnp.ndarray], jnp.ndarray] = None
) -> Callable[[float, jnp.ndarray, Any], jnp.ndarray]:
    """
    Create a dynamics function for use with ODE solvers.
    
    Args:
        params: Dictionary of X8 parameters
        control_function: Function that computes control inputs given time and state
        wind_function: Function that computes wind components given time and state
            (defaults to zero wind if not provided)
        
    Returns:
        Function that computes state derivatives given time and state
    """
    # Create default wind function if not provided
    if wind_function is None:
        def zero_wind(t: float, state: jnp.ndarray) -> jnp.ndarray:
            return jnp.zeros(6)
        wind_function = zero_wind
    
    def dynamics_function(t: float, state: jnp.ndarray, args=None) -> jnp.ndarray:
        """
        Compute state derivatives for the X8 UAV.
        
        Args:
            t: Current time (seconds)
            state: State vector [pos, Theta, vel, rates]
            args: Additional arguments (unused, included for compatibility with diffrax)
            
        Returns:
            State derivative vector [pos_dot, Theta_dot, vel_dot, rates_dot]
        """
        # Compute control inputs
        control_inputs = control_function(t, state, params)
        
        # Compute wind components
        wind = wind_function(t, state)
        
        # Compute state derivatives
        state_dot = rigid_body_dynamics(t, state, control_inputs, wind, params)
        
        return state_dot
    
    return dynamics_function