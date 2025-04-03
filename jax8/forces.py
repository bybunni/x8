"""Forces and moments computation for the X8 UAV model."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from jax8.parameters import X8Parameters
from jax8.transformations import inertial_to_body, wind_to_body
from jax8.utils.matrices import rzyx


@jax.jit
def compute_airspeed_angles(vel_body: jnp.ndarray, wind_body: jnp.ndarray) -> Tuple[float, float, float]:
    """
    Compute airspeed, angle of attack, and sideslip angle.
    
    Args:
        vel_body: Body-frame velocity [u, v, w]
        wind_body: Body-frame wind velocity [uw, vw, ww]
        
    Returns:
        Tuple of (airspeed, angle of attack, sideslip angle)
    """
    # Relative velocity
    vel_rel = vel_body - wind_body
    u_r, v_r, w_r = vel_rel
    
    # Compute airspeed, angle of attack, sideslip
    Va = jnp.sqrt(u_r**2 + v_r**2 + w_r**2)
    
    # Ensure Va is not exactly zero to avoid division by zero
    Va = jnp.maximum(Va, 1e-5)
    
    alpha = jnp.arctan2(w_r, u_r)
    beta = jnp.arcsin(v_r / Va)
    
    return Va, alpha, beta


@jax.jit
def compute_forces_moments(
    t: float,
    state: jnp.ndarray,
    control_inputs: jnp.ndarray,
    wind: jnp.ndarray,
    params: Dict
) -> jnp.ndarray:
    """
    Compute forces and moments acting on the X8 UAV.
    
    Args:
        t: Current time (seconds)
        state: State vector [pos, Theta, vel, rates]
        control_inputs: Control inputs [elevator, aileron, rudder, throttle]
        wind: Wind components [uw, vw, ww, wp, wq, wr] in body frame
        params: Dictionary of X8 parameters
        
    Returns:
        Vector of forces and moments [Fx, Fy, Fz, Mx, My, Mz]
    """
    # Extract state components
    pos = state[0:3]
    euler_angles = state[3:6]
    vel_body = state[6:9]
    rates = state[9:12]
    
    # Extract control inputs
    elevator, aileron, rudder, throttle = control_inputs
    
    # Extract parameters
    rho = params['rho']
    mass = params['mass']
    gravity = params['gravity']
    S_wing = params['S_wing']
    b = params['b']
    c = params['c']
    
    # Extract body rates and add wind components
    p = rates[0] + wind[3]
    q = rates[1] + wind[4]
    r = rates[2] + wind[5]
    
    # Compute airspeed, angle of attack, sideslip
    wind_body = wind[0:3]
    Va, alpha, beta = compute_airspeed_angles(vel_body, wind_body)
    
    # Compute gravitational force
    fg_inertial = jnp.array([0, 0, mass * gravity])
    fg_body = inertial_to_body(euler_angles, fg_inertial)
    
    # --- Longitudinal forces and moments ---
    
    # Lift coefficient
    C_L_alpha = params['C_L_0'] + params['C_L_alpha'] * alpha
    C_L = C_L_alpha + params['C_L_q'] * c / (2 * Va) * q + params['C_L_delta_e'] * elevator
    
    # Drag coefficient
    C_D_alpha = params['C_D_0'] + params['C_D_alpha1'] * alpha + params['C_D_alpha2'] * alpha**2
    C_D_beta = params['C_D_beta1'] * beta + params['C_D_beta2'] * beta**2
    C_D = C_D_alpha + C_D_beta + params['C_D_q'] * c / (2 * Va) * q + params['C_D_delta_e'] * elevator**2
    
    # Compute lift and drag forces in stability frame
    f_lift_stab = 0.5 * rho * Va**2 * S_wing * C_L
    f_drag_stab = 0.5 * rho * Va**2 * S_wing * C_D
    
    # Pitching moment
    m_a = params['C_m_0'] + params['C_m_alpha'] * alpha
    m_pitch = 0.5 * rho * Va**2 * S_wing * c * (
        m_a + params['C_m_q'] * c / (2 * Va) * q + params['C_m_delta_e'] * elevator
    )
    
    # --- Lateral forces and moments ---
    
    # Side force
    C_Y = params['C_Y_0'] + params['C_Y_beta'] * beta + params['C_Y_p'] * b / (2 * Va) * p + \
          params['C_Y_r'] * b / (2 * Va) * r + params['C_Y_delta_a'] * aileron + \
          params['C_Y_delta_r'] * rudder
    f_side = 0.5 * rho * Va**2 * S_wing * C_Y
    
    # Rolling moment
    C_l = params['C_l_0'] + params['C_l_beta'] * beta + params['C_l_p'] * b / (2 * Va) * p + \
          params['C_l_r'] * b / (2 * Va) * r + params['C_l_delta_a'] * aileron + \
          params['C_l_delta_r'] * rudder
    l_roll = 0.5 * rho * Va**2 * b * S_wing * C_l
    
    # Yawing moment
    C_n = params['C_n_0'] + params['C_n_beta'] * beta + params['C_n_p'] * b / (2 * Va) * p + \
          params['C_n_r'] * b / (2 * Va) * r + params['C_n_delta_a'] * aileron + \
          params['C_n_delta_r'] * rudder
    n_yaw = 0.5 * rho * Va**2 * b * S_wing * C_n
    
    # Convert aerodynamic forces from stability to body frame
    F_aero_stab = jnp.array([-f_drag_stab, f_side, -f_lift_stab])
    F_aero = wind_to_body(alpha, beta, F_aero_stab)
    
    # Aerodynamic moments in body frame
    T_aero = jnp.array([l_roll, m_pitch, n_yaw])
    
    # Propulsive forces
    Vd = Va + throttle * (params['k_motor'] - Va)  # Discharge velocity
    F_prop = jnp.array([
        0.5 * rho * params['S_prop'] * params['C_prop'] * Vd * (Vd - Va),
        0,
        0
    ])
    
    # Propulsive moments
    T_prop = jnp.array([
        -params['k_T_P'] * (params['k_Omega'] * throttle)**2,
        0,
        0
    ])
    
    # Sum all forces and moments
    forces = F_prop + fg_body + F_aero
    moments = T_aero + T_prop
    
    # Combine forces and moments
    output = jnp.concatenate([forces, moments])
    
    return output