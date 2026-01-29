"""
Force and moment calculations for the Skywalker X8 UAV.

Computes aerodynamic forces, gravitational forces, and propulsive forces/moments
in the body frame.
"""

import numpy as np

from .parameters import X8Parameters
from .transforms import rotation_matrix_zyx


def compute_forces_and_moments(
    state: np.ndarray,
    control: np.ndarray,
    params: X8Parameters,
    wind: np.ndarray = None
) -> np.ndarray:
    """
    Compute the total forces and moments acting on the aircraft.

    Args:
        state: State vector [x, y, z, phi, theta, psi, u, v, w, p, q, r] (12,)
        control: Control vector [delta_e, delta_a, delta_r, delta_t] (4,)
        params: Aircraft parameters
        wind: Wind vector [w_n, w_e, w_d, w_p, w_q, w_r] in body frame (6,)
              First 3 are wind velocities, last 3 are wind angular rates.
              If None, assumes no wind.

    Returns:
        Force/moment vector [Fx, Fy, Fz, Mx, My, Mz] (6,) in body frame
    """
    if wind is None:
        wind = np.zeros(6)

    # Extract state variables
    phi, theta, psi = state[3:6]
    vel = state[6:9]  # [u, v, w]
    rate = state[9:12]  # [p, q, r]

    # Extract control inputs
    elevator = control[0]
    aileron = control[1]
    rudder = control[2]
    throttle = control[3]

    # Body rates including wind
    p = rate[0] + wind[3]
    q = rate[1] + wind[4]
    r = rate[2] + wind[5]

    # Relative velocity (account for wind)
    wind_b = wind[0:3]
    vel_r = vel - wind_b
    u_r, v_r, w_r = vel_r

    # Compute airspeed, angle of attack, and sideslip
    Va = np.sqrt(u_r**2 + v_r**2 + w_r**2)
    if Va < 1e-5:
        Va = 1e-5  # Avoid division by zero

    alpha = np.arctan2(w_r, u_r)  # Angle of attack
    beta = np.arcsin(np.clip(v_r / Va, -1.0, 1.0))  # Sideslip angle

    # Gravitational force in body frame
    fg_N = np.array([0, 0, params.mass * params.gravity])  # Gravity in NED frame
    R = rotation_matrix_zyx(phi, theta, psi)
    fg_b = R.T @ fg_N  # Transform to body frame

    # Dynamic pressure term
    qbar = 0.5 * params.rho * Va**2

    # =====================
    # Longitudinal aerodynamics
    # =====================

    # Lift coefficient and force
    C_L_alpha = params.C_L_0 + params.C_L_alpha * alpha
    f_lift_s = qbar * params.S_wing * (
        C_L_alpha
        + params.C_L_q * params.c / (2 * Va) * q
        + params.C_L_delta_e * elevator
    )

    # Drag coefficient and force
    C_D_alpha = params.C_D_0 + params.C_D_alpha1 * alpha + params.C_D_alpha2 * alpha**2
    C_D_beta = params.C_D_beta1 * beta + params.C_D_beta2 * beta**2
    f_drag_s = qbar * params.S_wing * (
        C_D_alpha + C_D_beta
        + params.C_D_q * params.c / (2 * Va) * q
        + params.C_D_delta_e * elevator**2
    )

    # Pitch moment
    m_a = params.C_m_0 + params.C_m_alpha * alpha
    m = qbar * params.S_wing * params.c * (
        m_a
        + params.C_m_q * params.c / (2 * Va) * q
        + params.C_m_delta_e * elevator
    )

    # =====================
    # Lateral aerodynamics
    # =====================

    # Side force
    f_y = qbar * params.S_wing * (
        params.C_Y_0
        + params.C_Y_beta * beta
        + params.C_Y_p * params.b / (2 * Va) * p
        + params.C_Y_r * params.b / (2 * Va) * r
        + params.C_Y_delta_a * aileron
        + params.C_Y_delta_r * rudder
    )

    # Roll moment
    l = qbar * params.b * params.S_wing * (
        params.C_l_0
        + params.C_l_beta * beta
        + params.C_l_p * params.b / (2 * Va) * p
        + params.C_l_r * params.b / (2 * Va) * r
        + params.C_l_delta_a * aileron
        + params.C_l_delta_r * rudder
    )

    # Yaw moment
    n = qbar * params.b * params.S_wing * (
        params.C_n_0
        + params.C_n_beta * beta
        + params.C_n_p * params.b / (2 * Va) * p
        + params.C_n_r * params.b / (2 * Va) * r
        + params.C_n_delta_a * aileron
        + params.C_n_delta_r * rudder
    )

    # Convert aerodynamic forces from stability frame to body frame
    # Stability frame: x aligned with airspeed, z perpendicular
    R_stab = rotation_matrix_zyx(0, alpha, beta)
    F_aero = R_stab.T @ np.array([-f_drag_s, f_y, -f_lift_s])
    T_aero = np.array([l, m, n])

    # =====================
    # Propulsion
    # =====================

    # Discharge velocity model
    Vd = Va + throttle * (params.k_motor - Va)

    # Thrust force
    F_prop = np.array([
        0.5 * params.rho * params.S_prop * params.C_prop * Vd * (Vd - Va),
        0,
        0
    ])

    # Propeller torque
    T_prop = np.array([
        -params.k_T_P * (params.k_Omega * throttle)**2,
        0,
        0
    ])

    # =====================
    # Total forces and moments
    # =====================
    Force = F_prop + fg_b + F_aero
    Torque = T_aero + T_prop

    return np.concatenate([Force, Torque])
