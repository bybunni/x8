"""
State space dynamics model for the Skywalker X8 UAV.

Implements the nonlinear 6-DOF rigid-body dynamics based on Fossen's formulation.
Provides a dxdt method for use with numerical integrators.
"""

import numpy as np
from typing import Optional, Callable

from .parameters import X8Parameters
from .transforms import skew_symmetric, rotation_matrix_zyx, euler_kinematic_matrix
from .forces import compute_forces_and_moments


class X8Dynamics:
    """
    State space model for the Skywalker X8 fixed-wing UAV.

    State vector (12 elements):
        x[0:3]  - Position in NED frame [x_n, x_e, x_d] (m)
        x[3:6]  - Euler angles [phi, theta, psi] (rad)
        x[6:9]  - Body-frame velocity [u, v, w] (m/s)
        x[9:12] - Body-frame angular rates [p, q, r] (rad/s)

    Control vector (4 elements):
        u[0] - Elevator deflection [-1, 1]
        u[1] - Aileron deflection [-1, 1]
        u[2] - Rudder deflection [-1, 1]
        u[3] - Throttle [0, 1]

    Usage:
        params = X8Parameters()
        dynamics = X8Dynamics(params)

        # Compute state derivative for integration
        x_dot = dynamics.dxdt(x, u)

        # Integrate one step with Euler method
        x_next = x + dt * x_dot

        # Or use with scipy.integrate.odeint
        from scipy.integrate import odeint
        solution = odeint(lambda x, t: dynamics.dxdt(x, u), x0, t_span)
    """

    STATE_DIM = 12
    CONTROL_DIM = 4

    # State indices
    POS_X, POS_Y, POS_Z = 0, 1, 2
    PHI, THETA, PSI = 3, 4, 5
    U, V, W = 6, 7, 8
    P, Q, R = 9, 10, 11

    def __init__(self, params: X8Parameters = None):
        """
        Initialize the dynamics model.

        Args:
            params: Aircraft parameters. If None, uses default X8 parameters.
        """
        self.params = params if params is not None else X8Parameters()
        self._wind = np.zeros(6)

    @property
    def wind(self) -> np.ndarray:
        """Get current wind vector [w_n, w_e, w_d, w_p, w_q, w_r]."""
        return self._wind

    @wind.setter
    def wind(self, value: np.ndarray):
        """Set wind vector [w_n, w_e, w_d, w_p, w_q, w_r]."""
        self._wind = np.asarray(value)

    def dxdt(
        self,
        x: np.ndarray,
        u: np.ndarray,
        t: float = 0.0,
        wind: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the state derivative (x_dot) for the given state and control.

        This is the core method for numerical integration. Given the current
        state x and control input u, it returns the time derivative of the
        state vector.

        Args:
            x: State vector (12,) - [pos(3), euler(3), vel(3), omega(3)]
            u: Control vector (4,) - [elevator, aileron, rudder, throttle]
            t: Current time (optional, for time-varying wind/control)
            wind: Wind vector (6,) - [w_n, w_e, w_d, w_p, w_q, w_r]
                  If None, uses self.wind

        Returns:
            x_dot: State derivative vector (12,)
        """
        if wind is None:
            wind = self._wind

        # Ensure inputs are numpy arrays
        x = np.asarray(x)
        u = np.asarray(u)

        # Saturate control inputs
        u = self._saturate_control(u)

        # Extract state components
        pos = x[0:3]
        Theta = x[3:6]  # [phi, theta, psi]
        vel = x[6:9]  # [u, v, w]
        Omega = x[9:12]  # [p, q, r]

        phi, theta, psi = Theta

        # Compute forces and moments
        tau = compute_forces_and_moments(x, u, self.params, wind)

        # Build Coriolis-centrifugal matrix (Fossen 3.56)
        S_vel = skew_symmetric(vel)
        S_Omega = skew_symmetric(Omega)
        S_r_cg = skew_symmetric(self.params.r_cg)

        C_rb = np.block([
            [np.zeros((3, 3)), -self.params.mass * S_vel - self.params.mass * S_Omega @ S_r_cg],
            [-self.params.mass * S_vel + self.params.mass * S_r_cg @ S_Omega, -skew_symmetric(self.params.I_cg @ Omega)]
        ])

        # Solve for velocity and angular rate derivatives
        # M_rb * nu_dot = tau - C_rb * nu
        nu = np.concatenate([vel, Omega])
        nu_dot = np.linalg.solve(self.params.M_rb, tau - C_rb @ nu)

        vel_dot = nu_dot[0:3]
        Omega_dot = nu_dot[3:6]

        # Position derivative (transform body velocity to NED frame)
        R = rotation_matrix_zyx(phi, theta, psi)
        pos_dot = R @ vel

        # Euler angle derivatives
        T = euler_kinematic_matrix(phi, theta, psi)
        Theta_dot = T @ Omega

        # Assemble full state derivative
        x_dot = np.concatenate([pos_dot, Theta_dot, vel_dot, Omega_dot])

        return x_dot

    def _saturate_control(self, u: np.ndarray) -> np.ndarray:
        """
        Apply saturation limits to control inputs.

        Args:
            u: Control vector [elevator, aileron, rudder, throttle]

        Returns:
            Saturated control vector
        """
        u_sat = np.clip(u, -1.0, 1.0)
        u_sat[3] = np.clip(u[3], 0.0, 1.0)  # Throttle is 0-1
        return u_sat

    def get_airspeed(self, x: np.ndarray, wind: np.ndarray = None) -> float:
        """
        Compute the airspeed from the current state.

        Args:
            x: State vector
            wind: Wind vector (first 3 components are wind velocity in body frame)

        Returns:
            Airspeed magnitude (m/s)
        """
        if wind is None:
            wind = self._wind
        vel_r = x[6:9] - wind[0:3]
        return np.linalg.norm(vel_r)

    def get_angle_of_attack(self, x: np.ndarray, wind: np.ndarray = None) -> float:
        """
        Compute the angle of attack from the current state.

        Args:
            x: State vector
            wind: Wind vector

        Returns:
            Angle of attack (rad)
        """
        if wind is None:
            wind = self._wind
        vel_r = x[6:9] - wind[0:3]
        return np.arctan2(vel_r[2], vel_r[0])

    def get_sideslip(self, x: np.ndarray, wind: np.ndarray = None) -> float:
        """
        Compute the sideslip angle from the current state.

        Args:
            x: State vector
            wind: Wind vector

        Returns:
            Sideslip angle (rad)
        """
        if wind is None:
            wind = self._wind
        vel_r = x[6:9] - wind[0:3]
        Va = np.linalg.norm(vel_r)
        if Va < 1e-5:
            return 0.0
        return np.arcsin(np.clip(vel_r[1] / Va, -1.0, 1.0))

    def get_heading(self, x: np.ndarray) -> float:
        """
        Compute the course/heading angle from the current state.

        The heading is computed from the ground velocity in the NED frame.

        Args:
            x: State vector

        Returns:
            Heading angle (rad)
        """
        phi, theta, psi = x[3:6]
        vel = x[6:9]
        R = rotation_matrix_zyx(phi, theta, psi)
        v_n = R @ vel  # Velocity in NED frame
        return np.arctan2(v_n[1], v_n[0])

    def get_altitude(self, x: np.ndarray) -> float:
        """
        Get altitude (positive up) from state.

        Args:
            x: State vector

        Returns:
            Altitude (m) - positive above ground
        """
        return -x[2]  # NED convention: down is positive

    @staticmethod
    def create_initial_state(
        position: tuple = (0, 0, -200),
        euler_angles: tuple = (0, 0, 0),
        velocity: tuple = (18, 0, 0),
        angular_rates: tuple = (0, 0, 0)
    ) -> np.ndarray:
        """
        Create an initial state vector.

        Args:
            position: (x_n, x_e, x_d) in NED frame (m). x_d should be negative for altitude.
            euler_angles: (phi, theta, psi) roll, pitch, yaw (rad)
            velocity: (u, v, w) body-frame velocity (m/s)
            angular_rates: (p, q, r) body angular rates (rad/s)

        Returns:
            State vector (12,)
        """
        return np.array([
            *position,
            *euler_angles,
            *velocity,
            *angular_rates
        ])
