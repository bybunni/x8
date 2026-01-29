"""
Autopilot controller for the Skywalker X8 UAV.

Implements cascaded PID control for altitude, heading, and velocity tracking.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

from .transforms import rotation_matrix_zyx
from .parameters import DEFAULT_TRIM_CONTROL


@dataclass
class ControllerGains:
    """PID gains for the X8 autopilot."""

    # Altitude loop -> pitch angle reference
    kp_h: float = -0.025
    ki_h: float = 0.0

    # Pitch angle -> elevator
    kp_theta: float = 0.1
    ki_theta: float = 0.0
    kd_theta: float = -0.01

    # Velocity -> throttle
    kp_V: float = -0.05
    ki_V: float = -0.01

    # Heading -> roll angle reference
    kp_chi: float = -0.05
    ki_chi: float = 0.0

    # Roll angle -> aileron
    kp_phi: float = -0.5
    ki_phi: float = 0.0
    kd_phi: float = 0.0


class X8Controller:
    """
    Cascaded PID autopilot for the Skywalker X8.

    Control architecture:
        - Heading control: chi_ref -> phi_ref -> delta_a (aileron)
        - Altitude control: h_ref -> theta_ref -> delta_e (elevator)
        - Velocity control: V_ref -> delta_t (throttle)
        - Rudder: set to 0 (not used)

    Usage:
        controller = X8Controller()
        controller.set_reference(velocity=18, heading=0, altitude=200)

        # In simulation loop:
        u = controller.compute_control(state, dt)
    """

    def __init__(
        self,
        gains: ControllerGains = None,
        trim_control: np.ndarray = None
    ):
        """
        Initialize the controller.

        Args:
            gains: Controller gains. If None, uses default gains.
            trim_control: Trim control values to add to computed deltas.
        """
        self.gains = gains if gains is not None else ControllerGains()
        self.trim_control = trim_control if trim_control is not None else DEFAULT_TRIM_CONTROL.copy()

        # Reference commands
        self.V_ref = 18.0  # Airspeed (m/s)
        self.chi_ref = 0.0  # Heading (rad)
        self.h_ref = 200.0  # Altitude (m)

        # Integral states
        self._i_phi = 0.0
        self._i_theta = 0.0
        self._i_h = 0.0
        self._i_V = 0.0
        self._i_chi = 0.0

        # Store last time for dt computation if needed
        self._last_t = None

    def reset(self):
        """Reset integral states to zero."""
        self._i_phi = 0.0
        self._i_theta = 0.0
        self._i_h = 0.0
        self._i_V = 0.0
        self._i_chi = 0.0
        self._last_t = None

    def set_reference(
        self,
        velocity: float = None,
        heading: float = None,
        altitude: float = None
    ):
        """
        Set reference commands for the autopilot.

        Args:
            velocity: Desired airspeed (m/s)
            heading: Desired heading angle (rad)
            altitude: Desired altitude (m, positive up)
        """
        if velocity is not None:
            self.V_ref = velocity
        if heading is not None:
            self.chi_ref = heading
        if altitude is not None:
            self.h_ref = altitude

    def compute_control(
        self,
        state: np.ndarray,
        dt: float = None,
        t: float = None
    ) -> np.ndarray:
        """
        Compute control inputs based on current state and references.

        Args:
            state: Current state vector (12,)
            dt: Time step for integral update. If None and t is provided,
                computed from last call time.
            t: Current simulation time. Used to compute dt if dt not provided.

        Returns:
            Control vector [delta_e, delta_a, delta_r, delta_t] (4,)
        """
        # Compute dt if not provided
        if dt is None:
            if t is not None and self._last_t is not None:
                dt = t - self._last_t
            else:
                dt = 0.01  # Default 10ms step
        if t is not None:
            self._last_t = t

        # Extract state components
        pos = state[0:3]
        Theta = state[3:6]
        vel = state[6:9]
        Omega = state[9:12]

        phi = Theta[0]
        theta = Theta[1]
        psi = Theta[2]
        p = Omega[0]
        q = Omega[1]

        # Current measurements
        V = np.linalg.norm(vel)  # Airspeed
        h = -pos[2]  # Altitude (NED: down is positive)

        # Compute heading from ground velocity
        R = rotation_matrix_zyx(phi, theta, psi)
        v_n = R @ vel  # Velocity in NED frame
        chi = np.arctan2(v_n[1], v_n[0])  # Course angle

        # =====================
        # Heading control loop
        # =====================
        chi_error = self._wrap_angle(chi - self.chi_ref)
        self._i_chi += chi_error * dt
        phi_ref = self.gains.kp_chi * chi_error + self.gains.ki_chi * self._i_chi

        # Roll control loop
        phi_error = phi - phi_ref
        self._i_phi += phi_error * dt
        delta_a = (
            self.gains.kp_phi * phi_error
            + self.gains.ki_phi * self._i_phi
            - self.gains.kd_phi * p
        )

        # =====================
        # Altitude control loop
        # =====================
        h_error = h - self.h_ref
        self._i_h += h_error * dt
        theta_ref = self.gains.kp_h * h_error + self.gains.ki_h * self._i_h

        # Pitch control loop
        theta_error = theta - theta_ref
        self._i_theta += theta_error * dt
        delta_e = (
            self.gains.kp_theta * theta_error
            + self.gains.ki_theta * self._i_theta
            - self.gains.kd_theta * q
        )

        # =====================
        # Velocity control loop
        # =====================
        V_error = V - self.V_ref
        self._i_V += V_error * dt
        delta_t = self.gains.kp_V * V_error + self.gains.ki_V * self._i_V

        # Rudder (not used)
        delta_r = 0.0

        # Assemble control vector and add trim
        u = np.array([delta_e, delta_a, delta_r, delta_t]) + self.trim_control

        # Apply saturation
        u = self._saturate(u)

        return u

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _saturate(u: np.ndarray) -> np.ndarray:
        """Apply control saturation limits."""
        u_sat = np.clip(u, -1.0, 1.0)
        u_sat[3] = np.clip(u[3], 0.0, 1.0)  # Throttle is 0-1
        return u_sat


class ReferenceTrajectory:
    """
    Reference trajectory generator matching the original MATLAB simulation.

    Generates time-varying references for heading and altitude.
    """

    def __init__(
        self,
        V_ref: float = 18.0,
        initial_heading: float = 0.0,
        final_heading: float = np.deg2rad(15),
        heading_ramp_start: float = 20.0,
        heading_ramp_end: float = 35.0,
        initial_altitude: float = 200.0,
        final_altitude: float = 250.0,
        climb_start: float = 75.0,
        climb_end: float = 325.0
    ):
        """
        Initialize reference trajectory.

        Args:
            V_ref: Reference airspeed (m/s)
            initial_heading: Initial heading (rad)
            final_heading: Final heading after ramp (rad)
            heading_ramp_start: Time to start heading ramp (s)
            heading_ramp_end: Time to end heading ramp (s)
            initial_altitude: Initial altitude reference (m)
            final_altitude: Final altitude reference (m)
            climb_start: Time to start climb (s)
            climb_end: Time to end climb (s)
        """
        self.V_ref = V_ref
        self.initial_heading = initial_heading
        self.final_heading = final_heading
        self.heading_ramp_start = heading_ramp_start
        self.heading_ramp_end = heading_ramp_end
        self.initial_altitude = initial_altitude
        self.final_altitude = final_altitude
        self.climb_start = climb_start
        self.climb_end = climb_end

        # Compute rates
        self.heading_rate = (final_heading - initial_heading) / (heading_ramp_end - heading_ramp_start)
        self.climb_rate = (final_altitude - initial_altitude) / (climb_end - climb_start)

    def get_reference(self, t: float) -> Tuple[float, float, float]:
        """
        Get reference values at time t.

        Args:
            t: Current simulation time (s)

        Returns:
            Tuple of (V_ref, chi_ref, h_ref)
        """
        # Heading reference
        if t < self.heading_ramp_start:
            chi_ref = self.initial_heading
        elif t < self.heading_ramp_end:
            chi_ref = self.initial_heading + (t - self.heading_ramp_start) * self.heading_rate
        else:
            chi_ref = self.final_heading

        # Altitude reference
        if t < self.climb_start:
            h_ref = self.initial_altitude
        elif t < self.climb_end:
            h_ref = self.initial_altitude + (t - self.climb_start) * self.climb_rate
        else:
            h_ref = self.final_altitude

        return self.V_ref, chi_ref, h_ref
