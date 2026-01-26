"""
X8 Dynamics - Python module for Skywalker X8 UAV simulation.

This module provides a state-space model of the Skywalker X8 fixed-wing UAV
dynamics, based on the MATLAB implementation from Gryte et al. (2018).

Main Components:
    - X8Dynamics: Core dynamics model with dxdt() method for integration
    - X8Parameters: Aircraft parameters (loadable from .mat file)
    - X8Controller: Cascaded PID autopilot
    - X8Simulator: Discrete-time simulation wrapper

Basic Usage:
    from x8_dynamics import X8Dynamics, X8Parameters

    # Create dynamics model
    params = X8Parameters()
    dynamics = X8Dynamics(params)

    # Create initial state
    x = dynamics.create_initial_state(
        position=(0, 0, -200),  # NED coordinates
        velocity=(18, 0, 0)     # Forward flight at 18 m/s
    )

    # Define control
    u = np.array([0.037, 0.0, 0.0, 0.12])  # Trim control

    # Compute state derivative
    x_dot = dynamics.dxdt(x, u)

    # Integrate (e.g., Euler)
    dt = 0.01
    x_next = x + dt * x_dot

Reference:
    K. Gryte, R. Hann, M. Alam, J. Rohac, T. A. Johansen, T. I. Fossen,
    "Aerodynamic modeling of the Skywalker X8 Fixed-Wing Unmanned Aerial Vehicle",
    International Conference on Unmanned Aircraft Systems, Dallas, 2018.
"""

from .parameters import (
    X8Parameters,
    DEFAULT_TRIM_STATE,
    DEFAULT_TRIM_CONTROL,
)

from .dynamics import X8Dynamics

from .controller import (
    X8Controller,
    ControllerGains,
    ReferenceTrajectory,
)

from .simulator import (
    X8Simulator,
    SimulationResult,
    simulate_x8,
)

from .forces import compute_forces_and_moments

from .transforms import (
    rotation_matrix_zyx,
    euler_kinematic_matrix,
    skew_symmetric,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "X8Dynamics",
    "X8Parameters",
    "X8Controller",
    "X8Simulator",
    # Data classes
    "ControllerGains",
    "SimulationResult",
    "ReferenceTrajectory",
    # Functions
    "simulate_x8",
    "compute_forces_and_moments",
    # Utilities
    "rotation_matrix_zyx",
    "euler_kinematic_matrix",
    "skew_symmetric",
    # Constants
    "DEFAULT_TRIM_STATE",
    "DEFAULT_TRIM_CONTROL",
]
