"""
Aircraft parameters for the Skywalker X8 UAV.

Parameters can be loaded from a MATLAB .mat file or use default values.
Based on: Gryte et al. "Aerodynamic modeling of the Skywalker X8 Fixed-Wing
Unmanned Aerial Vehicle", ICUAS 2018.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .transforms import skew_symmetric


@dataclass
class X8Parameters:
    """
    Skywalker X8 aircraft parameters.

    All units are SI (kg, m, s, rad).
    """

    # Mass and inertia properties
    mass: float = 3.364  # Aircraft mass (kg)
    Jx: float = 1.229  # Moment of inertia about x-axis (kg·m²)
    Jy: float = 0.1702  # Moment of inertia about y-axis (kg·m²)
    Jz: float = 0.8808  # Moment of inertia about z-axis (kg·m²)
    Jxz: float = 0.9343  # Product of inertia (kg·m²)
    r_cg: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # CG position (m)

    # Geometric properties
    S_wing: float = 0.75  # Wing reference area (m²)
    b: float = 2.1  # Wing span (m)
    c: float = 0.35714285714285715  # Mean aerodynamic chord (m)
    S_prop: float = 0.10178760197630929  # Propeller disk area (m²)

    # Propulsion parameters
    k_motor: float = 40.0  # Maximum discharge velocity factor
    k_T_P: float = 0.0  # Motor torque/power coefficient
    k_Omega: float = 0.0  # Motor angular velocity constant
    C_prop: float = 1.0  # Propeller efficiency coefficient

    # Longitudinal aerodynamic coefficients
    C_L_0: float = 0.08673556671610734  # Lift coefficient at α = 0
    C_L_alpha: float = 4.020328244000679  # Lift curve slope (1/rad)
    C_L_q: float = 3.87  # Pitch rate lift coefficient
    C_L_delta_e: float = 0.2780736201734713  # Elevator lift coefficient

    C_D_0: float = 0.01970001181915082  # Drag coefficient at α = 0
    C_D_alpha1: float = 0.07909146315766297  # Linear drag due to α
    C_D_alpha2: float = 1.0554699867680841  # Quadratic drag due to α
    C_D_beta1: float = -0.005842980345415388  # Linear drag due to β
    C_D_beta2: float = 0.14781193079241584  # Quadratic drag due to β
    C_D_q: float = 0.0  # Pitch rate drag coefficient
    C_D_delta_e: float = 0.06334739678180232  # Elevator drag coefficient

    C_m_0: float = 0.02275  # Pitch moment coefficient at α = 0
    C_m_alpha: float = -0.4629  # Pitch moment slope
    C_m_q: float = -1.3012370370370372  # Pitch damping coefficient
    C_m_delta_e: float = -0.2292  # Elevator pitch moment coefficient

    # Lateral aerodynamic coefficients
    C_Y_0: float = 0.0  # Side force coefficient at β = 0
    C_Y_beta: float = -0.22387215700254048  # Side force slope
    C_Y_p: float = -0.13735505263157893  # Roll rate side force coefficient
    C_Y_r: float = 0.08386876842105263  # Yaw rate side force coefficient
    C_Y_delta_a: float = 0.043276402502774876  # Aileron side force coefficient
    C_Y_delta_r: float = 0.0  # Rudder side force coefficient

    C_l_0: float = 0.0  # Roll moment coefficient at β = 0
    C_l_beta: float = -0.08489628639662417  # Roll moment due to sideslip
    C_l_p: float = -0.40419799999999995  # Roll damping coefficient
    C_l_r: float = 0.055520599999999996  # Roll moment due to yaw rate
    C_l_delta_a: float = 0.12018814125782745  # Aileron roll moment coefficient
    C_l_delta_r: float = 0.0  # Rudder roll moment coefficient

    C_n_0: float = 0.0  # Yaw moment coefficient at β = 0
    C_n_beta: float = 0.0283  # Yaw moment due to sideslip
    C_n_p: float = 0.004365511578947368  # Yaw moment due to roll rate
    C_n_r: float = -0.07200000000000001  # Yaw damping coefficient
    C_n_delta_a: float = -0.00339  # Aileron yaw moment coefficient
    C_n_delta_r: float = 0.0  # Rudder yaw moment coefficient

    # Physical constants
    rho: float = 1.225  # Air density at sea level (kg/m³)
    gravity: float = 9.81  # Gravitational acceleration (m/s²)

    # Derived quantities (computed in __post_init__)
    I_cg: np.ndarray = field(default=None, repr=False)
    M_rb: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        """Compute derived quantities after initialization."""
        # Ensure r_cg is a numpy array
        if not isinstance(self.r_cg, np.ndarray):
            self.r_cg = np.array(self.r_cg)

        # Inertia matrix (assuming symmetry wrt xz-plane -> Jxy=Jyz=0)
        self.I_cg = np.array([
            [self.Jx, 0, -self.Jxz],
            [0, self.Jy, 0],
            [-self.Jxz, 0, self.Jz]
        ])

        # Mass-inertia matrix (6x6 rigid body mass matrix)
        S_r_cg = skew_symmetric(self.r_cg)
        self.M_rb = np.block([
            [np.eye(3) * self.mass, -self.mass * S_r_cg],
            [self.mass * S_r_cg, self.I_cg]
        ])

    @classmethod
    def from_mat_file(cls, filepath: Union[str, Path]) -> "X8Parameters":
        """
        Load parameters from a MATLAB .mat file.

        Args:
            filepath: Path to the x8_param.mat file

        Returns:
            X8Parameters instance with values from the file
        """
        try:
            import scipy.io
        except ImportError:
            raise ImportError("scipy is required to load .mat files. Install with: pip install scipy")

        mat_data = scipy.io.loadmat(str(filepath))

        # Extract scalar values from the mat file
        params = {}
        scalar_keys = [
            'mass', 'Jx', 'Jy', 'Jz', 'Jxz', 'S_wing', 'b', 'c', 'S_prop',
            'k_motor', 'k_T_P', 'k_Omega', 'C_prop',
            'C_L_0', 'C_L_alpha', 'C_L_q', 'C_L_delta_e',
            'C_D_0', 'C_D_alpha1', 'C_D_alpha2', 'C_D_beta1', 'C_D_beta2', 'C_D_q', 'C_D_delta_e',
            'C_m_0', 'C_m_alpha', 'C_m_q', 'C_m_delta_e',
            'C_Y_0', 'C_Y_beta', 'C_Y_p', 'C_Y_r', 'C_Y_delta_a', 'C_Y_delta_r',
            'C_l_0', 'C_l_beta', 'C_l_p', 'C_l_r', 'C_l_delta_a', 'C_l_delta_r',
            'C_n_0', 'C_n_beta', 'C_n_p', 'C_n_r', 'C_n_delta_a', 'C_n_delta_r',
        ]

        for key in scalar_keys:
            if key in mat_data:
                value = mat_data[key]
                params[key] = float(value.flatten()[0]) if value.size == 1 else float(value)

        # Handle r_cg separately as it's a vector
        if 'r_cg' in mat_data:
            params['r_cg'] = mat_data['r_cg'].flatten()

        return cls(**params)


# Default trim conditions for level flight at 18 m/s
DEFAULT_TRIM_STATE = np.array([
    0.0, 0.0, -200.0,  # position (N, E, D) - 200m altitude
    0.0, 0.0308, 0.0,  # Euler angles (phi, theta, psi)
    17.9914, 0.0, 0.5551,  # velocity (u, v, w)
    0.0, 0.0, 0.0  # angular rates (p, q, r)
])

DEFAULT_TRIM_CONTROL = np.array([
    0.0370,  # elevator
    0.0000,  # aileron
    0.0,  # rudder
    0.1219  # throttle
])
