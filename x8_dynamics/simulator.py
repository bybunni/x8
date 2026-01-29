"""
Simulation wrapper for the X8 dynamics model.

Provides convenient methods for running discrete-time simulations
with various integration schemes.
"""

import numpy as np
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass

from .dynamics import X8Dynamics
from .parameters import X8Parameters, DEFAULT_TRIM_STATE, DEFAULT_TRIM_CONTROL
from .controller import X8Controller, ReferenceTrajectory


@dataclass
class SimulationResult:
    """Container for simulation results."""
    time: np.ndarray  # Time vector (N,)
    states: np.ndarray  # State history (N, 12)
    controls: np.ndarray  # Control history (N, 4)

    @property
    def position(self) -> np.ndarray:
        """Position [x_n, x_e, x_d] (N, 3)."""
        return self.states[:, 0:3]

    @property
    def euler_angles(self) -> np.ndarray:
        """Euler angles [phi, theta, psi] in rad (N, 3)."""
        return self.states[:, 3:6]

    @property
    def velocity(self) -> np.ndarray:
        """Body velocity [u, v, w] (N, 3)."""
        return self.states[:, 6:9]

    @property
    def angular_rates(self) -> np.ndarray:
        """Body angular rates [p, q, r] (N, 3)."""
        return self.states[:, 9:12]

    @property
    def altitude(self) -> np.ndarray:
        """Altitude (positive up) (N,)."""
        return -self.states[:, 2]

    @property
    def airspeed(self) -> np.ndarray:
        """Airspeed magnitude (N,)."""
        return np.linalg.norm(self.states[:, 6:9], axis=1)


class X8Simulator:
    """
    Discrete-time simulator for the X8 dynamics.

    Supports various integration methods and can be used with external
    simulation engines that step in discrete timestamps.

    Usage:
        # Create simulator
        sim = X8Simulator(dt=0.01)

        # Initialize state
        x = sim.dynamics.create_initial_state()

        # Step simulation
        for t in range(1000):
            u = my_controller(x)  # Your control law
            x = sim.step(x, u)

        # Or run complete simulation with built-in controller
        result = sim.run_simulation(duration=400, use_controller=True)
    """

    def __init__(
        self,
        dt: float = 0.01,
        params: X8Parameters = None,
        integration_method: str = "rk4"
    ):
        """
        Initialize the simulator.

        Args:
            dt: Time step for integration (s)
            params: Aircraft parameters. If None, uses defaults.
            integration_method: Integration method ('euler', 'rk2', 'rk4')
        """
        self.dt = dt
        self.dynamics = X8Dynamics(params)
        self.integration_method = integration_method.lower()

        if self.integration_method not in ['euler', 'rk2', 'rk4']:
            raise ValueError(f"Unknown integration method: {integration_method}")

    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: float = None,
        wind: np.ndarray = None
    ) -> np.ndarray:
        """
        Advance the simulation by one time step.

        Args:
            x: Current state vector (12,)
            u: Control vector (4,)
            dt: Time step. If None, uses self.dt
            wind: Wind vector (6,). If None, uses dynamics.wind

        Returns:
            Next state vector (12,)
        """
        if dt is None:
            dt = self.dt

        if self.integration_method == 'euler':
            return self._euler_step(x, u, dt, wind)
        elif self.integration_method == 'rk2':
            return self._rk2_step(x, u, dt, wind)
        else:  # rk4
            return self._rk4_step(x, u, dt, wind)

    def _euler_step(self, x: np.ndarray, u: np.ndarray, dt: float, wind: np.ndarray) -> np.ndarray:
        """Forward Euler integration."""
        x_dot = self.dynamics.dxdt(x, u, wind=wind)
        return x + dt * x_dot

    def _rk2_step(self, x: np.ndarray, u: np.ndarray, dt: float, wind: np.ndarray) -> np.ndarray:
        """Midpoint (RK2) integration."""
        k1 = self.dynamics.dxdt(x, u, wind=wind)
        k2 = self.dynamics.dxdt(x + 0.5 * dt * k1, u, wind=wind)
        return x + dt * k2

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float, wind: np.ndarray) -> np.ndarray:
        """4th-order Runge-Kutta integration."""
        k1 = self.dynamics.dxdt(x, u, wind=wind)
        k2 = self.dynamics.dxdt(x + 0.5 * dt * k1, u, wind=wind)
        k3 = self.dynamics.dxdt(x + 0.5 * dt * k2, u, wind=wind)
        k4 = self.dynamics.dxdt(x + dt * k3, u, wind=wind)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run_simulation(
        self,
        duration: float,
        x0: np.ndarray = None,
        control_fn: Callable[[np.ndarray, float], np.ndarray] = None,
        use_controller: bool = False,
        controller: X8Controller = None,
        trajectory: ReferenceTrajectory = None
    ) -> SimulationResult:
        """
        Run a complete simulation.

        Args:
            duration: Simulation duration (s)
            x0: Initial state. If None, uses trim state.
            control_fn: Control function (state, time) -> control.
                        If None and use_controller=False, uses trim control.
            use_controller: If True, uses built-in X8Controller
            controller: Custom X8Controller instance. If None and use_controller=True,
                       creates default controller.
            trajectory: Reference trajectory for controller. If None, uses default.

        Returns:
            SimulationResult with time, states, and controls history
        """
        # Initialize state
        if x0 is None:
            x0 = DEFAULT_TRIM_STATE.copy()

        # Set up control
        if use_controller:
            if controller is None:
                controller = X8Controller()
            if trajectory is None:
                trajectory = ReferenceTrajectory()
            control_fn = self._make_controller_fn(controller, trajectory)
        elif control_fn is None:
            trim_control = DEFAULT_TRIM_CONTROL.copy()
            control_fn = lambda x, t: trim_control

        # Allocate storage
        n_steps = int(duration / self.dt) + 1
        time = np.linspace(0, duration, n_steps)
        states = np.zeros((n_steps, 12))
        controls = np.zeros((n_steps, 4))

        # Initial conditions
        x = x0.copy()
        states[0] = x

        # Run simulation
        for i in range(n_steps - 1):
            t = time[i]
            u = control_fn(x, t)
            controls[i] = u

            x = self.step(x, u)
            states[i + 1] = x

        # Final control
        controls[-1] = control_fn(states[-1], time[-1])

        return SimulationResult(time=time, states=states, controls=controls)

    def _make_controller_fn(
        self,
        controller: X8Controller,
        trajectory: ReferenceTrajectory
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """Create control function from controller and trajectory."""
        def control_fn(x: np.ndarray, t: float) -> np.ndarray:
            V_ref, chi_ref, h_ref = trajectory.get_reference(t)
            controller.set_reference(velocity=V_ref, heading=chi_ref, altitude=h_ref)
            return controller.compute_control(x, dt=self.dt, t=t)
        return control_fn


def simulate_x8(
    duration: float = 400.0,
    dt: float = 0.01,
    x0: np.ndarray = None,
    use_controller: bool = True,
    params: X8Parameters = None
) -> SimulationResult:
    """
    Convenience function to run a simulation matching the MATLAB example.

    Args:
        duration: Simulation duration (s)
        dt: Time step (s)
        x0: Initial state. If None, uses trim state.
        use_controller: If True, uses autopilot. If False, uses trim control.
        params: Aircraft parameters

    Returns:
        SimulationResult
    """
    sim = X8Simulator(dt=dt, params=params)
    return sim.run_simulation(
        duration=duration,
        x0=x0,
        use_controller=use_controller
    )
