"""Simulation utilities for the X8 UAV model."""

from typing import Dict, Callable, Tuple, Optional, Any

import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import numpy as np

from jax8.dynamics import create_dynamics_function
from jax8.controller import create_controller_function
from jax8.parameters import X8Parameters


def run_simulation(
    params: Dict,
    initial_state: jnp.ndarray,
    controller_fn: Optional[Callable] = None,
    wind_fn: Optional[Callable] = None,
    t_span: Tuple[float, float] = (0.0, 400.0),
    dt: float = 0.05,
    solver_method: str = 'dopri5',
    trim_values: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
) -> Dict[str, Any]:
    """
    Run a simulation of the X8 UAV dynamics.
    
    Args:
        params: Dictionary of X8 parameters
        initial_state: Initial state vector [pos, Theta, vel, rates]
        controller_fn: Function that computes control inputs given time and state
            (if None, a default controller will be created)
        wind_fn: Function that computes wind components given time and state
            (if None, zero wind will be used)
        t_span: Tuple of (start_time, end_time)
        dt: Time step for output (seconds)
        solver_method: ODE solver method ('dopri5', 'euler', etc.)
        trim_values: Optional tuple of (y_trim, u_trim) to use for trim values
            
    Returns:
        Dictionary containing simulation results:
            't': Time points
            'state': State trajectories
            'control': Control input trajectories
    """
    # Create default controller if not provided
    if controller_fn is None:
        # Use default trim values if not provided
        if trim_values is None:
            # From MATLAB:
            u_trim = jnp.array([0.0370, 0.0000, 0.0, 0.1219])
            # Add trim values to parameters
            params['u_trim'] = u_trim
        else:
            y_trim, u_trim = trim_values
            params['u_trim'] = u_trim
            
        # Create controller function
        controller_fn = create_controller_function(params)
    
    # Create dynamics function
    dynamics_fn = create_dynamics_function(params, controller_fn, wind_fn)
    
    # Create time points
    t0, tf = t_span
    ts = jnp.arange(t0, tf, dt)
    
    # Set up ODE solver
    if solver_method == 'dopri5':
        solver = diffrax.Dopri5()
    elif solver_method == 'euler':
        solver = diffrax.Euler()
    else:
        raise ValueError(f"Unknown solver method: {solver_method}")
    
    # Create term for the ODE solver
    term = diffrax.ODETerm(dynamics_fn)
    
    # Create saveat for the ODE solver
    saveat = diffrax.SaveAt(ts=ts)
    
    # Solve the ODE
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=tf,
        dt0=dt,
        y0=initial_state,
        saveat=saveat,
        max_steps=jnp.iinfo(jnp.int32).max,
    )
    
    # Get solution
    t = sol.ts
    state = sol.ys
    
    # Compute control inputs at each time point
    control = jnp.stack([controller_fn(t_i, state_i, params) for t_i, state_i in zip(t, state)])
    
    # Return results
    return {
        't': np.array(t),
        'state': np.array(state),
        'control': np.array(control)
    }


def plot_simulation_results(results: Dict[str, Any]):
    """
    Plot simulation results.
    
    Args:
        results: Dictionary containing simulation results from run_simulation
    """
    t = results['t']
    state = results['state']
    control = results['control']
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot altitude
    plt.subplot(4, 1, 1)
    plt.plot(t, state[:, 2])
    plt.ylabel('Down (m)')
    plt.title('Altitude')
    plt.grid(True)
    
    # Plot Euler angles
    plt.subplot(4, 1, 2)
    plt.plot(t, np.rad2deg(state[:, 3]), label='Roll (deg)')
    plt.plot(t, np.rad2deg(state[:, 4]), label='Pitch (deg)')
    plt.plot(t, np.rad2deg(state[:, 5]), label='Yaw (deg)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.grid(True)
    
    # Plot velocities
    plt.subplot(4, 1, 3)
    plt.plot(t, state[:, 6], label='u (m/s)')
    plt.plot(t, state[:, 7], label='v (m/s)')
    plt.plot(t, state[:, 8], label='w (m/s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    # Plot angular rates
    plt.subplot(4, 1, 4)
    plt.plot(t, np.rad2deg(state[:, 9]), label='p (deg/s)')
    plt.plot(t, np.rad2deg(state[:, 10]), label='q (deg/s)')
    plt.plot(t, np.rad2deg(state[:, 11]), label='r (deg/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Rate (deg/s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create another figure for control inputs
    plt.figure(figsize=(12, 5))
    
    # Plot control inputs
    plt.subplot(2, 1, 1)
    plt.plot(t, control[:, 0], label='Elevator')
    plt.plot(t, control[:, 1], label='Aileron')
    plt.plot(t, control[:, 2], label='Rudder')
    plt.ylabel('Control Surface (-)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, control[:, 3], label='Throttle')
    plt.xlabel('Time (s)')
    plt.ylabel('Throttle (-)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def find_trim(
    initial_guess: jnp.ndarray,
    params: Dict,
    altitude: float = -200.0,
    airspeed: float = 18.0,
    flight_path_angle: float = 0.0,
    turning_radius: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find trim conditions for the X8 UAV.
    
    This is a placeholder implementation that would need to be replaced
    with a proper optimization-based trim finder.
    
    Args:
        initial_guess: Initial guess for state vector
        params: Dictionary of X8 parameters
        altitude: Desired altitude (m)
        airspeed: Desired airspeed (m/s)
        flight_path_angle: Desired flight path angle (rad)
        turning_radius: Desired turning radius (m), None for straight flight
        
    Returns:
        Tuple of (trimmed_state, trimmed_control)
    """
    # This is a placeholder that returns predefined trim values
    # In a full implementation, this would solve an optimization problem
    
    # From MATLAB:
    u_trim = jnp.array([0.0370, 0.0000, 0.0, 0.1219])
    y_trim = jnp.array([
        0.0, 0.0, altitude,  # Position
        0.0, 0.0308, 0.0,     # Euler angles
        17.9914, 0.0, 0.5551, # Velocity
        0.0, 0.0, 0.0         # Angular rates
    ])
    
    # TODO: Implement proper trim optimization
    
    return y_trim, u_trim