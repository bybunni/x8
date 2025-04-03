"""Controller implementation for the X8 UAV."""

from typing import Dict, Callable, Tuple

import jax
import jax.numpy as jnp

from jax8.transformations import body_to_inertial
from jax8.utils.matrices import rzyx


class X8Controller:
    """
    Implementation of the reference controller for the X8 UAV.
    
    This is a basic autopilot with cascaded PID loops for:
    - Airspeed control via throttle
    - Heading control via roll angle
    - Roll angle control via aileron
    - Altitude control via pitch angle
    - Pitch angle control via elevator
    """
    
    def __init__(self, params: Dict):
        """
        Initialize controller with parameters.
        
        Args:
            params: Dictionary containing controller gains and aircraft parameters
        """
        # Store parameters
        self.params = params
        
        # Initialize integrator states
        self.reset_integrators()
    
    def reset_integrators(self):
        """Reset all integrator states to zero."""
        self.integrator_states = {
            'phi': 0.0,    # Roll angle
            'theta': 0.0,  # Pitch angle
            'h': 0.0,      # Altitude
            'V': 0.0,      # Airspeed
            'chi': 0.0,    # Course angle
        }
        self.prev_time = 0.0
    
    def get_reference(self, t: float, state: jnp.ndarray) -> Tuple[float, float, float]:
        """
        Compute reference values for airspeed, course, and altitude.
        
        This is a simple version that mimics the MATLAB implementation.
        In a real implementation, this would likely be replaced with 
        a more sophisticated trajectory generation approach.
        
        Args:
            t: Current time (seconds)
            state: Current state vector
            
        Returns:
            Tuple of (airspeed_ref, course_ref, altitude_ref)
        """
        # Reference airspeed (constant)
        V_ref = 18.0  # m/s
        
        # Reference course angle (time-varying)
        if t < 20:
            chi_ref = 0
        elif t < 35:
            chi_ref = (t - 20) * 1 * jnp.pi / 180  # 1 deg/s turn rate
        else:
            chi_ref = 15 * jnp.pi / 180  # 15 deg
        
        # Reference altitude (time-varying)
        climb_rate = 0.2  # m/s
        if t < 75:
            h_ref = 200
        elif t < 325:
            h_ref = 200 + (t - 75) * climb_rate
        else:
            h_ref = 250
        
        return V_ref, chi_ref, h_ref
    
    def __call__(self, t: float, state: jnp.ndarray, params: Dict) -> jnp.ndarray:
        """
        Compute control inputs based on current state.
        
        Args:
            t: Current time (seconds)
            state: Current state vector [pos, Theta, vel, rates]
            params: Dictionary of aircraft parameters
            
        Returns:
            Control inputs [elevator, aileron, rudder, throttle]
        """
        # Extract state components
        pos = state[0:3]
        euler_angles = state[3:6]
        vel_body = state[6:9]
        omega = state[9:12]
        
        # Compute time step for integrators
        dt = t - self.prev_time
        dt = jnp.clip(dt, 0.0, 0.1)  # Limit dt to avoid large steps
        self.prev_time = t
        
        # Get reference values
        V_ref, chi_ref, h_ref = self.get_reference(t, state)
        
        # Compute current course angle
        vel_ned = body_to_inertial(euler_angles, vel_body)
        chi = jnp.arctan2(vel_ned[1], vel_ned[0])
        
        # --- Course control ---
        # Update course integrator
        self.integrator_states['chi'] += (chi - chi_ref) * dt
        
        # Compute desired roll angle from course error
        phi_ref = params['kp_chi'] * (chi - chi_ref) + params['ki_chi'] * self.integrator_states['chi']
        
        # --- Altitude control ---
        # Update altitude integrator
        self.integrator_states['h'] += (-pos[2] - h_ref) * dt
        
        # Compute desired pitch angle from altitude error
        theta_ref = params['kp_h'] * (-pos[2] - h_ref) + params['ki_h'] * self.integrator_states['h']
        
        # --- Pitch control ---
        # Update pitch integrator
        self.integrator_states['theta'] += (euler_angles[1] - theta_ref) * dt
        
        # Compute elevator command from pitch error
        delta_e = (params['kp_theta'] * (euler_angles[1] - theta_ref) + 
                  params['ki_theta'] * self.integrator_states['theta'] - 
                  params['kd_theta'] * omega[1])
        
        # --- Roll control ---
        # Update roll integrator
        self.integrator_states['phi'] += (euler_angles[0] - phi_ref) * dt
        
        # Compute aileron command from roll error
        delta_a = (params['kp_phi'] * (euler_angles[0] - phi_ref) + 
                  params['ki_phi'] * self.integrator_states['phi'] - 
                  params['kd_phi'] * omega[0])
        
        # --- Airspeed control ---
        # Compute current airspeed
        V = jnp.linalg.norm(vel_body)
        
        # Update airspeed integrator
        self.integrator_states['V'] += (V - V_ref) * dt
        
        # Compute throttle command from airspeed error
        delta_t = params['kp_V'] * (V - V_ref) + params['ki_V'] * self.integrator_states['V']
        
        # No rudder control
        delta_r = 0.0
        
        # Assemble control vector
        u = jnp.array([delta_e, delta_a, delta_r, delta_t])
        
        # Add trim values
        # Note: In the MATLAB code, trim values are added externally
        # u = u + u_trim
        
        # Saturate controls
        u = jnp.clip(u, -1.0, 1.0)  # All controls saturate at ±1
        u = u.at[3].set(jnp.clip(u[3], 0.0, 1.0))  # Throttle saturates at 0 to 1
        
        return u


def create_controller_function(params: Dict) -> Callable:
    """
    Create a stateless controller function using JAX.
    
    This creates a pure function version of the controller that can be used
    with JAX transformations like jit and vmap.
    
    Args:
        params: Dictionary of X8 parameters
        
    Returns:
        Function that computes control inputs given time and state
    """
    # Create a stateless controller function
    @jax.jit
    def controller_fn(
        t: float,
        state: jnp.ndarray,
        params: Dict,
        integrator_states: Dict,
        prev_time: float
    ) -> Tuple[jnp.ndarray, Dict, float]:
        """
        Compute control inputs based on current state.
        
        Args:
            t: Current time (seconds)
            state: Current state vector [pos, Theta, vel, rates]
            params: Dictionary of aircraft parameters
            integrator_states: Dictionary of integrator states
            prev_time: Previous time for dt calculation
            
        Returns:
            Tuple of (control_inputs, new_integrator_states, new_prev_time)
        """
        # Extract state components
        pos = state[0:3]
        euler_angles = state[3:6]
        vel_body = state[6:9]
        omega = state[9:12]
        
        # Compute time step for integrators
        dt = t - prev_time
        dt = jnp.clip(dt, 0.0, 0.1)  # Limit dt to avoid large steps
        new_prev_time = t
        
        # Get reference values (hardcoded for now)
        V_ref = 18.0  # m/s
        
        # Reference course angle (time-varying)
        chi_ref = jnp.where(
            t < 20,
            0.0,
            jnp.where(
                t < 35,
                (t - 20) * 1 * jnp.pi / 180,  # 1 deg/s turn rate
                15 * jnp.pi / 180  # 15 deg
            )
        )
        
        # Reference altitude (time-varying)
        climb_rate = 0.2  # m/s
        h_ref = jnp.where(
            t < 75,
            200.0,
            jnp.where(
                t < 325,
                200 + (t - 75) * climb_rate,
                250.0
            )
        )
        
        # Compute current course angle
        vel_ned = body_to_inertial(euler_angles, vel_body)
        chi = jnp.arctan2(vel_ned[1], vel_ned[0])
        
        # --- Course control ---
        # Update course integrator
        new_int_chi = integrator_states['chi'] + (chi - chi_ref) * dt
        
        # Compute desired roll angle from course error
        phi_ref = params['kp_chi'] * (chi - chi_ref) + params['ki_chi'] * new_int_chi
        
        # --- Altitude control ---
        # Update altitude integrator
        new_int_h = integrator_states['h'] + (-pos[2] - h_ref) * dt
        
        # Compute desired pitch angle from altitude error
        theta_ref = params['kp_h'] * (-pos[2] - h_ref) + params['ki_h'] * new_int_h
        
        # --- Pitch control ---
        # Update pitch integrator
        new_int_theta = integrator_states['theta'] + (euler_angles[1] - theta_ref) * dt
        
        # Compute elevator command from pitch error
        delta_e = (params['kp_theta'] * (euler_angles[1] - theta_ref) + 
                  params['ki_theta'] * new_int_theta - 
                  params['kd_theta'] * omega[1])
        
        # --- Roll control ---
        # Update roll integrator
        new_int_phi = integrator_states['phi'] + (euler_angles[0] - phi_ref) * dt
        
        # Compute aileron command from roll error
        delta_a = (params['kp_phi'] * (euler_angles[0] - phi_ref) + 
                  params['ki_phi'] * new_int_phi - 
                  params['kd_phi'] * omega[0])
        
        # --- Airspeed control ---
        # Compute current airspeed
        V = jnp.linalg.norm(vel_body)
        
        # Update airspeed integrator
        new_int_V = integrator_states['V'] + (V - V_ref) * dt
        
        # Compute throttle command from airspeed error
        delta_t = params['kp_V'] * (V - V_ref) + params['ki_V'] * new_int_V
        
        # No rudder control
        delta_r = 0.0
        
        # Assemble control vector
        u = jnp.array([delta_e, delta_a, delta_r, delta_t])
        
        # Add trim values (if provided)
        u_trim = params.get('u_trim', jnp.zeros(4))
        u = u + u_trim
        
        # Saturate controls
        u = jnp.clip(u, -1.0, 1.0)  # All controls saturate at ±1
        u = u.at[3].set(jnp.clip(u[3], 0.0, 1.0))  # Throttle saturates at 0 to 1
        
        # Update integrator states dictionary
        new_integrator_states = {
            'phi': new_int_phi,
            'theta': new_int_theta,
            'h': new_int_h,
            'V': new_int_V,
            'chi': new_int_chi,
        }
        
        return u, new_integrator_states, new_prev_time
    
    # Create initial integrator states
    integrator_states = {
        'phi': 0.0,
        'theta': 0.0,
        'h': 0.0,
        'V': 0.0,
        'chi': 0.0,
    }
    prev_time = 0.0
    
    # Create a function that maintains internal state
    def stateful_controller(t: float, state: jnp.ndarray, _: Dict) -> jnp.ndarray:
        nonlocal integrator_states, prev_time
        
        # Compute control inputs and update internal state
        u, integrator_states, prev_time = controller_fn(
            t, state, params, integrator_states, prev_time
        )
        
        return u
    
    return stateful_controller