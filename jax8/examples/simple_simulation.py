"""Example script demonstrating a simple simulation of the X8 UAV."""

import os
import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Ensure the jax8 package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from jax8.parameters import X8Parameters
from jax8.controller import create_controller_function
from jax8.simulation import run_simulation, plot_simulation_results


def main():
    """Run a simple simulation of the X8 UAV."""
    # Load parameters
    params = X8Parameters('x8_param.mat')
    
    # Add missing controller parameters
    controller_params = params.to_dict()

    # Add controller gains if they don't exist
    controller_gains = {
        # Course control gains
        'kp_chi': 0.5,     # Proportional gain for course control
        'ki_chi': 0.1,     # Integral gain for course control
        
        # Altitude control gains
        'kp_h': 0.05,      # Proportional gain for altitude control
        'ki_h': 0.01,      # Integral gain for altitude control
        
        # Pitch control gains
        'kp_theta': 1.0,   # Proportional gain for pitch control
        'ki_theta': 0.1,   # Integral gain for pitch control
        'kd_theta': 0.2,   # Derivative gain for pitch control
        
        # Roll control gains
        'kp_phi': 1.0,     # Proportional gain for roll control
        'ki_phi': 0.1,     # Integral gain for roll control
        'kd_phi': 0.2,     # Derivative gain for roll control
        
        # Airspeed control gains
        'kp_V': 0.5,       # Proportional gain for airspeed control
        'ki_V': 0.1,       # Integral gain for airspeed control
    }

    # Update parameters with controller gains
    controller_params.update(controller_gains)

    # Create controller
    controller_fn = create_controller_function(controller_params)
    
    # Define initial state
    initial_state = jnp.array([
        0.0, 0.0, -200.0,       # Position (NED)
        0.0, 0.0, 0.0,         # Euler angles
        18.0, 0.0, 0.0,        # Velocity (body)
        0.0, 0.0, 0.0          # Angular rates
    ])
    
    # Run simulation
    results = run_simulation(
        params=controller_params,
        initial_state=initial_state,
        controller_fn=controller_fn,
        t_span=(0.0, 400.0),
        dt=0.1
    )
    
    # Plot results
    plot_simulation_results(results)


if __name__ == "__main__":
    main()