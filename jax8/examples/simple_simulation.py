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
    
    # Create controller
    controller_fn = create_controller_function(params.to_dict())
    
    # Define initial state
    initial_state = jnp.array([
        0.0, 0.0, -200.0,       # Position (NED)
        0.0, 0.0, 0.0,         # Euler angles
        18.0, 0.0, 0.0,        # Velocity (body)
        0.0, 0.0, 0.0          # Angular rates
    ])
    
    # Run simulation
    results = run_simulation(
        params=params.to_dict(),
        initial_state=initial_state,
        controller_fn=controller_fn,
        t_span=(0.0, 400.0),
        dt=0.1
    )
    
    # Plot results
    plot_simulation_results(results)


if __name__ == "__main__":
    main()