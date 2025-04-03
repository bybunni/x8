"""
Tests validating the JAX implementation against MATLAB reference outputs.

These tests require generating reference output data from the MATLAB code first.
Instructions for generating this data are provided in the comments.
"""

import os
import numpy as np
import pytest
from pathlib import Path
import scipy.io

import jax.numpy as jnp

from jax8.parameters import X8Parameters
from jax8.utils.matrices import smtrx, rzyx, transformation_matrix
from jax8.transformations import body_to_inertial, body_rate_to_euler_rate
from jax8.forces import compute_forces_moments
from jax8.dynamics import rigid_body_dynamics
from jax8.simulation import run_simulation


# Skip these tests if MATLAB reference data is not available
matlab_data_available = Path("./tests/matlab_reference_data.mat").exists()
skip_if_no_matlab_data = pytest.mark.skipif(
    not matlab_data_available, 
    reason="MATLAB reference data not available"
)


@pytest.fixture
def reference_data():
    """
    Load reference data from MATLAB.
    
    To generate this data in MATLAB, run the following:
    
    ```matlab
    % Run simulation
    simX8;
    
    % Extract reference data
    reference_data = struct();
    reference_data.t = t;
    reference_data.y = y;
    
    % Save test vectors for individual functions
    % Smtrx test
    v_test = [1.0; 2.0; 3.0];
    reference_data.smtrx_input = v_test;
    reference_data.smtrx_output = Smtrx(v_test);
    
    % Rzyx test
    phi_test = 0.1;
    theta_test = 0.2;
    psi_test = 0.3;
    reference_data.rzyx_input = [phi_test; theta_test; psi_test];
    reference_data.rzyx_output = Rzyx(phi_test, theta_test, psi_test);
    
    % Save to file
    save('matlab_reference_data.mat', 'reference_data');
    ```
    """
    if not matlab_data_available:
        return None
    
    # Load MATLAB reference data
    mat_data = scipy.io.loadmat("./tests/matlab_reference_data.mat")
    return mat_data["reference_data"]


@skip_if_no_matlab_data
def test_smtrx_against_matlab(reference_data):
    """Test the skew-symmetric matrix function against MATLAB reference."""
    # Extract reference data
    v_test = reference_data["smtrx_input"].flatten()
    smtrx_matlab = reference_data["smtrx_output"]
    
    # Compute with JAX implementation
    smtrx_jax = smtrx(jnp.array(v_test))
    
    # Compare results
    np.testing.assert_allclose(smtrx_jax, smtrx_matlab, rtol=1e-5)


@skip_if_no_matlab_data
def test_rzyx_against_matlab(reference_data):
    """Test the rotation matrix function against MATLAB reference."""
    # Extract reference data
    phi_test = reference_data["rzyx_input"][0, 0]
    theta_test = reference_data["rzyx_input"][1, 0]
    psi_test = reference_data["rzyx_input"][2, 0]
    rzyx_matlab = reference_data["rzyx_output"]
    
    # Compute with JAX implementation
    rzyx_jax = rzyx(phi_test, theta_test, psi_test)
    
    # Compare results
    np.testing.assert_allclose(rzyx_jax, rzyx_matlab, rtol=1e-5)


@skip_if_no_matlab_data
def test_simulation_trajectory(reference_data):
    """
    Test the full simulation trajectory against MATLAB reference.
    
    This test validates that the time evolution of the state vectors
    is consistent with the MATLAB implementation.
    """
    # This test would need to be implemented once we have generated
    # the reference data from MATLAB
    
    # For now, we'll include a skeleton of what this would look like
    if matlab_data_available:
        # Extract reference data
        t_matlab = reference_data["t"].flatten()
        y_matlab = reference_data["y"]
        
        # Run JAX simulation with the same parameters
        # ...
        
        # Compare results
        # np.testing.assert_allclose(y_jax, y_matlab, rtol=1e-3, atol=1e-3)
        pass


# Example of how to add more validation tests:
"""
@skip_if_no_matlab_data
def test_forces_against_matlab(reference_data):
    # Extract reference data for a specific time point
    t_idx = 10  # Use 10th time point for comparison
    state_matlab = reference_data["y"][t_idx, :]
    control_matlab = reference_data["control_inputs"][t_idx, :]
    forces_moments_matlab = reference_data["forces_moments"][t_idx, :]
    
    # Compute with JAX implementation
    params = X8Parameters("x8_param.mat").to_dict()
    wind = jnp.zeros(6)
    forces_moments_jax = compute_forces_moments(
        t_matlab[t_idx], jnp.array(state_matlab), 
        jnp.array(control_matlab), wind, params
    )
    
    # Compare results
    np.testing.assert_allclose(
        forces_moments_jax, forces_moments_matlab, rtol=1e-3, atol=1e-3
    )
"""