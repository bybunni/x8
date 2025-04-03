"""Tests for parameter loading and management."""

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from jax8.parameters import X8Parameters


# Mock the load_matlab_params function to avoid actual file loading
@pytest.fixture
def mock_params():
    """Create mock parameters for testing."""
    return {
        'mass': 5.0,
        'Jx': 0.8244,
        'Jy': 1.135,
        'Jz': 1.759,
        'Jxz': 0.1204,
        'r_cg': np.array([[0.0, 0.0, 0.0]]).T,
        'S_wing': 0.75,
        'b': 2.5,
        'c': 0.3,
        'S_prop': 0.2,
        'C_prop': 1.0,
        'k_motor': 80.0,
        'k_T_P': 0.0,
        'k_Omega': 0.0,
        'C_L_0': 0.28,
        'C_L_alpha': 3.45,
        'C_L_q': 0.0,
        'C_L_delta_e': 0.36,
        'C_D_0': 0.03,
        'C_D_alpha1': 0.3,
        'C_D_alpha2': 0.0,
        'C_D_beta1': 0.0,
        'C_D_beta2': 0.0,
        'C_D_q': 0.0,
        'C_D_delta_e': 0.0,
        'C_m_0': 0.0,
        'C_m_alpha': -0.38,
        'C_m_q': -3.6,
        'C_m_delta_e': -0.5,
        'C_Y_0': 0.0,
        'C_Y_beta': -0.98,
        'C_Y_p': 0.0,
        'C_Y_r': 0.0,
        'C_Y_delta_a': 0.0,
        'C_Y_delta_r': 0.0,
        'C_l_0': 0.0,
        'C_l_beta': -0.12,
        'C_l_p': -0.26,
        'C_l_r': 0.14,
        'C_l_delta_a': 0.08,
        'C_l_delta_r': 0.105,
        'C_n_0': 0.0,
        'C_n_beta': 0.25,
        'C_n_p': 0.022,
        'C_n_r': -0.35,
        'C_n_delta_a': 0.06,
        'C_n_delta_r': -0.032
    }


def test_x8_parameters_init(mock_params):
    """Test initializing parameters from mock data."""
    with patch('jax8.parameters.load_matlab_params', return_value=mock_params):
        params = X8Parameters('dummy.mat')
        
        # Check that parameters were loaded
        assert params['mass'] == 5.0
        assert params['S_wing'] == 0.75
        
        # Check that derived parameters were computed
        assert 'I_cg' in params
        assert 'M_rb' in params
        
        # Check that default parameters were set
        assert params['rho'] == 1.2250
        assert params['gravity'] == 9.81


def test_x8_parameters_getitem(mock_params):
    """Test accessing parameters using the getitem interface."""
    with patch('jax8.parameters.load_matlab_params', return_value=mock_params):
        params = X8Parameters('dummy.mat')
        
        # Check accessing existing parameters
        assert params['mass'] == 5.0
        
        # Check accessing non-existent parameters
        with pytest.raises(KeyError):
            _ = params['non_existent_param']


def test_x8_parameters_contains(mock_params):
    """Test checking parameter existence using the contains interface."""
    with patch('jax8.parameters.load_matlab_params', return_value=mock_params):
        params = X8Parameters('dummy.mat')
        
        # Check existing parameters
        assert 'mass' in params
        
        # Check non-existent parameters
        assert 'non_existent_param' not in params


def test_x8_parameters_to_dict(mock_params):
    """Test converting parameters to a dictionary."""
    with patch('jax8.parameters.load_matlab_params', return_value=mock_params):
        params = X8Parameters('dummy.mat')
        
        # Convert to dictionary
        params_dict = params.to_dict()
        
        # Check that it's a dictionary
        assert isinstance(params_dict, dict)
        
        # Check that it contains the parameters
        assert 'mass' in params_dict
        assert params_dict['mass'] == 5.0