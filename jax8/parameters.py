"""Parameter loading and management for the X8 UAV model."""

from typing import Dict, Any, Optional

import jax.numpy as jnp

from jax8.utils.io import load_matlab_params
from jax8.utils.matrices import smtrx


class X8Parameters:
    """
    Class to hold and manage parameters for the X8 UAV model.
    
    This class loads parameters from a MATLAB .mat file and computes
    derived parameters needed for the dynamics model.
    """
    
    def __init__(self, param_file: str):
        """
        Initialize parameters from a MATLAB .mat file.
        
        Args:
            param_file: Path to the MATLAB .mat file containing parameters
        """
        # Load raw parameters from MATLAB file
        self.params = load_matlab_params(param_file)
        
        # Add computed parameters
        self._compute_derived_parameters()
    
    def _compute_derived_parameters(self):
        """Compute derived parameters needed for dynamics modeling."""
        # Compute inertia matrix (assuming symmetry wrt xz-plane, i.e., Jxy=Jyz=0)
        self.params['I_cg'] = jnp.array([
            [self.params['Jx'], 0, -self.params['Jxz']],
            [0, self.params['Jy'], 0],
            [-self.params['Jxz'], 0, self.params['Jz']]
        ], dtype=jnp.float32)
        
        # Compute mass matrix
        r_cg = self.params['r_cg']
        mass = self.params['mass']
        self.params['M_rb'] = jnp.block([
            [jnp.eye(3) * mass, -mass * smtrx(r_cg)],
            [mass * smtrx(r_cg), self.params['I_cg']]
        ])
        
        # Set environmental constants if not present
        if 'rho' not in self.params:
            self.params['rho'] = 1.2250  # air density (kg/m^3)
        
        if 'gravity' not in self.params:
            self.params['gravity'] = 9.81  # gravitational acceleration (m/s^2)
    
    def __getitem__(self, key: str) -> Any:
        """
        Access a parameter value by key.
        
        Args:
            key: Parameter name
            
        Returns:
            Parameter value
            
        Raises:
            KeyError: If the parameter does not exist
        """
        return self.params[key]
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a parameter exists.
        
        Args:
            key: Parameter name
            
        Returns:
            True if the parameter exists, False otherwise
        """
        return key in self.params
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to a dictionary.
        
        Returns:
            Dictionary containing all parameters
        """
        return self.params.copy()
    
    @classmethod
    def default(cls) -> 'X8Parameters':
        """
        Create a parameters object with the default parameter file.
        
        Returns:
            X8Parameters object with default parameters
        """
        return cls('x8_param.mat')