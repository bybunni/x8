"""I/O utilities for loading MATLAB data files."""

from typing import Dict, Any

import jax.numpy as jnp
import numpy as np
import scipy.io


def load_matlab_params(filepath: str) -> Dict[str, Any]:
    """
    Load parameters from a MATLAB .mat file.
    
    Args:
        filepath: Path to the MATLAB .mat file
        
    Returns:
        Dictionary containing the parameters from the .mat file,
        converted to JAX arrays where appropriate
    """
    # Load the MATLAB file
    mat_data = scipy.io.loadmat(filepath)
    
    # Convert numpy arrays to JAX arrays
    params = {}
    for key, value in mat_data.items():
        # Skip MATLAB internal variables
        if key.startswith('__') and key.endswith('__'):
            continue
            
        # Convert numpy arrays to JAX arrays
        if isinstance(value, np.ndarray):
            # Handle scalar values stored as arrays
            if value.size == 1:
                params[key] = float(value)
            else:
                params[key] = jnp.array(value)
        else:
            params[key] = value
    
    return params