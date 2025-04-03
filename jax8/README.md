# JAX8: JAX Implementation of Skywalker X8 UAV Dynamics

This package provides a JAX-based implementation of the Skywalker X8 UAV dynamics model, based on the paper:

K. Gryte, R. Hann, M. Alam, J. Rohác, T. A. Johansen, T. I. Fossen, [*Aerodynamic modeling of the Skywalker X8 Fixed-Wing Unmanned Aerial Vehicle*](https://folk.ntnu.no/torarnj/icuasX8.pdf), International Conference on Unmanned Aircraft Systems, Dallas, 2018

## Features

- Full 6-DOF rigid body dynamics
- Aerodynamic model from the paper
- Propulsion model
- Basic autopilot implementation
- JAX-based implementation for hardware acceleration and automatic differentiation
- Trimming functionality
- Simulation utilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jax8.git
cd jax8

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .
```

## Usage

```python
import jax.numpy as jnp
from jax8.parameters import X8Parameters
from jax8.simulation import run_simulation, plot_simulation_results

# Load parameters
params = X8Parameters('x8_param.mat')

# Define initial state
initial_state = jnp.array([
    0.0, 0.0, -200.0,     # Position (NED)
    0.0, 0.0, 0.0,        # Euler angles
    18.0, 0.0, 0.0,       # Velocity (body)
    0.0, 0.0, 0.0         # Angular rates
])

# Run simulation
results = run_simulation(
    params=params.to_dict(),
    initial_state=initial_state,
    t_span=(0.0, 400.0),
    dt=0.1
)

# Plot results
plot_simulation_results(results)
```

## Structure

- `dynamics.py`: Core dynamics equations
- `forces.py`: Aerodynamic and propulsion forces
- `transformations.py`: Rotation matrices and transformations
- `simulation.py`: Simulation runner (ODE integration)
- `trim.py`: Trim calculation utilities
- `controller.py`: Reference controller implementation
- `parameters.py`: Parameter loading and storage
- `visualization.py`: Plotting utilities
- `utils/`: Utility functions

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check jax8 tests

# Run type checking
mypy jax8

# Run formatting
black jax8 tests
```