# JAX8: Python JAX Port of X8 UAV Dynamics

This document outlines the development plan for porting the MATLAB-based Skywalker X8 UAV dynamics model to Python using JAX for hardware acceleration and automatic differentiation capabilities.

## Project Structure

```
jax8/
├── __init__.py              # Package exports
├── dynamics.py              # Core dynamics equations
├── forces.py                # Aerodynamic and propulsion forces
├── transformations.py       # Rotation matrices and transformations
├── simulation.py            # Simulation runner (ODE integration)
├── trim.py                  # Trim calculation utilities
├── controller.py            # Reference controller implementation
├── parameters.py            # Parameter loading and storage
├── visualization.py         # Plotting utilities
└── utils/
    ├── __init__.py
    ├── matrices.py          # Matrix utilities (Smtrx, etc.)
    └── io.py                # MATLAB file loading
```

## Development Standards

### Environment Setup
- Python 3.10+
- Virtual environment managed through `.venv`
- Package management with `uv`

### Code Quality Tools

- **Formatting**: black with 88 character line limit
- **Linting**: ruff for fast, comprehensive linting
- **Type checking**: mypy with strict settings
- **Testing**: pytest with coverage reporting
- **Build system**: hatchling

### Testing Strategy

1. **Unit Tests**:
   - Test each mathematical function individually
   - Compare against hardcoded MATLAB outputs for selected inputs
   - Test vector/matrix algebra operations

2. **Integration Tests**:
   - Test complete simulation runs
   - Compare trajectory data against MATLAB reference outputs
   - Validate trim calculations

3. **Performance Tests**:
   - Benchmark JAX implementation against NumPy implementation
   - Measure compilation overhead vs runtime performance

### Development Workflow

1. Port matrix utilities first (`Smtrx.m`, `Rzyx.m`, `TransformationMatrix.m`)
2. Implement parameter loading from MATLAB files
3. Port core dynamics equations
4. Port forces model
5. Implement ODE solver and simulation loop
6. Port trim functionality
7. Port reference controller
8. Add visualization tools
9. Create validation tests against MATLAB reference

### JAX-Specific Considerations

- Use `jax.numpy` as a drop-in replacement for numpy
- Leverage `jax.jit` for compilation acceleration
- Structure code to avoid JAX tracer errors
- Use pytrees for parameter structures
- Implement batch processing capabilities
- Consider using `jax.vmap` for vectorized operations

## Configuration Files

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "jax8"
version = "0.1.0"
description = "JAX implementation of Skywalker X8 UAV dynamics"
readme = "README.md"
requires-python = ">=3.10"
license = {file = "LICENSE"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
dependencies = [
    "jax>=0.4.13",
    "jaxlib>=0.4.13",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "diffrax>=0.4.0",  # JAX-based ODE solvers
    "scipy-matlab-api>=0.1.0",  # For loading MATLAB files
]

[project.optional-dependencies]
dev = [
    "black>=23.1.0",
    "ruff>=0.0.262",
    "mypy>=1.2.0",
    "pytest>=7.3.1",
    "pytest-cov>=4.1.0",
]

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.ruff]
select = ["E", "F", "B", "I", "N", "UP", "ANN", "A"]
ignore = ["ANN101", "ANN102"]
line-length = 88
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

## Installation and Setup

```bash
# Assuming .venv is already set up
source .venv/bin/activate

# Install build tools
uv pip install build hatch

# Install dev dependencies
uv pip install -e ".[dev]"

# Run code quality checks
black jax8 tests
ruff check jax8 tests
mypy jax8

# Run tests
pytest
```

## Implementation Notes

### MATLAB vs Python/JAX Differences

1. **Indexing**: MATLAB uses 1-based indexing, Python uses 0-based indexing
2. **Matrix operations**: MATLAB's `*` is matrix multiplication, Python/NumPy/JAX require `@` operator
3. **Division**: MATLAB's `/` is right division, Python/NumPy/JAX's `/` is element-wise division
4. **Transpose**: MATLAB's `'` is conjugate transpose, Python/NumPy/JAX's `.T` is transpose
5. **Function declarations**: MATLAB uses `function`, Python uses `def`
6. **Function returns**: MATLAB returns last variable, Python requires explicit return
7. **ODE solvers**: MATLAB's `ode45` vs JAX's `diffrax` library