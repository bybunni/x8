# Contributing to the X8 UAV Simulator

Thank you for your interest in contributing to the X8 UAV Simulator project! This document provides guidelines and instructions for contributing to both the MATLAB and JAX implementations.

## Development Setup

### MATLAB Implementation

1. Clone the repository
2. Open MATLAB (version 2018b or newer recommended)
3. Run `simX8.m` to verify the simulation works

### JAX Implementation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests to verify setup:
   ```bash
   pytest
   ```

## Development Workflow

1. Create a feature branch from `master`:
   ```bash
   git checkout -b feature-name
   ```
2. Make your changes
3. Ensure code quality:
   - For JAX implementation:
     ```bash
     black jax8 tests
     ruff check jax8 tests
     mypy jax8
     pytest
     ```
4. Commit your changes
5. Push to your branch and create a pull request

## Code Style Guidelines

### MATLAB Implementation

- Follow MATLAB naming conventions (camelCase for variables, PascalCase for functions)
- Use matrix operations instead of loops when possible
- Document function inputs and outputs with comments
- Include physical units in comments when needed

### JAX Implementation

- Follow PEP 8 (enforced by Black and Ruff)
- Use type hints for all function parameters and return values
- Document all functions and classes with docstrings
- Include test coverage for new functionality
- Follow JAX-specific best practices (use pure functions, avoid in-place operations)

## Adding New Features

When adding new features, please consider implementing them in both the MATLAB and JAX versions to maintain parity between the implementations.

### Documentation

All new features should be documented:
- In-code documentation (comments in MATLAB, docstrings in Python)
- Update relevant README files
- Add example usage when appropriate

### Testing

- For the JAX implementation, add appropriate tests in the `tests/` directory
- For cross-implementation validation, add test cases that compare results between MATLAB and JAX

## Submitting Pull Requests

1. Ensure your code follows the style guidelines
2. Include test cases for new functionality
3. Update documentation as needed
4. Provide a clear description of your changes in the PR
5. Link to any related issues

Thank you for contributing to the X8 UAV Simulator project!