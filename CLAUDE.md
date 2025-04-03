# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Run simulation: `simX8`
- Debug controllers: Individual functions can be tested by importing `x8_param.mat` and calling directly
- Trim calculation: Set `do_trim = true` in simX8.m to calculate trim conditions

## Code Style Guidelines
- MATLAB naming conventions: camelCase for variables, PascalCase for functions
- Physical variables use lowercase letters (u, v, w)
- Matrices use uppercase (R, T)
- Constants in uppercase (P.S_wing, P.C_L_0)
- Parameters stored in P struct and passed to functions
- Use matrix operations instead of loops when possible
- Document function inputs and outputs with comments
- Keep line length under 100 characters
- Include mathematical references (e.g., equation numbers from papers)
- Document physical units when needed
- Aerodynamic coefficient naming follows standard conventions (C_L_α, C_D_β)
- Use appropriate precision for physical constants