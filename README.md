# 3R Manipulator Trajectory Planning - Refactored

Clean, modular codebase with trajectory caching for improved performance.

## Project Structure

```
3r/
├── trajectory/              # Trajectory planning package
│   ├── __init__.py         # Package exports
│   ├── kinematics.py       # CasADi kinematics functions
│   ├── optimization.py     # Trajectory solvers with caching
│   ├── plotting.py         # Visualization and animation
│   ├── utils.py            # Helper utilities
│   └── cache.py            # Trajectory caching system
├── config.py               # Robot configuration parameters
├── kinematics.py           # Basic kinematics (FK/IK)
├── exec_refactored.py      # Main execution script
├── manage_cache.py         # Cache management utility
└── exec.py                 # Original script (legacy)
```

## Features

### Trajectory Caching

Maximum speed trajectory profiles are **automatically cached** to avoid expensive recomputation:

- **First run**: Solves optimization problem and saves to cache (~5s)
- **Subsequent runs**: Loads from cache instantly (<0.1s)
- Cache location: `~/.cache/3r_trajectories/`
- Cache key based on: `q0`, `xT`, `v_final`, `T`, and configuration

### Usage

Run trajectory planning:
```bash
python exec_refactored.py
```

Manage cache:
```bash
# List cached trajectories
python manage_cache.py list

# Clear cache
python manage_cache.py clear
```

Import in code:
```python
from trajectory import (
    solve_trajectory_problem,           # Solves with automatic caching
    scale_casadi_solution_taskspace_peak,  # Fast approximation
    refine_trajectory_for_exact_speed,     # Exact refinement
    animate_3r_trajectory,
    clear_trajectory_cache,
    list_cached_trajectories
)

# Two-phase approach for exact speed control:
# 1. Compute/load max speed trajectory (cached)
max_solution = solve_trajectory_problem(v_final_max, T, q0, xT)

# 2. Fast scaling for initial approximation
scaled_sol, _ = scale_casadi_solution_taskspace_peak(max_solution, v_desired=1.9)

# 3. Refine to exact desired speed (uses scaled as initial guess)
refined_sol, info = refine_trajectory_for_exact_speed(scaled_sol, v_desired=1.9, q0=q0, xT=xT)
print(f"Achieved: {info['achieved_peak']:.6f} m/s, Error: {info['error_percent']:.6f}%")
```

## Module Organization

### `trajectory/kinematics.py`
- CasADi symbolic kinematics functions
- Forward kinematics, Jacobian, and derivatives

### `trajectory/optimization.py`
- Trajectory optimization with automatic caching
- **Two-phase speed control:**
  - Phase 1: Fast scaling (approximation, ~1% error)
  - Phase 2: Refinement optimization (exact speed, <0.001% error)
- Smooth stop segment generation

### `trajectory/plotting.py`
- 3R manipulator animation
- Joint and end-effector trajectory plots

### `trajectory/utils.py`
- Array interpolation utilities
- Full trajectory assembly

### `trajectory/cache.py`
- Save/load trajectory solutions
- Cache key computation based on parameters
- Cache management utilities

## Why Numeric Inverse Kinematics?

This project uses **numeric methods** for inverse kinematics because:

1. **Complexity**: Nonlinear equations make symbolic solutions impractical
2. **Multiple Solutions**: Numeric methods handle multiple valid configurations
3. **Generality**: Works for various robot configurations and constraints
4. **Efficiency**: Fast approximate solutions vs expensive symbolic computation

The trajectory optimization uses **CasADi** for automatic differentiation and efficient nonlinear programming with IPOPT solver.

## Performance

- **Without cache**: ~5s (optimization + visualization)
- **With cache**: ~3s (cache load + visualization)
- **Cache savings**: ~40% reduction in computation time
