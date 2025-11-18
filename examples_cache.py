#!/usr/bin/env python3
"""
Example: Using trajectory caching for different scenarios.

This script demonstrates how to use the trajectory cache system
for precomputing and reusing maximum speed trajectory profiles.
"""

import numpy as np
from kinematics import inverse_kinematics_3r
from trajectory import (
    solve_trajectory_problem,
    scale_casadi_solution_taskspace_peak,
    list_cached_trajectories,
    clear_trajectory_cache
)


def example_basic_usage():
    """Basic example: solve and cache a trajectory."""
    print("\n=== Basic Usage Example ===\n")
    
    # Define trajectory
    q0 = inverse_kinematics_3r(np.array([0.175, 0.025, -2.094395102393195]))
    xT = np.array([0.825, 0.0, 0.0])
    T = 0.7
    v_final = np.array([1.909, 1.909, 0.0])
    
    # First call: computes and caches
    print("First call (will compute):")
    solution = solve_trajectory_problem(v_final, T, q0, xT)
    
    # Second call: loads from cache
    print("\nSecond call (will load from cache):")
    solution = solve_trajectory_problem(v_final, T, q0, xT)
    
    print(f"\nSolution shape: Q={solution['Q'].shape}, time={solution['time'].shape}")


def example_multiple_speeds():
    """Example: precompute max speed, then scale to multiple speeds."""
    print("\n=== Multiple Speed Scaling Example ===\n")
    
    q0 = inverse_kinematics_3r(np.array([0.175, 0.025, -2.094395102393195]))
    xT = np.array([0.825, 0.0, 0.0])
    T = 0.7
    angle = 45.0 * (np.pi / 180.0)
    max_speed = 2.7
    v_final = np.array([max_speed * np.cos(angle), max_speed * np.sin(angle), 0.0])
    
    # Compute/load max speed trajectory (cached)
    print("Computing maximum speed trajectory...")
    max_solution = solve_trajectory_problem(v_final, T, q0, xT)
    
    # Scale to different speeds instantly
    speeds = [2.5, 2.0, 1.5, 1.0, 0.5]
    print(f"\nScaling to {len(speeds)} different speeds:")
    
    for desired_speed in speeds:
        scaled_sol, info = scale_casadi_solution_taskspace_peak(
            max_solution, 
            v_desired=desired_speed,
            verbose=False
        )
        print(f"  Speed {desired_speed:.1f} m/s: "
              f"T={info['T_new']:.3f}s, "
              f"achieved={info['achieved_peak']:.3f} m/s")


def example_cache_management():
    """Example: managing the cache."""
    print("\n=== Cache Management Example ===\n")
    
    # List cached trajectories
    cached = list_cached_trajectories()
    print(f"Currently cached trajectories: {len(cached)}")
    
    for i, item in enumerate(cached, 1):
        print(f"\n{i}. Cache file: {item['file']}")
        print(f"   Duration: {item['T']:.2f}s")
        print(f"   Initial q0: {item['q0']}")
        print(f"   Final xT: {item['xT']}")


def example_force_recompute():
    """Example: force recomputation without using cache."""
    print("\n=== Force Recompute Example ===\n")
    
    q0 = inverse_kinematics_3r(np.array([0.175, 0.025, -2.094395102393195]))
    xT = np.array([0.825, 0.0, 0.0])
    T = 0.7
    v_final = np.array([1.909, 1.909, 0.0])
    
    print("Computing with cache disabled:")
    solution = solve_trajectory_problem(v_final, T, q0, xT, use_cache=False)
    print("Done (result not saved to cache)")


if __name__ == "__main__":
    print("=" * 60)
    print("Trajectory Caching Examples")
    print("=" * 60)
    
    example_basic_usage()
    example_multiple_speeds()
    example_cache_management()
    # example_force_recompute()  # Uncomment to test
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
