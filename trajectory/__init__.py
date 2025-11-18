"""Trajectory planning and optimization package for 3R manipulator."""

from .kinematics import get_kinematics_functions
from .optimization import solve_trajectory_problem, scale_casadi_solution_taskspace_peak, refine_trajectory_for_exact_speed, smooth_stop_segment, append_stop_trajectory
from .plotting import animate_3r_trajectory, plot_joint_trajectories, plot_ee_kinematics
from .utils import linspace_arrays, assemble_full_trajectory
from .cache import clear_trajectory_cache, list_cached_trajectories

__all__ = [
    'get_kinematics_functions',
    'solve_trajectory_problem',
    'scale_casadi_solution_taskspace_peak',
    'refine_trajectory_for_exact_speed',
    'smooth_stop_segment',
    'append_stop_trajectory',
    'animate_3r_trajectory',
    'plot_joint_trajectories',
    'plot_ee_kinematics',
    'linspace_arrays',
    'assemble_full_trajectory',
    'clear_trajectory_cache',
    'list_cached_trajectories',
]
