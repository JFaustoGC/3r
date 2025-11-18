"""Main execution script for 3R manipulator trajectory planning and visualization."""

import numpy as np
from kinematics import inverse_kinematics_3r
from trajectory import (
    solve_trajectory_problem,
    scale_casadi_solution_taskspace_peak,
    refine_trajectory_for_exact_speed,
    append_stop_trajectory,
    animate_3r_trajectory,
    plot_joint_trajectories,
    plot_ee_kinematics
)


def main():
    """Main execution function."""
    # Initial and final configurations
    q0 = inverse_kinematics_3r(np.array([0.175, 0.025, -2.094395102393195]))
    xT = np.array([0.825, 0.0, 0.0])
    
    # Trajectory parameters
    T = 0.7
    angle = 45.0 * (np.pi / 180.0)
    speed = 2.7
    v_final = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])
    dt = 0.01
    desired_speed = 2.7  # Desired end-effector speed after scaling and refinement
    
    # Solve trajectory optimization problem (maximum speed)
    print("Solving trajectory optimization problem...")
    solution = solve_trajectory_problem(v_final, T, q0, xT)
    
    # Scale to desired speed
    print("Scaling trajectory to desired speed...")
    scaled_sol, meta = scale_casadi_solution_taskspace_peak(solution, v_desired=desired_speed, verbose=True)
    
    # Refine to achieve exact desired speed
    print("Refining trajectory for exact speed...")
    refined_sol, refine_info = refine_trajectory_for_exact_speed(
        scaled_sol, v_desired=desired_speed, q0=q0, xT=xT, verbose=True
    )
    solution = refined_sol
    
    # Add smooth stop segment - return towards initial position
    motion_time = solution['time'][-1]
    stop_time = 2.0  # Time to stop (longer allows more return)
    return_home_weight = 5.0  # Weight for returning to initial position (higher = stronger return)
    total_time = motion_time + stop_time
    total_time_arr = np.arange(0.0, total_time + dt/2, dt)
    
    print(f"Adding smooth stop segment (return_home_weight={return_home_weight})...")
    q_full, qd_full, qdd_full = append_stop_trajectory(
        solution['Q'], solution['Qd'], solution['Qdd'],
        dt=dt, T_stop=stop_time, return_home_weight=return_home_weight
    )
    
    # Update solution with stop segment
    solution['Q'] = q_full
    solution['Qd'] = qd_full
    solution['Qdd'] = qdd_full
    solution['time'] = total_time_arr
    
    # Visualize results
    print("Plotting joint trajectories...")
    plot_joint_trajectories(solution)
    
    print("Plotting end-effector kinematics...")
    plot_ee_kinematics(solution)
    
    print("Starting animation...")
    animate_3r_trajectory(q_traj=solution['Q'].T, dt=dt)
    
    print("Done!")


if __name__ == '__main__':
    main()
