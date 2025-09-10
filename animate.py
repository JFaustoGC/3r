import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation
from kinematics import forward_kinematics_3r, inverse_kinematics_3r
from config import l1, l2, l3, base_offset_x, base_offset_z, global_gripper_width, global_gripper_length, gripper
import casadi as ca

joint_limits = {
    "slow": {
        "vel": np.array([0.2825, 0.415, 0.74]),
        "accel": np.array([1.5, 3.0, 3.0])
    },
    "medium": {
        "vel": np.array([0.678, 0.996, 1.776]),
        "accel": np.array([2.5, 5.0, 5.0])
    },
    "fast": {
        "vel": np.array([1.13, 1.66, 2.96]),
        "accel": np.array([5.0, 8.0, 8.0])
    },
    "express": {
        "vel": np.array([1.13, 1.66, 2.96]),
        "accel": np.array([8.0, 10.0, 12.0])
    }
}


def animate_3r_trajectory(q_traj: np.ndarray, dt: float) -> None:
    """
    Animate a 3R manipulator given a trajectory of joint configurations (q arrays).
    Uses CasADi FK for exact EE tip position to guarantee it matches plots.
    """
    q_traj = np.asarray(q_traj)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Estimate total reach for axis limits
    total_length = l1 + l2 + l3 + gripper
    ax.set_xlim(-0.1, 1.3)
    ax.set_ylim(base_offset_z - total_length/2, base_offset_z + total_length/2)
    
    # Precompute FK path
    ee_path = np.zeros((len(q_traj), 2))
    joint_positions_all = []
    for i, q in enumerate(q_traj):
        x, z, theta = forward_kinematics_3r(q)
        ee_path[i] = [x, z]

        # Compute joint positions for animation links
        t1, t2, t3 = -q[0], -q[0]-q[1], -q[0]-q[1]-q[2]
        x0, y0 = base_offset_x, base_offset_z
        x1 = x0 + l1 * np.cos(t1); y1 = y0 + l1 * np.sin(t1)
        x2 = x1 + l2 * np.cos(t2); y2 = y1 + l2 * np.sin(t2)
        x3 = x2 + l3 * np.cos(t3); y3 = y2 + l3 * np.sin(t3)
        joint_positions_all.append([(x0, y0), (x1, y1), (x2, y2), (x3, y3), (x, z)])

    ax.plot(ee_path[:,0], ee_path[:,1], 'k--', label='Reference Path', zorder=5)
    ax.legend()

    # Plot elements
    line, = ax.plot([], [], '-', lw=4, color='blue')
    joint_circles = [Circle((0,0), 0.015, color='black', zorder=20) for _ in range(4)]
    for circ in joint_circles:
        ax.add_patch(circ)
    gripper_patch = Rectangle((0,0), global_gripper_length, global_gripper_width, angle=0, color='red', zorder=10)
    ax.add_patch(gripper_patch)

    ee_x_arrow = None
    ee_z_arrow = None

    def init():
        line.set_data([], [])
        for circ in joint_circles:
            circ.center = (0,0)
        gripper_patch.set_xy((0,0))
        return (line, gripper_patch, *joint_circles)

    def animate(i):
        nonlocal ee_x_arrow, ee_z_arrow
        joints = joint_positions_all[i]
        line.set_data([p[0] for p in joints], [p[1] for p in joints])
        for circ, pos in zip(joint_circles, joints[:-1]):
            circ.center = pos

        # Tip position from FK
        ee_x, ee_y = joints[-1]
        theta = -q_traj[i,0] - q_traj[i,1] - q_traj[i,2]  # same FK convention

        # Gripper rectangle (small, just for visuals)
        gripper_patch.set_width(global_gripper_length)
        gripper_patch.set_height(global_gripper_width)
        gripper_patch.set_angle(np.rad2deg(theta))
        center_dx = global_gripper_length / 2 * np.cos(theta)
        center_dy = global_gripper_length / 2 * np.sin(theta)
        gripper_patch.set_xy((ee_x - center_dx - global_gripper_width/2 * np.sin(theta),
                              ee_y - center_dy + global_gripper_width/2 * np.cos(theta)))

        # EE frame arrows
        if ee_x_arrow: ee_x_arrow.remove()
        if ee_z_arrow: ee_z_arrow.remove()
        ee_x_arrow = ax.arrow(ee_x, ee_y, 0.07*np.cos(theta), 0.07*np.sin(theta),
                              head_width=0.02, head_length=0.04, fc='r', ec='r', zorder=30)
        ee_z_arrow = ax.arrow(ee_x, ee_y, -0.07*np.sin(theta), 0.07*np.cos(theta),
                              head_width=0.02, head_length=0.04, fc='g', ec='g', zorder=30)
        return (line, gripper_patch, *joint_circles, ee_x_arrow, ee_z_arrow)

    interval_ms = int(dt*1000)
    ani = animation.FuncAnimation(fig, animate, frames=len(q_traj), init_func=init,
                                  blit=True, interval=interval_ms, repeat=False)
    plt.show()

    
    
    
# ==============================================================================
# ## 1. Robot Model and Kinematics
# ==============================================================================
def get_kinematics_functions():
    """Builds and returns callable CasADi functions for the robot model."""
    q_s = ca.SX.sym('q', 3); qd_s = ca.SX.sym('qd', 3)
    t1_s = -q_s[0]; t2_s = -q_s[0] - q_s[1]; t3_s = -q_s[0] - q_s[1] - q_s[2]; L3_gripper = l3 + gripper
    x_s = base_offset_x + l1 * ca.cos(t1_s) + l2 * ca.cos(t2_s) + L3_gripper * ca.cos(t3_s)
    z_s = base_offset_z + l1 * ca.sin(t1_s) + l2 * ca.sin(t2_s) + L3_gripper * ca.sin(t3_s)
    theta_s = t3_s; pose_s = ca.vertcat(x_s, z_s, theta_s)
    J_s = ca.jacobian(pose_s, q_s); p_dot_s = J_s @ qd_s; Jdqd_s = ca.jtimes(p_dot_s, q_s, qd_s)
    return {
        'fk': ca.Function('fk', [q_s], [pose_s]),
        'jacobian': ca.Function('jacobian', [q_s], [J_s]),
        'jdot_qdot': ca.Function('jdot_qdot', [q_s, qd_s], [Jdqd_s])
    }



# ==============================================================================
# ## 2. Core Optimization Function
# ==============================================================================
def solve_trajectory_problem(kinematics, params):
    T = params['T']; N = params['N']; h = T / N; n_joints = 3
    
    opti = ca.Opti()

    # Decision Variables
    Q = opti.variable(n_joints, N + 1)
    Qd = opti.variable(n_joints, N + 1)
    Qdd = opti.variable(n_joints, N + 1)

    # Objective Function
    cost = 0
    w = params['weights']
    for k in range(N):
        # penalize vi - vi-1 to smooth velocities
        cost += w['vel'] * ca.sumsqr(Qd[:, k+1] - Qd[:, k])
        cost += w['accel'] * ca.sumsqr(Qdd[:, k+1] - Qdd[:, k])
        cost += w['pos'] * ca.sumsqr(Q[:, k+1] - Q[:, k])
        
    opti.minimize(cost)

    # Dynamics Constraints
    for k in range(N):
        q_next, qd_next = Q[:, k+1], Qd[:, k+1]
        q_curr, qd_curr = Q[:, k], Qd[:, k]
        qdd_curr, qdd_next = Qdd[:, k], Qdd[:, k+1]
        opti.subject_to(q_next == q_curr + h/2 * (qd_curr + qd_next))
        opti.subject_to(qd_next == qd_curr + h/2 * (qdd_curr + qdd_next))

    opti.subject_to(kinematics['fk'](Q[:, 0])[0:2] == params['p_initial'][0:2])
    opti.subject_to(kinematics['fk'](Q[:, -1])[0:2] == params['p_final'][0:2])
    opti.subject_to((kinematics['jacobian'](Q[:, -1]) @ Qd[:, -1])[0:2] == params['v_final'][0:2])

    q_vel_limits = joint_limits[params['setup']]["vel"]
    q_accel_limits = joint_limits[params['setup']]["accel"]
    
    # Apply bounds to each joint's trajectory
    for i in range(n_joints):
        # Use opti.bounded for a clean way to set lower and upper bounds
        opti.subject_to(opti.bounded(-q_vel_limits[i], Qd[i, :], q_vel_limits[i]))
        opti.subject_to(opti.bounded(-q_accel_limits[i], Qdd[i, :], q_accel_limits[i]))
    # --------------------------------------------------------------------------

    q3_fixed_value = np.pi / 2.0
    # opti.subject_to(Q[2, :] == q3_fixed_value)  # Constrain position
    # opti.subject_to(Qd[2, :] == 0)  # Constrain velocity
    # opti.subject_to(Qdd[2, :] == 0)  # Constrain acceleration

    # Initial Guess
    try:
        q_initial_guess = inverse_kinematics_3r(params['p_initial'])
        q_final_guess = inverse_kinematics_3r(params['p_final'])
        q_guess_traj = np.linspace(q_initial_guess, q_final_guess, N + 1).T
        opti.set_initial(Q, q_guess_traj)
    except ValueError as e:
        print(f"Could not generate initial guess: {e}")
        return None

    # Solve
    opti.solver('ipopt')
    try:
        sol = opti.solve()
        print("✅ Trajectory optimization successful!")
        return {
            "Q": sol.value(Q), "Qd": sol.value(Qd), "Qdd": sol.value(Qdd),
            "time": np.linspace(0, T, N + 1)
        }
    except RuntimeError as e:
        print(f"❌ Solver failed: {e}")
        return None

# ==============================================================================
# ## 3. Post-Processing and Plotting
# ==============================================================================
def post_process_and_plot(solution, kinematics, params):
    """Calculates task-space trajectories and plots all results."""
    Q_opt, Qd_opt, Qdd_opt = solution['Q'], solution['Qd'], solution['Qdd']
    time = solution['time']
    N = params['N']
    
    # Calculate EE position, velocity, and acceleration trajectories
    p_opt = np.zeros((3, N + 1)); p_dot_opt = np.zeros((3, N + 1)); p_ddot_opt = np.zeros((3, N + 1))
    for k in range(N + 1):
        q_k, qd_k, qdd_k = Q_opt[:, k], Qd_opt[:, k], Qdd_opt[:, k]
        p_opt[:, k] = kinematics['fk'](q_k).full().flatten()
        p_dot_opt[:, k] = (kinematics['jacobian'](q_k) @ qd_k).full().flatten()
        p_ddot_opt[:, k] = (kinematics['jacobian'](q_k) @ qdd_k + kinematics['jdot_qdot'](q_k, qd_k)).full().flatten()

    # Plotting
    fig, axs = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)
    fig.suptitle('Direct Collocation Trajectory Optimization Results', fontsize=18)
    # ... (Plotting code for all 8 subplots is identical to the previous version)
    axs[0, 0].plot(time, Q_opt.T); axs[0, 0].set_title('Joint Positions (q)'); axs[0, 0].set_ylabel('Angle [rad]'); axs[0, 0].legend([f'q{i}' for i in range(3)]); axs[0, 0].grid(True)
    axs[1, 0].plot(time, Qd_opt.T); axs[1, 0].set_title('Joint Velocities (qd)'); axs[1, 0].set_ylabel('rad/s'); axs[1, 0].grid(True)
    axs[2, 0].plot(time, Qdd_opt.T); axs[2, 0].set_title('Joint Accelerations (qdd)'); axs[2, 0].set_ylabel('rad/s^2'); axs[2, 0].set_xlabel('Time [s]'); axs[2, 0].grid(True)
    axs[3, 0].plot(p_opt[0, :], p_opt[1, :], '-o', markersize=3, color='purple'); axs[3, 0].plot(params['p_initial'][0], params['p_initial'][1], 'go', markersize=10, label='Start'); axs[3, 0].plot(params['p_final'][0], params['p_final'][1], 'ro', markersize=10, label='End'); axs[3, 0].set_title('End-Effector Spatial Path'); axs[3, 0].set_xlabel('x [m]'); axs[3, 0].set_ylabel('z [m]'); axs[3, 0].axis('equal'); axs[3, 0].legend(); axs[3, 0].grid(True)
    axs[0, 1].plot(time, p_opt.T); axs[0, 1].set_title('End-Effector Position (p)'); axs[0, 1].set_ylabel('Position / Angle'); axs[0, 1].legend(['x', 'z', '$\\theta$']); axs[0, 1].grid(True)
    axs[1, 1].plot(time, p_dot_opt.T); axs[1, 1].set_title('End-Effector Velocity (p_dot)'); axs[1, 1].set_ylabel('Velocity / Angular Vel.'); axs[1, 1].legend(['$v_x$', '$v_z$', '$\\omega$']); axs[1, 1].grid(True)
    axs[2, 1].plot(time, p_ddot_opt.T); axs[2, 1].set_title('End-Effector Acceleration (p_ddot)'); axs[2, 1].set_ylabel('Accel. / Angular Accel.'); axs[2, 1].set_xlabel('Time [s]'); axs[2, 1].legend(['$a_x$', '$a_z$', '$\\alpha$']); axs[2, 1].grid(True)
    fig.delaxes(axs[3,1]); plt.show()


import numpy as np


def assemble_full_trajectory(q, qd, qdd, fixed_values=None):

    if fixed_values is None:
        fixed_values = {'j0': 0.0, 'j2': 0.0, 'j4': 0.0, 'j6': 1.7659902159840577}
    N = q.shape[1]
    # Indices: [j0, j1, j2, j3, j4, j5, j6]
    joint_order = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    # Map: joint name -> column in q/q_full
    q_full = np.zeros((N, 7))
    qd_full = np.zeros((N, 7))
    qdd_full = np.zeros((N, 7))

    # Insert variable joints (assume q[0,:]=j1, q[1,:]=j3, q[2,:]=j5)
    # j1 at index 1, j3 at index 3, j5 at index 5
    q_full[:, 1] = q[0]
    q_full[:, 3] = q[1]
    q_full[:, 5] = q[2]
    qd_full[:, 1] = qd[0]
    qd_full[:, 3] = qd[1]
    qd_full[:, 5] = qd[2]
    qdd_full[:, 1] = qdd[0]
    qdd_full[:, 3] = qdd[1]
    qdd_full[:, 5] = qdd[2]

    # Fixed joints: j0 (0), j2 (2), j4 (4), j6 (6)
    for idx, joint in zip([0, 2, 4, 6], ['j0', 'j2', 'j4', 'j6']):
        val = fixed_values[joint]
        q_full[:, idx] = val
        qd_full[:, idx] = 0.0
        qdd_full[:, idx] = 0.0

    return q_full, qd_full, qdd_full



if __name__ == '__main__':

    q0 = np.deg2rad([-45, 135, 90])
    qT = np.deg2rad([-55, 30, 90])
    x0 = forward_kinematics_3r(q0)
    xT = forward_kinematics_3r(qT)
    T = 0.7  # seconds
    DT = 0.01  # seconds
    N = int(T / DT)

    desired_speed = 1  # m/s
    desired_angle = np.deg2rad(30) # radians
    desired_velocity = np.array([np.cos(desired_angle), np.sin(desired_angle), 0.0]) * desired_speed
    print(f"Start Pose: {x0}, Goal Pose: {xT}, Desired Velocity: {desired_velocity}")

    setup = "express"  # could be "slow", "medium", "fast", or "express"

    # 1. Define the problem in a parameters dictionary
    problem_params = {
        'T': T,  # Total trajectory time [s]
        'N': N,   # Number of control intervals
        'p_initial': x0,
        'p_final': xT,
        'v_final': desired_velocity,
        'weights': {
            'accel': 1.0,
            'vel': 1.0,
            'pos': 1.0
        },
        'setup': setup
    }

    # 2. Generate the robot's kinematic functions
    kinematic_functions = get_kinematics_functions()

    # 3. Solve the optimization problem
    solution = solve_trajectory_problem(kinematic_functions, problem_params)

    # 4. Plot the results if a solution was found
    if solution:
        post_process_and_plot(solution, kinematic_functions, problem_params)

        # 5. Animate the resulting trajectory

        animate_3r_trajectory(solution['Q'].T, dt=problem_params['T']/problem_params['N'])

        full_q, full_qd, full_qdd = assemble_full_trajectory(solution['Q'], solution['Qd'], solution['Qdd'])
        np.savez('3r_trajectory.npz', q=full_q, qd=full_qd, qdd=full_qdd, time=solution['time'])
    else:
        print("No solution found; skipping plotting and animation.")


    # print(forward_kinematics_3r(q0))
    # print(forward_kinematics_3r(qT))
    # animate_3r_trajectory(np.linspace(q0, qT, 100), dt=0.1)

