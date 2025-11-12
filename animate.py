import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation
from kinematics import forward_kinematics_3r, inverse_kinematics_3r
from config import l1, l2, l3, base_offset_x, base_offset_z, global_gripper_width, global_gripper_length, gripper
import casadi as ca


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
    ax.set_ylim(base_offset_z - total_length / 2, base_offset_z + total_length / 2)

    # Precompute FK path
    ee_path = np.zeros((len(q_traj), 2))
    joint_positions_all = []
    for i, q in enumerate(q_traj):
        x, z, theta = forward_kinematics_3r(q)
        ee_path[i] = [x, z]


        # Compute joint positions for animation links
        t1, t2, t3 = -q[0], -q[0] - q[1], -q[0] - q[1] - q[2]
        x0, y0 = base_offset_x, base_offset_z
        x1 = x0 + l1 * np.cos(t1)
        y1 = y0 + l1 * np.sin(t1)
        x2 = x1 + l2 * np.cos(t2)
        y2 = y1 + l2 * np.sin(t2)
        x3 = x2 + l3 * np.cos(t3)
        y3 = y2 + l3 * np.sin(t3)
        joint_positions_all.append([(x0, y0), (x1, y1), (x2, y2), (x3, y3), (x, z)])

    ax.plot(ee_path[:, 0], ee_path[:, 1], 'k--', label='Reference Path', zorder=5)
    ax.legend()

    # Plot elements
    line, = ax.plot([], [], '-', lw=4, color='blue')
    joint_circles = [Circle((0, 0), 0.015, color='black', zorder=20) for _ in range(4)]
    for circ in joint_circles:
        ax.add_patch(circ)
    gripper_patch = Rectangle((0, 0), global_gripper_length, global_gripper_width, angle=0, color='red', zorder=10)
    ax.add_patch(gripper_patch)

    ee_x_arrow = None
    ee_z_arrow = None

    def init():
        line.set_data([], [])
        for circ in joint_circles:
            circ.center = (0, 0)
        gripper_patch.set_xy((0, 0))
        return (line, gripper_patch, *joint_circles)

    def animate(i):
        nonlocal ee_x_arrow, ee_z_arrow
        joints = joint_positions_all[i]
        line.set_data([p[0] for p in joints], [p[1] for p in joints])
        for circ, pos in zip(joint_circles, joints[:-1]):
            circ.center = pos

        # Tip position from FK
        ee_x, ee_y = joints[-1]
        theta = -q_traj[i, 0] - q_traj[i, 1] - q_traj[i, 2]  # same FK convention

        # Gripper rectangle (small, just for visuals)
        gripper_patch.set_width(global_gripper_length)
        gripper_patch.set_height(global_gripper_width)
        gripper_patch.set_angle(np.rad2deg(theta))
        center_dx = global_gripper_length / 2 * np.cos(theta)
        center_dy = global_gripper_length / 2 * np.sin(theta)
        gripper_patch.set_xy((ee_x - center_dx - global_gripper_width / 2 * np.sin(theta),
                              ee_y - center_dy + global_gripper_width / 2 * np.cos(theta)))

        # EE frame arrows
        if ee_x_arrow: ee_x_arrow.remove()
        if ee_z_arrow: ee_z_arrow.remove()
        ee_x_arrow = ax.arrow(ee_x, ee_y, 0.07 * np.cos(theta), 0.07 * np.sin(theta),
                              head_width=0.02, head_length=0.04, fc='r', ec='r', zorder=30)
        ee_z_arrow = ax.arrow(ee_x, ee_y, -0.07 * np.sin(theta), 0.07 * np.cos(theta),
                              head_width=0.02, head_length=0.04, fc='g', ec='g', zorder=30)
        return (line, gripper_patch, *joint_circles, ee_x_arrow, ee_z_arrow)

    interval_ms = int(dt * 1000)
    ani = animation.FuncAnimation(fig, animate, frames=len(q_traj), init_func=init,
                                  blit=True, interval=interval_ms, repeat=False)
    plt.show()


# ==============================================================================
# ## 1. Robot Model and Kinematics
# ==============================================================================
def get_kinematics_functions():
    """Builds and returns callable CasADi functions for the robot model."""
    q_s = ca.SX.sym('q', 3)
    qd_s = ca.SX.sym('qd', 3)
    t1_s = -q_s[0]
    t2_s = -q_s[0] - q_s[1]
    t3_s = -q_s[0] - q_s[1] - q_s[2]
    L3_gripper = l3 + gripper
    x_s = base_offset_x + l1 * ca.cos(t1_s) + l2 * ca.cos(t2_s) + L3_gripper * ca.cos(t3_s)
    z_s = base_offset_z + l1 * ca.sin(t1_s) + l2 * ca.sin(t2_s) + L3_gripper * ca.sin(t3_s)
    theta_s = t3_s
    pose_s = ca.vertcat(x_s, z_s, theta_s)
    J_s = ca.jacobian(pose_s, q_s)
    p_dot_s = J_s @ qd_s
    Jdqd_s = ca.jtimes(p_dot_s, q_s, qd_s)
    return {
        'fk': ca.Function('fk', [q_s], [pose_s]),
        'jacobian': ca.Function('jacobian', [q_s], [J_s]),
        'jdot_qdot': ca.Function('jdot_qdot', [q_s, qd_s], [Jdqd_s])
    }


config = {
    "joint_limits": {
        "pos": {
            "min": np.array([-3.8095, -3.0439, -2.9761]),
            "max": np.array([2.2736, 3.0439, 2.9761]),
        },
        "slow": {
            "vel": np.array([0.2825, 0.415, 0.74]),
            "accel": np.array([1.5, 3.0, 3.0]),
        },
        "medium": {
            "vel": np.array([0.678, 0.996, 1.776]),
            "accel": np.array([2.5, 5.0, 5.0]),
        },
        "fast": {
            "vel": np.array([1.13, 1.66, 2.96]),
            "accel": np.array([5.0, 8.0, 8.0]),
        },
        "express": {
            "vel": np.array([1.13, 1.66, 2.96]),
            "accel": np.array([8.0, 10.0, 12.0]),
        },
    },
    "weights": {
        "accel": 1.0,
        "vel": 1.0,
        "pos": 1.0,
        "vel_dir": 1.0,
    },
    "start_tolerance_x": 0.0,  # A small value, e.g., 10 cm
    "start_tolerance_z": 0.0,
    "setup": "express",
    "dt" : 0.01,
    "release_angle" : 45.0 * (np.pi / 180.0),  # radians
}






def linspace_arrays(start, stop, num):
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    result = np.empty((num, start.size))
    for i in range(num):
        alpha = float(i) / (num - 1) if num > 1 else 0.0
        result[i] = (1 - alpha) * start + alpha * stop
    return result

def solve_trajectory_problem(v_final, T, q0, xT):
    kinematics = get_kinematics_functions()
    h = config["dt"]
    N = int(T / h)
    n_joints = 3

    opti = ca.Opti()

    Q = opti.variable(n_joints, N + 1)
    Qd = opti.variable(n_joints, N + 1)
    Qdd = opti.variable(n_joints, N + 1)
    cost = 0
    w = config["weights"]
    for k in range(N):
        cost += w["vel"] * ca.sumsqr(Qd[:, k + 1] - Qd[:, k])
        cost += w["accel"] * ca.sumsqr(Qdd[:, k + 1] - Qdd[:, k])
        cost += w["pos"] * ca.sumsqr(Q[:, k + 1] - Q[:, k])

    opti.minimize(cost)

    for k in range(N):
        q_next, qd_next = Q[:, k + 1], Qd[:, k + 1]
        q_curr, qd_curr = Q[:, k], Qd[:, k]
        qdd_curr, qdd_next = Qdd[:, k], Qdd[:, k + 1]
        opti.subject_to(q_next == q_curr + h / 2.0 * (qd_curr + qd_next))
        opti.subject_to(qd_next == qd_curr + h / 2.0 * (qdd_curr + qdd_next))



    opti.subject_to(kinematics["fk"](Q[:, -1])[0:2] == xT[0:2])
    opti.subject_to(
        (ca.mtimes(kinematics["jacobian"](Q[:, -1]), Qd[:, -1])[0:2])
        == v_final[0:2]
    )

    opti.subject_to(Q[:, 0] == q0)

    joint_limits = config["joint_limits"]
    q_vel_limits = joint_limits[config["setup"]]["vel"]
    q_accel_limits = joint_limits[config["setup"]]["accel"]
    pos_max = joint_limits["pos"]["max"]
    pos_min = joint_limits["pos"]["min"]

    for i in range(n_joints):
        opti.subject_to(opti.bounded(pos_min[i], Q[i, :], pos_max[i]))
        opti.subject_to(opti.bounded(-q_vel_limits[i], Qd[i, :], q_vel_limits[i]))
        opti.subject_to(opti.bounded(-q_accel_limits[i], Qdd[i, :], q_accel_limits[i]))



    angles = [0.0, -0.50, 0.0, 2.07, 0.0, 2.19, 1.76599022]
    q_initial_guess = np.array(angles)[[1, 3, 5]]
    q_final_guess = inverse_kinematics_3r(xT)
    q_guess_traj = linspace_arrays(q_initial_guess, q_final_guess, N + 1).T
    opti.set_initial(Q, q_guess_traj)

    opti.solver("ipopt", {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False})
    sol = opti.solve()
    return {
        "Q": sol.value(Q),
        "Qd": sol.value(Qd),
        "Qdd": sol.value(Qdd),
        "time": np.linspace(0, T, N + 1),
    }

def post_process_and_plot(solution):
    kinematics = get_kinematics_functions()
    """Calculates task-space trajectories and plots all results."""
    Q_opt, Qd_opt, Qdd_opt = solution['Q'], solution['Qd'], solution['Qdd']

    N = len(Q_opt)
    dt = 0.01
    time = np.linspace(0, dt * (N - 1), N)

    # Calculate EE position, velocity, and acceleration trajectories
    p_opt = np.zeros((3, N + 1))
    p_dot_opt = np.zeros((3, N + 1))
    p_ddot_opt = np.zeros((3, N + 1))
    for k in range(N + 1):
        q_k, qd_k, qdd_k = Q_opt[:, k], Qd_opt[:, k], Qdd_opt[:, k]
        p_opt[:, k] = kinematics['fk'](q_k).full().flatten()
        p_dot_opt[:, k] = (kinematics['jacobian'](q_k) @ qd_k).full().flatten()
        p_ddot_opt[:, k] = (kinematics['jacobian'](q_k) @ qdd_k + kinematics['jdot_qdot'](q_k, qd_k)).full().flatten()

    # Plotting
    fig, axs = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)
    fig.suptitle('Direct Collocation Trajectory Optimization Results', fontsize=18)
    # ... (Plotting code for all 8 subplots is identical to the previous version)
    axs[0, 0].plot(time, Q_opt.T)
    axs[0, 0].set_title('Joint Positions (q)')
    axs[0, 0].set_ylabel('Angle [rad]')
    axs[0, 0].legend([f'q{i}' for i in range(3)])
    axs[0, 0].grid(True)
    axs[1, 0].plot(time, Qd_opt.T)
    axs[1, 0].set_title('Joint Velocities (qd)')
    axs[1, 0].set_ylabel('rad/s')
    axs[1, 0].grid(True)
    axs[2, 0].plot(time, Qdd_opt.T)
    axs[2, 0].set_title('Joint Accelerations (qdd)')
    axs[2, 0].set_ylabel('rad/s^2')
    axs[2, 0].set_xlabel('Time [s]')
    axs[2, 0].grid(True)
    axs[3, 0].plot(p_opt[0, :], p_opt[1, :], '-o', markersize=3, color='purple')
    # axs[3, 0].plot(params['p_initial'][0], params['p_initial'][1], 'go', markersize=10, label='Start')
    # axs[3, 0].plot(params['p_final'][0], params['p_final'][1], 'ro', markersize=10, label='End')
    axs[3, 0].set_title('End-Effector Spatial Path')
    axs[3, 0].set_xlabel('x [m]')
    axs[3, 0].set_ylabel('z [m]')
    axs[3, 0].axis('equal')
    axs[3, 0].legend()
    axs[3, 0].grid(True)
    axs[0, 1].plot(time, p_opt.T)
    axs[0, 1].set_title('End-Effector Position (p)')
    axs[0, 1].set_ylabel('Position / Angle')
    axs[0, 1].legend(['x', 'z', '$\\theta$'])
    axs[0, 1].grid(True)
    axs[1, 1].plot(time, p_dot_opt.T)
    axs[1, 1].set_title('End-Effector Velocity (p_dot)')
    axs[1, 1].set_ylabel('Velocity / Angular Vel.')
    axs[1, 1].legend(['$v_x$', '$v_z$', '$\\omega$'])
    axs[1, 1].grid(True)
    axs[2, 1].plot(time, p_ddot_opt.T)
    axs[2, 1].set_title('End-Effector Acceleration (p_ddot)')
    axs[2, 1].set_ylabel('Accel. / Angular Accel.')
    axs[2, 1].set_xlabel('Time [s]')
    axs[2, 1].legend(['$a_x$', '$a_z$', '$\\alpha$'])
    axs[2, 1].grid(True)
    fig.delaxes(axs[3, 1])
    plt.show()



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


from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
from tqdm import tqdm
import threading


def find_best_configuration():
    min_start_x = 0.1
    max_start_x = 0.3
    min_end_x = 0.7
    max_end_x = 0.8
    min_start_y = 0.0
    max_start_y = 0.2
    min_end_y = 0.0
    max_end_y = 0.2
    delta = 0.025

    top_solutions = []
    max_heap_size = 100
    best_distance = [0.0]  # Use a mutable object for thread-safe updates
    best_distance_lock = threading.Lock()

    x_start_vals = np.arange(min_start_x, max_start_x + delta, delta)
    y_start_vals = np.arange(min_start_y, max_start_y + delta, delta)
    x_end_vals = np.arange(min_end_x, max_end_x + delta, delta)
    y_end_vals = np.arange(min_end_y, max_end_y + delta, delta)
    angle_vals = np.array([4.1887902047863905])
    T_vals = np.array([0.7])

    total = (
        len(x_start_vals)
        * len(y_start_vals)
        * len(x_end_vals)
        * len(y_end_vals)
        * len(angle_vals)
        * len(T_vals)  # include T in the combinations
    )

    g = 9.81

    def get_best_distance():
        with best_distance_lock:
            return best_distance[0]

    def update_best_distance(new_distance):
        with best_distance_lock:
            if new_distance > best_distance[0]:
                best_distance[0] = new_distance
                return True
            return False

    def ballistic_distance(y0, v, theta):
        vy = v * np.sin(theta)
        vx = v * np.cos(theta)
        if vy < 0:
            return 0.0
        t_flight = (vy + np.sqrt(vy ** 2 + 2 * g * y0)) / g
        return vx * t_flight

    def try_config(x_start, y_start, x_end, y_end, angle, T):
        try:
            q_start = inverse_kinematics_3r(np.array([x_start, y_start, angle]))
            xT = np.array([x_end, y_end, 0])
            release_angle = 45.0 * (np.pi / 180.0)
            speed = 2.0
            max_speed = 3.0
            speed_step = 0.1
            last_solution = None
            while speed <= max_speed:
                v_final = np.array([
                    speed * np.cos(release_angle),
                    speed * np.sin(release_angle),
                    0.0
                ])
                try:
                    solution = solve_trajectory_problem(v_final, T, q_start, xT)
                    q_release = solution["Q"][:, -1]
                    x_release, y_release, _ = forward_kinematics_3r(q_release)
                    dist = x_release + ballistic_distance(y_release, speed, release_angle)
                    last_solution = (
                        dist,
                        (x_start, y_start, angle, x_end, y_end, speed, T),
                        solution
                    )
                    speed += speed_step
                except Exception:
                    break
            return last_solution
        except Exception:
            return None

    configs = [
        (x_start, y_start, x_end, y_end, angle, T)
        for x_start in x_start_vals
        for y_start in y_start_vals
        for x_end in x_end_vals
        for y_end in y_end_vals
        for angle in angle_vals
        for T in T_vals
    ]

    with tqdm(total=total, desc="Searching configurations") as pbar, ThreadPoolExecutor() as executor:
        futures = {executor.submit(try_config, *cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            result = future.result()
            if result:
                dist = result[0]
                if update_best_distance(dist):
                    print(f"New max distance: {dist:.3f} at config: {result[1]}")
                heapq.heappush(top_solutions, (dist, result))
                if len(top_solutions) > max_heap_size:
                    heapq.heappop(top_solutions)
            pbar.update(1)

    if top_solutions:
        top_solutions.sort(reverse=True)
        print(f"Top {max_heap_size} solutions:")
        for i, (dist, result) in enumerate(top_solutions):
            print(f"{i+1}: Distance={dist:.3f}, Config={result[1]}")
    else:
        print("No feasible configuration found.")


if __name__ == '__main__':
    find_best_configuration()
    # q = np.array([-0.88449375 , 1.96303457 , 2.52638987])  # 10 repeated points as a trajectory
    # print(forward_kinematics_3r(q))
    # qT = np.deg2rad([-55, 30, 90])
    # xT = forward_kinematics_3r(qT) + np.array([-0.1, -0.2, 0.0])
    # angle = 45.0 * (np.pi / 180.0)  # radians
    # speed = 1.8
    # v_final = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])
    #
    # # 3. Solve the optimization problem
    # solution = solve_trajectory_problem(v_final, 2.0, q, xT)
    #
    # # 4. Plot the results if a solution was found
    # if solution:
    #     # post_process_and_plot(solution)
    #
    #     # 5. Animate the resulting trajectory
    #
    #     animate_3r_trajectory(solution['Q'].T[:], dt=0.01)
    #     print("first drawn q:", solution['Q'][:, 100])
    #     #Last q: [-0.85302408  0.90456257  0.95857157]
    #     full_q, full_qd, full_qdd = assemble_full_trajectory(solution['Q'], solution['Qd'], solution['Qdd'])
    #     np.savez('3r_trajectory.npz', q=full_q, qd=full_qd, qdd=full_qdd, time=solution['time'])
    #     final_pose = forward_kinematics_3r(solution['Q'][:, -1])
    #     print(f"Final Pose from FK: {final_pose}, Target Pose: {xT}")
    #     print("Error in final pose:", final_pose - xT)
    # else:
    #     print("No solution found skipping plotting and animation.")
