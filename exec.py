import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation
from kinematics import forward_kinematics_3r, inverse_kinematics_3r, diff_inverse_kinematics_pos_only
from config import l1, l2, l3, base_offset_x, base_offset_z, global_gripper_width, global_gripper_length, gripper
import casadi as ca
from scipy.interpolate import interp1d


# ------------------- Animation Function -------------------
def animate_3r_trajectory(q_traj: np.ndarray, dt: float) -> None:
    """
    Animate a 3R manipulator given a trajectory of joint configurations (q arrays).
    """
    q_traj = np.asarray(q_traj)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True)

    # Axis limits
    total_length = l1 + l2 + l3 + gripper
    ax.set_xlim(-0.1, 1.3)
    ax.set_ylim(base_offset_z - total_length / 2, base_offset_z + total_length / 2)

    # Precompute FK path
    ee_path = np.zeros((len(q_traj), 2))
    joint_positions_all = []
    for i, q in enumerate(q_traj):
        x, z, theta = forward_kinematics_3r(q)
        ee_path[i] = [x, z]

        # Joint positions
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

        ee_x, ee_y = joints[-1]
        theta = -q_traj[i, 0] - q_traj[i, 1] - q_traj[i, 2]

        # Gripper rectangle
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


# ------------------- CasADi Kinematics -------------------
def get_kinematics_functions():
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


# ------------------- Config -------------------
config = {
    "joint_limits": {
        "pos": {"min": np.array([-3.8095, -3.0439, -2.9761]), "max": np.array([2.2736, 3.0439, 2.9761])},
        "slow": {"vel": np.array([0.2825, 0.415, 0.74]), "accel": np.array([1.5, 3.0, 3.0])},
        "medium": {"vel": np.array([0.678, 0.996, 1.776]), "accel": np.array([2.5, 5.0, 5.0])},
        "fast": {"vel": np.array([1.13, 1.66, 2.96]), "accel": np.array([5.0, 8.0, 8.0])},
        "express": {"vel": np.array([1.13, 1.66, 2.96]), "accel": np.array([8.0, 10.0, 12.0])},
    },
    "weights": {"accel": 1.0, "vel": 1.0, "pos": 1.0, "vel_dir": 1.0},
    "dt": 0.01,
}


# ------------------- Helper Functions -------------------
def linspace_arrays(start, stop, num):
    start, stop = np.asarray(start), np.asarray(stop)
    result = np.empty((num, start.size))
    for i in range(num):
        alpha = float(i) / (num - 1) if num > 1 else 0.0
        result[i] = (1 - alpha) * start + alpha * stop
    return result


def assemble_full_trajectory(q, qd, qdd, fixed_values=None):
    if fixed_values is None:
        fixed_values = {'j0': 0.0, 'j2': 0.0, 'j4': 0.0, 'j6': 1.7659902159840577}
    N = q.shape[1]
    q_full = np.zeros((N, 7))
    qd_full = np.zeros((N, 7))
    qdd_full = np.zeros((N, 7))

    q_full[:, 1] = np.mod(q[0] + np.pi, 2 * np.pi) - np.pi
    q_full[:, 3] = np.mod(q[1] + np.pi, 2 * np.pi) - np.pi
    q_full[:, 5] = np.mod(q[2] + np.pi, 2 * np.pi) - np.pi
    qd_full[:, 1] = qd[0]
    qd_full[:, 3] = qd[1]
    qd_full[:, 5] = qd[2]
    qdd_full[:, 1] = qdd[0]
    qdd_full[:, 3] = qdd[1]
    qdd_full[:, 5] = qdd[2]

    for idx, joint in zip([0, 2, 4, 6], ['j0', 'j2', 'j4', 'j6']):
        val = fixed_values[joint]
        q_full[:, idx] = val
        qd_full[:, idx] = 0.0
        qdd_full[:, idx] = 0.0

    return q_full, qd_full, qdd_full


# ------------------- Trajectory Solver -------------------
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
        spatial_velocity = ca.mtimes(kinematics["jacobian"](Q[:, k]), Qd[:, k])
        cost -= ca.dot(spatial_velocity[0:2], spatial_velocity[0:2])
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
    opti.subject_to(ca.mtimes(kinematics["jacobian"](Q[:, -1]), Qd[:, -1])[0:2] == v_final[0:2])
    opti.subject_to(Q[:, 0] == q0)

    joint_limits = config["joint_limits"]
    q_vel_limits = joint_limits["express"]["vel"]
    q_accel_limits = joint_limits["express"]["accel"]
    pos_max = joint_limits["pos"]["max"]
    pos_min = joint_limits["pos"]["min"]

    for i in range(n_joints):
        opti.subject_to(opti.bounded(pos_min[i], Q[i, :], pos_max[i]))
        opti.subject_to(opti.bounded(-q_vel_limits[i], Qd[i, :], q_vel_limits[i]))
        opti.subject_to(opti.bounded(-q_accel_limits[i], Qdd[i, :], q_accel_limits[i]))

    q_guess_traj = linspace_arrays(q0, inverse_kinematics_3r(xT), N + 1).T
    opti.set_initial(Q, q_guess_traj)

    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})
    sol = opti.solve()
    return {"Q": sol.value(Q), "Qd": sol.value(Qd), "Qdd": sol.value(Qdd), "time": np.linspace(0, T, N + 1)}


def resample_trajectory_fixed_dt(solution, speed_scale, dt):
    """
    Resample a trajectory to change speed while keeping dt constant.
    speed_scale < 1.0 -> slower (more samples)
    speed_scale = 1.0 -> same
    """
    import numpy as np
    from scipy.interpolate import interp1d

    t = solution["time"]
    Q = solution["Q"].T      # (N, dof)
    Qd = solution["Qd"].T
    Qdd = solution["Qdd"].T

    T_old = t[-1]
    T_new = T_old / speed_scale

    # New time vector, preventing floating point overshoot
    t_new = np.arange(0, T_new + dt/2, dt)
    t_new = np.minimum(t_new, t[-1])  # clamp final value to stay inside interpolation range

    # Build interpolators
    interp_Q = interp1d(t, Q, axis=0, kind='linear')
    interp_Qd = interp1d(t, Qd * speed_scale, axis=0, kind='linear')
    interp_Qdd = interp1d(t, Qdd * (speed_scale**2), axis=0, kind='linear')

    Q_new = interp_Q(t_new).T
    Qd_new = interp_Qd(t_new).T
    Qdd_new = interp_Qdd(t_new).T

    return {"time": t_new, "Q": Q_new, "Qd": Qd_new, "Qdd": Qdd_new}

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def scale_casadi_solution_taskspace_peak(solution_max, v_desired, dt=0.01, verbose=True):
    """
    Scale a CasADi-computed (max-speed) solution so the peak end-effector speed
    becomes v_desired while preserving profile shape. Time is stretched accordingly.
    Inputs:
      - solution_max: dict with keys "time", "Q", "Qd", "Qdd"
                      shapes: time (N,), Q (3,N), Qd (3,N), Qdd (3,N)
      - v_desired: desired peak EE speed (m/s) <= current peak
      - dt: desired control timestep for output (seconds)
      - verbose: print and plot diagnostics
    Returns:
      - solution_scaled: dict with "time", "Q", "Qd", "Qdd" resampled at dt
      - info: dict with scaling factor 's', original_peak, achieved_peak
    """
    # kinematics helper to compute EE velocity from q, qdot
    kinematics = get_kinematics_functions()

    t_old = np.asarray(solution_max["time"])
    Q_old = np.asarray(solution_max["Q"])    # shape (3, N)
    Qd_old = np.asarray(solution_max["Qd"])
    Qdd_old = np.asarray(solution_max["Qdd"])

    # compute EE speed profile (magnitudes) at old sample times
    N_old = t_old.size
    ee_speed_old = np.zeros(N_old)
    for k in range(N_old):
        J = kinematics["jacobian"](Q_old[:, k]).full()[0:2, :]   # 2x3
        v2 = J @ Qd_old[:, k]
        ee_speed_old[k] = np.linalg.norm(v2)

    v_peak_old = np.max(ee_speed_old)
    if v_peak_old <= 1e-8:
        raise RuntimeError("Original trajectory has near-zero EE speed; cannot scale.")

    # clamp desired if above original peak
    if v_desired > v_peak_old:
        print(f"[scale] v_desired ({v_desired}) > original peak ({v_peak_old:.4f}) -> clamping to original.")
        v_desired = float(v_peak_old)

    # scaling factor: velocities scale by s, accelerations by s^2, time by 1/s
    s = v_desired / v_peak_old
    if verbose:
        print(f"[scale] original peak = {v_peak_old:.4f} m/s, desired = {v_desired:.4f}, scale s = {s:.6f}")

    # new total time
    T_old = t_old[-1]
    T_new = T_old / s

    # build uniform time vector at controller dt (include final point)
    t_new = np.arange(0.0, T_new + dt/2, dt)

    # to sample old trajectories at the times corresponding to new timeline:
    # old_time_at_tnew = s * t_new  because t_old = s * t_new  (since t_new = t_old / s)
    t_query = s * t_new
    # ensure query lies within original time domain (tiny epsilon ok)
    t_query = np.clip(t_query, t_old[0], t_old[-1])

    # build interpolators based on old solution (interpolate along time for each column)
    interp_Q   = interp1d(t_old, Q_old.T, axis=0, kind='linear', fill_value='extrapolate')
    interp_Qd  = interp1d(t_old, Qd_old.T, axis=0, kind='linear', fill_value='extrapolate')
    interp_Qdd = interp1d(t_old, Qdd_old.T, axis=0, kind='linear', fill_value='extrapolate')

    Q_at_query   = interp_Q(t_query)   # shape (len(t_new), 3)
    Qd_at_query  = interp_Qd(t_query)
    Qdd_at_query = interp_Qdd(t_query)

    # Now apply scaling to velocities and accelerations
    Q_new   = Q_at_query.T              # shape (3, M)
    Qd_new  = (s * Qd_at_query).T
    Qdd_new = (s**2 * Qdd_at_query).T

    # Sanity: compute achieved EE speed peak on resampled timeline
    ee_speed_new = np.zeros(t_new.size)
    for i in range(t_new.size):
        J = kinematics["jacobian"](Q_new[:, i]).full()[0:2, :]
        ee_v = J @ Qd_new[:, i]
        ee_speed_new[i] = np.linalg.norm(ee_v)
    achieved_peak = np.max(ee_speed_new)

    # Pack result
    solution_scaled = {"time": t_new, "Q": Q_new, "Qd": Qd_new, "Qdd": Qdd_new}
    info = {"s": s, "original_peak": v_peak_old, "achieved_peak": achieved_peak, "T_old": T_old, "T_new": T_new}

    if verbose:
        print(f"[scale] T_old={T_old:.4f} s -> T_new={T_new:.4f} s, achieved_peak={achieved_peak:.6f} m/s")

        # Quick plots: EE speed old vs new, and joint velocities before/after (sampled)
        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(t_old, ee_speed_old, label='EE speed (original)')
        plt.plot(t_new, ee_speed_new, '--', label='EE speed (scaled)')
        plt.axhline(v_desired, color='r', linestyle='--', label='v_desired')
        plt.xlabel('time [s]'); plt.ylabel('EE speed [m/s]'); plt.legend(); plt.grid(True)
        plt.title('EE speed: original vs scaled')

        plt.subplot(2,1,2)
        # plot first joint velocity comparison (you can plot all)
        # for clarity, plot qd original interpolated to t_new before scaling and after scaling
        qd_orig_interp_at_tnew = interp_Qd(t_query)
        plt.plot(t_new, qd_orig_interp_at_tnew[:,0], label='qdot0 original (mapped to new times)')
        plt.plot(t_new, (s * qd_orig_interp_at_tnew)[:,0], '--', label='qdot0 scaled')
        plt.xlabel('time [s]'); plt.ylabel('qdot [rad/s]'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.show()

    return solution_scaled, info


def smooth_stop_segment(q_last, qd_last, qdd_last, dt=0.01, T=1.5, a_max=5.0):

    N = int(T / dt)
    n_joints = len(q_last)

    opti = ca.Opti()

    # Variables
    q = opti.variable(n_joints, N+1)
    qd = opti.variable(n_joints, N+1)
    qdd = opti.variable(n_joints, N+1)
    jerk = opti.variable(n_joints, N)

    # Initial conditions
    opti.subject_to(q[:,0] == q_last)
    opti.subject_to(qd[:,0] == qd_last)
    opti.subject_to(qdd[:,0] == qdd_last)

    # Dynamics constraints
    for k in range(N):
        opti.subject_to(q[:,k+1] == q[:,k] + dt * qd[:,k])
        opti.subject_to(qd[:,k+1] == qd[:,k] + dt * qdd[:,k])
        opti.subject_to(qdd[:,k+1] == qdd[:,k] + dt * jerk[:,k])
        opti.subject_to(ca.fabs(qdd[:,k]) <= a_max)

    # Final rest conditions
    opti.subject_to(qd[:,N] == 0)
    opti.subject_to(qdd[:,N] == 0)

    # Smoothness objective (minimize jerk²)
    opti.minimize(ca.sumsqr(jerk))

    opti.solver('ipopt', {"print_time": 0}, {"print_level": 0})
    sol = opti.solve()

    q_val = np.array(sol.value(q))
    qd_val = np.array(sol.value(qd))
    qdd_val = np.array(sol.value(qdd))
    return q_val, qd_val, qdd_val


def append_stop_trajectory(q_full, qd_full, qdd_full, dt=0.01, T_stop=1.5, a_max=5.0):
    """Appends a smooth deceleration phase to existing trajectory"""
    q_last = q_full[:,-1]
    qd_last = qd_full[:,-1]
    qdd_last = qdd_full[:,-1]

    q_stop, qd_stop, qdd_stop = smooth_stop_segment(q_last, qd_last, qdd_last, dt, T_stop, a_max)

    # Avoid duplicate at junction
    q_concat = np.concatenate((q_full, q_stop[:,1:]), axis=1)
    qd_concat = np.concatenate((qd_full, qd_stop[:,1:]), axis=1)
    qdd_concat = np.concatenate((qdd_full, qdd_stop[:,1:]), axis=1)

    return q_concat, qd_concat, qdd_concat



# ------------------- MAIN -------------------
if __name__ == '__main__':
    q0 = inverse_kinematics_3r(np.array([0.175, 0.025, -2.094395102393195]))
    T = 0.7
    xT = np.array([0.825, 0.0, 0.0])
    angle = 45.0 * (np.pi / 180.0)
    speed = 2.7
    v_final = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])

    # solution = solve_trajectory_problem(v_final, T, q0, xT)

    # Original (maximum speed)
    solution = solve_trajectory_problem(v_final, T, q0, xT)
    scaled_sol, meta = scale_casadi_solution_taskspace_peak(solution, v_desired=1.9)
    solution = scaled_sol  # use scaled trajectory
    dt = 0.01

    motion_time = solution['time'][-1]
    stop_time = 0.5
    total_time = motion_time + stop_time
    total_time_arr = np.arange(0.0, total_time + dt/2, dt)

    q_full, qd_full, qdd_full = append_stop_trajectory(solution['Q'], solution['Qd'], solution['Qdd'], dt=0.01, T_stop=stop_time, a_max=5.0)

    solution['Q'] = q_full
    solution['Qd'] = qd_full
    solution['Qdd'] = qdd_full
    solution['time'] = total_time_arr





    # ------------------- Joint plots -------------------
    time = solution['time']
    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(time, solution['Q'].T)
    plt.title('Joint Positions (q)')
    plt.ylabel('rad')
    plt.legend([f'q{i}' for i in range(3)])
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(time, solution['Qd'].T)
    plt.title('Joint Velocities (qd)')
    plt.ylabel('rad/s')
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(time, solution['Qdd'].T)
    plt.title('Joint Accelerations (qdd)')
    plt.ylabel('rad/s²')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------- Compute EE Position, Velocity, Acceleration Magnitudes -------------------
    kinematics = get_kinematics_functions()
    Q_opt, Qd_opt, Qdd_opt = solution['Q'], solution['Qd'], solution['Qdd']
    N = Q_opt.shape[1]

    ee_pos = np.zeros((2, N))
    ee_vel = np.zeros((2, N))
    ee_acc = np.zeros((2, N))

    for k in range(N):
        q_k = Q_opt[:, k]
        qd_k = Qd_opt[:, k]
        qdd_k = Qdd_opt[:, k]

        fk = kinematics['fk'](q_k).full().flatten()
        ee_pos[:, k] = fk[0:2]

        jac = kinematics['jacobian'](q_k).full()
        ee_vel[:, k] = jac[0:2, :] @ qd_k

        jdot_qdot = kinematics['jdot_qdot'](q_k, qd_k).full().flatten()
        ee_acc[:, k] = jdot_qdot[0:2] + jac[0:2, :] @ qdd_k

    # Magnitudes
    ee_vel_mag = np.linalg.norm(ee_vel, axis=0)
    ee_acc_mag = np.linalg.norm(ee_acc, axis=0)

    # ------------------- Plot EE Kinematics -------------------
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, ee_pos[0, :], label='x')
    plt.plot(time, ee_pos[1, :], label='z')
    plt.title('End-Effector Position')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, ee_vel_mag, label='|v|', color='r')
    plt.title('End-Effector Velocity Magnitude')
    plt.ylabel('Velocity [m/s]')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, ee_acc_mag, label='|a|', color='g')
    plt.title('End-Effector Acceleration Magnitude')
    plt.ylabel('Acceleration [m/s²]')
    plt.xlabel('Time [s]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(Q_opt.T.shape)
    animate_3r_trajectory(q_traj=Q_opt.T, dt=dt)