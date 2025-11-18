"""Plotting and animation functions for 3R manipulator trajectories."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation
from kinematics import forward_kinematics_3r
from config import l1, l2, l3, base_offset_x, base_offset_z, global_gripper_width, global_gripper_length, gripper
from .kinematics import get_kinematics_functions


def animate_3r_trajectory(q_traj: np.ndarray, dt: float) -> None:
    """
    Animate a 3R manipulator given a trajectory of joint configurations.
    
    Args:
        q_traj: Trajectory of joint configurations (N x 3 array)
        dt: Time step between configurations
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

    def animate_frame(i):
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
    ani = animation.FuncAnimation(fig, animate_frame, frames=len(q_traj), init_func=init,
                                  blit=True, interval=interval_ms, repeat=False)
    plt.show()


def plot_joint_trajectories(solution):
    """
    Plot joint positions, velocities, and accelerations over time.
    
    Args:
        solution: Dictionary containing 'time', 'Q', 'Qd', 'Qdd' arrays
    """
    time = solution['time']
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, solution['Q'].T)
    plt.title('Joint Positions (q)')
    plt.ylabel('rad')
    plt.legend([f'q{i}' for i in range(3)])
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, solution['Qd'].T)
    plt.title('Joint Velocities (qd)')
    plt.ylabel('rad/s')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, solution['Qdd'].T)
    plt.title('Joint Accelerations (qdd)')
    plt.ylabel('rad/s²')
    plt.xlabel('Time [s]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_ee_kinematics(solution):
    """
    Plot end-effector position, velocity magnitude, and acceleration magnitude.
    
    Args:
        solution: Dictionary containing 'time', 'Q', 'Qd', 'Qdd' arrays
    """
    kinematics = get_kinematics_functions()
    Q_opt, Qd_opt, Qdd_opt = solution['Q'], solution['Qd'], solution['Qdd']
    time = solution['time']
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

    # Plot
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
