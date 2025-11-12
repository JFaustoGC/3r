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

g = 9.81

def ballistic_distance(y0, v, theta, g=g):
    vy = v * ca.sin(theta)
    vx = v * ca.cos(theta)
    t_flight = (vy + ca.sqrt(vy ** 2 + 2 * g * y0)) / g
    return vx * t_flight



def linspace_arrays(start, stop, num):
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    result = np.empty((num, start.size))
    for i in range(num):
        alpha = float(i) / (num - 1) if num > 1 else 0.0
        result[i] = (1 - alpha) * start + alpha * stop
    return result


def solve_configurations():
    # Configs
    N = 200
    n_joints = 3


    opti = ca.Opti()
    T = opti.variable()
    dt = T / N

    Q = opti.variable(n_joints, N + 1)     # Joint positions (trajectory)
    Qd = opti.variable(n_joints, N + 1)    # Joint velocities
    Qdd = opti.variable(n_joints, N + 1)   # Joint accelerations


    # Solver options
    opti.solver("ipopt", {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 2000,  # <-- Use ipopt.max_iter, not max_iter
        "ipopt.tol": 1e-6,  # <-- Use ipopt.tol, not tol
        "print_time": False,
    })
    sol = opti.solve()

    return {
        "Q": sol.value(Q),
        "Qd": sol.value(Qd),
        "Qdd": sol.value(Qdd),
        "v_release": sol.value(v_release),
        "distance": sol.value(distance),
        "time": np.linspace(0, T, N + 1),
    }

if __name__ == '__main__':

    # q = np.array([-0.88449375 , 1.96303457 , 2.52638987])  # 10 repeated points as a trajectory
    # print(forward_kinematics_3r(q))
    # qT = np.deg2rad([-55, 30, 90])
    # xT = forward_kinematics_3r(qT) + np.array([-0.1, -0.2, 0.0])
    # angle = 45.0 * (np.pi / 180.0)  # radians
    # speed = 1.8
    # v_final = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])
    #
    # 3. Solve the optimization problem
    solution = solve_configurations()
    print("Optimized release velocity:", solution['v_release'])
    print("Optimized throw distance:", solution['distance'])

    # 4. Plot the results if a solution was found
    if solution:
        # post_process_and_plot(solution)

        # 5. Animate the resulting trajectory

        animate_3r_trajectory(solution['Q'].T[:], dt=0.01)
        print("first drawn q:", solution['Q'][:, 100])
        #Last q: [-0.85302408  0.90456257  0.95857157]

    else:
        print("No solution found skipping plotting and animation.")
