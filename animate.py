import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation
from kinematics import forward_kinematics_3r, inverse_kinematics_3r
from config import l1, l2, l3, base_offset_x, base_offset_z, global_gripper_width, global_gripper_length


def animate_3r_trajectory(q_traj: np.ndarray) -> None:
    """
    Animate a 3R manipulator given a trajectory of joint configurations (q arrays).
    Considers the total length including the gripper, and draws the end-effector reference frame.
    q_traj: (N, 3) array, each row is [q1, q2, q3]
    """
    q_traj = np.asarray(q_traj)
    total_length = l1 + l2 + l3 + global_gripper_length
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(-0.1, 1.3)
    ax.set_ylim(base_offset_z - total_length/2, base_offset_z + total_length/2)

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
        q = q_traj[i]
        # Forward kinematics to get all joint positions
        t1 = -q[0]
        t2 = -q[0] - q[1]
        t3 = -q[0] - q[1] - q[2]
        x0, y0 = base_offset_x, base_offset_z
        x1 = x0 + l1 * np.cos(t1)
        y1 = y0 + l1 * np.sin(t1)
        x2 = x1 + l2 * np.cos(t2)
        y2 = y1 + l2 * np.sin(t2)
        x3 = x2 + l3 * np.cos(t3)
        y3 = y2 + l3 * np.sin(t3)
        # End-effector (gripper) center
        ee_x = x3 + (global_gripper_length/2) * np.cos(t3)
        ee_y = y3 + (global_gripper_length/2) * np.sin(t3)
        # Draw links
        line.set_data([x0, x1, x2, x3, ee_x], [y0, y1, y2, y3, ee_y])
        # Draw joint circles
        joint_positions = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        for circ, pos in zip(joint_circles, joint_positions):
            circ.center = pos
        # Gripper as a square
        theta = t3
        dx = global_gripper_length/2 * np.cos(theta)
        dy = global_gripper_length/2 * np.sin(theta)
        gripper_patch.set_width(global_gripper_length)
        gripper_patch.set_height(global_gripper_width)
        gripper_patch.set_angle(np.rad2deg(theta))
        gripper_patch.set_xy((ee_x - dx - global_gripper_width/2 * np.sin(theta),
                              ee_y - dy + global_gripper_width/2 * np.cos(theta)))
        # Draw end-effector reference frame (x: red, z: green)
        # Remove old arrows if they exist
        if ee_x_arrow is not None:
            ee_x_arrow.remove()
        if ee_z_arrow is not None:
            ee_z_arrow.remove()
        # X axis (red)
        ee_x_arrow = ax.arrow(ee_x, ee_y, 0.07*np.cos(theta), 0.07*np.sin(theta), head_width=0.02, head_length=0.04, fc='r', ec='r', zorder=30)
        # Z axis (green, perpendicular)
        ee_z_arrow = ax.arrow(ee_x, ee_y, -0.07*np.sin(theta), 0.07*np.cos(theta), head_width=0.02, head_length=0.04, fc='g', ec='g', zorder=30)
        return (line, gripper_patch, *joint_circles, ee_x_arrow, ee_z_arrow)

    ani = animation.FuncAnimation(fig, animate, frames=len(q_traj), init_func=init,
                                  blit=True, interval=50, repeat=False)
    plt.show()
    
    
    
if __name__ == "__main__":
    q = np.array([-0.54, 0.79, -0.60])
    q_traj = np.tile(q, (1, 1))
    animate_3r_trajectory(q_traj)

