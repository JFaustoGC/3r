import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from kinematics import forward_kinematics_3r, inverse_kinematics_3r     

def animate_3r_trajectory(q_traj: np.ndarray, lengths: np.ndarray, gripper_width=0.2, gripper_length=0.3) -> None:
    q_traj = np.asarray(q_traj)
    l1, l2, l3 = lengths
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(np.min(q_traj[:,0])-l1-l2, np.max(q_traj[:,0])+l1+l2)
    ax.set_ylim(np.min(q_traj[:,1])-l1-l2, np.max(q_traj[:,1])+l1+l2)

    # Plot elements
    line, = ax.plot([], [], 'o-', lw=4, color='blue')
    gripper_patch = Rectangle((0,0), gripper_length, gripper_width, angle=0, color='red', zorder=10)
    ax.add_patch(gripper_patch)

    def init():
        line.set_data([], [])
        gripper_patch.set_xy((0,0))
        return line, gripper_patch

    def animate(i):
        q = q_traj[i]
        try:
            th1, th2, th3 = inverse_kinematics_3r(q, lengths)
        except ValueError:
            return line, gripper_patch
        # Joint positions
        x0, y0 = 0, 0
        x1 = l1 * np.cos(th1)
        y1 = l1 * np.sin(th1)
        x2 = x1 + l2 * np.cos(th1 + th2)
        y2 = y1 + l2 * np.sin(th1 + th2)
        x3 = x2 + l3 * np.cos(th1 + th2 + th3)
        y3 = y2 + l3 * np.sin(th1 + th2 + th3)
        line.set_data([x0, x1, x2, x3], [y0, y1, y2, y3])
        # Gripper position and orientation
        gripper_center_x = x3
        gripper_center_y = y3
        theta = q[2]
        gripper_angle_deg = np.rad2deg(theta)
        dx = gripper_length/2 * np.cos(theta)
        dy = gripper_length/2 * np.sin(theta)
        gripper_patch.set_width(gripper_length)
        gripper_patch.set_height(gripper_width)
        gripper_patch.set_angle(gripper_angle_deg)
        gripper_patch.set_xy((gripper_center_x - dx - gripper_width/2 * np.sin(theta),
                              gripper_center_y - dy + gripper_width/2 * np.cos(theta)))
        return line, gripper_patch

    ani = animation.FuncAnimation(fig, animate, frames=len(q_traj), init_func=init,
                                  blit=True, interval=50, repeat=False)
    plt.show()
    
    
if __name__ == "__main__":
    lengths = np.array([1.0, 1.0, 1.0])
    n_points = 100
    x_traj = np.linspace(1.5, 2.0, n_points)
    y_traj = np.linspace(0.5, 1.0, n_points)
    theta_traj = np.linspace(np.deg2rad(60), np.deg2rad(90), n_points)
    q_traj = np.stack([x_traj, y_traj, theta_traj], axis=1)
    animate_3r_trajectory(q_traj, lengths)