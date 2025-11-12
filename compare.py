import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation
from kinematics import forward_kinematics_3r, inverse_kinematics_3r
from config import l1, l2, l3, base_offset_x, base_offset_z, global_gripper_width, global_gripper_length, gripper
import casadi as ca
from animate import animate_3r_trajectory, get_kinematics_functions, post_process_and_plot

data1 = np.load('3r_trajectory2.npz')
q = data1['q']  # shape (N, 7)
qd = data1['qd']  # shape (N, 7)
qdd = data1['qdd']  # shape (N, 7)
# Extract the moving joints: j1, j3, j5 (columns 1, 3, 5)
q_3r = q[:, [1, 3, 5]]
dq_3r = qd[:, [1, 3, 5]]
ddq_3r = qdd[:, [1, 3, 5]]

# plot joint trajectories
time = np.linspace(0, 1, q_3r.shape[0])
plt.figure(figsize=(12, 8))
joint_names = ['Joint 1', 'Joint 2', 'Joint 3']
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(time, q_3r[:, i], label='Position (rad)')
    plt.plot(time, dq_3r[:, i], label='Velocity (rad/s)')
    plt.plot(time, ddq_3r[:, i], label='Acceleration (rad/s²)')
    plt.title(f'{joint_names[i]} Trajectory')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()