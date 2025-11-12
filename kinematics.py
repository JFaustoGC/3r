
import numpy as np
from config import l1, l2, l3, base_offset_x, base_offset_z, gripper

def forward_kinematics_3r(q: np.ndarray) -> np.ndarray:
    t1 = -q[0]
    t2 = -q[0] - q[1]
    t3 = -q[0] - q[1] - q[2]
    x = base_offset_x + l1 * np.cos(t1) + l2 * np.cos(t2) + (l3 + gripper) * np.cos(t3)
    z = l1 * np.sin(t1) + l2 * np.sin(t2) + (l3 + gripper) * np.sin(t3) + base_offset_z
    return np.array([x, z, t3])

def inverse_kinematics_3r(pose: np.ndarray) -> np.ndarray:
    x, z, t3 = pose
    # subtract base offsets
    x_rel = x - base_offset_x
    z_rel = z - base_offset_z
    
    # wrist position
    x_wrist = x_rel - (l3 + gripper) * np.cos(t3)
    z_wrist = z_rel - (l3 + gripper) * np.sin(t3)

    # compute D
    D = (x_wrist**2 + z_wrist**2 - l1**2 - l2**2) / (2*l1*l2)
    if abs(D) > 1:
        raise ValueError(f"Position is unreachable, D={D}")

    q2 = np.arctan2(-np.sqrt(1 - D**2), D)
    q1 = np.arctan2(z_wrist, x_wrist) - np.arctan2(l2*np.sin(q2), l1+l2*np.cos(q2))
    q3 = t3 - q1 - q2

    # match FK sign convention
    return np.array([-q1, -q2, -q3])

def diff_inverse_kinematics_pos_only(x, y, q0):
    def fk_xy(q):
        pos = forward_kinematics_3r(q)
        return pos[0:2]

    def objective(q):
        pos = fk_xy(q)
        return np.sum((pos - np.array([x, y]))**2)

    from scipy.optimize import minimize
    res = minimize(objective, q0, method='BFGS')
    if not res.success:
        raise ValueError("Optimization failed in inverse kinematics")
    return res.x
