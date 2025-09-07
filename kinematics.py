
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
    x, z, theta = pose
    # Calculate wrist position
    x_wrist = x - (l3 + gripper) * np.cos(theta)
    z_wrist = z - (l3 + gripper) * np.sin(theta)
    D = (x_wrist**2 + z_wrist**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if abs(D) > 1:
        raise ValueError("Position is unreachable")
    q2 = np.arctan2(np.sqrt(1 - D**2), D)
    q1 = np.arctan2(z_wrist, x_wrist) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q3 = theta - q1 - q2
    return np.array([q1, q2, q3])


# ---- analytic Jacobian & Jdot (consistent with your FK signs) ----
def jacobian_3r(q):
    # q = [q0,q1,q2]
    t1 = -q[0]
    t2 = -q[0] - q[1]
    t3 = -q[0] - q[1] - q[2]
    L3 = l3 + gripper
    J = np.zeros((3,3))
    # ∂x/∂q_i
    J[0,0] =  l1 * np.sin(t1) + l2 * np.sin(t2) + L3 * np.sin(t3)
    J[0,1] =  l2 * np.sin(t2) + L3 * np.sin(t3)
    J[0,2] =  L3 * np.sin(t3)
    # ∂z/∂q_i
    J[1,0] = -l1 * np.cos(t1) - l2 * np.cos(t2) - L3 * np.cos(t3)
    J[1,1] = -l2 * np.cos(t2) - L3 * np.cos(t3)
    J[1,2] = -L3 * np.cos(t3)
    # orientation row: theta = t3 = -q0 - q1 - q2
    J[2,:] = np.array([-1.0, -1.0, -1.0])
    return J

def jacobian_dot_3r(q, qdot):
    # compute time derivatives using chain rule
    t1 = -q[0]
    t2 = -q[0] - q[1]
    t3 = -q[0] - q[1] - q[2]
    L3 = l3 + gripper
    qd0, qd1, qd2 = qdot
    t1d = -qd0
    t2d = -(qd0 + qd1)
    t3d = -(qd0 + qd1 + qd2)

    Jd = np.zeros((3,3))
    # derivatives for J[0,*] (sine terms)
    Jd[0,0] =  l1 * np.cos(t1) * t1d + l2 * np.cos(t2) * t2d + L3 * np.cos(t3) * t3d
    Jd[0,1] =  l2 * np.cos(t2) * t2d + L3 * np.cos(t3) * t3d
    Jd[0,2] =  L3 * np.cos(t3) * t3d
    # derivatives for J[1,*] (cosine terms)
    Jd[1,0] = -l1 * np.sin(t1) * t1d - l2 * np.sin(t2) * t2d - L3 * np.sin(t3) * t3d
    Jd[1,1] = -l2 * np.sin(t2) * t2d - L3 * np.sin(t3) * t3d
    Jd[1,2] = -L3 * np.sin(t3) * t3d
    # orientation row derivative = 0 (since ∂(-q_sum)/∂t = -qdot_sum but derivative of that row w.r.t time is zero in Jdot)
    Jd[2,:] = 0.0
    return Jd