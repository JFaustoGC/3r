"""Kinematics functions for 3R manipulator using CasADi."""

import casadi as ca
from config import l1, l2, l3, base_offset_x, base_offset_z, gripper


def get_kinematics_functions():
    """
    Create CasADi symbolic functions for forward kinematics, Jacobian, and Jdot*qdot.
    
    Returns:
        dict: Dictionary containing CasADi functions:
            - 'fk': Forward kinematics function (q -> [x, z, theta])
            - 'jacobian': Jacobian matrix function (q -> J)
            - 'jdot_qdot': Product of Jacobian time derivative and joint velocities
    """
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
