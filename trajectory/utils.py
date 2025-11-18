"""Utility functions for trajectory planning."""

import numpy as np


def linspace_arrays(start, stop, num):
    """
    Create linearly spaced arrays between start and stop arrays.
    
    Args:
        start: Starting array
        stop: Ending array
        num: Number of samples
        
    Returns:
        np.ndarray: Array of shape (num, len(start)) with interpolated values
    """
    start, stop = np.asarray(start), np.asarray(stop)
    result = np.empty((num, start.size))
    for i in range(num):
        alpha = float(i) / (num - 1) if num > 1 else 0.0
        result[i] = (1 - alpha) * start + alpha * stop
    return result


def assemble_full_trajectory(q, qd, qdd, fixed_values=None):
    """
    Assemble a full 7-DOF trajectory from a 3-DOF trajectory by inserting fixed joint values.
    
    Args:
        q: Position trajectory (3 x N)
        qd: Velocity trajectory (3 x N)
        qdd: Acceleration trajectory (3 x N)
        fixed_values: Dict with fixed values for joints j0, j2, j4, j6
        
    Returns:
        tuple: (q_full, qd_full, qdd_full) as (N x 7) arrays
    """
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
