import numpy as np

def forward_kinematics_3r(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    l1, l2, l3 = lengths
    t1 = q[0]
    t2 = q[0] + q[1]
    t3 = q[0] + q[1] + q[2]
    x = l1 * np.cos(t1) + l2 * np.cos(t2) + l3 * np.cos(t3)
    y = l1 * np.sin(t1) + l2 * np.sin(t2) + l3 * np.sin(t3)
    return np.array([x, y, t3])

def inverse_kinematics_3r(pose: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    x, y, theta = pose
    l1, l2, l3 = lengths
    x_wrist = x - l3 * np.cos(theta)
    y_wrist = y - l3 * np.sin(theta)
    D = (x_wrist**2 + y_wrist**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if abs(D) > 1:
        raise ValueError("Position is unreachable")
    q2 = np.arctan2(np.sqrt(1 - D**2), D)
    q1 = np.arctan2(y_wrist, x_wrist) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q3 = theta - q1 - q2
    return np.array([q1, q2, q3])





