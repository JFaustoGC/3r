import numpy as np
import matplotlib.pyplot as plt
from config import l1, l2, l3, base_offset_x, base_offset_z, global_gripper_width, global_gripper_length, gripper


def inverse_kinematics_3r(pose: np.ndarray) -> np.ndarray:
    x, z, t3 = pose
    # subtract base offsets
    x_rel = x - base_offset_x
    z_rel = z - base_offset_z

    # wrist position
    x_wrist = x_rel - (l3 + gripper) * np.cos(t3)
    z_wrist = z_rel - (l3 + gripper) * np.sin(t3)

    # compute D
    D = (x_wrist ** 2 + z_wrist ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    if abs(D) > 1:
        raise ValueError(f"Position is unreachable, D={D}")

    q2 = np.arctan2(-np.sqrt(1 - D ** 2), D)
    q1 = np.arctan2(z_wrist, x_wrist) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q3 = t3 - q1 - q2

    # match FK sign convention
    return np.array([-q1, -q2, -q3])
def jacobian_3r(q: np.ndarray) -> np.ndarray:
    """
    Jacobian d[x,z,t3] / dq for the planar 3R defined by your FK:
      t1 = -q0
      t2 = -q0 - q1
      t3 = -q0 - q1 - q2

    Returns a 3x3 numpy array J where:
      [vx; vz; vt3] = J @ qdot
    """
    q0, q1, q2 = q
    # joint-angle substitutions (same as in your FK)
    t1 = -q0
    t2 = -q0 - q1
    t3 = -q0 - q1 - q2

    L3 = l3 + gripper

    # dx/dq
    dx_dq0 = l1 * np.sin(t1) + l2 * np.sin(t2) + L3 * np.sin(t3)
    dx_dq1 =        l2 * np.sin(t2) + L3 * np.sin(t3)
    dx_dq2 =                     L3 * np.sin(t3)

    # dz/dq
    dz_dq0 = - (l1 * np.cos(t1) + l2 * np.cos(t2) + L3 * np.cos(t3))
    dz_dq1 = - (       l2 * np.cos(t2) + L3 * np.cos(t3))
    dz_dq2 = - (                      L3 * np.cos(t3))

    # dt3/dq = d(theta3)/dq = -1 for each q0,q1,q2
    J = np.array([
        [dx_dq0, dx_dq1, dx_dq2],
        [dz_dq0, dz_dq1, dz_dq2],
        [-1.0,   -1.0,   -1.0  ]
    ], dtype=float)

    return J

def cubic_bezier_points(P0, P1, P2, P3, t):
    """Return point and derivative d/dt at parameter t (0..1)."""
    # position
    B = (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3
    # derivative dB/dt
    dB = 3*(1-t)**2 * (P1 - P0) + 6*(1-t)*t * (P2 - P1) + 3*t**2 * (P3 - P2)
    return B, dB

# ---------- Bézier helpers ----------
def bezier_cubic(P0, P1, P2, P3, t):
    return ((1-t)**3)*P0 + 3*((1-t)**2)*t*P1 + 3*(1-t)*(t**2)*P2 + (t**3)*P3

def bezier_cubic_derivative(P0, P1, P2, P3, t):
    return 3*((1-t)**2)*(P1-P0) + 6*(1-t)*t*(P2-P1) + 3*(t**2)*(P3-P2)

def build_u_bezier(x0, xT, depth=1.5, apex_rel=0.5, final_tangent_mag=None):
    P0 = np.array(x0, dtype=float)
    P3 = np.array(xT, dtype=float)
    dx = P3 - P0
    if final_tangent_mag is None:
        final_tangent_mag = 0.25 * np.linalg.norm(dx)

    xa = P0[0] + apex_rel * (P3[0] - P0[0])
    ya = P0[1] - depth   # dip down (depth positive means down)

    # put P1 near apex (simple heuristic)
    P1 = np.array([xa * 0.9 + P0[0]*0.1, ya])

    # Enforce final tangent 45 deg (you can change sign if you want down/right)
    theta45 = np.pi/4
    v_dir = np.array([np.cos(theta45), np.sin(theta45)])
    desired_final_deriv = final_tangent_mag * v_dir
    P2 = P3 - desired_final_deriv / 3.0

    return P0, P1, P2, P3

# ---------- Parameters ----------
x0 = np.array([0.175, 0.025])
xT = np.array([0.825, 0.0])
depth = 1.5                    # as you requested
apex_rel = 0.4
final_tangent_mag = 0.25
N = 300                        # sampling resolution along bezier param t
desired_speed = 1.5            # desired EE speed at the end (m/s)

# build curve
P0, P1, P2, P3 = build_u_bezier(x0, xT, depth=depth, apex_rel=apex_rel, final_tangent_mag=0)

ts = np.linspace(0.0, 1.0, N)
dt_param = ts[1] - ts[0]

# allocate
X = np.zeros((N,2))
Xd = np.zeros((N,2))   # derivative w.r.t param t
Theta = np.zeros(N)    # orientation (tangent angle)
Thetad = np.zeros(N)   # derivative of orientation w.r.t t (param)
Q = np.zeros((N,3))
Qd_param = np.zeros((N,3))  # qdot w.r.t param t
Qdd_param = np.zeros((N,3)) # qdd w.r.t param t (second derivative w.r.t param)

# 1) compute x(t) and x'(t) and theta(t)
for i,t in enumerate(ts):
    x = bezier_cubic(P0,P1,P2,P3,t)
    xd = bezier_cubic_derivative(P0,P1,P2,P3,t)
    X[i,:] = x
    Xd[i,:] = xd
    Theta[i] = np.arctan2(xd[1], xd[0])

# 2) compute Theta derivative w.r.t param t (central finite differences)
Thetad = np.gradient(Theta, dt_param)   # dtheta/dt_param

# 3) for each sample compute IK and qdot solving J qdot = [xd; thetad]
lam = 1e-6  # damping for stability
for i in range(N):
    x = X[i]
    xd = Xd[i]
    th = Theta[i]
    thd = Thetad[i]
    pose = np.array([x[0], x[1], th])           # (x,z,t3) naming consistent with your FK/IK
    q = inverse_kinematics_3r(pose)             # must be in scope
    J = jacobian_3r(q)                          # 3x3

    # form desired twist in task space (x_dot, z_dot, theta_dot)
    v = np.array([xd[0], xd[1], thd])

    # Solve (J^T J + lam I) qdot = J^T v  (Tikhonov)
    A = J.T @ J + lam * np.eye(3)
    b = J.T @ v
    qdot = np.linalg.solve(A, b)

    Q[i,:] = q
    Qd_param[i,:] = qdot

# 4) finite-difference qdd w.r.t param t
# central differences for interior, forward/backward for endpoints
Qdd_param[1:-1] = (Qd_param[2:] - Qd_param[:-2]) / (2*dt_param)
Qdd_param[0]   = (Qd_param[1] - Qd_param[0]) / dt_param
Qdd_param[-1]  = (Qd_param[-1] - Qd_param[-2]) / dt_param

# 5) compute end-effector speed at end in current paramization (magnitude of Xd at t=1)
speed_end_param = np.linalg.norm(Xd[-1])   # this is derivative w.r.t param t
# We want final physical speed to be desired_speed (m/s).
# If we set real time scaling factor alpha so that physical time t_phys = t_param / alpha,
# then x_dot_phys = alpha * x_dot_param. So choose alpha = desired_speed / speed_end_param
if speed_end_param <= 0:
    raise RuntimeError("End-point path derivative is zero, cannot scale to desired speed.")
alpha = desired_speed / speed_end_param

# scale qdot (w.r.t physical time) and qdd (w.r.t physical time)
Qd_time = Qd_param * alpha                # qdot in 1/s (physical)
Qdd_time = Qdd_param * (alpha**2)         # qdd in 1/s^2 (physical)

# For completeness compute physical time vector (parameter t runs 0..1, so t_phys length = 1/alpha)
total_time = 1.0 / alpha
time = np.linspace(0.0, total_time, N)

# 6) compute actual end-effector speed along scaled trajectory (should end at desired_speed)
Xd_time = Xd * alpha
speed_time = np.linalg.norm(Xd_time, axis=1)

# ---------- Plots ----------
plt.figure(figsize=(6,4))
plt.plot(X[:,0], X[:,1], 'k-', linewidth=2)
plt.scatter([P0[0],P1[0],P2[0],P3[0]],[P0[1],P1[1],P2[1],P3[1]], color='red', s=30)
plt.title("Bezier U-curve (depth=%.3f)" % depth)
plt.xlabel("x")
plt.ylabel("z")
plt.grid(True)

plt.figure(figsize=(6,4))
plt.plot(time, speed_time, label='EE speed (scaled)')
plt.hlines(desired_speed, time[0], time[-1], colors='r', linestyles='--', label='desired end speed')
plt.xlabel("time (s)")
plt.ylabel("speed (m/s)")
plt.legend()
plt.grid(True)

plt.figure(figsize=(6,4))
plt.plot(time, Qd_time)
plt.title("Joint velocities (qdot) vs time")
plt.xlabel("time (s)")
plt.ylabel("qdot (rad/s)")
plt.grid(True)

plt.figure(figsize=(6,4))
plt.plot(time, Qdd_time)
plt.title("Joint accelerations (qddot) vs time")
plt.xlabel("time (s)")
plt.ylabel("qddot (rad/s^2)")
plt.grid(True)

plt.show()

# ---------- Return / print summary ----------
print("samples:", N)
print("alpha (time-scale factor):", alpha)
print("total_time (s):", total_time)
print("end EE speed (scaled):", speed_time[-1])
# Q: joint positions (N x 3)
# Qd_time: joint velocities (N x 3) in physical time
# Qdd_time: joint accelerations (N x 3) in physical time