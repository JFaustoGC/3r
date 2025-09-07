import numpy as np
from config import l1, l2, l3, base_offset_x, base_offset_z, gripper
from kinematics import forward_kinematics_3r, inverse_kinematics_3r, jacobian_3r, jacobian_dot_3r
import matplotlib.pyplot as plt

# ---- utilities: quintic and eval ----
def quintic_coeffs(s0, sT, sd0, sdT, sdd0, sddT, T):
    A = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,2,0,0,0],
        [1,T,T**2,T**3,T**4,T**5],
        [0,1,2*T,3*T**2,4*T**3,5*T**4],
        [0,0,2,6*T,12*T**2,20*T**3]
    ], dtype=float)
    b = np.array([s0, sd0, sdd0, sT, sdT, sddT], dtype=float)
    return np.linalg.solve(A, b)

def eval_poly(a, t):
    tpow = np.array([1, t, t**2, t**3, t**4, t**5])
    s = a.dot(tpow)
    sd = a.dot(np.array([0,1,2*t,3*t**2,4*t**3,5*t**4]))
    sdd = a.dot(np.array([0,0,2,6*t,12*t**2,20*t**3]))
    return s, sd, sdd

def shortest_angular_difference(a, b):
    """return delta in (-pi, pi] so we take shortest rotation from a -> b"""
    d = (b - a + np.pi) % (2*np.pi) - np.pi
    return d



# ---- damped pseudoinverse helper ----
def damped_pinv(J, lam=1e-6):
    # J is 3x3; produce 3x3 "inverse" robustly
    JJt = J.dot(J.T)
    inv = np.linalg.inv(JJt + (lam**2) * np.eye(3))
    return J.T.dot(inv)

# ---- main pipeline that includes theta interpolation ----
def task_to_joint_traj_with_theta(pose0, poseT, v_goal_xy, T=1.0, dt=0.005,lam=1e-6):
    """
    pose0, poseT: [x, z, theta]
    v_goal_xy: (vx, vz) -- desired translational task velocity at final pose
    T: total time
    dt: timestep
    lam: damping factor for pseudoinverse
    """
    x0, z0, th0 = pose0
    xT, zT, thT = poseT
    vx_goal, vz_goal = v_goal_xy

    # translational quintics (start vel = 0 as you required)
    ax = quintic_coeffs(x0, xT, 0.0, vx_goal, 0.0, 0.0, T)
    az = quintic_coeffs(z0, zT, 0.0, vz_goal, 0.0, 0.0, T)

 
    dth = shortest_angular_difference(th0, thT)
    def theta_traj_linear(t):
        th = th0 + dth * (t / T)
        thd = dth / T
        thdd = 0.0
        return th, thd, thdd
    theta_traj = theta_traj_linear


    times = np.arange(0.0, T + dt*0.5, dt)
    N = len(times)
    Q = np.zeros((N, 3))
    Qd = np.zeros((N, 3))
    Qdd = np.zeros((N, 3))

    # initial joint config from your IK
    q = inverse_kinematics_3r(pose0)
    # initial desired task xdot: translational 0, rotational theta_dot as produced by theta_traj(0)
    _, thd0, _ = theta_traj(0.0)
    xdot0 = np.array([0.0, 0.0, thd0])

    # compute initial qdot solving J qdot = xdot (damped)
    J0 = jacobian_3r(q)
    Jp = damped_pinv(J0, lam=lam)
    qdot = Jp.dot(xdot0)
    qdd = np.zeros(3)

    for i, t in enumerate(times):
        xs, xds, xdds = eval_poly(ax, t)
        zs, zds, zdds = eval_poly(az, t)
        ths, thds, thdds = theta_traj(t)

        x_vec = np.array([xs, zs, ths])
        xd_vec = np.array([xds, zds, thds])
        xdd_vec = np.array([xdds, zdds, thdds])

        J = jacobian_3r(q)
        Jd = jacobian_dot_3r(q, qdot)

        Jp = damped_pinv(J, lam=lam)  # 3x3
        # resolved-rate (direct solve); damped pseudoinverse is robust
        qdot = Jp.dot(xd_vec)

        # resolved-acceleration
        qdd = Jp.dot(xdd_vec - Jd.dot(qdot))

        # integrate q (simple Euler; replace with RK4 if you want better accuracy)
        q = q + qdot * dt

        Q[i, :] = q
        Qd[i, :] = qdot
        Qdd[i, :] = qdd

    return times, Q, Qd, Qdd


# ---- plotting helpers ----
def compute_task_and_joint(times, Q, Qd, Qdd, ax, az, theta_traj):
    """
    Compute actual and reference task space and joint space values.
    Returns: dict with all arrays needed for plotting.
    """
    N = len(times)
    X = np.zeros((N,3))
    Xd = np.zeros((N,3))
    Xdd = np.zeros((N,3))

    for i in range(N):
        q, qd, qdd = Q[i], Qd[i], Qdd[i]
        X[i] = forward_kinematics_3r(q)
        J = jacobian_3r(q)
        Jd = jacobian_dot_3r(q, qd)
        Xd[i] = J.dot(qd)
        Xdd[i] = J.dot(qdd) + Jd.dot(qd)

    # also compute the reference task trajectory for comparison
    Xref = np.zeros((N,3))
    Xdref = np.zeros((N,3))
    Xddref = np.zeros((N,3))
    for i, t in enumerate(times):
        xs, xds, xdds = eval_poly(ax, t)
        zs, zds, zdds = eval_poly(az, t)
        ths, thds, thdds = theta_traj(t)
        Xref[i] = [xs, zs, ths]
        Xdref[i] = [xds, zds, thds]
        Xddref[i] = [xdds, zdds, thdds]

    return dict(
        times=times, Q=Q, Qd=Qd, Qdd=Qdd,
        X=X, Xd=Xd, Xdd=Xdd,
        Xref=Xref, Xdref=Xdref, Xddref=Xddref
    )

def plot_task_and_joint(data):
    """
    Plot task space and joint space trajectories and their derivatives from precomputed data.
    """
    times = data['times']
    Q, Qd, Qdd = data['Q'], data['Qd'], data['Qdd']
    X, Xd, Xdd = data['X'], data['Xd'], data['Xdd']
    Xref, Xdref, Xddref = data['Xref'], data['Xdref'], data['Xddref']

    # --- plot task space ---
    labels = ["x", "z", "theta"]
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i in range(3):
        axs1[i].plot(times, Xref[:, i], 'k--', label=f"{labels[i]} ref")
        axs1[i].plot(times, X[:, i], label=f"{labels[i]} actual")
        axs1[i].set_ylabel(labels[i])
        axs1[i].legend()
    axs1[-1].set_xlabel("time [s]")
    fig1.suptitle("Task Space Position")

    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i in range(3):
        axs2[i].plot(times, Xdref[:, i], 'k--', label=f"{labels[i]}_dot ref")
        axs2[i].plot(times, Xd[:, i], label=f"{labels[i]}_dot actual")
        axs2[i].set_ylabel(f"{labels[i]}_dot")
        axs2[i].legend()
    axs2[-1].set_xlabel("time [s]")
    fig2.suptitle("Task Space Velocity")

    fig3, axs3 = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i in range(3):
        axs3[i].plot(times, Xddref[:, i], 'k--', label=f"{labels[i]}_ddot ref")
        axs3[i].plot(times, Xdd[:, i], label=f"{labels[i]}_ddot actual")
        axs3[i].set_ylabel(f"{labels[i]}_ddot")
        axs3[i].legend()
    axs3[-1].set_xlabel("time [s]")
    fig3.suptitle("Task Space Acceleration")

    # --- plot joint space ---
    q_labels = ["q0", "q1", "q2"]
    fig4, axs4 = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i in range(3):
        axs4[i].plot(times, Q[:, i], label=q_labels[i])
        axs4[i].set_ylabel(q_labels[i])
        axs4[i].legend()
    axs4[-1].set_xlabel("time [s]")
    fig4.suptitle("Joint Position")

    fig5, axs5 = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i in range(3):
        axs5[i].plot(times, Qd[:, i], label=q_labels[i]+"dot")
        axs5[i].set_ylabel(q_labels[i]+"dot")
        axs5[i].legend()
    axs5[-1].set_xlabel("time [s]")
    fig5.suptitle("Joint Velocity")

    fig6, axs6 = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i in range(3):
        axs6[i].plot(times, Qdd[:, i], label=q_labels[i]+"ddot")
        axs6[i].set_ylabel(q_labels[i]+"ddot")
        axs6[i].legend()
    axs6[-1].set_xlabel("time [s]")
    fig6.suptitle("Joint Acceleration")

    plt.show()

# ---- Example usage ----
if __name__ == "__main__":
    p0 = np.array([0.4, 0.3, 0.0])
    pT = np.array([0.5, 0.4, np.pi/4])
    vx_goal, vz_goal = 0.1, 0.0
    T = 1.2
    dt = 0.01

    times, Q, Qd, Qdd = task_to_joint_traj_with_theta(
        p0, pT, (vx_goal, vz_goal),
        T=T, dt=dt,
    )

    # recompute interpolation coeffs for task-space reference
    ax = quintic_coeffs(p0[0], pT[0], 0.0, vx_goal, 0.0, 0.0, T)
    az = quintic_coeffs(p0[1], pT[1], 0.0, vz_goal, 0.0, 0.0, T)
    dth = shortest_angular_difference(p0[2], pT[2])
    def theta_traj(t):
        th = p0[2] + dth * (t / T)
        thd = dth / T
        thdd = 0.0
        return th, thd, thdd

    # Compute all values for plotting
    data = compute_task_and_joint(times, Q, Qd, Qdd, ax, az, theta_traj)
    plot_task_and_joint(data)
