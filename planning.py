import numpy as np
from kinematics import inverse_kinematics_3r, jacobian_3r, jacobian_dot_3r
from animate import animate_3r_trajectory
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

def quintic_coeffs(p0, pT, v0, vT, a0, aT, T):
    """
    Compute coefficients of a quintic polynomial for trajectory generation.
    p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    Given boundary conditions:
    p(0) = p0, p(T) = pT
    p'(0) = v0, p'(T) = vT
    p''(0) = a0, p''(T) = aT
    Returns coefficients [a0, a1, a2, a3, a4, a5]
    """
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3]
    ])
    b = np.array([p0, v0, a0, pT, vT, aT])
    coeffs = np.linalg.solve(A, b)
    return coeffs

def eval_poly(coeffs, t):
    """
    Evaluate quintic polynomial and its first two derivatives at time t.
    coeffs: array of coefficients [a0, a1, a2, a3, a4, a5]
    Returns position, velocity, acceleration at time t.
    """
    a0, a1, a2, a3, a4, a5 = coeffs
    p = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    v = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    a = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    return p, v, a  

def shortest_angular_difference(th0, thT):
    """
    Compute the shortest angular difference between two angles th0 and thT.
    Returns the difference in radians, in the range [-pi, pi].
    """
    dth = thT - th0
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    return dth



# ---- Task Space Trajectory Generation (Unchanged) ----
def generate_task_space_trajectory(pose0, poseT, v_goal_xy, T=1.0, dt=0.005):
    """
    Generate task space trajectory arrays (positions, velocities, accelerations) for x, z, theta.
    Returns: times, X_task (N,3), Xd_task (N,3), Xdd_task (N,3), ax, az, theta_traj_fn
    """
    x0, z0, th0 = pose0
    xT, zT, thT = poseT
    vx_goal, vz_goal = v_goal_xy

    ax = quintic_coeffs(x0, xT, 0.0, vx_goal, 0.0, 0.0, T)
    az = quintic_coeffs(z0, zT, 0.0, vz_goal, 0.0, 0.0, T)
    dth = shortest_angular_difference(th0, thT)
    def theta_traj(t):
        th = th0 + dth * (t / T)
        thd = dth / T
        thdd = 0.0
        return th, thd, thdd

    times = np.arange(0.0, T + dt*0.5, dt)
    N = len(times)
    X_task = np.zeros((N,3))
    Xd_task = np.zeros((N,3))
    Xdd_task = np.zeros((N,3))
    for i, t in enumerate(times):
        xs, xds, xdds = eval_poly(ax, t)
        zs, zds, zdds = eval_poly(az, t)
        ths, thds, thdds = theta_traj(t)
        X_task[i] = [xs, zs, ths]
        Xd_task[i] = [xds, zds, thds]
        Xdd_task[i] = [xdds, zdds, thdds]
    return times, X_task, Xd_task, Xdd_task, ax, az, theta_traj



def compute_joint_trajectory_ik_spline(times, X_task, Xd_task, Xdd_task, dt, lam=None):
    print("Solving Inverse Kinematics for all waypoints...")
    N = X_task.shape[0]
    joint_waypoints = np.zeros((N, 3))
    q_guess = inverse_kinematics_3r(X_task[0]) # Initial guess from the start pose
    joint_waypoints[0] = q_guess

    for i in range(1, N):
        # Use the previous solution as the initial guess for the next
        q_sol = inverse_kinematics_3r(X_task[i])
        joint_waypoints[i] = q_sol
        q_guess = q_sol # Update the guess
    print("IK solving complete.")

    # 2. Fit a smooth B-Spline to the joint waypoints
    tck_q0 = splrep(times, joint_waypoints[:, 0])
    tck_q1 = splrep(times, joint_waypoints[:, 1])
    tck_q2 = splrep(times, joint_waypoints[:, 2])

    # 3. Differentiate the spline to find Q, Qd, and Qdd
    Q = np.vstack([splev(times, tck) for tck in [tck_q0, tck_q1, tck_q2]]).T
    Qd = np.vstack([splev(times, tck, der=1) for tck in [tck_q0, tck_q1, tck_q2]]).T
    Qdd = np.vstack([splev(times, tck, der=2) for tck in [tck_q0, tck_q1, tck_q2]]).T

    return Q, Qd, Qdd



# ---- Example usage ----
if __name__ == "__main__":
    p0 = np.array([0.2, -0.1, -np.deg2rad(135)])
    pT = np.array([0.8, 0.6, -np.deg2rad(315)])
    vx_goal, vz_goal = 0.1, 0.0
    T = 4.0
    dt = 0.01

    # Generate task space trajectory
    times, X_task, Xd_task, Xdd_task, ax, az, theta_traj = generate_task_space_trajectory(
        p0, pT, (vx_goal, vz_goal), T=T, dt=dt
    )
    
   
    # Compute joint space trajectory
    Q, Qd, Qdd = compute_joint_trajectory_ik_spline(times, X_task, Xd_task, Xdd_task, dt)
    
    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # First subplot: X_task components
    axs[0].plot(times, X_task[:, 0], label="X_x")
    axs[0].plot(times, X_task[:, 1], label="X_y")
    axs[0].plot(times, X_task[:, 2], label="X_z")
    axs[0].set_ylabel("Position")
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: Xd_task components
    axs[1].plot(times, Xd_task[:, 0], label="Xd_x")
    axs[1].plot(times, Xd_task[:, 1], label="Xd_y")
    axs[1].plot(times, Xd_task[:, 2], label="Xd_z")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()
    axs[1].grid(True)

    # Third subplot: Xdd_task components
    axs[2].plot(times, Xdd_task[:, 0], label="Xdd_x")
    axs[2].plot(times, Xdd_task[:, 1], label="Xdd_y")
    axs[2].plot(times, Xdd_task[:, 2], label="Xdd_z")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Acceleration")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    
    animate_3r_trajectory(Q, dt)

