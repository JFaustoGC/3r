import casadi as cas

def min_time_double_integrator(x0, xT, umax):
    N = 1000
    T = cas.SX.sym('T')  # Decision variable: total time
    dt = T / N

    # Decision variables
    u = cas.SX.sym('u', N)           # Control trajectory
    X = cas.SX.sym('X', 2, N + 1)    # State trajectory [pos, vel]

    w = [T]                          # optimization variables list
    w += [cas.vec(u)]
    w += [cas.vec(X)]

    # Initial guess
    w0 = [1.0]
    w0 += [0.0] * N
    w0 += [0.0] * 2 * (N + 1)

    # Constraints and bounds
    g = []
    lbg = []
    ubg = []

    # Initial state
    g += [X[:, 0] - x0]
    lbg += [0, 0]
    ubg += [0, 0]

    # Dynamics constraints (Euler integration)
    for k in range(N):
        x_next = X[:, k] + dt * cas.vertcat(X[1, k], u[k])
        g += [X[:, k + 1] - x_next]
        lbg += [0, 0]
        ubg += [0, 0]

    # Final state
    g += [X[:, -1] - xT]
    lbg += [0, 0]
    ubg += [0, 0]

    # Control bounds
    lbw = [0.01]               # T > 0
    ubw = [100.0]
    lbw += [-umax] * N
    ubw += [umax] * N
    lbw += [-cas.inf] * 2 * (N + 1)
    ubw += [cas.inf] * 2 * (N + 1)

    # Objective: Minimize T (final time)
    J = T

    # NLP setup
    w = cas.vertcat(*w)
    g = cas.vertcat(*g)

    prob = {'f': J, 'x': w, 'g': g}
    solver = cas.nlpsol('solver', 'ipopt', prob,
                        {'ipopt.print_level': 0, 'print_time': False})

    sol = solver(x0=w0, lbg=lbg, ubg=ubg, lbx=lbw, ubx=ubw)
    w_opt = sol['x'].full().flatten()

    # Extract solution
    T_opt = w_opt[0]
    u_opt = w_opt[1:1 + N]
    X_opt = w_opt[1 + N:].reshape(2, N + 1)

    return T_opt, X_opt, u_opt

if __name__ == "__main__":
    res = min_time_double_integrator([0, 0], [10, 0], umax=1.0)

    # plot results
    import matplotlib.pyplot as plt
    T_opt, X_opt, u_opt = res
    N = len(u_opt)
    t_grid = [i * T_opt / N for i in range(N + 1)]
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_grid, X_opt[0, :], label='Position')
    plt.ylabel('Position')
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(t_grid, X_opt[1, :], label='Velocity')
    plt.ylabel('Velocity')
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.step(t_grid[:-1], u_opt, label='Control', where='post')
    plt.ylabel('Control')
    plt.xlabel('Time')
    plt.grid()
    plt.tight_layout()
    plt.show()