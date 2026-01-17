def rk54(f, y, dt):
    """
    4th-order Runge–Kutta integrator.

    f : function(t, y) -> dy/dt
    y : state vector (numpy array)
    dt: timestep

    Returns y_next
    """
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)

    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)