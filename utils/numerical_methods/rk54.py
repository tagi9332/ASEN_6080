import numpy as np

def rk54(f, y, dt):
    """
    Fehlberg RK5(4) integrator with fixed timestep.
    f(y) must return dy/dt.
    """

    k1 = f(y)
    k2 = f(y + dt*(1/4)*k1)
    k3 = f(y + dt*(3/32*k1 + 9/32*k2))
    k4 = f(y + dt*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3))
    k5 = f(y + dt*(439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4))
    k6 = f(y + dt*(-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5))

    # 5th-order solution
    y5 = y + dt * (
        16/135*k1 +
        6656/12825*k2 +
        28561/56430*k3 +
        -9/50*k4 +
        2/55*k5
    )

    return y5
