import numpy as np

def rho_expo_model(alts):
    """
    Estimate mass densities at given geometric altitudes (km) above sea level
    using a simple exponential atmospheric model (Vallado 2013).

    Inputs:
        alts -> [float or array] geometric altitudes [km]

    Outputs:
        rhos -> [float array] densities at given altitudes [kg/m^3]
    """

    # Convert input to NumPy array of floats
    zs = np.atleast_1d(alts).astype(float)

    # Base altitudes [km] for exponential model
    zb = np.array([0., 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                   130, 140, 150, 180, 200, 250, 300, 350, 400, 450,
                   500, 600, 700, 800, 900, 1000])

    # Nominal densities [kg/m^3]
    rhob = np.array([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3,
                     3.206e-4, 8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7,
                     9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9,
                     5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11,
                     9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13,
                     1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15])

    # Scale heights [km]
    ZS = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549,
                   5.799, 5.382, 5.877, 7.263, 9.473, 12.636, 16.149,
                   22.523, 29.740, 37.105, 45.546, 53.628, 53.298,
                   58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00])

    # Warn if altitude is out of bounds
    if np.any(zs < -0.611):
        print("Warning: Altitude is outside valid range [-0.611, inf] km.")

    elif np.any(zs > 1000):
        rhos = 0.0
        return rhos

    # Extrapolation endpoints
    zb_expand = zb.copy()
    zb_expand[0], zb_expand[-1] = -np.inf, np.inf

    # Initialize output array (floating-point)
    rhos = np.zeros_like(zs, dtype=float)

    # Compute densities using exponential model
    for i, z in enumerate(zs):
        ind = np.where((z - zb_expand) >= 0)[0][-1]
        rhos[i] = rhob[ind] * np.exp(-(z - zb[ind]) / ZS[ind])

    return rhos
