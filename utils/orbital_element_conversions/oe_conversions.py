import numpy as np
import warnings
from utils.frame_conversions.rel_to_inertial_functions import rel_vector_to_inertial
from resources.constants import MU_EARTH

def inertial_to_oes(R, V, mu=MU_EARTH, units='rad'):
    """
    Convert inertial state vectors (R, V) to classical orbital elements.
    Returns: a, e, i, RAAN, ARGP, TA
    """

    warnings.filterwarnings("error", category=RuntimeWarning)

    R = np.array(R, dtype=float)
    V = np.array(V, dtype=float)
    r_norm = np.linalg.norm(R)
    v_norm = np.linalg.norm(V)

    # Angular momentum vector
    H = np.cross(R, V)
    h = np.linalg.norm(H)

    # Line of nodes
    N = np.cross([0, 0, 1], H)
    n_norm = np.linalg.norm(N)

    # Eccentricity vector
    e_vec = (1/mu) * ((v_norm**2 - mu / r_norm) * R - np.dot(R, V) * V)
    ecc = np.linalg.norm(e_vec)

    # ----------------------
    #  CIRCULAR ORBIT CHECK
    # ----------------------
    tol = 1e-8
    if ecc < tol:
        # Inclination
        cos_i = H[2] / h
        cos_i = np.clip(cos_i, -1, 1)
        i = np.arccos(cos_i)

        # RAAN
        if n_norm < tol:  # equatorial circular
            raan = 0.0
        else:
            Nhat = N / n_norm
            cos_raan = np.clip(Nhat[0], -1, 1)
            raan = np.arccos(cos_raan)
            if Nhat[1] < 0:
                raan = 2*np.pi - raan

        # Argument of periapsis undefined → 0 by convention
        argp = 0.0

        # True anomaly depends on equatorial/inclined
        if n_norm > tol:
            cos_ta = np.dot(N, R) / (n_norm * r_norm)
            cos_ta = np.clip(cos_ta, -1, 1)
            ta = np.arccos(cos_ta)
            if R[2] < 0:
                ta = 2*np.pi - ta
        else:
            # Equatorial circular: atan2 in inertial plane
            ta = np.arctan2(R[1], R[0]) % (2*np.pi)

        # Semi-major axis
        a = 1.0 / ((2.0 / r_norm) - (v_norm**2 / mu))

        # Units
        if units == 'deg':
            return a, ecc, np.degrees(i), np.degrees(raan), np.degrees(argp), np.degrees(ta)
        else:
            return a, ecc, i, raan, argp, ta

    # ------------------------------
    # NONCIRCULAR ORBIT (ecc ≥ tol)
    # ------------------------------

    # Inclination
    cos_i = H[2] / h
    cos_i = np.clip(cos_i, -1, 1)
    i = np.arccos(cos_i)

    # RAAN
    if n_norm < tol:
        raan = 0.0
        Nhat = np.array([1,0,0])
    else:
        Nhat = N / n_norm
        cos_raan = np.clip(Nhat[0], -1, 1)
        raan = np.arccos(cos_raan)
        if Nhat[1] < 0:
            raan = 2*np.pi - raan

    # Argument of periapsis
    cos_argp = np.dot(Nhat, e_vec) / ecc
    cos_argp = np.clip(cos_argp, -1, 1)
    argp = np.arccos(cos_argp)
    if e_vec[2] < 0:
        argp = 2*np.pi - argp

    # True anomaly
    cos_ta = np.dot(e_vec, R) / (ecc * r_norm)
    cos_ta = np.clip(cos_ta, -1, 1)
    ta = np.arccos(cos_ta)
    if np.dot(R, V) < 0:
        ta = 2*np.pi - ta

    # Semi-major axis
    a = 1 / ((2/r_norm) - (v_norm**2/mu))

    if units == 'deg':
        return a, ecc, np.degrees(i), np.degrees(raan), np.degrees(argp), np.degrees(ta)
    else:
        return a, ecc, i, raan, argp, ta


def orbital_elements_to_inertial(a, e, i, raan, argp, ta, mu=MU_EARTH, units='rad'):
    """
    Classical Orbital Elements -> inertial position/velocity.

    Inputs:
        a     : semi-major axis [km] (a<0 allowed for hyperbolic)
        e     : eccentricity (scalar,e!=1)
        i     : inclination [deg]
        RAAN  : right ascension of ascending node Ω [deg]
        AOP  : argument of perigee ω [deg]
        TA    : true anomaly f [deg]
        mu    : gravitational parameter [km^3/s^2]
        units : 'rad' for radians, 'deg' for degrees

    Returns:
        r : position vector in inertial frame [km]
        v : velocity vector in inertial frame [km/s]
    """

    # Convert angles from degrees to radians if units is 'deg'
    if units == 'deg':
        i = np.radians(i)
        raan = np.radians(raan)
        argp = np.radians(argp)
        ta = np.radians(ta)

    # semi-latus rectum (works for elliptical & hyperbolic; not defined for parabolic e=1)
    if np.isclose(e, 1.0, atol=1e-12):
        raise ValueError("Parabolic case (e close to 1) not supported.")
    p = a * (1.0 - e**2)

    cnu, snu = np.cos(ta), np.sin(ta)
    r_pf = (p / (1.0 + e * cnu)) * np.array([cnu, snu, 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-snu, e + cnu, 0.0])

    # Rotation PQW -> IJK
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(argp), np.sin(argp)

    R3_O = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    R1_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    R3_o = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])

    Q_pqw_to_ijk = R3_O @ R1_i @ R3_o

    r = Q_pqw_to_ijk @ r_pf
    v = Q_pqw_to_ijk @ v_pf
    return r, v

# compute mean motion from inertial state vectors
def compute_mean_motion_from_ECI(r, v, mu=398600.4418):
    """
    Compute the mean motion from inertial state vectors (r, v).
    r: position vector (km)
    v: velocity vector (km/s)
    mu: gravitational parameter (km^3/s^2), default is Earth's

    Returns: mean motion n (rad/s)
    """
    r = np.array(r)
    v = np.array(v)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    # Specific orbital energy
    energy = v_norm**2 / 2 - mu / r_norm

    # Semi-major axis
    a = -mu / (2 * energy)

    # Mean motion
    n = np.sqrt(mu / a**3)

    return n

# Compute inertial state vector from LROEs
def lroes_to_inertial(t, chief_r, chief_v, lroes, mu=MU_EARTH):
    """
    Convert LROEs to inertial state vectors (r, v) of the deputy.

    Inputs:
        t: time [s]
        chief_r: chief inertial position vector [km]
        chief_v: chief inertial velocity vector [km/s]
        lroes: linear relative orbital elements [A_0, B_0, alpha, beta, x_offset, y_offset] in km and !!RADIANS!!
        mu: gravitational parameter [km^3/s^2]

    Returns:
        r_deputy: deputy inertial position vector [km]
        v_deputy: deputy inertial velocity vector [km/s]
    """
    chief_r = np.array(chief_r)
    chief_v = np.array(chief_v)
    A_0, B_0, alpha, beta, x_offset, y_offset = lroes

    # Compute chief mean motion
    n = compute_mean_motion_from_ECI(chief_r, chief_v, mu)

    # Compute A_1 and A_2
    A_1 = A_0 * np.cos(alpha)
    A_2 = A_0 * np.sin(alpha)

    # Compute B_1 and B_2
    B_1 = B_0 * np.cos(beta)
    B_2 = B_0 * np.sin(beta)

    # Compute position state
    x = A_1 * np.cos(n*t) - A_2 * np.sin(n*t) + x_offset
    y = -2*A_1 * np.sin(n*t) - 2*A_2 * np.cos(n*t) - (3/2)*(n*x_offset*t) + y_offset
    z = B_1 * np.cos(n*t) + B_2 * np.sin(n*t)

    # Compute position state
    x_dot = -A_1*n*np.sin(n*t) - A_2*n*np.cos(n*t)
    y_dot = -2*A_1*n*np.cos(n*t) + 2*A_2*n*np.sin(n*t) - (3/2)*n*x_offset
    z_dot = -B_1*n*np.sin(n*t) + B_2*n*np.cos(n*t)

    # LVLH relative state vectors
    deputy_rho = np.array([x, y, z])
    deputy_rho_dot = np.array([x_dot, y_dot, z_dot])

    # Convert to inertial frame
    deputy_r, deputy_v = rel_vector_to_inertial(deputy_rho, deputy_rho_dot, chief_r, chief_v) 

    return deputy_r, deputy_v


def _wrap01(x):
    # wrap to [0, 2π)
    return np.mod(x, 2*np.pi)

def ta_to_m(TA, e):
    """
    True anomaly -> Mean anomaly (elliptic 0<=e<1).
    Handles scalar or ndarray TA. Returns M in [0,2π).
    """
    TA = np.atleast_1d(TA).astype(float)
    # principal eccentric anomaly via atan2 form (robust)
    # tan(E/2) = sqrt((1-e)/(1+e)) * tan(TA/2)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(TA/2.0),
                         np.sqrt(1.0 + e) * np.cos(TA/2.0))
    E = _wrap01(E)
    M = E - e * np.sin(E)
    M = _wrap01(M)
    return M.squeeze()

def m_to_ta(M, e, tol=1e-12, max_iter=100):
    """
    Mean anomaly -> True anomaly (elliptic 0<=e<1) using Newton-Raphson.
    Returns TA in [0, 2π).
    TA and E match the shape of input M (scalar or array).
    """
    M_in = np.atleast_1d(M).astype(float)
    M_wrapped = _wrap01(M_in)

    # initial guess for E
    E = np.copy(M_wrapped)
    # better guess for larger e
    mask = (e >= 0.8)
    if np.isscalar(E):
        if e >= 0.8:
            E = np.pi
    else:
        E[mask] = np.pi

    # Newton-Raphson solve for E
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M_wrapped
        fprime = 1.0 - e * np.cos(E)
        dE = f / fprime
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    else:
        raise RuntimeError("Kepler solve did not converge")

    E = _wrap01(E)
    # convert E->TA robustly using atan2 form
    sinv = np.sqrt(1.0 + e) * np.sin(E/2.0)
    cosv = np.sqrt(1.0 - e) * np.cos(E/2.0)
    TA = 2.0 * np.arctan2(sinv, cosv)
    TA = _wrap01(TA)

    # return shapes consistent with scalar input
    if np.isscalar(M):
        return float(TA.squeeze())
    return TA.squeeze()

def normalize_angle(angle):
    """Wrap angle to [-pi, pi)."""
    a = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return a
