import numpy as np
from data.resources.constants import MU_EARTH

# --- Kepler/coordinate helpers -------------------------------------------
def true_to_E(f, e):
    """True anomaly -> eccentric anomaly (principal value)."""
    return 2.0 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f / 2.0))

def E_to_M(E, e):
    return E - e * np.sin(E)

def solve_kepler(M, e, tol=1e-12, maxit=200):
    """Solve Kepler's equation M = E - e sin E (Newton method)."""
    # initial guess
    E = M if e < 0.8 else np.pi
    for _ in range(maxit):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E

def E_to_true(E, e):
    """Eccentric anomaly -> true anomaly."""
    return 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2.0),
                            np.sqrt(1 - e) * np.cos(E / 2.0))

def keplerian_to_cartesian(a, e, inc, raan, argp, ta, mu=MU_EARTH):
    """Classical orbital elements (angles in radians) -> ECI r, v."""
    p = a * (1 - e**2)
    r_pf = (p / (1 + e * np.cos(ta))) * np.array([np.cos(ta), np.sin(ta), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(ta), e + np.cos(ta), 0.0])

    # rotation from perifocal to ECI: R3(RAAN) * R1(inc) * R3(AOP)
    R3_W = np.array([[np.cos(raan), -np.sin(raan), 0.0],
                     [np.sin(raan),  np.cos(raan), 0.0],
                     [0.0, 0.0, 1.0]])
    R1_i = np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(inc), -np.sin(inc)],
                     [0.0, np.sin(inc),  np.cos(inc)]])
    R3_w = np.array([[np.cos(argp), -np.sin(argp), 0.0],
                     [np.sin(argp),  np.cos(argp), 0.0],
                     [0.0, 0.0, 1.0]])
    Q_pX = R3_W @ R1_i @ R3_w

    r = Q_pX @ r_pf
    v = Q_pX @ v_pf
    return r, v

# --- Hill/relative helpers -----------------------------------------------
def inertial_to_hill_relative(rd, vd, rc, vc):
    """Return rho_H and rho_dot_H (Hill frame) given inertial r,v of deputy & chief."""
    rc = np.array(rc); vc = np.array(vc); rd = np.array(rd); vd = np.array(vd)

    # Hill unit vectors (ECI basis)
    r_hat = rc / np.linalg.norm(rc)
    h_vec = np.cross(rc, vc)
    h_hat = h_vec / np.linalg.norm(h_vec)
    theta_hat = np.cross(h_hat, r_hat)

    # rotation ECI -> Hill (columns are r_hat, theta_hat, h_hat)
    C_HI = np.vstack((r_hat, theta_hat, h_hat)).T

    rho = rd - rc
    drho = vd - vc

    # angular velocity of Hill frame (ECI) used for velocity correction
    omega_HI = np.cross(rc, vc) / (np.linalg.norm(rc)**2)

    rho_H = C_HI.T @ rho
    rho_dot_H = C_HI.T @ (drho - np.cross(omega_HI, rho))

    return rho_H, rho_dot_H

# --- Main mapping: chief OE + dOE -> rho_H, rho_dot_H ---------------------
def doe_to_rho(a_c, e_c, i_c, raan_c, argp_c, f_c,
                            da, de, di, draan, dargp, dM):
    """
    Inputs:
      - chief elements a_c, e_c, i_c, RAAN_c, AOP_c, f_c (angles in radians)
      - differences da (km), de, di (rad), dRAAN (rad), dAOP (rad), dM (rad)
    Returns:
      - rho_H (km), rho_dot_H (km/s) at chief true anomaly f_c
    Method:
      - compute chief r_c, v_c from (a_c,e_c,i_c,RAAN_c,AOP_c,f_c)
      - compute chief mean anomaly, add dM to get deputy mean anomaly
      - solve Kepler for deputy E -> deputy true anomaly f_d (with deputy e)
      - deputy classical elements = chief + dOE (use da,de,...)
      - convert both to inertial states and compute hill relative (rho, rho_dot)
    """
    # Chief inertial
    r_c, v_c = keplerian_to_cartesian(a_c, e_c, i_c, raan_c, argp_c, f_c)

    # Chief eccentric anomaly and mean anomaly
    E_c = true_to_E(f_c, e_c)
    M_c = E_c - e_c * np.sin(E_c)

    # Deputy elements (classical)
    a_d = a_c + da
    e_d = e_c + de
    i_d = i_c + di
    raan_d = raan_c + draan
    argp_d = argp_c + dargp

    # Deputy mean anomaly and true anomaly (solve Kepler with deputy e)
    M_d = M_c + dM
    E_d = solve_kepler(M_d, e_d)
    f_d = E_to_true(E_d, e_d)

    # Deputy inertial
    r_d, v_d = keplerian_to_cartesian(a_d, e_d, i_d, raan_d, argp_d, f_d)

    # Hill relative
    rho_H, rho_dot_H = inertial_to_hill_relative(r_d, v_d, r_c, v_c)
    return rho_H, rho_dot_H