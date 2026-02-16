## Relative to inertial functions
import numpy as np

#  Compute LVLH (Hill) Direction Cosine Matrix (DCM) from inertial position and velocity
def LVLH_DCM(r, v):
    """
    Compute LVLH (Hill) Direction Cosine Matrix (DCM) from inertial state to LVLH frame.
    """
    r = np.array(r)
    v = np.array(v)

    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)
    theta_hat = np.cross(h_hat, r_hat)

    C_H_N = np.vstack((r_hat, theta_hat, h_hat))

    return C_H_N


def inertial_to_rel_LVLH(r_d, v_d, r_c, v_c):
    """
    Compute relative position and velocity in the Hill (RTN) frame.

    Parameters
    ----------
    r_c : ndarray, shape (3,)
        Chief position vector in ECI [km]
    v_c : ndarray, shape (3,)
        Chief velocity vector in ECI [km/s]
    r_d : ndarray, shape (3,)
        Deputy position vector in ECI [km]
    v_d : ndarray, shape (3,)
        Deputy velocity vector in ECI [km/s]

    Returns
    -------
    rho_hill : ndarray, shape (3,)
        Relative position vector in Hill frame [km]
    rho_dot_hill : ndarray, shape (3,)
        Relative velocity vector in Hill frame [km/s]
    """

    # --- Hill frame unit vectors (ECI basis) ---
    r_hat = r_c / np.linalg.norm(r_c)
    h_hat = np.cross(r_c, v_c)
    h_hat /= np.linalg.norm(h_hat)
    theta_hat = np.cross(h_hat, r_hat)

    # Rotation matrix from ECI -> Hill
    C_HI = np.vstack((r_hat, theta_hat, h_hat)).T

    # --- Relative position and velocity in ECI ---
    rho = r_d - r_c
    drho = v_d - v_c

    # Chief angular velocity in ECI
    omega_HI = np.cross(r_c, v_c) / np.linalg.norm(r_c)**2

    # --- Transform to Hill frame ---
    rho_hill = C_HI.T @ rho
    rho_dot_hill = C_HI.T @ (drho - np.cross(omega_HI, rho))

    return rho_hill, rho_dot_hill


def inertial_to_LVLH(r,v):
    """
    Transform a state from LVLH frame to inertial frame.
    """
    # Compute DCM from inertial to LVLH
    C_N_H = LVLH_DCM(r, v)

    # Compute specific angular momentum
    h = np.cross(r, v)

    # Compute angular velocity of LVLH(Hill) frame
    omega = np.array([0, 0, np.linalg.norm(h) / np.dot(r, r)])

    # Transform state from inertial to LVLH
    r_H = C_N_H @ r
    v_H = C_N_H @ v - np.cross(omega, r_H)
    
    return r_H, v_H


def LVLH_basis_vectors(r, v):
    """
    Compute LVLH (Hill) basis vectors from inertial position and velocity.
    """
    r = np.array(r)
    v = np.array(v)

    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)
    theta_hat = np.cross(h_hat, r_hat)

    return r_hat, theta_hat, h_hat


def rel_vector_to_LVLH(rho_H,rho_dot_H,rc_H,vc_H):
    """
    Compute deputy relative position and velocity in LVLH frame.
    """

    # Deputy in LVLH (Hill) frame
    rd_H = rc_H + rho_H
    vd_H = vc_H + rho_dot_H

    return rd_H, vd_H

def rel_vector_to_inertial(rho_H,rho_dot_H,rc_N,vc_N):
    """
    Compute deputy inertial position and velocity from chief inertial state and deputy relative state in LVLH frame.
    """

    # Chief in LVLH (Hill) frame
    rc_H, vc_H = inertial_to_LVLH(rc_N, vc_N)

    # Deputy in LVLH (Hill) frame
    rd_H, vd_H = rel_vector_to_LVLH(rho_H,rho_dot_H,rc_H,vc_H)

    # DCM from inertial to LVLH
    C_N_H = LVLH_DCM(rc_N, vc_N)

    # Specific angular momentum
    h = np.cross(rc_N, vc_N)

    # Angular velocity of LVLH(Hill) frame
    omega = np.array([0, 0, np.linalg.norm(h) / np.dot(rc_N, rc_N)])

    # Transform deputy state from LVLH to inertial
    rd_N = C_N_H.T @ rd_H
    vd_N = C_N_H.T @ (vd_H + np.cross(omega, rd_H))

    return rd_N, vd_N

def compute_omega(r, v):
    """
    Compute angular velocity of LVLH(Hill) frame given inertial position and velocity.
    """
    r = np.array(r)
    v = np.array(v)

    h = np.cross(r, v)
    omega = np.array([0, 0, np.linalg.norm(h) / np.dot(r, r)])

    return omega

def compute_Q_ECI_from_RIC(r_eci, v_eci, sigma_R, sigma_I, sigma_C):
    """
    Computes the Process Noise PSD (3x3) in ECI frame given sigmas in RIC.
    
    Args:
        r_eci (np.array): Position vector [x, y, z]
        v_eci (np.array): Velocity vector [vx, vy, vz]
        sigma_R, sigma_I, sigma_C: Sigmas for Radial, In-Track, Cross-Track noise
        
    Returns:
        Q_ECI (3x3 matrix): The covariance rotated into the inertial frame.
    """
    # 1. Compute Unit Vectors for RIC Frame
    # Radial: Unit vector in direction of position
    u_r = r_eci / np.linalg.norm(r_eci)
    
    # Cross-track: Unit vector normal to orbital plane (h = r x v)
    h = np.cross(r_eci, v_eci)
    u_c = h / np.linalg.norm(h)
    
    # In-track: Completes the triad (i = c x r)
    u_i = np.cross(u_c, u_r)
    
    # 2. Rotation Matrix (RIC -> ECI)
    # The columns are the RIC basis vectors expressed in ECI
    R_RIC_to_ECI = np.column_stack((u_r, u_i, u_c))
    
    # 3. Define Q in RIC (Diagonal)
    Q_RIC = np.diag([sigma_R**2, sigma_I**2, sigma_C**2])
    
    # 4. Rotate Covariance: Q_ECI = R * Q_RIC * R^T
    Q_ECI = R_RIC_to_ECI @ Q_RIC @ R_RIC_to_ECI.T
    
    return Q_ECI