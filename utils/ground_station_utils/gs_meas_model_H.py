import numpy as np

def compute_H_matrix(R, V, Rs, Vs):
    """
    Compute the measurement model H matrix and predicted measurements
    for range and range-rate between a spacecraft and a ground station.

    :param R: Spacecraft position vector (3,)
    :param V: Spacecraft velocity vector (3,)
    :param Rs: Ground station position vector (3,)
    :param Vs: Ground station velocity vector (3,)
    """
    rho_vec = R - Rs
    v_rel = V - Vs
    rho = np.linalg.norm(rho_vec)
    rho_dot = np.dot(rho_vec, v_rel) / rho

    d_rho_dR = rho_vec / rho
    d_rhodot_dR = (v_rel - (rho_dot * d_rho_dR)) / rho
    d_rhodot_dV = d_rho_dR

    H = np.zeros((2, 6))
    H[0, 0:3] = d_rho_dR
    H[1, 0:3] = d_rhodot_dR
    H[1, 3:6] = d_rhodot_dV
    return np.array([rho, rho_dot]), H