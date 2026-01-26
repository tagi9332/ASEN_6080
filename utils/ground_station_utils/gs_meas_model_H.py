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
    return H

def compute_rho_rhodot(X_sc, X_station):
    """
    Predicts range and range-rate measurements.
    
    :param X_sc: Spacecraft state vector [x, y, z, vx, vy, vz]
    :param X_station: Ground station state vector [xs, ys, zs, vxs, vys, vzs]
    :return: numpy array [rho, rho_dot]
    """
    # Extract position and velocity vectors
    R = X_sc[0:3]
    V = X_sc[3:6]
    
    Rs = X_station[0:3]
    Vs = X_station[3:6]
    
    # Relative position vector
    rho_vec = R - Rs
    rho = np.linalg.norm(rho_vec)
    
    # Guard against division by zero (singular case)
    if rho == 0.0:
        return np.array([0.0, 0.0])
    
    # Relative velocity vector
    v_rel = V - Vs
    
    # Range-rate: (rho_vec dot v_rel) / rho
    rho_dot = np.dot(rho_vec, v_rel) / rho
    
    return np.array([rho, rho_dot])