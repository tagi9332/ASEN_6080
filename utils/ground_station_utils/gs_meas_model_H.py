import numpy as np
from resources.constants import OMEGA_EARTH

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

def compute_H_tilde_18_state(state, stat_idx, meas_include=[True, True]):
    """
    Computes the measurement partials matrix (H_tilde) for Range and Range-Rate.
    
    Inputs:
        state (np.array): 18-element state vector 
                          [x, y, z, vx, vy, vz, mu, J2, Cd, 
                           gs1_x, gs1_y, gs1_z, gs2_x, gs2_y, gs2_z, gs3_x, gs3_y, gs3_z]
        stat_idx (int):   Index of the observing station (0, 1, or 2).
        meas_include (list): Boolean list [include_range, include_range_rate].
                             Default is [True, True].
    
    Outputs:
        H_tilde (np.array): Measurement partials matrix (Mx18), where M is 1 or 2 
                            depending on meas_include.
    """
    
    # --- 1. Extract Spacecraft State ---
    x, y, z = state[0:3]
    vx, vy, vz = state[3:6]
    
    # --- 2. Extract Visible Station State ---
    # The stations start at index 9. Each station has 3 coordinates.
    base_idx = 9 + (3 * stat_idx)
    xs = state[base_idx]
    ys = state[base_idx + 1]
    zs = state[base_idx + 2]
    
    # Calculate Station Velocity ( Inertial v_s = w_earth x r_s )
    w_vec = np.array([0, 0, OMEGA_EARTH])
    r_s_vec = np.array([xs, ys, zs])
    v_s_vec = np.cross(w_vec, r_s_vec)
    vxs, vys, vzs = v_s_vec
    
    # --- 3. Relative Vectors ---
    dx = x - xs
    dy = y - ys
    dz = z - zs
    rho = np.sqrt(dx**2 + dy**2 + dz**2)
    
    dvx = vx - vxs
    dvy = vy - vys
    dvz = vz - vzs
    
    # Range Rate (rho_dot) term for convenience
    # rho_dot = (dx*dvx + dy*dvy + dz*dvz) / rho
    dot_prod = dx*dvx + dy*dvy + dz*dvz
    
    # --- 4. Partials w.r.t Spacecraft State (Block 1) ---
    
    # d(rho)/d(r)
    d_rho_dx = dx / rho
    d_rho_dy = dy / rho
    d_rho_dz = dz / rho
    
    # d(rho_dot)/d(r)
    # Formula: (rho^2 * dv - dr * (dr.dv)) / rho^3
    rho2 = rho**2
    rho3 = rho**3
    
    d_rhodot_dx = (rho2 * dvx - dx * dot_prod) / rho3
    d_rhodot_dy = (rho2 * dvy - dy * dot_prod) / rho3
    d_rhodot_dz = (rho2 * dvz - dz * dot_prod) / rho3
    
    # d(rho_dot)/d(v) -> Same as d(rho)/d(r)
    d_rhodot_dvx = d_rho_dx
    d_rhodot_dvy = d_rho_dy
    d_rhodot_dvz = d_rho_dz
    
    # Assemble Block 1 (Spacecraft)
    h_block_1 = []
    
    if meas_include[0]: # Include Range
        # [dRho/dr, dRho/dv(0), dRho/dParams(0)]
        row_rng = [d_rho_dx, d_rho_dy, d_rho_dz, 0, 0, 0, 0, 0, 0]
        h_block_1.append(row_rng)
        
    if meas_include[1]: # Include Range Rate
        # [dRhoDot/dr, dRhoDot/dv, dRhoDot/dParams(0)]
        row_rate = [d_rhodot_dx, d_rhodot_dy, d_rhodot_dz, 
                    d_rhodot_dvx, d_rhodot_dvy, d_rhodot_dvz, 
                    0, 0, 0]
        h_block_1.append(row_rate)
        
    h_block_1 = np.array(h_block_1)

    # --- 5. Partials w.r.t Ground Stations (Block 2) ---
    # We need to iterate through all 3 possible stations (indices 0, 1, 2)
    # Only the visible station (stat_idx) has non-zero partials.
    
    h_block_2 = []
    
    # Earth rotation cross-product matrix (tilde w)
    w_tilde = np.array([
        [0, -OMEGA_EARTH, 0],
        [OMEGA_EARTH, 0, 0],
        [0, 0, 0]
    ])
    
    # Vectors for calculation
    r_vec = np.array([x, y, z])
    v_vec = np.array([vx, vy, vz])
    
    # Calculate rows for Range and Range-Rate separately so we can stack them horizontally later
    row_rng_stations = []
    row_rate_stations = []
    
    for k in range(3):
        if k == stat_idx:
            # --- Non-zero partials ---
            r_s_curr = np.array([xs, ys, zs]) # Current station pos
            rho_vec = r_vec - r_s_curr
            
            # 1. Range w.r.t Station Position
            # d(rho)/d(rs) = - d(rho)/d(r) = -rho_vec / rho
            d_rho_drs = -rho_vec / rho
            
            # 2. Range Rate w.r.t Station Position
            # This is complex because d(vs)/d(rs) is not zero (vs = w x rs)
            # The MATLAB formula: ((rho^2)*(wTilde*R - V) + rhoVec'*(V - wTilde*Rs)*rhoVec)/rho^3
            # Note: MATLAB's derivation simplifies d(vs)/d(rs) into the wTilde terms.
            
            term1 = (rho**2) * (np.dot(w_tilde, r_vec) - v_vec)
            
            # (V - wTilde*Rs) is effectively (V - Vs) = V_rel
            v_s_curr = np.dot(w_tilde, r_s_curr)
            term2_scalar = np.dot(rho_vec, (v_vec - v_s_curr)) # This is rho * rho_dot
            
            d_rhodot_drs = (term1 + term2_scalar * rho_vec) / rho3
            
            # Append to list
            row_rng_stations.extend(d_rho_drs)
            row_rate_stations.extend(d_rhodot_drs)
            
        else:
            # --- Zero partials for non-visible stations ---
            row_rng_stations.extend([0, 0, 0])
            row_rate_stations.extend([0, 0, 0])

    # Assemble Block 2
    if meas_include[0]:
        h_block_2.append(row_rng_stations)
    if meas_include[1]:
        h_block_2.append(row_rate_stations)
        
    h_block_2 = np.array(h_block_2)
    
    # --- 6. Concatenate Blocks ---
    # Structure: [ Spacecraft_Partials(Mx9) | Station_Partials(Mx9) ]
    H_tilde = np.hstack((h_block_1, h_block_2))
    
    return H_tilde