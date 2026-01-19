import numpy as np

def range_obs_partials(R, V, Rs, Vs):
    """
    Compute the observation partials for range and range-rate measurements.
    
    Inputs:
        R : (3,) np.array - Spacecraft position vector
        V : (3,) np.array - Spacecraft velocity vector
        Rs : (3,) np.array - Ground station position vector
        Vs : (3,) np.array - Ground station velocity vector
    """

    # Relative position and velocity
    rho_vec = R - Rs
    v_rel = V - Vs
    
    
    # Range and range rate
    rho = np.linalg.norm(rho_vec)
    rho_dot = np.dot(rho_vec, v_rel) / rho
    
    # Range partials
    d_rho_dR = rho_vec / rho
    d_rho_dV = np.zeros(3)

    # Range rate partials
    d_rhodot_dV = d_rho_dR  
    d_rhodot_dR = (rho * v_rel - rho_dot * rho_vec) / (rho**2)

    # Partials w.r.t SC position and velocity (2x6 measurement jacobian)
    obs_matrix_dR = np.zeros((2, 6))
    obs_matrix_dR[0, 0:3] = d_rho_dR
    obs_matrix_dR[1, 0:3] = d_rhodot_dR
    obs_matrix_dR[1, 3:6] = d_rhodot_dV

    # Partials w.r.t. GS position and velocity (2x3 measurement jacobian)
    obs_matrix_dRs = np.zeros((2, 3))
    obs_matrix_dRs[0, 0:3] = -d_rho_dR
    obs_matrix_dRs[1, 0:3] = -d_rhodot_dR\
    
    # Output Dictionary of observation partials
    obs_measurements = np.array([rho, rho_dot])
    obs_matrix = {
        'wrt_R': obs_matrix_dR,
        'wrt_Rs': obs_matrix_dRs
    }

    return obs_measurements, obs_matrix
if __name__ == "__main__":
    # Example usage
    R = np.array([0.42286036448769, 1.29952829655200, -1.04979323447507])  # km
    V = np.array([-1.78641172211092, 0.81604308103192, -0.32820854314251])   # km/s
    Rs = np.array([-1.21456561358767, 1.11183287253465, -0.50749695482985]) # km
    Vs = np.array([-0.00008107614118, -0.00008856753168, 0.0])    # km/s
    
    # Print to 14 decimal places for comparison
    np.set_printoptions(precision=14, suppress=True)
    obs_measurements, obs_matrix = range_obs_partials(R, V, Rs, Vs)
    print("Observation Partials:\n", obs_matrix['wrt_R'])

    # Compare to truth values
    partials_truth_wrtR = np.array([[0.94372160948022, 0.10817724282958, -0.31254952876921, 0.0, 0.0, 0.0],
        [-0.21643606559054, 0.56357804886802, -0.45845237165164, 0.94372160948022, 0.10817724282958, -0.31254952876921]])

    partials_truth_wrtRs = np.array([[-0.94372160948022, -0.10817724282958, 0.31254952876921],
                                       [0.21642817718066, -0.56350923159490, 0.45845237165164]])


    # Error
    error = obs_matrix['wrt_R'] - partials_truth_wrtR
    print("Satellite Measurement Matrix Error:\n", error)
    print("Max Error:", np.max(np.abs(error)))

    error_Rs = obs_matrix['wrt_Rs'] - partials_truth_wrtRs
    print("Ground Station Measurement Matrix Error:\n", error_Rs)
    print("Max Error:", np.max(np.abs(error_Rs)))
