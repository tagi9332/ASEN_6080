import numpy as np

def get_zonal_jacobian(r_vec, v_vec, mu, coeffs, Re=6378.0):
    """
    Computes the 9x9 Jacobian (A-matrix) using analytic formulas.
    State: [x, y, z, vx, vy, vz, mu, J2, J3]
    """
    x, y, z = r_vec
    r2 = x*x + y*y + z*z
    r = np.sqrt(r2)
    
    # Precompute powers of r for speed
    r3, r5, r7, r9, r11 = r**3, r**5, r**7, r**9, r**11
    
    # Extract coefficients (Assuming coeffs = [J1, J2, J3])
    J2, J3 = coeffs[1], coeffs[2]
    
    # --- 1. Gravity Gradient (G = del_a / del_r) ---
    # Point Mass
    G = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)
    
    # J2 Gradient Contribution
    j2_c = -1.5 * mu * J2 * Re**2
    G_j2 = j2_c * np.array([
        [1/r5 - 5*(x**2+z**2)/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9,         -15*x*z/r7 + 35*x*z**3/r9],
        [-5*x*y/r7 + 35*x*y*z**2/r9,         1/r5 - 5*(y**2+z**2)/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
        [-15*x*z/r7 + 35*x*z**3/r9,         -15*y*z/r7 + 35*y*z**3/r9,                 3/r5 - 30*z**2/r7 + 35*z**4/r9]
    ])
    
    # J3 Gradient Contribution
    j3_c = -2.5 * mu * J3 * Re**3
    G_j3 = j3_c * np.array([
        [3*z/r7 - 21*x**2*z/r9 - 7*z**3/r9 + 63*x**2*z**3/r11, -21*x*y*z/r9 + 63*x*y*z**3/r11, 3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11],
        [-21*x*y*z/r9 + 63*x*y*z**3/r11, 3*z/r7 - 21*y**2*z/r9 - 7*z**3/r9 + 63*y**2*z**3/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11],
        [3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11,         15*z/r7 - 70*z**3/r9 + 63*z**5/r11]
    ])
    
    G_total = G + G_j2 + G_j3

    # --- 2. Sensitivity (S = del_a / del_parameters) ---
    # Accelerations
    a_pm = -(mu/r3) * r_vec
    
    a_j2 = j2_c * np.array([
        x/r5 * (1 - 5*z**2/r2),
        y/r5 * (1 - 5*z**2/r2),
        z/r5 * (3 - 5*z**2/r2)
    ])
    
    a_j3 = j3_c * np.array([
        x/r7 * (3*z - 7*z**3/r2),
        y/r7 * (3*z - 7*z**3/r2),
        1/r7 * (6*z**2 - 7*z**4/r2 - 0.6*r2)
    ])
    
    S = np.zeros((3, 3))
    S[:, 0] = (a_pm + a_j2 + a_j3) / mu  # partial wrt mu
    S[:, 1] = a_j2 / J2                # partial wrt J2
    S[:, 2] = a_j3 / J3                # partial wrt J3

    # --- 3. Assemble Full 9x9 A-Matrix ---
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3)   # dr/dv
    A[3:6, 0:3] = G_total     # dv/dr
    A[3:6, 6:9] = S           # dv/dp [mu, J2, J3]
    
    return A

# --- Example Usage with your Test Case ---
state_test = np.array([-0.64901376519124, 1.18116604196553, -0.75845329728369, 
                       -1.10961303850152, -0.84555124000780, -0.57266486645795, 
                       -0.55868076447397, 0.17838022584977, -0.19686144647594])

A_res = get_zonal_jacobian(state_test[0:3], state_test[3:6], 
                           state_test[6], [0, state_test[7], state_test[8]])


# Print max error
target_A = np.array([
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [733162737.5603306, 4792269202.1485796, -7567430282.272636, 0.0, 0.0, 0.0, 3910671593.4771023, 506834.93002139014, 11098706444.52816],
    [4792269202.1485796, -5355277247.564346, 13772268253.398691, 0.0, 0.0, 0.0, -7117187238.907471, -922409.1079899368, -20198978612.044998],
    [-7567430282.272636, 13772268253.398691, 4622114510.004021, 0.0, 0.0, 0.0, 6327041227.415253, -5253592.802495056, 17950996276.141552],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
])

error_A = A_res - target_A
print(f"Max Error in A-Matrix: {np.max(np.abs(error_A))}")