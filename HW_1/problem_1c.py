import numpy as np

def dfdx_wJ2J3(r_vec, v_vec, mu, J2, J3, Re=6378.0,use_J2=True,use_J3=True):
    
    x, y, z = r_vec
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2)
    r3, r5, r7, r9, r11 = r**3, r**5, r**7, r**9, r**11
    
    # --- 2. Build the 3x3 Gravity Gradient (G = del_a / del_r) ---
    # Point Mass
    G = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)
    
    # J2 Contribution
    if use_J2:
        j2_c = -1.5 * mu * J2 * Re**2
        G_j2 = np.array([
            [1/r5 - 5*x**2/r7 - 5*z**2/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9, -15*x*z/r7 + 35*x*z**3/r9],
            [-5*x*y/r7 + 35*x*y*z**2/r9, 1/r5 - 5*y**2/r7 - 5*z**2/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
            [-15*x*z/r7 + 35*x*z**3/r9, -15*y*z/r7 + 35*y*z**3/r9, 3/r5 - 30*z**2/r7 + 35*z**4/r9]
        ])
        G += j2_c * G_j2
    
    # J3 Contribution if enabled
    if use_J3:
        j3_c = -2.5 * mu * J3 * Re**3
        G_j3 = np.array([
            [3*z/r7 - 21*x**2*z/r9 - 7*z**3/r9 + 63*x**2*z**3/r11, -21*x*y*z/r9 + 63*x*y*z**3/r11, 3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11],
            [-21*x*y*z/r9 + 63*x*y*z**3/r11, 3*z/r7 - 21*y**2*z/r9 - 7*z**3/r9 + 63*y**2*z**3/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11],
            [3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11, 15*z/r7 - 70*z**3/r9 + 63*z**5/r11]
        ])
        G += j3_c * G_j3

# --- 3. Build the 3x3 Sensitivity Matrix (S = del_a / del_params) ---
    
    # Use the same constants as the G-block to ensure sign consistency
    # (Note: mu is already included in these j_c constants)
    j2_c = -1.5 * mu * J2 * Re**2
    j3_c = -2.5 * mu * J3 * Re**3

    # Acceleration - Point Mass
    a_pm = -(mu / r3) * r_vec
    
    # Acceleration - J2 (Matches G_j2 bracket logic)
    # Target JSON uses: x*(1 - 5z^2/r^2)
    vec_j2 = np.array([
        x * (1/r5 - 5*z**2/r7),
        y * (1/r5 - 5*z**2/r7),
        z * (3/r5 - 5*z**2/r7)
    ])
    a_j2 = j2_c * vec_j2
    
    # Acceleration - J3 (Matches G_j3 bracket logic)
    # Target JSON uses: x*(3z - 7z^3/r^2)
    vec_j3 = np.array([
        x * (3*z/r7 - 7*z**3/r9),
        y * (3*z/r7 - 7*z**3/r9),
        (6*z**2/r7 - 7*z**4/r9 - 0.6/r5) 
    ])
    a_j3 = j3_c * vec_j3

    # --- Assemble S Matrix ---
    S = np.zeros((3, 3))
    
    # Column 6: Partial wrt mu (Total Accel / mu)
    # This now includes pm, j2, AND j3 correctly
    S[:, 0] = (a_pm + a_j2 + a_j3) / mu
    
    # Column 7: Partial wrt J2
    S[:, 1] = a_j2 / J2
    
    # Column 8: Partial wrt J3
    S[:, 2] = a_j3 / J3
    
    # REMOVED: S[2,2] *= -1 and S[2,0] *= -1
    # The signs are now handled natively by the j2_c and j3_c definitions

    # --- 4. Assemble the Full 9x9 A-Matrix ---
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3) # dr/dv block
    A[3:6, 0:3] = G        # dv/dr block (Gravity Gradient)
    A[3:6, 6:9] = S        # dv/dp block (Sensitivity)
    
    return A

# --- TEST CASE USING JSON VALUES ---
state_json = np.array([
    -0.64901376519124, 1.18116604196553, -0.75845329728369, # r
    -1.10961303850152, -0.84555124000780, -0.57266486645795, # v
    -0.55868076447397, 0.17838022584977, -0.19686144647594  # mu, J2, J3
])

r_vec = state_json[0:3]
v_vec = state_json[3:6]
mu = state_json[6]
J2 = state_json[7]
J3 = state_json[8]


target_vals = [
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [733162737.5603306, 4792269202.1485796, -7567430282.272636, 0.0, 0.0, 0.0, 3910671593.4771023, 506834.93002139014, 11098706444.52816],
    [4792269202.1485796, -5355277247.564346, 13772268253.398691, 0.0, 0.0, 0.0, -7117187238.907471, -922409.1079899368, -20198978612.044998],
    [-7567430282.272636, 13772268253.398691, 4622114510.004021, 0.0, 0.0, 0.0, 6327041227.415253, -5253592.802495056, 17950996276.141552],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]

A_matrix = dfdx_wJ2J3(r_vec, v_vec, mu, J2, J3)

np.set_printoptions(precision=23, suppress=True)
print("A-Matrix:\n", A_matrix)

# Print out Gradient and Sensitivity for verification in a nice format
G_block = A_matrix[3:6, 0:3]
S_block = A_matrix[3:6, 6:9]
print("\nGravity Gradient (G) Block:\n", G_block)
print("\nSensitivity (S) Block:\n", S_block)

# Compute error values and display max error
error_matrix = A_matrix - np.array(target_vals)
max_error = np.max(np.abs(error_matrix))
print("\nMax Error:", max_error)

