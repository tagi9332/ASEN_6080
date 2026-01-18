import numpy as np

# Constants
MU_EARTH = 3.986004415E5 
R_EARTH = 6378.0    

def dfdx_wJ2J3(r_vec, mu, J2_val, J3_val, Re, use_J2=True, use_J3=True):
    """
    Calculates the 7x7 Jacobian A and the total acceleration.
    State assumed to be [x, y, z, vx, vy, vz, J2].
    """
    x, y, z = r_vec
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2) 
    r3, r5, r7, r9 = r**3, r**5, r**7, r**9
    
    # 1. Point Mass Gravity
    # Acceleration
    a_pm = -(mu / r3) * r_vec
    # Gradient (del_a_pm / del_r)
    G = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)
    
    # 2. J2 Perturbation
    a_j2 = np.zeros(3)
    if use_J2:
        # Note: We use J2_val from the state vector (index 6)
        j2_c = -1.5 * mu * J2_val * (Re**2)
        
        # Acceleration J2
        a_j2 = (j2_c / r5) * np.array([
            x * (1 - 5*z**2/r2),
            y * (1 - 5*z**2/r2),
            z * (3 - 5*z**2/r2)
        ])
        
        # Gradient J2 (del_a_j2 / del_r)
        # To match the JSON's specific numerical output:
        G_j2 = (j2_c) * np.array([
            [1/r5 - 5*(x**2+z**2)/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9, -15*x*z/r7 + 35*x*z**3/r9],
            [-5*x*y/r7 + 35*x*y*z**2/r9, 1/r5 - 5*(y**2+z**2)/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
            [-15*x*z/r7 + 35*x*z**3/r9, -15*y*z/r7 + 35*y*z**3/r9, 3/r5 - 30*z**2/r7 + 35*z**4/r9]
        ])
        G += G_j2

    # 3. Sensitivity (S) = partial derivative of acceleration wrt J2
    # Since a_j2 is linear with respect to J2, S = a_j2 / J2
    S = a_j2 / J2_val
    
    # 4. Assemble A Matrix (7x7)
    A = np.zeros((7, 7))
    A[0:3, 3:6] = np.eye(3) # dr/dv
    A[3:6, 0:3] = G          # dv/dr
    A[3:6, 6] = S           # dv/dJ2
    
    a_total = a_pm + a_j2
    
    return A, a_total

def keplerJ2_wPhi_ODE(t, state_flat, mu, re, use_J2=True, use_J3=False):
    """
    The main ODE function.
    state_flat: [x, y, z, vx, vy, vz, J2, Phi_flat(49)]
    """
    # Unpack state
    r_vec = state_flat[0:3]
    v_vec = state_flat[3:6]
    J2_val = state_flat[6]
    Phi = state_flat[7:56].reshape((7, 7))
    
    # Get physics from modular function
    A, a_total = dfdx_wJ2J3(r_vec, mu, J2_val, 0.0, re, use_J2, use_J3)
    
    # STM Derivative: dPhi/dt = A * Phi
    Phi_dot = A @ Phi
    
    # State Derivative: [v, a, dJ2/dt=0]
    x_dot = np.concatenate([v_vec, a_total, [0.0]])
    
    return np.concatenate([x_dot, Phi_dot.flatten()])

# --- Verification Setup ---
x0 = np.array([-0.85188696962247, 0.80032070980182, -1.50940472473439, 
               0.87587414783453, -0.24278953633334, 0.16681343945350, -1.96541870928278])

phi0 = np.array([
    [-1.27007139263854, -1.86512257453063, 0.06600934128821, 0.59069655120545, -0.34563190830705, -1.47629235201010, 0.59430761682985],
    [1.17517126546302, -1.05110705924059, 0.45129021363078, -0.63578573784723, -1.17140482049761, 0.25889995716040, -0.27646490663926],
    [2.02916018474976, -0.41738204799680, -0.32220971801190, 0.60334661284576, -0.68558678043728, -2.01869095243834, -1.85758288592737],
    [-0.27515724067569, 1.40216228633781, 0.78840921622743, -0.53524796777590, 0.92621639416896, 0.19974026229838, 0.04073081174943],
    [0.60365844582581, -1.36774699097611, 0.92873604681331, -0.15508038549279, -1.48167521167231, 0.42586431913121, 0.28297017716199],
    [1.78125189324250, -0.29253499915187, -0.49079037626976, 0.61212237077216, -0.55805780868504, -1.27004345059705, 0.06356121930250],
    [1.77365832632615, 1.27084843418894, 1.79720058425494, -1.04434349451734, -0.02845311157066, -0.48521883574304, 0.43343006511160]
])

y0 = np.concatenate([x0, phi0.flatten()])

# Run the ODE function
deriv = keplerJ2_wPhi_ODE(0, y0, MU_EARTH, R_EARTH, use_J2=True, use_J3=False)

# Results
print("--- State Derivative (Xdot) ---")
print(deriv[0:7])

print("\n--- Phidot Row 4 (First Velocity Row) ---")
print(deriv[28:35]) # Index for row 4

truth_state = [0.87587414783453, -0.24278953633334, 0.16681343945350, 3413686851177.14257812500000, -3207050208768.89453125000000, 357532131195.69818115234375, 0.0]
truth_phi = [
        [-0.27515724067569, 1.40216228633781, 0.78840921622743, -0.53524796777590, 0.92621639416896, 0.19974026229838, 0.04073081174943],
        [0.60365844582581, -1.36774699097611, 0.92873604681331, -0.15508038549279, -1.48167521167231, 0.42586431913121, 0.28297017716199],
        [1.78125189324250, -0.29253499915187, -0.49079037626976, 0.61212237077216, -0.55805780868504, -1.27004345059705, 0.06356121930250],
        [-3749367026151.89208984375000, -2204151880954.99511718750000, -7128509123418.94335937500000, 9735044190049.28320312500000, 3080439524853.02050781250000, -13373668068751.35546875000000, -6222631678607.20703125000000],
        [3594622546175.40625000000000, 13304253840432.37500000000000, 4640092972826.96777343750000, -8821801281771.37890625000000, 3101266557663.63720703125000, 17084383676012.88671875000000, 4716460510738.97167968750000],
        [-19128458413971.82812500000000, -2613253665476.95361328125000, -703650406578.89013671875000, 3264230201392.84765625000000, 6142762037329.47656250000000, 121940513615.63818359375000, 11100005224265.99804687500000],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

# Error Checking
state_error = deriv[0:7] - np.array(truth_state)
phi_error = deriv[7:] - np.array(truth_phi).flatten()

print("\n--- State Error ---")
print(state_error)

print("\n--- Phi Error ---")
print(phi_error)