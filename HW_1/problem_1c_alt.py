import numpy as np
import sympy as sp

# --- 1. Symbolic Precomputation ---
def precompute_zonal_sph(Re_val, l_max):
    """Generates numerical functions for acceleration and the A matrix (Jacobian)"""
    x, y, z, vx, vy, vz, mu_sym = sp.symbols('x y z vx vy vz mu_sym')
    coeffs = sp.symbols(f'J1:{l_max+1}')
    
    r = sp.sqrt(x**2 + y**2 + z**2)
    phi = sp.asin(z / r)
    
    # Gravitational Potential U
    U = mu_sym / r
    for l in range(1, l_max + 1):
        P_l = sp.legendre(l, sp.sin(phi))
        U -= (mu_sym / r) * coeffs[l-1] * (Re_val / r)**l * P_l
        
    acc_sym = [sp.diff(U, char) for char in (x, y, z)]
    
    # Augmented State: [x, y, z, vx, vy, vz, mu, J1, J2, ... Jl]
    state_vars = [x, y, z, vx, vy, vz, mu_sym] + list(coeffs)
    full_dynamics = [vx, vy, vz] + acc_sym + [0] * (1 + l_max)
    
    A_sym = sp.Matrix(full_dynamics).jacobian(state_vars)
    
    # Lambdafy for numerical evaluation
    acc_func = sp.lambdify((x, y, z, mu_sym, coeffs), acc_sym, 'numpy')
    A_func = sp.lambdify((x, y, z, vx, vy, vz, mu_sym, coeffs), A_sym, 'numpy')
    
    return acc_func, A_func

# --- 2. Setup Constants and Functions ---
R_EARTH = 6378.0
L_MAX = 3 # We need J3, so we go to L=3
acc_f, A_f = precompute_zonal_sph(R_EARTH, L_MAX)

# --- 3. Test Case Data (from your manual script) ---
state_json = np.array([
    -0.64901376519124, 1.18116604196553, -0.75845329728369, # r (km)
    -1.10961303850152, -0.84555124000780, -0.57266486645795, # v (km/s)
    -0.55868076447397, 0.17838022584977, -0.19686144647594   # mu, J2, J3
])

# Re-mapping to match the A_f function signature
r_vec = state_json[0:3]
v_vec = state_json[3:6]
mu_val = state_json[6]
# Note: Symbolic coeffs expects J1, J2, J3. Your JSON gives mu, J2, J3.
# We assume J1 = 0 based on standard zonal models.
coeffs_val = (0, state_json[7], state_json[8]) 

# --- 4. Evaluate the A-Matrix numerically ---
# Signature: A_func(x, y, z, vx, vy, vz, mu, coeffs)
A_matrix = A_f(r_vec[0], r_vec[1], r_vec[2], 
               v_vec[0], v_vec[1], v_vec[2], 
               mu_val, coeffs_val)

# --- 5. Comparison and Error Analysis ---
target_vals = np.array([
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

np.set_printoptions(precision=6, suppress=True, linewidth=150)
print("Symbolic A-Matrix (from zonal_sph logic):")
print(A_matrix)

# Slice for specific block verification
# In the symbolic version, J1 is index 7, J2 is 8, J3 is 9.
# The target_vals script used mu(6), J2(7), J3(8).
# We adjust the slicing to compare relevant columns.

print("\n--- Verification Block (Gravity Gradient G) ---")
G_sym = A_matrix[3:6, 0:3]
print(G_sym)

print("\n--- Sensitivity (S) Block Error vs Target ---")
# Adjusting indices: Symbolic S block for [mu, J2, J3] are cols 6, 8, 9
S_sym = A_matrix[3:6, [6, 8, 9]] 
S_target = target_vals[3:6, 6:9]
error_S = S_sym - S_target

print(f"Max Error in Sensitivity: {np.max(np.abs(error_S))}")



