import numpy as np
from resources.constants import R_EARTH

def zonal_jacobian_nxn(r_vec, v_vec, coeffs):
    """
    Computes an nxn Jacobian matrix based on active gravity coefficients.
    coeffs = [mu, j2, j3]
    """
    re = R_EARTH
    mu, j2, j3 = coeffs
    
    # 1. Determine dimension (n) and parameter mapping
    # Base: Pos(3) + Vel(3) + Mu(1) = 7
    n = 7
    j2_col = None
    j3_col = None
    
    if j2 != 0:
        j2_col = n
        n += 1
    if j3 != 0:
        j3_col = n
        n += 1

    x, y, z = r_vec
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    r3, r5, r7, r9 = r**3, r**5, r**7, r**9

    # 2. Gravity Gradients
    # Point Mass
    G_total = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)
    a_total = -(mu/r3) * r_vec # Accumulator for d(accel)/d(mu)

    # J2 Gradient and Sensitivity
    a_j2 = np.zeros(3)
    if j2 != 0:
        j2_c = -1.5 * mu * j2 * re**2
        G_total += j2_c * np.array([
            [1/r5 - 5*(x**2+z**2)/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9, -15*x*z/r7 + 35*x*z**3/r9],
            [-5*x*y/r7 + 35*x*y*z**2/r9, 1/r5 - 5*(y**2+z**2)/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
            [-15*x*z/r7 + 35*x*z**3/r9, -15*y*z/r7 + 35*y*z**3/r9, 3/r5 - 30*z**2/r7 + 35*z**4/r9]
        ])
        a_j2 = j2_c * np.array([x/r5*(1-5*z**2/r2), y/r5*(1-5*z**2/r2), z/r5*(3-5*z**2/r2)])
        a_total += a_j2

    # J3 Gradient and Sensitivity
    a_j3 = np.zeros(3)
    if j3 != 0:
        r11 = r**11
        j3_c = -2.5 * mu * j3 * re**3
        G_total += j3_c * np.array([
            [3*z/r7 - 21*x**2*z/r9 - 7*z**3/r9 + 63*x**2*z**3/r11, -21*x*y*z/r9 + 63*x*y*z**3/r11, 3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11],
            [-21*x*y*z/r9 + 63*x*y*z**3/r11, 3*z/r7 - 21*y**2*z/r9 - 7*z**3/r9 + 63*y**2*z**3/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11],
            [3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11, 15*z/r7 - 70*z**3/r9 + 63*z**5/r11]
        ])
        a_j3 = j3_c * np.array([x/r7*(3*z-7*z**3/r2), y/r7*(3*z-7*z**3/r2), 1/r7*(6*z**2-7*z**4/r2-0.6*r2)])
        a_total += a_j3

    # 3. Assemble Matrix
    A = np.zeros((n, n))
    A[0:3, 3:6] = np.eye(3)   # dr/dv
    A[3:6, 0:3] = G_total    # dv/dr (Gravity Gradient)
    A[3:6, 6]   = a_total/mu # dv/dmu
    
    if j2_col: A[3:6, j2_col] = a_j2 / j2
    if j3_col: A[3:6, j3_col] = a_j3 / j3
    
    return A


def zonal_sph_ode_nxn(t, state, coeffs):
    """
    ODE that handles dynamic state vector: [Pos(3), Vel(3), Params(1-3), Phi(n*n)]
    """
    re = R_EARTH
    mu, j2, j3 = coeffs
    
    # Determine n based on the same logic as the Jacobian
    n = 7
    if j2 != 0: n += 1
    if j3 != 0: n += 1
    
    # 1. Unpack State
    r_vec = state[0:3]
    v_vec = state[3:6]
    # The STM starts immediately after the parameters
    Phi = state[n:].reshape((n, n))
    
    r = np.linalg.norm(r_vec)
    x, y, z = r_vec
    z_r_sq = (z / r)**2
    
    # 2. Accelerations
    dvdt = -(mu / r**3) * r_vec
    
    if j2 != 0:
        j2_f = 1.5 * j2 * mu * (re**2 / r**5)
        dvdt += j2_f * np.array([x*(5*z_r_sq-1), y*(5*z_r_sq-1), z*(5*z_r_sq-3)])
        
    if j3 != 0:
        j3_f = 0.5 * j3 * mu * (re**3 / r**7)
        dvdt += j3_f * np.array([
            5*x*z*(7*z_r_sq - 3),
            5*y*z*(7*z_r_sq - 3),
            35*z**4/r**2 - 30*z**2 + 3*r**2
        ])

    # 3. STM Propagation
    A = zonal_jacobian_nxn(r_vec, v_vec, coeffs)
    Phi_dot = A @ Phi
    
    # 4. Parameter derivatives (all constant)
    d_params = np.zeros(n - 6)
    
    return np.concatenate([
        v_vec, 
        dvdt, 
        d_params, 
        Phi_dot.flatten()
    ])

def get_zonal_jacobian_6x6(r_vec, v_vec, coeffs):
    """
    Fixed 6x6 Jacobian for [pos, vel].
    Includes J2/J3 in the gravity gradient math but excludes them from the state.
    """
    re = R_EARTH
    mu, j2, j3 = coeffs
    
    x, y, z = r_vec
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    r3, r5, r7, r9 = r**3, r**5, r**7, r**9

    # 1. Start with Point Mass Gradient (3x3)
    G_total = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)

    # 2. Add J2 Gradient contribution to the 3x3 block
    if j2 != 0:
        j2_c = -1.5 * mu * j2 * re**2
        G_total += j2_c * np.array([
            [1/r5 - 5*(x**2+z**2)/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9, -15*x*z/r7 + 35*x*z**3/r9],
            [-5*x*y/r7 + 35*x*y*z**2/r9, 1/r5 - 5*(y**2+z**2)/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
            [-15*x*z/r7 + 35*x*z**3/r9, -15*y*z/r7 + 35*y*z**3/r9, 3/r5 - 30*z**2/r7 + 35*z**4/r9]
        ])

    # 3. Add J3 Gradient contribution to the 3x3 block
    if j3 != 0:
        r11 = r**11
        j3_c = -2.5 * mu * j3 * re**3
        G_total += j3_c * np.array([
            [3*z/r7 - 21*x**2*z/r9 - 7*z**3/r9 + 63*x**2*z**3/r11, -21*x*y*z/r9 + 63*x*y*z**3/r11, 3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11],
            [-21*x*y*z/r9 + 63*x*y*z**3/r11, 3*z/r7 - 21*y**2*z/r9 - 7*z**3/r9 + 63*y**2*z**3/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11],
            [3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11, 15*z/r7 - 70*z**3/r9 + 63*z**5/r11]
        ])

    # 4. Assemble standard 6x6 A matrix
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)   # dr/dv
    A[3:6, 0:3] = G_total    # dv/dr
    
    return A


def zonal_sph_ode_6x6(t, state, coeffs):
    """
    Fixed 6-state ODE: [x, y, z, vx, vy, vz, Phi(36)]
    High-fidelity physics, low-fidelity filter state.
    """
    re = R_EARTH
    mu, j2, j3 = coeffs
    
    # 1. Unpack State (Always 6 + 36 = 42 elements)
    r_vec = state[0:3]
    v_vec = state[3:6]
    Phi = state[6:42].reshape((6, 6))
    
    r = np.linalg.norm(r_vec)
    x, y, z = r_vec
    z_r_sq = (z / r)**2
    
    # 2. Physics: Accelerations (Including J2 and J3)
    dvdt = -(mu / r**3) * r_vec
    
    if j2 != 0:
        j2_f = 1.5 * j2 * mu * (re**2 / r**5)
        dvdt += j2_f * np.array([x*(5*z_r_sq-1), y*(5*z_r_sq-1), z*(5*z_r_sq-3)])
        
    if j3 != 0:
        j3_f = 0.5 * j3 * mu * (re**3 / r**7)
        dvdt += j3_f * np.array([
            5*x*z*(7*z_r_sq - 3),
            5*y*z*(7*z_r_sq - 3),
            35*z**4/r**2 - 30*z**2 + 3*r**2
        ])

    # 3. STM Propagation (6x6)
    A = get_zonal_jacobian_6x6(r_vec, v_vec, coeffs)
    Phi_dot = A @ Phi
    
    return np.concatenate([
        v_vec,             # dr/dt
        dvdt,              # dv/dt
        Phi_dot.flatten()  # dPhi/dt
    ])

import numpy as np

# Assuming R_EARTH is defined globally as in your snippet
# R_EARTH = 6378.137 

def get_zonal_jacobian_dmc(r_vec, v_vec, coeffs):
    """
    Returns 9x9 Jacobian for [pos, vel, acc_dmc].
    Structure:
    [  0   I   0 ]
    [  G   0   I ]
    [  0   0  -B ]
    """
    # Unpack coefficients (Now includes B matrix for DMC)
    mu, j2, j3, B = coeffs
    re = R_EARTH
    
    x, y, z = r_vec
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    r3, r5, r7, r9 = r**3, r**5, r**7, r**9

    # --- 1. Compute Gravity Gradient G (Same as before) ---
    # Point Mass
    G_total = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)

    # J2 Contribution
    if j2 != 0:
        j2_c = -1.5 * mu * j2 * re**2
        G_j2 = np.array([
            [1/r5 - 5*(x**2+z**2)/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9, -15*x*z/r7 + 35*x*z**3/r9],
            [-5*x*y/r7 + 35*x*y*z**2/r9, 1/r5 - 5*(y**2+z**2)/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
            [-15*x*z/r7 + 35*x*z**3/r9, -15*y*z/r7 + 35*y*z**3/r9, 3/r5 - 30*z**2/r7 + 35*z**4/r9]
        ])
        G_total += j2_c * G_j2

    # J3 Contribution
    if j3 != 0:
        r11 = r**11
        j3_c = -2.5 * mu * j3 * re**3
        G_j3 = np.array([
            [3*z/r7 - 21*x**2*z/r9 - 7*z**3/r9 + 63*x**2*z**3/r11, -21*x*y*z/r9 + 63*x*y*z**3/r11, 3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11],
            [-21*x*y*z/r9 + 63*x*y*z**3/r11, 3*z/r7 - 21*y**2*z/r9 - 7*z**3/r9 + 63*y**2*z**3/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11],
            [3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11, 15*z/r7 - 70*z**3/r9 + 63*z**5/r11]
        ])
        G_total += j3_c * G_j3

    # --- 2. Assemble 9x9 A matrix ---
    A = np.zeros((9, 9))
    
    # Block (0,1): Velocity -> Position (Identity)
    A[0:3, 3:6] = np.eye(3)
    
    # Block (1,0): Position -> Velocity (Gravity Gradient)
    A[3:6, 0:3] = G_total
    
    # Block (1,2): DMC Acceleration -> Velocity (Identity)
    # This couples the estimated acceleration into the velocity state
    A[3:6, 6:9] = np.eye(3) 

    # Block (2,2): DMC Acceleration -> DMC Acceleration (-B)
    # This is the Gauss-Markov decay
    A[6:9, 6:9] = -B
    
    return A


def zonal_sph_ode_dmc(t, state, coeffs):
    """
    DMC 9-state ODE: [x, y, z, vx, vy, vz, ax, ay, az, Phi(81)]
    Total elements = 9 + 81 = 90
    """
    re = R_EARTH
    # coeffs must now include B (3x3 diagonal matrix of 1/tau)
    mu, j2, j3, B = coeffs 
    
    # 1. Unpack State
    r_vec = state[0:3]
    v_vec = state[3:6]
    a_dmc = state[6:9] # The 3 new DMC acceleration states
    Phi = state[9:90].reshape((9, 9))
    
    r = np.linalg.norm(r_vec)
    x, y, z = r_vec
    z_r_sq = (z / r)**2
    
    # 2. Physics: Gravity Accelerations
    dvdt_grav = -(mu / r**3) * r_vec
    
    if j2 != 0:
        j2_f = -1.5 * j2 * mu * (re**2 / r**5)
        dvdt_grav += j2_f * np.array([x*(5*z_r_sq-1), y*(5*z_r_sq-1), z*(5*z_r_sq-3)])
        
    if j3 != 0:
        j3_f = 0.5 * j3 * mu * (re**3 / r**7)
        dvdt_grav += j3_f * np.array([
            5*x*z*(7*z_r_sq - 3),
            5*y*z*(7*z_r_sq - 3),
            35*z**4/r**2 - 30*z**2 + 3*r**2
        ])
    
    # --- DMC SPECIFIC UPDATES ---
    
    # Total Acceleration = Gravity + DMC State
    dvdt_total = dvdt_grav + a_dmc
    
    # DMC State Derivative: da/dt = -B * a
    dadMCdt = -B @ a_dmc

    # 3. STM Propagation (9x9)
    A = get_zonal_jacobian_dmc(r_vec, v_vec, coeffs)
    Phi_dot = A @ Phi
    
    return np.concatenate([
        v_vec,          # dr/dt
        dvdt_total,     # dv/dt
        dadMCdt,        # da_dmc/dt
        Phi_dot.flatten() # dPhi/dt
    ])