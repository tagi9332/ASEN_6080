import numpy as np
from resources.constants import R_EARTH, OMEGA_EARTH, H
from utils.drag_model.compute_drag_accel import get_drag_acceleration
from utils.drag_model.compute_atm_rho import compute_atm_rho

def skew_symmetric(v):
    """Computes the skew-symmetric matrix [v x]"""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0]
    ])

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

def stm_eom_mu_j2_drag(t, x_phi_augmented):
    """
    Equation of motion for the State AND the State Transition Matrix (STM).
    
    Inputs:
        t (float): Current time
        x_phi_augmented (np.array): (18 + 18^2) x 1 vector.
                                    Contains State[18] stacked with Flat_STM[324].
    
    Outputs:
        d_x_phi (np.array): Derivative vector of the same size.
    """
    
    # Unpack the Augmented Vector
    n_state = 18
    state = x_phi_augmented[:n_state]
    phi_flat = x_phi_augmented[n_state:]
    
    # Reshape STM to 18x18 Matrix
    phi_mat = phi_flat.reshape((n_state, n_state))
    
    # Compute State Derivatives (dX)
    d_state = orbit_eom_mu_j2_drag(t, state)
    
    # Compute STM Derivatives (dPhi)
    # Get the Jacobian Matrix A (18x18)
    A = compute_jacobian_18x18(state)
    
    # dPhi/dt = A * Phi
    d_phi_mat = A @ phi_mat
    
    # Flatten back to vector
    d_phi_flat = d_phi_mat.flatten()
    
    # Concatenate
    d_x_phi = np.concatenate([d_state, d_phi_flat])
    
    return d_x_phi


def orbit_eom_mu_j2_drag(t, state, area=3.0, mass=970.0):
    """
    Computes EOM in METERS.
    Matches C++ force models for Central Gravity, J2, and Drag.
    """
    # 1. Unpack State
    r_vec = state[0:3]
    v_vec = state[3:6]
    mu = state[6]
    J2 = state[7]
    Cd = state[8]
    
    # Ground Stations (Pass-through dynamics)
    rs_1 = state[9:12]
    rs_2 = state[12:15]
    rs_3 = state[15:18]

    # 2. Geometry
    r_norm = np.linalg.norm(r_vec)
    r2 = r_norm**2
    r3 = r_norm**3
    
    # 3. Earth Rotation Vector & Relative Velocity
    w_vec = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel = v_vec - np.cross(w_vec, r_vec)
    v_rel_norm = np.linalg.norm(v_rel)

    # 4. Forces
    
    # A. Central Gravity: -mu/r^3 * r
    a_mu = -mu / r3 * r_vec
    
    # B. J2 Perturbation
    # Matches standard J2 formulation used in C++ j2::dr derivation
    # a_J2_x = - (3/2) J2 (mu/r^2) (R/r)^2 * (x/r) * (1 - 5(z/r)^2)
    # This vector form is equivalent:
    c_J2 = (3.0/2.0) * J2 * (mu / r2) * (R_EARTH / r_norm)**2
    z_sq_ratio = (r_vec[2] / r_norm)**2
    
    rj2_x = (r_vec[0]/r_norm) * (5 * z_sq_ratio - 1)
    rj2_y = (r_vec[1]/r_norm) * (5 * z_sq_ratio - 1)
    rj2_z = (r_vec[2]/r_norm) * (5 * z_sq_ratio - 3)
    
    a_j2 = c_J2 * np.array([rj2_x, rj2_y, rj2_z])

    # C. Atmospheric Drag
    rho = compute_atm_rho(r_norm)
    # alpha matches C++: -0.5 * Cd * (A/m)
    alpha = -0.5 * Cd * (area / mass)
    a_drag = alpha * rho * v_rel_norm * v_rel

    # 5. Total Acceleration
    a_total = a_mu + a_j2 + a_drag

    # 6. Station Dynamics (Rotation)
    rs_1_dot = np.cross(w_vec, rs_1)
    rs_2_dot = np.cross(w_vec, rs_2)
    rs_3_dot = np.cross(w_vec, rs_3)

    # 7. Assemble
    d_state = np.zeros(18)
    d_state[0:3] = v_vec
    d_state[3:6] = a_total
    d_state[6:9] = 0.0 # Parameters are constant
    d_state[9:12]  = rs_1_dot
    d_state[12:15] = rs_2_dot
    d_state[15:18] = rs_3_dot
    
    return d_state

def compute_jacobian_18x18(state):
    """
    Computes 18x18 Jacobian matching C++ implementation.
    Vectorized for exact mathematical consistency.
    """
    # --- 0. Constants & Setup ---
    r_vec = state[0:3]
    v_vec = state[3:6]
    mu = state[6]
    J2 = state[7]
    Cd = state[8]

    # Physics Params
    area = 3.0
    mass = 970.0
    R = R_EARTH
    omega_scalar = OMEGA_EARTH
    
    # Vectors & Norms
    r = np.linalg.norm(r_vec)
    r2 = r*r
    r3 = r*r*r
    
    w_vec = np.array([0.0, 0.0, omega_scalar])
    OmegaX = skew_symmetric(w_vec) # [wx] matrix
    
    # Relative Velocity: u = v - w x r
    u = v_vec - OmegaX @ r_vec
    u_norm = np.linalg.norm(u)
    
    # Atmosphere
    rho = compute_atm_rho(r)
    # Gradient of rho (Assuming exponential model: grad_rho = -rho/H * r_hat)
    if H > 0:
        grad_rho = (-rho / H) * (r_vec / r)
    else:
        grad_rho = np.zeros(3)

    # --- 1. da/dr Terms ---
    
    # 1.1 Central Gravity da/dr (Matches jacobians::mu::dr)
    # -mu/r^3 * (I - 3 * r * r^T / r^2)
    I3 = np.eye(3)
    dr_mu = (-mu / r3) * (I3 - 3.0 * np.outer(r_vec, r_vec) / r2)

    # 1.2 J2 Gravity da/dr (Matches jacobians::j2::dr)
    # Transliteration of C++ code
    x, y, z = r_vec
    r7 = r**7
    inv_r7 = 1.0 / r7
    R2 = R * R
    
    # k = (3/2) * (mu * inv_r7 * R2) * (-J2)
    k_j2 = 1.5 * (mu * inv_r7 * R2) * (-J2)
    
    # Matrix m construction
    x2, y2, z2 = x*x, y*y, z*z
    m = np.zeros((3,3))
    m[0,0] = (r2 - 5*z2) + 2*x2
    m[0,1] = 2*x*y
    m[0,2] = -8*x*z
    m[1,0] = 2*x*y
    m[1,1] = (r2 - 5*z2) + 2*y2
    m[1,2] = -8*y*z
    m[2,0] = 6*x*z
    m[2,1] = 6*y*z
    m[2,2] = (3*r2 - 5*z2) - 4*z2
    
    # Vector v construction
    v_vec_j2 = np.array([
        x * (r2 - 5*z2),
        y * (r2 - 5*z2),
        z * (3*r2 - 5*z2)
    ])
    
    # J2 Jacobian assembly
    dr_j2 = k_j2 * (m + np.outer(v_vec_j2 * (-7.0/r2), r_vec))

    # 1.3 Drag da/dr (Matches jacobians::drag::dr)
    alpha = -0.5 * Cd * area / mass
    
    # Term 1: Density Gradient -> (||u|| u) * grad_rho^T
    term_rho = np.outer(u_norm * u, grad_rho)
    
    # Term 2: Velocity Variation -> rho * d(||u||u)/du * du/dr
    # d(||u||u)/du = (u u^T)/||u|| + ||u|| I
    d_su_du = (np.outer(u, u) / u_norm) + (u_norm * I3)
    
    # du/dr = -OmegaX (The missing physics term in original Python code)
    du_dr = -OmegaX 
    
    term_u = rho * (d_su_du @ du_dr)
    
    dr_drag = alpha * (term_rho + term_u)

    # Combine all da/dr
    da_dr = dr_mu + dr_j2 + dr_drag


    # --- 2. da/dv Terms ---
    # Gravity has no velocity dependence.
    # Drag da/dv (Matches jacobians::drag::dv)
    # alpha * rho * ( (u u^T)/s + s I )
    da_dv = alpha * rho * ((np.outer(u, u) / u_norm) + (u_norm * I3))


    # --- 3. Parameter Sensitivities ---
    
    # 3.1 da/dMu (Matches jacobians::dmu)
    # (1/mu) * a_grav_total. 
    # Note: Logic assumes drag does not depend on mu, which is true.
    # Re-calculate gravity acceleration for this term:
    a_grav_mu = -mu/r3 * r_vec
    
    # Re-calculate J2 acceleration vector for this term:
    # We can reuse the J2 Jacobian logic or just compute acceleration directly.
    # Using direct acceleration formula for clarity:
    # a_j2 = -3/2 * J2 * mu * R^2 / r^5 * [vector]
    inv_r5 = 1.0 / (r2 * r3)
    c_j2_acc = -1.5 * J2 * mu * R2 * inv_r5
    acc_j2_vec = c_j2_acc * np.array([
        x * (1 - 5 * z2/r2),
        y * (1 - 5 * z2/r2),
        z * (3 - 5 * z2/r2)
    ])
    
    da_dmu = (1.0 / mu) * (a_grav_mu + acc_j2_vec)

    # 3.2 da/dJ2 (Matches jacobians::dJ2)
    k_dJ2 = -1.5 * (mu * inv_r7 * R2) * r2 # Note: C++ uses inv_r7, then multiplies terms approx r2
    # C++ dJ2: k * [x(r2-5z2), y(r2-5z2), z(3r2-5z2)]
    # where k = -1.5 * mu * inv_r7 * R2
    k_sens_j2 = -1.5 * mu * inv_r7 * R2
    da_dJ2 = k_sens_j2 * np.array([
        x * (r2 - 5*z2),
        y * (r2 - 5*z2),
        z * (3*r2 - 5*z2)
    ])

    # 3.3 da/dCd (Matches jacobians::dcd)
    # a_drag / Cd
    da_dCd = (1.0 / Cd) * (alpha * rho * u_norm * u)


    # --- 4. Assemble 18x18 Matrix ---
    A = np.zeros((18, 18))

    # Top Right: dr/dv
    A[0:3, 3:6] = I3

    # Row 3-5: Accelerations
    A[3:6, 0:3] = da_dr   # da/dr
    A[3:6, 3:6] = da_dv   # da/dv
    A[3:6, 6]   = da_dmu  # da/dmu
    A[3:6, 7]   = da_dJ2  # da/dJ2
    A[3:6, 8]   = da_dCd  # da/dCd

    # Station Dynamics (indices 9-17)
    # d(r_gs)/dt = Omega x r_gs => Matrix form: OmegaX @ r_gs
    # The Jacobian block for d(r_gs_dot)/d(r_gs) is OmegaX
    A[9:12,  9:12]  = OmegaX
    A[12:15, 12:15] = OmegaX
    A[15:18, 15:18] = OmegaX

    return A