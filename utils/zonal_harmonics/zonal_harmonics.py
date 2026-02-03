import numpy as np
from resources.constants import R_EARTH, OMEGA_EARTH, H
from utils.drag_model.compute_drag_accel import get_drag_acceleration
from utils.drag_model.compute_atm_rho import compute_atm_rho

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


def orbit_eom_mu_j2_drag(t, state, area=3.0, mass=970.0):
    """
    Computes EOM in METERS.
    state: 18-element vector [pos, vel, mu, J2, Cd, stations...]
    """
    
    # Extract State
    x, y, z = state[0:3]
    vx, vy, vz = state[3:6]
    mu = state[6]
    j2 = state[7]
    cd = state[8]
    
    # Ground Stations
    rs_1 = state[9:12]
    rs_2 = state[12:15]
    rs_3 = state[15:18]

    # Derived Quantities
    r_sq = x**2 + y**2 + z**2
    r_mag = np.sqrt(r_sq)
    v_sq = vx**2 + vy**2 + vz**2
    v_mag = np.sqrt(v_sq)
    
    # Units: [kg/m^3]
    rho = compute_atm_rho(r_mag)
    
    # Drag Ballistic Term
    # [m^2 / kg]
    drag_b_term = (cd * area) / mass 
    
    # Calculate Accelerations
    
    # Common J2 Terms
    ri_r_sq = (R_EARTH / r_mag)**2
    z_r_sq = (z / r_mag)**2
    
    j2_factor_x_y = 1 + ri_r_sq * j2 * (7.5 * z_r_sq - 1.5)
    j2_factor_z   = 1 + ri_r_sq * j2 * (7.5 * z_r_sq - 4.5)
    
    mu_r3 = mu / (r_mag**3)
    
    # Drag Force per unit mass = 0.5 * rho * v^2 * (Cd*A/m) * (v_vec / v)
    # Simplifies to: 0.5 * rho * v * (Cd*A/m) * v_vec
    # Units: [kg/m^3] * [m/s] * [m^2/kg] * [m/s] = [m/s^2] -> OK
    drag_factor = 0.5 * rho * drag_b_term * v_mag
    
    ax = -(mu_r3 * x) * j2_factor_x_y - drag_factor * vx
    ay = -(mu_r3 * y) * j2_factor_x_y - drag_factor * vy
    az = -(mu_r3 * z) * j2_factor_z   - drag_factor * vz

    # Calculate station velocities (Earth Rotation)
    w_earth = np.array([0, 0, OMEGA_EARTH])
    
    rs_1_dot = np.cross(w_earth, rs_1)
    rs_2_dot = np.cross(w_earth, rs_2)
    rs_3_dot = np.cross(w_earth, rs_3)

    # Assemble derivative vector
    d_state = np.zeros(18)
    d_state[0:3] = [vx, vy, vz]
    d_state[3:6] = [ax, ay, az]
    d_state[6:9] = 0.0 # Parameters (mu, J2, Cd) are constant
    d_state[9:12]  = rs_1_dot
    d_state[12:15] = rs_2_dot
    d_state[15:18] = rs_3_dot
    
    return d_state

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

def compute_jacobian_18x18(state):
    """
    Computes 18x18 Jacobian for Orbital Dynamics in METERS.
    State: [x, y, z, vx, vy, vz, mu, J2, Cd, GS1(3), GS2(3), GS3(3)]
    """
    # 1. Extract State
    x, y, z = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]
    mu = state[6]
    J2_curr = state[7]
    Cd = state[8]

    # 2. Constants & Spacecraft Physical Params
    Ri = R_EARTH         # [m]
    Ri2 = Ri**2
    Asc = 3.0            # [m^2]
    m_sat = 970.0        # [kg]

    # 3. Derived Quantities
    r_sq = x**2 + y**2 + z**2
    r = np.sqrt(r_sq)
    
    v_sq = vx**2 + vy**2 + vz**2
    v = np.sqrt(v_sq)

    # Compute Atmospheric Density [kg/m^3]
    # NOTE: Ensure this function accepts Meters and returns kg/m^3
    rho = compute_atm_rho(r) 

    # Ballistic Coefficient (1/2 * Cd * A / m) [m^2/kg]
    B_star = 0.5 * (Cd * Asc / m_sat)
    
    # Pre-compute common terms
    # Partial of density w.r.t position (assuming exponential model d_rho/dr = -rho/H)
    # Result units: [(kg/m^3) * (1/m)] = [kg/m^4]
    if H > 0:
        d_rho_dr = -rho / H 
    else:
        d_rho_dr = 0.0

    # Common Drag Gradient Term: B_star * d_rho/dr * v
    atmos_grad_pre = B_star * d_rho_dr * v 
    
    # --- BLOCK 1: d(Acc)/d(Pos) ---
    r5, r7, r9 = r**5, r**7, r**9
    
    # X-Row Gradients
    # Gravity (Point Mass + J2)
    term_grav_x = -mu * ( 
        ((r_sq - 3*x**2)/r5) + 
        ((Ri2 * J2_curr) / 2) * ( (15*(x**2 + z**2)/r7) - ((105*x**2*z**2)/r9) - (3/r5) ) 
    )
    # Drag: d(a_drag)/dx = d(a)/d(rho) * d(rho)/dr * dr/dx
    term_drag_x = atmos_grad_pre * vx * (x/r)
    delXddDelX = term_grav_x + term_drag_x

    # X-Y Cross Terms
    term_grav_xy = mu * ( 
        ((3*x*y)/r5) - 
        ((Ri2 * J2_curr * x * y) / 2) * ((15/r7) - ((105*z**2)/r9)) 
    )
    term_drag_xy = atmos_grad_pre * vx * (y/r)
    delXddDelY = term_grav_xy + term_drag_xy

    # X-Z Cross Terms
    term_grav_xz = mu * ( 
        ((3*x*z)/r5) - 
        ((Ri2 * J2_curr * x * z) / 2) * ((45/r7) - ((105*z**2)/r9)) 
    )
    term_drag_xz = atmos_grad_pre * vx * (z/r)
    delXddDelZ = term_grav_xz + term_drag_xz

    # Y-Row Gradients
    delYddDelX = term_grav_xy + atmos_grad_pre * vy * (x/r)
    
    term_grav_y = -mu * ( 
        ((r_sq - 3*y**2)/r5) + 
        ((Ri2 * J2_curr) / 2) * ( (15*(y**2 + z**2)/r7) - ((105*y**2*z**2)/r9) - (3/r5) ) 
    )
    term_drag_y = atmos_grad_pre * vy * (y/r)
    delYddDelY = term_grav_y + term_drag_y

    term_grav_yz = mu * ( 
        ((3*y*z)/r5) - 
        ((Ri2 * J2_curr * y * z) / 2) * ((45/r7) - ((105*z**2)/r9)) 
    )
    term_drag_yz = atmos_grad_pre * vy * (z/r)
    delYddDelZ = term_grav_yz + term_drag_yz

    # Z-Row Gradients
    delZddDelX = term_grav_xz + atmos_grad_pre * vz * (x/r)
    delZddDelY = term_grav_yz + atmos_grad_pre * vz * (y/r)
    
    term_grav_z = -mu * ( 
        ((r_sq - 3*z**2)/r5) + 
        ((Ri2 * J2_curr) / 2) * ( (90*z**2/r7) - ((105*z**4)/r9) - (9/r5) ) 
    )
    term_drag_z = atmos_grad_pre * vz * (z/r)
    delZddDelZ = term_grav_z + term_drag_z

    # --- BLOCK 2: d(Acc)/d(Vel) ---
    # a_drag = -B_star * rho * v * v_vec
    # This requires derivative of (v * v_vec)
    C = -B_star * rho 
    
    delXddDelXd = C * ((vx**2 / v) + v)
    delXddDelYd = C * (vx * vy / v)
    delXddDelZd = C * (vx * vz / v)

    delYddDelXd = C * (vy * vx / v)
    delYddDelYd = C * ((vy**2 / v) + v)
    delYddDelZd = C * (vy * vz / v)

    delZddDelXd = C * (vz * vx / v)
    delZddDelYd = C * (vz * vy / v)
    delZddDelZd = C * ((vz**2 / v) + v)

    # --- BLOCK 3: d(Acc)/d(Params) ---
    
    # 1. d/dMu
    r3 = r**3
    z_r = z / r
    # J2 term scaling for Mu partial
    j2_term_mu = (1 + (Ri/r)**2 * J2_curr * (7.5 * z_r**2 - 1.5))
    
    delXddDelMu = (-x / r3) * j2_term_mu
    delYddDelMu = (-y / r3) * j2_term_mu
    delZddDelMu = (-z / r3) * (1 + (Ri/r)**2 * J2_curr * (7.5 * z_r**2 - 4.5))
    
    # 2. d/dJ2
    j2_common = (-mu / r3) * (Ri/r)**2
    delXddDelJ2 = j2_common * x * (7.5 * z_r**2 - 1.5)
    delYddDelJ2 = j2_common * y * (7.5 * z_r**2 - 1.5)
    delZddDelJ2 = j2_common * z * (7.5 * z_r**2 - 4.5)
    
    # 3. d/dCd
    # a_drag = -0.5 * (Cd * A / m) * rho * v * vec(v)
    # d(a)/dCd = a_drag / Cd
    # In METERS, no unit conversion needed.
    cd_common = -0.5 * (Asc / m_sat) * rho * v
    delXddDelCd = cd_common * vx
    delYddDelCd = cd_common * vy
    delZddDelCd = cd_common * vz

    # --- ASSEMBLE MATRIX ---
    A = np.zeros((18, 18))
    
    # Identity for Position -> Velocity
    A[0, 3] = 1.0; A[1, 4] = 1.0; A[2, 5] = 1.0

    # Dynamics Gradients (Acc / Pos)
    A[3,0]=delXddDelX; A[3,1]=delXddDelY; A[3,2]=delXddDelZ
    A[4,0]=delYddDelX; A[4,1]=delYddDelY; A[4,2]=delYddDelZ
    A[5,0]=delZddDelX; A[5,1]=delZddDelY; A[5,2]=delZddDelZ
    
    # Dynamics Gradients (Acc / Vel)
    A[3,3]=delXddDelXd; A[3,4]=delXddDelYd; A[3,5]=delXddDelZd
    A[4,3]=delYddDelXd; A[4,4]=delYddDelYd; A[4,5]=delYddDelZd
    A[5,3]=delZddDelXd; A[5,4]=delZddDelYd; A[5,5]=delZddDelZd
    
    # Parameter Sensitivities (Acc / Params)
    # Col 6: Mu, Col 7: J2, Col 8: Cd
    A[3,6]=delXddDelMu; A[3,7]=delXddDelJ2; A[3,8]=delXddDelCd
    A[4,6]=delYddDelMu; A[4,7]=delYddDelJ2; A[4,8]=delYddDelCd
    A[5,6]=delZddDelMu; A[5,7]=delZddDelJ2; A[5,8]=delZddDelCd
    
    # Station Rotation (Earth Fixed Frame to Inertial Frame derivative)
    # d(r_gs)/dt = Omega x r_gs
    w = OMEGA_EARTH
    
    # GS1 (Indices 9,10,11)
    A[9, 10] = -w; A[10, 9] = w
    # GS2 (Indices 12,13,14)
    A[12, 13] = -w; A[13, 12] = w
    # GS3 (Indices 15,16,17)
    A[15, 16] = -w; A[16, 15] = w

    return A