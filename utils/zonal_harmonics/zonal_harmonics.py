import numpy as np

def get_zonal_jacobian(r_vec, v_vec, mu, coeffs, Re=6378.0):
    """
    State: [x, y, z, vx, vy, vz, mu, J2, J3]
    coeffs[1] = J2, coeffs[2] = J3
    """
    x, y, z = r_vec
    r2 = x*x + y*y + z*z
    r = np.sqrt(r2)
    
    # Fundamental r-powers used by all components
    r3 = r**3
    r5 = r**5
    
    J2, J3 = coeffs[1], coeffs[2]
    
    # Point Mass Gravity Gradient
    G_total = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)
    
    # Point Mass Acceleration
    a_pm = -(mu/r3) * r_vec
    a_total = a_pm.copy()

    # J2 (if non-zero)
    if J2 != 0:
        r7 = r**7
        r9 = r**9
        j2_c = -1.5 * mu * J2 * Re**2
        
        G_j2 = j2_c * np.array([
            [1/r5 - 5*(x**2+z**2)/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9, -15*x*z/r7 + 35*x*z**3/r9],
            [-5*x*y/r7 + 35*x*y*z**2/r9, 1/r5 - 5*(y**2+z**2)/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
            [-15*x*z/r7 + 35*x*z**3/r9, -15*y*z/r7 + 35*y*z**3/r9, 3/r5 - 30*z**2/r7 + 35*z**4/r9]
        ])
        
        a_j2 = j2_c * np.array([
            x/r5 * (1 - 5*z**2/r2), 
            y/r5 * (1 - 5*z**2/r2), 
            z/r5 * (3 - 5*z**2/r2)
        ])
        
        G_total += G_j2
        a_total += a_j2
    else:
        a_j2 = np.zeros(3)

    # J3 (if non-zero)
    if J3 != 0:
        r7 = r**7 if J2 == 0 else r7 # Ensure r7 exists if J2 was skipped
        r9 = r**9 if J2 == 0 else r9
        r11 = r**11
        j3_c = -2.5 * mu * J3 * Re**3
        
        G_j3 = j3_c * np.array([
            [3*z/r7 - 21*x**2*z/r9 - 7*z**3/r9 + 63*x**2*z**3/r11, -21*x*y*z/r9 + 63*x*y*z**3/r11, 3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11],
            [-21*x*y*z/r9 + 63*x*y*z**3/r11, 3*z/r7 - 21*y**2*z/r9 - 7*z**3/r9 + 63*y**2*z**3/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11],
            [3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11, 15*z/r7 - 70*z**3/r9 + 63*z**5/r11]
        ])
        
        a_j3 = j3_c * np.array([
            x/r7 * (3*z - 7*z**3/r2), 
            y/r7 * (3*z - 7*z**3/r2), 
            1/r7 * (6*z**2 - 7*z**4/r2 - 0.6*r2)
        ])
        
        G_total += G_j3
        a_total += a_j3
    else:
        a_j3 = np.zeros(3)

    # Assemble Full 9x9 A-Matrix 
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3)      # dr/dv
    A[3:6, 0:3] = G_total       # da/dr
    
    # Sensitivities (da/d_params)
    A[3:6, 6] = a_total / mu    # wrt mu
    
    # If J2 is zero, the sensitivity to J2 is 0 
    if J2 != 0: A[3:6, 7] = a_j2 / J2
    if J3 != 0: A[3:6, 8] = a_j3 / J3
    
    return A


def zonal_sph_ode(t, state, mu, re, j2, j3):
    r_vec = state[0:3]
    v_vec = state[3:6]
    Phi = state[9:].reshape((9, 9))
    
    r = np.linalg.norm(r_vec)
    x, y, z = r_vec
    
    # Physics
    a_pm = -(mu / r**3) * r_vec
    factor_j2 = 1.5 * j2 * mu * (re**2 / r**5)
    dvdt = a_pm + factor_j2 * np.array([x*(5*(z/r)**2-1), y*(5*(z/r)**2-1), z*(5*(z/r)**2-3)])
    
    # STM Propagation
    A = get_zonal_jacobian(r_vec, v_vec, mu, [0, j2, j3], Re=re)
    Phi_dot = A @ Phi
    
    return np.concatenate([v_vec, dvdt, [0,0,0], Phi_dot.flatten()])