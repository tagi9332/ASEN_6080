import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. Your Custom OE to Inertial Function ---
def orbital_elements_to_inertial(a, e, i, raan, argp, ta, mu=398600.4415, units='rad'):
    """
    Classical Orbital Elements -> inertial position/velocity.
    """
    if units == 'deg':
        i = np.radians(i)
        raan = np.radians(raan)
        argp = np.radians(argp)
        ta = np.radians(ta)

    if np.isclose(e, 1.0, atol=1e-12):
        raise ValueError("Parabolic case (e close to 1) not supported.")
    
    p = a * (1.0 - e**2)

    cnu, snu = np.cos(ta), np.sin(ta)
    r_pf = (p / (1.0 + e * cnu)) * np.array([cnu, snu, 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-snu, e + cnu, 0.0])

    # Rotation PQW -> IJK
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(argp), np.sin(argp)

    R3_O = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    R1_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    R3_o = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])

    Q_pqw_to_ijk = R3_O @ R1_i @ R3_o

    r = Q_pqw_to_ijk @ r_pf
    v = Q_pqw_to_ijk @ v_pf
    return r, v

# --- 2. Your Analytic Jacobian Function ---
def get_zonal_jacobian(r_vec, v_vec, mu, coeffs, Re=6378.0):
    x, y, z = r_vec
    r2 = x*x + y*y + z*z
    r = np.sqrt(r2)
    r3, r5, r7, r9, r11 = r**3, r**5, r**7, r**9, r**11
    
    # coeffs[1] is J2, coeffs[2] is J3
    J2, J3 = coeffs[1], coeffs[2]
    
    G = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)
    
    j2_c = -1.5 * mu * J2 * Re**2
    G_j2 = j2_c * np.array([
        [1/r5 - 5*(x**2+z**2)/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9, -15*x*z/r7 + 35*x*z**3/r9],
        [-5*x*y/r7 + 35*x*y*z**2/r9, 1/r5 - 5*(y**2+z**2)/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
        [-15*x*z/r7 + 35*x*z**3/r9, -15*y*z/r7 + 35*y*z**3/r9, 3/r5 - 30*z**2/r7 + 35*z**4/r9]
    ])
    
    j3_c = -2.5 * mu * J3 * Re**3
    G_j3 = j3_c * np.array([
        [3*z/r7 - 21*x**2*z/r9 - 7*z**3/r9 + 63*x**2*z**3/r11, -21*x*y*z/r9 + 63*x*y*z**3/r11, 3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11],
        [-21*x*y*z/r9 + 63*x*y*z**3/r11, 3*z/r7 - 21*y**2*z/r9 - 7*z**3/r9 + 63*y**2*z**3/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11],
        [3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11, 15*z/r7 - 70*z**3/r9 + 63*z**5/r11]
    ])
    
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = G + G_j2 + G_j3

    a_pm = -(mu/r3) * r_vec
    a_j2 = j2_c * np.array([x/r5 * (1 - 5*z**2/r2), y/r5 * (1 - 5*z**2/r2), z/r5 * (3 - 5*z**2/r2)])
    a_j3 = j3_c * np.array([x/r7 * (3*z - 7*z**3/r2), y/r7 * (3*z - 7*z**3/r2), 1/r7 * (6*z**2 - 7*z**4/r2 - 0.6*r2)])
    
    A[3:6, 6] = (a_pm + a_j2 + a_j3) / mu  # wrt mu
    A[3:6, 7] = a_j2 / J2                # wrt J2
    A[3:6, 8] = a_j3 / J3                # wrt J3
    return A

# --- 3. Equations of Motion ---
def zonal_sph_ode(t, state, mu, re, j2, j3):
    r_vec = state[0:3]
    v_vec = state[3:6]
    Phi = state[9:].reshape((9, 9))
    
    r = np.linalg.norm(r_vec)
    
    # Dynamics include Point Mass + J2 only (J3=0 in force model)
    a_pm = -(mu / r**3) * r_vec
    factor_j2 = 1.5 * j2 * mu * (re**2 / r**5)
    a_j2 = factor_j2 * np.array([
        r_vec[0]*(5*(r_vec[2]/r)**2 - 1),
        r_vec[1]*(5*(r_vec[2]/r)**2 - 1),
        r_vec[2]*(5*(r_vec[2]/r)**2 - 3)
    ])
    
    dvdt = a_pm + a_j2
    
    # Jacobian for STM (Calculates how J2 and J3 would affect the path)
    A = get_zonal_jacobian(r_vec, v_vec, mu, [0, j2, j3], Re=re)
    Phi_dot = A @ Phi
    
    return np.concatenate([v_vec, dvdt, [0,0,0], Phi_dot.flatten()])

# --- 4. Main Simulation ---
R_EARTH = 6378.0
MU = 398600.4415
J2 = 1.0826269e-3
J3 = -2.532e-6 # Stored as a parameter but not used in dvdt

# Initial Conditions using your function
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, mu=MU, units='deg')
X0_ref = np.concatenate([r0, v0, [MU, J2, J3]])
state0 = np.concatenate([X0_ref, np.eye(9).flatten()])

# Perturbation
dx0 = np.zeros(9)
dx0[0], dx0[4] = 1.0, 0.01 # 1km in X, 10m/s in Vy
state_pert0 = np.concatenate([X0_ref + dx0, np.eye(9).flatten()])

T_orbit = 2 * np.pi * np.sqrt(10000**3 / MU)
t_span = (0, 15 * T_orbit)
t_eval = np.linspace(0, t_span[1], 1000)

sol_ref = solve_ivp(zonal_sph_ode, t_span, state0, t_eval=t_eval, args=(MU, R_EARTH, J2, J3), rtol=1e-11, atol=1e-11)
sol_pert = solve_ivp(zonal_sph_ode, t_span, state_pert0, t_eval=t_eval, args=(MU, R_EARTH, J2, J3), rtol=1e-11, atol=1e-11)

# --- 5. Deviation Analysis & Plotting ---
delta_x_true = sol_pert.y[:6, :].T - sol_ref.y[:6, :].T
delta_x_stm = np.zeros((len(t_eval), 6))
for k in range(len(t_eval)):
    Phi_t = sol_ref.y[9:, k].reshape((9, 9))
    delta_x_stm[k, :] = Phi_t[:6, :6] @ dx0[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
t_hrs = t_eval / 3600
labels = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']

for i in range(6):
    ax = axes[i//3, i%3]
    scale = 1000 if i >= 3 else 1
    ax.plot(t_hrs, delta_x_true[:, i] * scale, 'royalblue', label='Nonlinear Truth')
    ax.plot(t_hrs, delta_x_stm[:, i] * scale, 'k--', alpha=0.8, label='STM Prediction')
    ax.set_title(f'{labels[i]} Deviation')
    ax.legend(fontsize='small')

plt.tight_layout()
plt.show()