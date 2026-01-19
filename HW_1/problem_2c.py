import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Analytic Jacobian for Zonal Harmonics
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

# Helper Functions
def orbital_elements_to_cartesian(mu, a, e, i, raan, omega, nu):
    p = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(nu))
    v_mag = np.sqrt(mu / p)
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0])
    v_pqw = v_mag * np.array([-np.sin(nu), e + np.cos(nu), 0])
    
    def rot_z(ang): return np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    def rot_x(ang): return np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    
    R = rot_z(raan) @ rot_x(i) @ rot_z(omega)
    return R @ r_pqw, R @ v_pqw

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

# Main Simulation
R_EARTH = 6378.0
MU = 398600.4415
J2 = 1.0826269e-3
J3 = 0

r0, v0 = orbital_elements_to_cartesian(MU, 1e4, 0.001, np.radians(40), np.radians(80), np.radians(40), 0)
X0_ref = np.concatenate([r0, v0, [MU, J2, J3]])
state0 = np.concatenate([X0_ref, np.eye(9).flatten()])

# Perturbation
dx0 = np.zeros(9)
dx0[0], dx0[4] = 1.0, 0.01 # 1 km in x, 0.01 km/s in vy
state_perturbed0 = np.concatenate([X0_ref + dx0, np.eye(9).flatten()])

t_span = (0, 15 * 2 * np.pi * np.sqrt(1e4**3 / MU))
t_eval = np.linspace(0, t_span[1], 1000)

sol_ref = solve_ivp(zonal_sph_ode, t_span, state0, t_eval=t_eval, args=(MU, R_EARTH, J2, J3), rtol=1e-10, atol=1e-10)
sol_pert = solve_ivp(zonal_sph_ode, t_span, state_perturbed0, t_eval=t_eval, args=(MU, R_EARTH, J2, J3), rtol=1e-10, atol=1e-10)

# Plotting
delta_x_true = sol_pert.y[:6, :].T - sol_ref.y[:6, :].T
delta_x_stm = np.zeros((len(t_eval), 6))
for k in range(len(t_eval)):
    Phi_t = sol_ref.y[9:, k].reshape((9, 9))
    delta_x_stm[k, :] = Phi_t[:6, :6] @ dx0[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
t_hrs = t_eval / 3600
colors = ['royalblue', 'seagreen', 'indianred']
labels = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']

for i in range(6):
    ax = axes[i//3, i%3]
    scale = 1000 if i >= 3 else 1 # Convert vel to m/s
    ax.plot(t_hrs, delta_x_true[:, i] * scale, color=colors[i%3], label='Truth')
    ax.plot(t_hrs, delta_x_stm[:, i] * scale, 'k--', alpha=0.7, label='STM')
    ax.set_title(f'Dev: {labels[i]}')
    if i == 0: ax.set_ylabel('km')
    if i == 3: ax.set_ylabel('m/s')
    ax.legend()

plt.tight_layout()
plt.show()

# Plot residuals
residuals = delta_x_true - delta_x_stm

# Create a 2x3 grid for Residuals
fig_res, axes_res = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
fig_res.suptitle('Residuals: Nonlinear Truth minus STM Prediction', fontsize=16)

for i in range(6):
    ax = axes_res[i//3, i%3]
    scale = 1000 if i >= 3 else 1  # Convert velocity to m/s
    unit = "m/s" if i >= 3 else "km"
    
    ax.plot(t_hrs, residuals[:, i] * scale, color='crimson', linewidth=1.5)
    ax.set_title(f'Residual {labels[i]}')
    ax.grid(True, alpha=0.3)
    
    if i % 3 == 0:
        ax.set_ylabel(f'Error ({unit})')
    if i >= 3:
        ax.set_xlabel('Time (hours)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()