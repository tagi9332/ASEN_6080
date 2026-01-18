# Script to propagate initial conditions with a small perturbation
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from scipy.integrate import solve_ivp

def dfdx_wJ2J3(r_vec, mu, J2, J3, Re=6378.0,use_J2=True,use_J3=True):
    
    x, y, z = r_vec
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2) 
    r3, r5, r7, r9, r11 = r**3, r**5, r**7, r**9, r**11
    
    # Build the 3x3 Gravity Gradient
    # Point Mass
    G = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)
    
    # J2 Contribution if enabled
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

# Build the 3x3 Sensitivity Matrix (S = del_a / del_params)    
    j2_c = -1.5 * mu * J2 * Re**2
    j3_c = -2.5 * mu * J3 * Re**3

    # Point Mass Acceleration 
    a_pm = -(mu / r3) * r_vec
    
    # J2 Acceleration
    vec_j2 = np.array([
        x * (1/r5 - 5*z**2/r7),
        y * (1/r5 - 5*z**2/r7),
        z * (3/r5 - 5*z**2/r7)
    ])
    a_j2 = j2_c * vec_j2
    
    # J3 Acceleration
    vec_j3 = np.array([
        x * (3*z/r7 - 7*z**3/r9),
        y * (3*z/r7 - 7*z**3/r9),
        (6*z**2/r7 - 7*z**4/r9 - 0.6/r5) 
    ])
    a_j3 = j3_c * vec_j3

    # Assemble S Matrix
    S = np.zeros((3, 3))
    
    # Partial wrt mu (Total Accel / mu)
    S[:, 0] = (a_pm + a_j2 + a_j3) / mu
    
    # Partial wrt J2
    S[:, 1] = a_j2 / J2
    
    # Partial wrt J3
    S[:, 2] = a_j3 / J3

    # Assemble the Full 9x9 A-Matrix 
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3) # dr/dv block (zeros)
    A[3:6, 0:3] = G        # dv/dr block (Gravity Gradient)
    A[3:6, 6:9] = S        # dv/dp block (Sensitivity)
    
    return A


# Equations of Motion
def eom_with_stm(t, state_flat, mu, j2, j3, re):
    # Unpack state: [x, y, z, vx, vy, vz, Phi(81 elements flattened)]
    r_vec = state_flat[0:3]
    v_vec = state_flat[3:6]
    phi_flat = state_flat[6:]
    Phi = phi_flat.reshape((9, 9))
    
    r = np.linalg.norm(r_vec)
    
    # Calculate Accelerations
    # Point Mass
    a_vec = -(mu / r**3) * r_vec
    
    # J2 Perturbation
    x, y, z = r_vec
    z_r2 = (z/r)**2
    factor = 1.5 * j2 * mu * (re**2 / r**5)
    a_j2 = factor * np.array([
        x * (5*z_r2 - 1),
        y * (5*z_r2 - 1),
        z * (5*z_r2 - 3)
    ])
    
    # Total Acceleration
    dvdt = a_vec + a_j2
    
    # 2. STM Propagation: dPhi/dt = A * Phi
    A = dfdx_wJ2J3(r_vec, mu, j2, j3, Re=re, use_J3=False)
    Phi_dot = A @ Phi
    
    # Pack it back up
    return np.concatenate([v_vec, dvdt, Phi_dot.flatten()])

# 1. Setup Initial Conditions
r_init, v_init = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
period = 2 * np.pi * np.sqrt(10000**3 / MU_EARTH)
t_span = (0, 15 * period)
t_eval = np.arange(0, t_span[1], 10.0)

# Initial Identity STM (9x9)
phi_init = np.eye(9).flatten()

# --- RUN 1: Reference Trajectory ---
y0_ref = np.concatenate([r_init, v_init, phi_init])
sol_ref = solve_ivp(eom_with_stm, t_span, y0_ref, t_eval=t_eval, 
                    args=(MU_EARTH, J2, J3, R_EARTH), rtol=1e-10, atol=1e-10, method='DOP853')

# --- RUN 2: Perturbed Trajectory ---
# Add small disturbance to initial position and velocity
dr0 = np.array([1.0, 0, 0])      # 1 km shift in X
dv0 = np.array([0, 1e-3, 0])     # 1 m/s shift in Vy
y0_pert = np.concatenate([r_init + dr0, v_init + dv0, phi_init])

sol_pert = solve_ivp(eom_with_stm, t_span, y0_pert, t_eval=t_eval, 
                     args=(MU_EARTH, J2, J3, R_EARTH), rtol=1e-10, atol=1e-10, method='DOP853')

# 2. Compute Differences over Time
# We subtract the reference trajectory from the perturbed trajectory
# Indexing [:6, :] gets X, Y, Z, VX, VY, VZ for all time steps
diff = sol_pert.y[:6, :] - sol_ref.y[:6, :]

# Save differences to csv for analysis
output_path = os.path.join(os.path.dirname(__file__), 'problem_2c_differences.csv')

# Combine time and data
data_to_save = np.column_stack((sol_ref.t, diff.T))

# fmt argument: 
# '%.4f' makes the first column (Time) a decimal with 4 places
# '%.10e' keeps the rest in high-precision scientific notation (standard for small orbital errors)
formats = ['%.4f'] + ['%.10e'] * 6 

np.savetxt(output_path, 
           data_to_save, 
           delimiter=',', 
           header='Time(s),Delta_X(km),Delta_Y(km),Delta_Z(km),Delta_VX(km/s),Delta_VY(km/s),Delta_VZ(km/s)', 
           fmt=formats,
           comments='')

print(f"File saved to: {output_path}")


# 3. Plot Results
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
time_hours = sol_ref.t / 3600.0
axs[0].plot(time_hours, diff[0, :], label='ΔX (km)')
axs[0].plot(time_hours, diff[1, :], label='ΔY (km)')
axs[0].plot(time_hours, diff[2, :], label='ΔZ (km)')
axs[0].set_title('Position Differences Over Time')
axs[0].set_xlabel('Time (hours)')
axs[0].set_ylabel('Position Difference (km)')
axs[0].legend()
axs[0].grid()
axs[1].plot(time_hours, diff[3, :]*1000, label='ΔVX (m/s)')
axs[1].plot(time_hours, diff[4, :]*1000, label='ΔVY (m/s)')
axs[1].plot(time_hours, diff[5, :]*1000, label='ΔVZ (m/s)')
axs[1].set_title('Velocity Differences Over Time')
axs[1].set_xlabel('Time (hours)')
axs[1].set_ylabel('Velocity Difference (m/s)')
axs[1].legend()
axs[1].grid()
plt.tight_layout()
plt.show()

