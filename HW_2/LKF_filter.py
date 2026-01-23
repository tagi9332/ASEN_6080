import os, sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Note: Ensure these paths and constants match your local environment
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH

# Ground Stations (lat, lon) [rad]
stations_ll = np.deg2rad([
    [-35.398333, 148.981944], # Station 1 (Canberra, Australia)
    [ 40.427222, 355.749444], # Station 2 (Fort Davis, USA)
    [ 35.247164, 243.205000]  # Station 3 (Madrid, Spain)
])

# ============================================================
# Dynamics & Jacobian (Strictly 6x6)
# ============================================================
def get_zonal_jacobian(r_vec, params):
    mu, j2_val, _ = params  # J3 not used in this specific G matrix for simplicity
    x, y, z = r_vec
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2)
    r3, r5, r7 = r**3, r**5, r**7

    # Point-mass gravity gradient
    G = -(mu / r3) * np.eye(3) + (3 * mu / r5) * np.outer(r_vec, r_vec)

    # J2 Contribution to Gravity Gradient
    if j2_val != 0.0:
        j2c = -1.5 * mu * j2_val * R_EARTH**2
        G_j2 = j2c * np.array([
            [1/r5 - 5*(x*x+z*z)/r7, -5*x*y/r7,           -15*x*z/r7],
            [-5*x*y/r7,           1/r5 - 5*(y*y+z*z)/r7, -15*y*z/r7],
            [-15*x*z/r7,          -15*y*z/r7,            3/r5 - 30*z*z/r7]
        ])
        G += G_j2

    # Assemble 6x6 A matrix
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = G
    return A

def zonal_sph_ode(t, state):
    r = state[0:3]
    v = state[3:6]
    Phi = state[6:].reshape(6, 6)
    
    rnorm = np.linalg.norm(r)
    
    # Acceleration (Point Mass + J2)
    a_pm = -(MU_EARTH / rnorm**3) * r
    factor_j2 = 1.5 * J2 * MU_EARTH * (R_EARTH**2 / rnorm**5)
    a_j2 = factor_j2 * np.array([
        r[0]*(5*(r[2]/rnorm)**2 - 1),
        r[1]*(5*(r[2]/rnorm)**2 - 1),
        r[2]*(5*(r[2]/rnorm)**2 - 3)
    ])
    a = a_pm + a_j2

    # STM Dynamics (6x6)
    A = get_zonal_jacobian(r, [MU_EARTH, J2, J3])
    Phi_dot = A @ Phi

    return np.concatenate([v, a, Phi_dot.flatten()])

# ============================================================
# Measurement & Station Models
# ============================================================
def get_gs_eci_state(lat, lon, time, init_theta=122):
    omega_earth = 7.2921159e-5  
    theta_total = init_theta + (omega_earth * time) + lon
    cos_lat = np.cos(lat)
    
    Rs = np.array([
        R_EARTH * cos_lat * np.cos(theta_total),
        R_EARTH * cos_lat * np.sin(theta_total),
        R_EARTH * np.sin(lat)
    ])
    Vs = np.array([-omega_earth * Rs[1], omega_earth * Rs[0], 0.0])
    return Rs, Vs

def compute_H_matrix(R, V, Rs, Vs):
    rho_vec = R - Rs
    v_rel = V - Vs
    rho = np.linalg.norm(rho_vec)
    rho_dot = np.dot(rho_vec, v_rel) / rho

    d_rho_dR = rho_vec / rho
    d_rhodot_dR = (v_rel - (rho_dot * d_rho_dR)) / rho
    d_rhodot_dV = d_rho_dR

    H = np.zeros((2, 6))
    H[0, 0:3] = d_rho_dR
    H[1, 0:3] = d_rhodot_dR
    H[1, 3:6] = d_rhodot_dV
    return np.array([rho, rho_dot]), H

# ============================================================
# Linearized Kalman Filter
# ============================================================
def run_lkf(sol, meas_df, P0, Rk):
    # State deviation vector (6x1)
    x = np.array([0.5, -0.3, 0.8, 0.3e-3, -0.5e-3, 0.2e-3]) 
    P = P0.copy()

    x_hist = [x.copy()]
    P_hist = [P.copy()]
    innovations = np.zeros((len(sol.t), 2))
    prefit_residuals = np.zeros((len(sol.t), 2))

    for k in range(len(sol.t)):
            # 1. Extract STM for this step
            Phi_k = sol.y[6:, k].reshape(6, 6)
            
            # 2. Prediction (Time Update)
            x_pred = Phi_k @ x_hist[0] 
            P_pred = Phi_k @ P0 @ Phi_k.T

            # 3. GET THE ROW DATA (This was missing or out of order)
            row = meas_df.iloc[k] 
            
            # 4. Now you can use 'row' to get the station index
            station_idx = int(row['Station_ID']) - 1
            
            # 5. Get station state and compute H
            Rs, Vs = get_gs_eci_state(stations_ll[station_idx][0], 
                                    stations_ll[station_idx][1], 
                                    sol.t[k])

            # H-matrix and Reference Obs
            y_ref, H = compute_H_matrix(sol.y[0:3, k], sol.y[3:6, k], Rs, Vs)
            y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])

            # Innovations and Residuals
            innovation = y_obs - y_ref
            residual = innovation - H @ x_pred # Pre-fit residual

            innovations[k] = innovation
            prefit_residuals[k] = residual

            # Measurement Update
            S = H @ P_pred @ H.T + Rk
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ residual
            P = (np.eye(6) - K @ H) @ P_pred

            x_hist.append(x.copy())
            P_hist.append(P.copy())

    return np.array(x_hist), np.array(P_hist), innovations, prefit_residuals

# ============================================================
# Main Execution
# ============================================================
# Initial Reference State
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
state0 = np.concatenate([r0, v0, Phi0])

df_meas = pd.read_csv(r'C:\Users\tagi9332\OneDrive - UCB-O365\Documents\ASEN_6080\HW_2\measurements_truth.csv')
time_eval = df_meas['Time(s)'].values

print("Integrating 6x6 Reference Trajectory...")
sol = solve_ivp(zonal_sph_ode, (0, time_eval[-1]), state0, t_eval=time_eval, rtol=1e-10, atol=1e-10)

P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])
Rk = np.diag([1e-6, 1e-12])

x_hist, P_hist, innovations, prefit_residuals = run_lkf(sol, df_meas, P0, Rk)

# ============================================================
# Plotting (State Deviations)
# ============================================================
times = sol.t
sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
state_labels = ['x', 'y', 'z', 'vx', 'vy', 'vz']

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()
for i in range(6):
    axes[i].plot(times, x_hist[1:, i], 'b', label='Estimate')
    axes[i].plot(times, 3*sigmas[1:, i], 'r--', label=r'3$\sigma$')
    axes[i].plot(times, -3*sigmas[1:, i], 'r--', label=r'-3$\sigma$')
    axes[i].set_ylabel(state_labels[i])
    axes[i].grid(True)
plt.tight_layout()
plt.show()