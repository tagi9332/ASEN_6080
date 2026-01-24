import os, sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.stats as stats


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
    theta_total = np.deg2rad(init_theta) + (omega_earth * time) + lon
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
def run_lkf(sol, meas_df, dx_0, P0, Rk):
    n = 6 
    dx = dx_0.copy()
    P = P0.copy()
    I = np.eye(n)

    # --- Initialize all storage arrays ---
    dx_hist = []
    P_hist = []
    corrected_state_hist = []
    innovations = []
    prefit_residuals = [] 
    
    Phi_prev = np.eye(n)

    for k in range(len(sol.t)):
        Phi_global = sol.y[6:, k].reshape(n, n)
        
        # Incremental STM: Phi(tk, tk-1)
        Phi_incr = Phi_global @ np.linalg.inv(Phi_prev)
        
        # Prediction Step
        dx_pred = Phi_incr @ dx
        P_pred = Phi_incr @ P @ Phi_incr.T
        
        # Observation Data
        meas_row = meas_df.iloc[k]
        station_idx = int(meas_row['Station_ID']) - 1
        Rs, Vs = get_gs_eci_state(stations_ll[station_idx][0], 
                                 stations_ll[station_idx][1], 
                                 sol.t[k])
        
        x_ref = sol.y[0:6, k]
        y_pred, H = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs)
        y_obs = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])

        # Measurement Update
        prefit_residual = y_obs - y_pred
        innovation = prefit_residual - H @ dx_pred # The "post-prediction" residual

        # 4. Kalman Gain & Update
        S = H @ P_pred @ H.T + Rk
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        dx = dx_pred + K @ innovation
        
        # Covartiance Update (Joseph Form)
        IKH = I - K @ H
        P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T

        # --- Store all results ---
        dx_hist.append(dx.copy())
        P_hist.append(P.copy())
        corrected_state_hist.append(x_ref + dx)
        innovations.append(prefit_residual)
        prefit_residuals.append(innovation) # Store for returning

        Phi_prev = Phi_global

    return (np.array(dx_hist), 
            np.array(P_hist), 
            np.array(corrected_state_hist), 
            np.array(innovations), 
            np.array(prefit_residuals))

# ============================================================
# Main Execution
# ============================================================
# Initial Reference State
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
state0 = np.concatenate([r0, v0, Phi0])

df_meas = pd.read_csv(r'HW_2\measurements_noisy_2.csv')
time_eval = df_meas['Time(s)'].values

# Doing a nonlinear integration to get the reference trajectory (kinda cheating here)
print("Integrating 6x6 Reference Trajectory...")
sol = solve_ivp(zonal_sph_ode, (0, time_eval[-1]), state0, t_eval=time_eval, rtol=1e-10, atol=1e-10)

# Initial State Deviation & Covariances
dx_0 = np.array([0.5, 0, 0, 0.3e-3, 0, 0])
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])
Rk = np.diag([1e-6, 1e-12])

x_hist, P_hist, x_corrected, innovations, prefit_residuals = run_lkf(sol, df_meas, dx_0, P0, Rk)
# ============================================================
# Plotting (State Deviations)
# ============================================================
times = sol.t
sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
state_labels = ['x (km)', 'y (km)', 'z (km)', 'vx (km/s)', 'vy (km/s)', 'vz (km/s)']

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

for i in range(6):
    # Remove the [1:, i] and use [:, i] to match the full length of 'times'
    axes[i].plot(times, x_hist[:, i], 'b', label='Estimate')
    axes[i].plot(times, 3*sigmas[:, i], 'r--', alpha=0.7, label=fr'3$\sigma$')
    axes[i].plot(times, -3*sigmas[:, i], 'r--', alpha=0.7)
    
    axes[i].set_ylabel(state_labels[i])
    axes[i].set_title(f'State Deviation: {state_labels[i]}')
    axes[i].grid(True, linestyle=':', alpha=0.6)
    if i == 0:
        axes[i].legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()



# ============================================================
# Plotting (Residuals & Statistics)
# ============================================================
# Calculate 3-sigma bounds for the residuals (derived from Innovation Covariance S)
# Note: In a real LKF, S = H*P_pred*H' + Rk
res_sigmas = []
for k in range(len(sol.t)):
    # Recompute S for each step to get the measurement uncertainty bounds
    Phi_global = sol.y[6:, k].reshape(6, 6)
    # Using a simple approximation for the plotting bounds here:
    # Most of the residual uncertainty in a tuned LKF comes from Rk
    res_sigmas.append(np.sqrt(np.diag(Rk))) 

res_sigmas = np.array(res_sigmas)
prefit_residuals = np.array(prefit_residuals)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# --- Range Residuals ---
ax1.scatter(times, prefit_residuals[:, 0], s=2, c='black', label='Pre-fit Residual')
ax1.plot(times, 3*res_sigmas[:, 0], 'r--', alpha=0.8, label=fr'3$\sigma$ Threshold')
ax1.plot(times, -3*res_sigmas[:, 0], 'r--', alpha=0.8)
ax1.set_ylabel('Range Residual (km)')
ax1.set_title('Measurement Residuals (Innovation Checks)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- Range-Rate Residuals ---
ax2.scatter(times, prefit_residuals[:, 1], s=2, c='black')
ax2.plot(times, 3*res_sigmas[:, 1], 'r--', alpha=0.8)
ax2.plot(times, -3*res_sigmas[:, 1], 'r--', alpha=0.8)
ax2.set_ylabel('Range-Rate Residual (km/s)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Print Filter Statistics ---
rms_range = np.sqrt(np.mean(prefit_residuals[:, 0]**2))
rms_range_rate = np.sqrt(np.mean(prefit_residuals[:, 1]**2))
print(f"--- Filter Performance Metrics ---")
print(f"RMS Range Residual:      {rms_range:.6e} km")
print(f"RMS Range-Rate Residual: {rms_range_rate:.6e} km/s")




# ============================================================
# Histogram of Residuals (Gaussian Distribution Check)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Range Residual Histogram ---
range_res = prefit_residuals[:, 0]
mu_r, std_r = np.mean(range_res), np.std(range_res)

# Set a filter for outliers beyond 6 sigma
outlier_threshold = 6 * std_r
range_res = range_res[np.abs(range_res - mu_r) <= outlier_threshold]

# Plot Histogram
count, bins, ignored = ax1.hist(range_res, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
# Plot Theoretical Normal Distribution
x_axis = np.linspace(mu_r - 4*std_r, mu_r + 4*std_r, 100)
ax1.plot(x_axis, stats.norm.pdf(x_axis, mu_r, std_r), 'r-', lw=2, label='Fitted Normal')

ax1.set_title(f'Range Residual Distribution\n' + fr'$\mu$={mu_r:.2e}, $\sigma$={std_r:.2e}')
ax1.set_xlabel('Residual (km)')
ax1.set_ylabel('Probability Density')
ax1.legend()
ax1.grid(alpha=0.3)

# --- Range-Rate Residual Histogram ---
rr_res = prefit_residuals[:, 1]
mu_rr, std_rr = np.mean(rr_res), np.std(rr_res)

# Set a filter for outliers beyond 6 sigma
outlier_threshold_rr = 6 * std_rr
rr_res = rr_res[np.abs(rr_res - mu_rr) <= outlier_threshold_rr]

# Plot Histogram
count, bins, ignored = ax2.hist(rr_res, bins=50, density=True, alpha=0.6, color='salmon', edgecolor='black')
# Plot Theoretical Normal Distribution
x_axis_rr = np.linspace(mu_rr - 4*std_rr, mu_rr + 4*std_rr, 100)
ax2.plot(x_axis_rr, stats.norm.pdf(x_axis_rr, mu_rr, std_rr), 'b-', lw=2, label='Fitted Normal')

ax2.set_title(f'Range-Rate Residual Distribution\n' + fr'$\mu$={mu_rr:.2e}, $\sigma$={std_rr:.2e}')
ax2.set_xlabel('Residual (km/s)')
ax2.set_ylabel('Probability Density')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# def run_lkf(sol, meas_df, P0, Rk):
#     # State deviation vector (6x1)
#     x = np.array([0.5, -0.3, 0.8, 0.3e-3, -0.5e-3, 0.2e-3]) 
#     P = P0.copy()

#     x_hist = [x.copy()]
#     P_hist = [P.copy()]
#     innovations = np.zeros((len(sol.t), 2))
#     prefit_residuals = np.zeros((len(sol.t), 2))

#     for k in range(len(sol.t)):
#         # 1. Extract STM for this step
#         Phi_k = sol.y[6:, k].reshape(6, 6)
        
#         # 2. Prediction (Time Update)
#         x_pred = Phi_k @ x_hist[0] 
#         P_pred = Phi_k @ P0 @ Phi_k.T

#         # 3. GET THE ROW DATA (This was missing or out of order)
#         row = meas_df.iloc[k] 
        
#         # 4. Now you can use 'row' to get the station index
#         station_idx = int(row['Station_ID']) - 1
        
#         # 5. Get station state and compute H
#         Rs, Vs = get_gs_eci_state(stations_ll[station_idx][0], 
#                                 stations_ll[station_idx][1], 
#                                 sol.t[k])

#         # H-matrix and Reference Obs
#         y_ref, H = compute_H_matrix(sol.y[0:3, k], sol.y[3:6, k], Rs, Vs)
#         y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])

#         # Innovations and Residuals
#         innovation = y_obs - y_ref
#         residual = innovation - H @ x_pred # Pre-fit residual

#         innovations[k] = innovation
#         prefit_residuals[k] = residual

#         # Measurement Update
#         S = H @ P_pred @ H.T + Rk
#         K = P_pred @ H.T @ np.linalg.inv(S)

#         x = x_pred + K @ residual
#         P = (np.eye(6) - K @ H) @ P_pred

#         x_hist.append(x.copy())
#         P_hist.append(P.copy())

#     return np.array(x_hist), np.array(P_hist), innovations, prefit_residuals