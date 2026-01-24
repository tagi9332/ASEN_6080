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

def run_hybrid_filter(sol_ref, meas_df, dx_0, P0, Rk, switch_idx=100):
    n = 6
    I = np.eye(n)
    
    # Initial state for the filter
    # For LKF: we track deviation 'dx' from 'x_ref'
    # For EKF: we track the 'total_state' directly
    dx = dx_0.copy()
    P = P0.copy()
    
    # Storage
    corrected_state_hist = []
    P_hist = []
    innovations = []
    postfit_residuals = []
    
    # --------------------------------------------------------
    # PART 1: LKF BOOTSTRAP (First 100 measurements)
    # --------------------------------------------------------
    Phi_prev = np.eye(n)
    
    for k in range(switch_idx):
        Phi_global = sol_ref.y[6:, k].reshape(n, n)
        Phi_incr = Phi_global @ np.linalg.inv(Phi_prev)
        x_ref = sol_ref.y[0:6, k]
        
        # Prediction
        dx_pred = Phi_incr @ dx
        P_pred = Phi_incr @ P @ Phi_incr.T
        
        # Observation Logic
        row = meas_df.iloc[k]
        Rs, Vs = get_gs_eci_state(np.deg2rad(stations_ll[int(row['Station_ID'])-1][0]), 
                                  np.deg2rad(stations_ll[int(row['Station_ID'])-1][1]), 
                                  sol_ref.t[k])
        y_nom, H = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs)
        y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])
        
        # Update
        innovation = (y_obs - y_nom) - H @ dx_pred
        S = H @ P_pred @ H.T + Rk
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        dx = dx_pred + K @ innovation
        P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ Rk @ K.T # Joseph Form
        
        # Store
        current_total_state = x_ref + dx
        corrected_state_hist.append(current_total_state)
        P_hist.append(P)
        innovations.append(innovation)
        postfit_residuals.append(y_obs - compute_H_matrix(current_total_state[:3], current_total_state[3:6], Rs, Vs)[0])
        
        Phi_prev = Phi_global

    # --------------------------------------------------------
    # PART 2: EKF TRANSITION (Remainder)
    # --------------------------------------------------------
    # In EKF, our reference is the state we just updated
    current_state = corrected_state_hist[-1]
    last_time = sol_ref.t[switch_idx-1]
    
    for k in range(switch_idx, len(meas_df)):
        row = meas_df.iloc[k]
        current_time = row['Time(s)']
        dt = current_time - last_time
        
        # 1. EKF Propagate (Integrate state and STM over dt)
        # We integrate from the PREVIOUS updated state to the CURRENT time
        phi_init = np.eye(6).flatten()
        ode_init = np.concatenate([current_state, phi_init])
        
        ekf_sol = solve_ivp(zonal_sph_ode, (last_time, current_time), ode_init, 
                            rtol=1e-10, atol=1e-10)
        
        state_pred = ekf_sol.y[0:6, -1]
        Phi_incr = ekf_sol.y[6:, -1].reshape(6, 6)
        
        # Covariance Prediction
        P_pred = Phi_incr @ P @ Phi_incr.T
        
        # 2. EKF Observation
        Rs, Vs = get_gs_eci_state(np.deg2rad(stations_ll[int(row['Station_ID'])-1][0]), 
                                  np.deg2rad(stations_ll[int(row['Station_ID'])-1][1]), 
                                  current_time)
        y_pred, H = compute_H_matrix(state_pred[0:3], state_pred[3:6], Rs, Vs)
        y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])
        
        # 3. EKF Update
        innovation = y_obs - y_pred
        S = H @ P_pred @ H.T + Rk
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        current_state = state_pred + K @ innovation
        P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ Rk @ K.T
        
        # Store
        corrected_state_hist.append(current_state)
        P_hist.append(P)
        innovations.append(innovation)
        postfit_residuals.append(y_obs - compute_H_matrix(current_state[:3], current_state[3:6], Rs, Vs)[0])
        
        last_time = current_time

    return np.array(corrected_state_hist), np.array(P_hist), np.array(innovations), np.array(postfit_residuals)

# ============================================================
# Main Execution
# ============================================================
# Initial Reference State
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
state0 = np.concatenate([r0, v0, Phi0])

df_meas = pd.read_csv(r'HW_2\measurements_noisy_2.csv')
time_eval = df_meas['Time(s)'].values

print("Integrating 6x6 Reference Trajectory...")
sol = solve_ivp(zonal_sph_ode, (0, time_eval[-1]), state0, t_eval=time_eval, rtol=1e-10, atol=1e-10)

# Initial State Deviation & Covariances
dx_0 = np.array([0.5, 0, 0, 0.3e-3, 0, 0])
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])
Rk = np.diag([1e-6, 1e-12])

x_total_hist, P_hist, innovations, postfit_residuals = run_hybrid_filter(
    sol, 
    df_meas, 
    dx_0, 
    P0, 
    Rk, 
    switch_idx=100
)

# ============================================================
# Plotting (State Deviations)
# ============================================================
# Calculate deviation from the original reference trajectory
dx_plotting = x_total_hist - sol.y[0:6, :].T

times = sol.t
sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
state_labels = ['x (km)', 'y (km)', 'z (km)', 'vx (km/s)', 'vy (km/s)', 'vz (km/s)']

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

for i in range(6):
    axes[i].plot(times, dx_plotting[:, i], 'b', label='Deviation')
    axes[i].plot(times, 3*sigmas[:, i], 'r--', alpha=0.7, label=fr'3$\sigma$')
    axes[i].plot(times, -3*sigmas[:, i], 'r--', alpha=0.7)
    
    # Highlight the LKF -> EKF switch point
    axes[i].axvline(times[100], color='green', linestyle=':', alpha=0.8, label='EKF Switch')
    
    axes[i].set_ylabel(state_labels[i])
    axes[i].set_title(f'State Deviation: {state_labels[i]}')
    axes[i].grid(True, linestyle=':', alpha=0.6)
    
    if i == 0:
        axes[i].legend(loc='upper right', fontsize='small')

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()



# ============================================================
# Plotting (Residuals & Statistics)
# ============================================================
innovations = np.array(innovations)
res_sigmas = np.sqrt(np.diag(Rk))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# --- Range Residuals ---
ax1.scatter(times, innovations[:, 0], s=2, c='black', label='Innovation (Pre-fit)')
ax1.axhline(3*res_sigmas[0], color='r', linestyle='--', alpha=0.8, label=fr'3$\sigma$ Noise')
ax1.axhline(-3*res_sigmas[0], color='r', linestyle='--')
ax1.axvline(times[100], color='green', linestyle=':', label='EKF Switch')
ax1.set_ylabel('Range Residual (km)')
ax1.set_title('Measurement Innovations (Pre-fit)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- Range-Rate Residuals ---
ax2.scatter(times, innovations[:, 1], s=2, c='black')
ax2.axhline(3*res_sigmas[1], color='r', linestyle='--')
ax2.axhline(-3*res_sigmas[1], color='r', linestyle='--')
ax2.axvline(times[100], color='green', linestyle=':')
ax2.set_ylabel('Range-Rate Residual (km/s)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Print Filter Statistics ---
rms_range = np.sqrt(np.mean(innovations[:, 0]**2))
rms_range_rate = np.sqrt(np.mean(innovations[:, 1]**2))
print(f"--- Filter Performance Metrics ---")
print(f"RMS Range Innovation:      {rms_range:.6e} km")
print(f"RMS Range-Rate Innovation: {rms_range_rate:.6e} km/s")




# ============================================================
# Histogram of Residuals (Post-Switch EKF Performance)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# We only look at indices from 100 onwards for a "clean" Gaussian check
ekf_range_res = innovations[100:, 0]
ekf_rr_res = innovations[100:, 1]

def plot_res_hist(ax, data, label, color, noise_val):
    mu, std = np.mean(data), np.std(data)
    ax.hist(data, bins=40, density=True, alpha=0.6, color=color, edgecolor='black')
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'k-', lw=2, label='Fitted Normal')
    ax.set_title(rf'{label}\n$\mu$={mu:.1e}, $\sigma$={std:.1e}')
    ax.set_xlabel('Residual')
    ax.legend()

plot_res_hist(ax1, ekf_range_res, 'EKF Range Residuals (km)', 'skyblue', res_sigmas[0])
plot_res_hist(ax2, ekf_rr_res, 'EKF Range-Rate Residuals (km/s)', 'salmon', res_sigmas[1])

plt.tight_layout()
plt.show()
