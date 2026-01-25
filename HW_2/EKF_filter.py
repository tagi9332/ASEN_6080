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

# Local Imports
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix

# ============================================================
# EKF filter with LKF Bootstrap
# ============================================================
def run_hybrid_filter(sol_ref, meas_df, dx_0, P0, Rk, Q, switch_idx=100):
    n = 6
    I = np.eye(n)
    
    # Initial state for the filter
    # For LKF track deviation 'dx' from 'x_ref'
    # For EKF track the 'total_state' directly
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

        # Incremental STM
        Phi_global = sol_ref.y[6:, k].reshape(n, n)
        Phi_incr = Phi_global @ np.linalg.inv(Phi_prev)
        x_ref = sol_ref.y[0:6, k]
        
        # Prediction
        dx_pred = Phi_incr @ dx
        P_pred = Phi_incr @ P @ Phi_incr.T + (Q*(sol_ref.t[k] - sol_ref.t[k-1]) if k > 0 else 0)
        
        # Observation Logic
        row = meas_df.iloc[k]
        Rs, Vs = get_gs_eci_state((stations_ll[int(row['Station_ID'])-1][0]), 
                                  (stations_ll[int(row['Station_ID'])-1][1]), 
                                  sol_ref.t[k], init_theta=np.deg2rad(122))
        y_nom, H = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs)
        y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])

        # Residuals
        y_pred_total = y_nom + H @ dx_pred

        # Measurement Update
        prefit_residual = y_obs - y_pred_total

        # Kalman Gain & Update
        S = H @ P_pred @ H.T + Rk
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update state deviation and covariance
        dx = dx_pred + K @ prefit_residual
        post_fit_residual = y_obs - (y_nom + H @ dx)
        P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ Rk @ K.T # Joseph Form
        
        # Store
        current_total_state = x_ref + dx
        corrected_state_hist.append(current_total_state)
        P_hist.append(P)
        innovations.append(prefit_residual)
        postfit_residuals.append(post_fit_residual)
        
        Phi_prev = Phi_global

    # --------------------------------------------------------
    # PART 2: EKF TRANSITION
    # --------------------------------------------------------
    current_state = corrected_state_hist[-1]
    last_time = sol_ref.t[switch_idx-1]
    
    for k in range(switch_idx, len(meas_df)):
        row = meas_df.iloc[k]
        current_time = row['Time(s)']
        dt = current_time - last_time
        
        # EKF Propagate (Integrate state and STM over dt)
        phi_init = np.eye(6).flatten()
        ode_init = np.concatenate([current_state, phi_init])

        # Set ODE arguments
        coeffs = [MU_EARTH, J2, 0]  # Ignoring J3 for dynamics
        
        ekf_sol = solve_ivp(zonal_sph_ode_6x6, (last_time, current_time), ode_init, 
                            args=(coeffs,),
                            rtol=1e-10, atol=1e-10)
        
        state_pred = ekf_sol.y[0:6, -1]
        Phi_incr = ekf_sol.y[6:, -1].reshape(6, 6)
        
        # Covariance Prediction
        P_pred = Phi_incr @ P @ Phi_incr.T
        
        # EKF Observation
        Rs, Vs = get_gs_eci_state((stations_ll[int(row['Station_ID'])-1][0]), 
                                  (stations_ll[int(row['Station_ID'])-1][1]), 
                                  current_time, init_theta=np.deg2rad(122))
        y_pred, H = compute_H_matrix(state_pred[0:3], state_pred[3:6], Rs, Vs)
        y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])
        
        # EKF Update
        prefit_residual = y_obs - y_pred

        # Kalman Gain & Update
        S = H @ P_pred @ H.T + Rk
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Post-fit residual
        post_fit_residual = y_obs - (y_pred + H @ (K @ prefit_residual))
        
        # State and Covariance Update
        current_state = state_pred + K @ prefit_residual
        P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ Rk @ K.T
        
        # Store
        corrected_state_hist.append(current_state)
        P_hist.append(P)
        innovations.append(prefit_residual)
        postfit_residuals.append(post_fit_residual)
        
        last_time = current_time

    return np.array(corrected_state_hist), np.array(P_hist), np.array(innovations), np.array(postfit_residuals)

# ============================================================
# Main Execution
# ============================================================
# Initial Reference State
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
state0 = np.concatenate([r0, v0, Phi0])

# Number of measurements to bootstrap LKF before switching to EKF
switch_idx = 100

# Ground Stations (lat, lon) [rad]
stations_ll = np.deg2rad([
    [-35.398333, 148.981944], # Station 1 (Canberra, Australia)
    [ 40.427222, 355.749444], # Station 2 (Fort Davis, USA)
    [ 35.247164, 243.205000]  # Station 3 (Madrid, Spain)
])


df_meas = pd.read_csv(r'HW_2\measurements_noisy.csv')
time_eval = df_meas['Time(s)'].values

# Set ODE arguments
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for dynamics

# Integrate Reference Trajectory for LKF bootstrap
print("Integrating 6x6 Reference Trajectory...")
sol = solve_ivp(
    zonal_sph_ode_6x6,
    (0, time_eval[-1]),
    state0,
    t_eval=time_eval,
    args=(coeffs,),
    rtol=1e-10, atol=1e-10)

# Initial State Deviation & Covariances
# dx_0 = np.array([0, 0, 0, 0, 0, 0])
dx_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])
Rk = np.diag([1e-6, 1e-12])
Q = np.zeros((6, 6))  # No process noise

x_total_hist, P_hist, innovations, postfit_residuals = run_hybrid_filter(
    sol, 
    df_meas, 
    dx_0, 
    P0, 
    Rk,
    Q, 
    switch_idx=switch_idx
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
    axes[i].scatter(times, dx_plotting[:, i], c='b', label='Deviation', s=2)
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
# Plotting (Pre-fit vs Post-fit Residuals)
# ============================================================
innov_arr = np.array(innovations)
post_arr = np.array(postfit_residuals)
res_sigmas = np.sqrt(np.diag(Rk))

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

# Helper to plot residuals
def plot_residual_set(ax, data, sigma, title, ylabel, color):
    ax.scatter(times, data, s=2, c=color, alpha=0.5)
    ax.axhline(3*sigma, color='r', linestyle='--', alpha=0.8, label=r'3$\sigma$ Noise')
    ax.axhline(-3*sigma, color='r', linestyle='--')
    ax.axvline(times[100], color='green', linestyle=':', label='EKF Switch')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

# Row 1: Range
plot_residual_set(axes[0,0], innov_arr[:,0], res_sigmas[0], 'Range Innovations (Pre-fit)', 'km', 'black')
plot_residual_set(axes[0,1], post_arr[:,0], res_sigmas[0], 'Range Residuals (Post-fit)', 'km', 'blue')

# Row 2: Range-Rate
plot_residual_set(axes[1,0], innov_arr[:,1], res_sigmas[1], 'Range-Rate Innovations (Pre-fit)', 'km/s', 'black')
plot_residual_set(axes[1,1], post_arr[:,1], res_sigmas[1], 'Range-Rate Residuals (Post-fit)', 'km/s', 'green')

axes[0,0].legend(loc='upper right')
plt.tight_layout()
plt.show()

# ============================================================
# Histogram (Cleaned Bell Curve)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

def plot_cleaned_hist(ax, data, label, color):
    mu, std = np.mean(data), np.std(data)
    
    # CUT TAILS: Only plot data within 4 sigma for visual clarity
    filtered_data = data[np.abs(data - mu) < 4 * std]
    
    ax.hist(filtered_data, bins=50, density=True, alpha=0.6, color=color, edgecolor='black')
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'k-', lw=2, label='Fitted Normal')
    ax.set_title(rf'{label} (Centered)')
    ax.set_xlabel('Residual Value')
    ax.legend()

# Using Post-fit residuals for the Gaussian check
plot_cleaned_hist(ax1, post_arr[100:, 0], 'Post-fit Range', 'skyblue')
plot_cleaned_hist(ax2, post_arr[100:, 1], 'Post-fit Range-Rate', 'salmon')

plt.tight_layout()
plt.show()

# ============================================================
# Requirement iii: RMS Metrics (Total vs. Settled)
# ============================================================
def get_rms(data): return np.sqrt(np.mean(data**2))

print(f"--- Filter Metrics (Post-fit) ---")
print(f"Total Range RMS:    {get_rms(post_arr[:,0]):.6e} km")
print(f"Settled Range RMS:  {get_rms(post_arr[100:,0]):.6e} km (Excluding 1st pass)")
print(f"Total R-Rate RMS:   {get_rms(post_arr[:,1]):.6e} km/s")
print(f"Settled R-Rate RMS: {get_rms(post_arr[100:,1]):.6e} km/s")