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
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix

# ============================================================
# Batch Filter
# ============================================================
def run_iterative_batch(state0_initial, df_meas, P0_prior, Rk, max_iterations=10, tolerance=1e-10):
    """
    Performs Differential Correction (Iterative Least Squares).
    """
    # Initialize the state to be corrected (6-element vector)
    curr_x0 = state0_initial[:6].copy()
    n = 6
    
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        # Reset STM to identity for the new integration
        phi0 = np.eye(n).flatten()
        state_to_integrate = np.concatenate([curr_x0, phi0])
        
        # 1. Integrate the NEW reference trajectory
        sol = solve_ivp(zonal_sph_ode_6x6, (0, df_meas['Time(s)'].values[-1]), 
                        state_to_integrate, t_eval=df_meas['Time(s)'].values, 
                        args=([MU_EARTH, J2, 0],), rtol=1e-10, atol=1e-10)
        
        # 2. Accumulate Normal Equations
        info_matrix = np.linalg.inv(P0_prior)
        normal_vector = np.zeros(n) 
        inv_Rk = np.linalg.inv(Rk)
        
        for k in range(len(sol.t)):
            Phi_tk_t0 = sol.y[6:, k].reshape(n, n)
            
            row = df_meas.iloc[k]
            station_idx = int(row['Station_ID']) - 1
            Rs, Vs = get_gs_eci_state(stations_ll[station_idx][0], 
                                      stations_ll[station_idx][1], 
                                      sol.t[k], init_theta = np.deg2rad(122))
            
            x_ref = sol.y[0:6, k]
            y_ref, H_tilde = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs)
            y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])
            
            # Map H to epoch: H = H_tilde * Phi(tk, t0)
            H = H_tilde @ Phi_tk_t0
            residual = y_obs - y_ref
            
            info_matrix += H.T @ inv_Rk @ H
            normal_vector += H.T @ inv_Rk @ residual

        # 3. Solve for initial state correction
        dx0 = np.linalg.solve(info_matrix, normal_vector)
        
        # 4. Apply Correction
        curr_x0 += dx0
        
        correction_norm = np.linalg.norm(dx0)
        print(f"Correction Norm (RMS): {correction_norm:.6e}")
        
        # 5. Check Convergence
        if correction_norm < tolerance:
            print("Converged!")
            return compute_final_batch_stats(sol, df_meas, np.linalg.inv(info_matrix), dx0)

    print("Warning: Max iterations reached.")
    return compute_final_batch_stats(sol, df_meas, np.linalg.inv(info_matrix), np.zeros(n))

def compute_final_batch_stats(sol, df_meas, P0_final, dx0_final):
    """
    Final pass to propagate state deviations and covariance for plotting.
    """
    n = 6
    num_steps = len(sol.t)
    
    dx_hist = []
    P_hist = []
    corrected_state_hist = []
    postfit_residuals = []

    for k in range(num_steps):
        Phi_tk_t0 = sol.y[6:, k].reshape(n, n)
        x_ref = sol.y[0:6, k]
        
        # Propagate state deviation and covariance to time tk
        dx_k = Phi_tk_t0 @ dx0_final 
        P_k = Phi_tk_t0 @ P0_final @ Phi_tk_t0.T
        
        # Get post-fit residuals
        row = df_meas.iloc[k]
        station_idx = int(row['Station_ID']) - 1
        Rs, Vs = get_gs_eci_state(stations_ll[station_idx][0], 
                                  stations_ll[station_idx][1], 
                                  sol.t[k])
        
        y_ref, H_tilde = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs)
        y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])
        
        # Post-fit residual: measurement minus corrected reference
        postfit_res = (y_obs - y_ref) - H_tilde @ dx_k
        
        dx_hist.append(dx_k)
        P_hist.append(P_k)
        corrected_state_hist.append(x_ref + dx_k)
        postfit_residuals.append(postfit_res)

    return (np.array(dx_hist), 
            np.array(P_hist), 
            np.array(corrected_state_hist), 
            np.array(postfit_residuals))

# ============================================================
# Main Execution
# ============================================================
# Initial Reference State (Initial Guess)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
state0_initial = np.concatenate([r0, v0, Phi0])

# Ground Stations (lat, lon) [rad]
stations_ll = np.deg2rad([
    [-35.398333, 148.981944], # Station 1 (Canberra, Australia)
    [ 40.427222, 355.749444], # Station 2 (Fort Davis, USA)
    [ 35.247164, 243.205000]  # Station 3 (Madrid, Spain)
])

# Load Measurements
df_meas = pd.read_csv(r'HW_2\measurements_noisy_2.csv')

# Initial Covariances & Weights
# P0: Confidence in your initial r0, v0 guess
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])
# Rk: Measurement noise floor (Range and Range-Rate)
Rk = np.diag([1e-6, 1e-12])

print("Starting Iterative Batch Filter (Differential Correction)...")

dx_hist, P_hist, x_corrected, postfit_residuals = run_iterative_batch(
    state0_initial, 
    df_meas, 
    P0, 
    Rk, 
    max_iterations=10, 
    tolerance=1e-7
)

# ============================================================
# Plotting (State Deviations)
# ============================================================
times = df_meas['Time(s)'].values
sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
state_labels = ['x (m)', 'y (m)', 'z (m)', 'vx (m/s)', 'vy (m/s)', 'vz (m/s)']

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

for i in range(6):
    # Remove the [1:, i] and use [:, i] to match the full length of 'times'
    axes[i].scatter(times, dx_hist[:, i] * 1000, c='b', label='Estimate', s=2)  # Convert km to m for position states
    # axes[i].plot(times, 3*sigmas[:, i], 'r--', alpha=0.7, label=fr'3$\sigma$')
    # axes[i].plot(times, -3*sigmas[:, i], 'r--', alpha=0.7)
    
    axes[i].set_ylabel(state_labels[i])
    axes[i].set_title(f'State Deviation: {state_labels[i]}')
    axes[i].grid(True, linestyle=':', alpha=0.6)
    if i == 0:
        axes[i].legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# ============================================================
# Residual Plotting & Gaussian Distribution Analysis
# ============================================================
postfit_residuals = np.array(postfit_residuals)
res_sigmas = np.sqrt(np.diag(Rk))

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

# --- Range Residuals Time Series ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(times, postfit_residuals[:, 0], s=2, c='black', label='Post-fit Residual')
ax1.axhline(3*res_sigmas[0], color='r', linestyle='--', alpha=0.8, label=r'3$\sigma$ Noise Floor')
ax1.axhline(-3*res_sigmas[0], color='r', linestyle='--', alpha=0.8)
ax1.set_ylabel('Range Residual (km)')
ax1.set_title('Post-fit Measurement Residuals')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- Range Residual Histogram ---
ax2 = fig.add_subplot(gs[0, 1])
r_res = postfit_residuals[:, 0]
mu_r, std_r = np.mean(r_res), np.std(r_res)
count, bins, ignored = ax2.hist(r_res, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
x_range = np.linspace(mu_r - 4*std_r, mu_r + 4*std_r, 100)
ax2.plot(x_range, stats.norm.pdf(x_range, mu_r, std_r), 'r-', lw=2, label='Fitted Normal')
ax2.set_title(rf'Range PDF\n$\mu$={mu_r:.1e}, $\sigma$={std_r:.1e}')
ax2.legend(fontsize='small')

# --- Range-Rate Residual Time Series ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(times, postfit_residuals[:, 1], s=2, c='black')
ax3.axhline(3*res_sigmas[1], color='r', linestyle='--', alpha=0.8)
ax3.axhline(-3*res_sigmas[1], color='r', linestyle='--', alpha=0.8)
ax3.set_ylabel('Range-Rate Residual (km/s)')
ax3.set_xlabel('Time (s)')
ax3.grid(True, alpha=0.3)

# --- Range-Rate Residual Histogram ---
ax4 = fig.add_subplot(gs[1, 1])
rr_res = postfit_residuals[:, 1]
mu_rr, std_rr = np.mean(rr_res), np.std(rr_res)
count, bins, ignored = ax4.hist(rr_res, bins=40, density=True, alpha=0.6, color='salmon', edgecolor='black')
x_rr = np.linspace(mu_rr - 4*std_rr, mu_rr + 4*std_rr, 100)
ax4.plot(x_rr, stats.norm.pdf(x_rr, mu_rr, std_rr), 'b-', lw=2, label='Fitted Normal')
ax4.set_title(rf'Range-Rate PDF\n$\mu$={mu_rr:.1e}, $\sigma$={std_rr:.1e}')
ax4.set_xlabel('Residual')
ax4.legend(fontsize='small')

plt.tight_layout()
plt.show()

# --- Print Final Statistics ---
print(f"\n--- Final Batch Statistics ---")
print(f"RMS Range Residual:     {np.sqrt(np.mean(r_res**2)):.6e} km")
print(f"RMS Range-Rate Residual: {np.sqrt(np.mean(rr_res**2)):.6e} km/s")