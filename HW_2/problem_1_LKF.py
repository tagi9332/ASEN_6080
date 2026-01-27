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

# Local Imports
from utils.plotting.report_RMS_error import report_filter_metrics
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6
from utils.filters.lkf_class import LKF
from utils.plotting.compute_state_error import compute_state_error

# ============================================================
# Main Execution
# ============================================================
# Initial State Deviation & Covariances
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
# P0 = np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6])
P0 = np.diag([1e3, 1e3, 1e3, 1, 1, 1])**2
Rk = np.diag([1e-6, 1e-12])

# Initial Truth State (without deviation)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()

# Propagate the reference trajectory (truth plus deviation)
state0 = np.concatenate([r0+x_0[:3], v0+x_0[3:], Phi0])


# Load Measurements
df_meas = pd.read_csv(r'HW_2\measurements_noisy.csv')
time_eval = df_meas['Time(s)'].values

# ODE arguments
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for dynamics

# Doing a nonlinear integration to get the reference trajectory (kinda cheating here)
print("Integrating 6x6 Reference Trajectory...")
sol = solve_ivp(
    zonal_sph_ode_6x6, 
    (0, time_eval[-1]), 
    state0, 
    t_eval=time_eval, 
    args=(coeffs,),
    rtol=1e-10, 
    atol=1e-10
)

# Process noise
# Q = np.diag([1e-10, 1e-10, 1e-10, 1e-8, 1e-8, 1e-8])
Q = np.diag([0, 0, 0, 0, 0, 0])  # No process noise

lkf_filter = LKF(n_states=6)

results = lkf_filter.run(sol, df_meas, x_0, P0, Rk, Q)

# --- Unpack results ---
dx_hist = np.array(results.dx_hist)
corrected_state_hist = np.array(results.corrected_state_hist)
P_hist = np.array(results.P_hist)
innovations = np.array(results.innovations)
postfit_residuals = np.array(results.postfit_residuals)
S_hist = np.array(results.S_hist)

# Compute state errors against truth after bootstrap
state_errors = compute_state_error(
    x_corrected=corrected_state_hist, 
    df_meas=df_meas,
)
state_errors_m = np.array(state_errors)*1e3  # Convert to meters for error analysis

# ============================================================
# Plotting (State Errors)
# ============================================================
times = sol.t
sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
state_labels = ['x (m)', 'y (m)', 'z (m)', 'vx (m/s)', 'vy (m/s)', 'vz (m/s)']
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()
fig.suptitle('Linearized Kalman Filter State Estimation Errors with 3-Sigma Bounds', fontsize=16)  
for i in range(6):
    axes[i].scatter(times, state_errors_m[:, i], c='b', label='Estimate Error', s=2)
    axes[i].plot(times, 3*sigmas[:, i]*1e3, 'r--', alpha=0.7, label=fr'3$\sigma$')
    axes[i].plot(times, -3*sigmas[:, i]*1e3, 'r--', alpha=0.7)
    
    axes[i].set_ylabel(state_labels[i])
    axes[i].set_title(f'State Error: {state_labels[i]}')
    axes[i].grid(True, linestyle=':', alpha=0.6)
    if i == 0:
        axes[i].legend()
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# ============================================================
# Plotting (State Deviations)
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

for i in range(6):
    # Remove the [1:, i] and use [:, i] to match the full length of 'times'
    axes[i].scatter(times, dx_hist[:, i]*1e3, c='b', label='Estimate', s=2)
    axes[i].plot(times, 3*sigmas[:, i]*1e3, 'r--', alpha=0.7, label=fr'3$\sigma$')
    axes[i].plot(times, -3*sigmas[:, i]*1e3, 'r--', alpha=0.7)
    
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
# Update: Calculate 3-sigma bounds using Innovation Covariance (S)
# This matches the rigorous method used in the EKF script.
# S = H * P_pred * H.T + R  (Total uncertainty: Measurement + State)
S_arr = np.array(S_hist)
res_sigmas = np.sqrt(np.array([np.diag(S) for S in S_arr]))

postfit_residuals = np.array(postfit_residuals)

# Assuming 'times', 'postfit_residuals', and 'res_sigmas' are already defined
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('LKF Residual Analysis: Time History and Error Distributions', fontsize=18)

# Pre-calculations for Range (Row 0)
range_res = postfit_residuals[:, 0] * 1e3
mu_r, std_r = np.mean(range_res), np.std(range_res)

# Pre-calculations for Range-Rate (Row 1)
rr_res = postfit_residuals[:, 1] * 1e3
mu_rr, std_rr = np.mean(rr_res), np.std(rr_res)

# --- ROW 1, COL 1: Range Residual Time History ---
axs[0, 0].scatter(times, range_res, s=2, c='black', label='Post-fit Residual')
axs[0, 0].plot(times, 3 * res_sigmas[:, 0] * 1e3, 'r--', alpha=0.8, label=r'$3\sigma$ Bound')
axs[0, 0].plot(times, -3 * res_sigmas[:, 0] * 1e3, 'r--', alpha=0.8)
axs[0, 0].set_ylabel('Range Residual (m)')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend(loc='upper right')

# --- ROW 1, COL 2: Range Residual Histogram ---
range_filt = range_res[np.abs(range_res - mu_r) <= 6 * std_r]
axs[0, 1].hist(range_filt, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
x_r = np.linspace(mu_r - 4*std_r, mu_r + 4*std_r, 100)
axs[0, 1].plot(x_r, stats.norm.pdf(x_r, mu_r, std_r), 'r-', lw=2, label='Normal Fit')
axs[0, 1].set_title(fr'Range PDF ($\mu$={mu_r:.2e}, $\sigma$={std_r:.2e})')
axs[0, 1].grid(alpha=0.3)

# --- ROW 2, COL 1: Range-Rate Residual Time History ---
axs[1, 0].scatter(times, rr_res, s=2, c='black')
axs[1, 0].plot(times, 3 * res_sigmas[:, 1] * 1e3, 'r--', alpha=0.8)
axs[1, 0].plot(times, -3 * res_sigmas[:, 1] * 1e3, 'r--', alpha=0.8)
axs[1, 0].set_ylabel('Range-Rate Residual (m/s)')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].grid(True, alpha=0.3)

# --- ROW 2, COL 2: Range-Rate Residual Histogram ---
rr_filt = rr_res[np.abs(rr_res - mu_rr) <= 6 * std_rr]
axs[1, 1].hist(rr_filt, bins=40, density=True, alpha=0.6, color='salmon', edgecolor='black')
x_rr = np.linspace(mu_rr - 4*std_rr, mu_rr + 4*std_rr, 100)
axs[1, 1].plot(x_rr, stats.norm.pdf(x_rr, mu_rr, std_rr), 'b-', lw=2, label='Normal Fit')
axs[1, 1].set_title(fr'Range-Rate PDF ($\mu$={mu_rr:.2e}, $\sigma$={std_rr:.2e})')
axs[1, 1].set_xlabel('Residual (m/s)')
axs[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Plot of innovations over time
# ============================================================
# Get the innovations and S from your filter run
# Assuming: innovations.shape = (N, 2), S_hist.shape = (N, 2, 2)
# Convert lists to numpy arrays if they aren't already
innov_arr = np.array(innovations)
S_arr = np.array(S_hist)

# Calculate 3-sigma bounds from S
# S[k, 0, 0] is the variance for Range, S[k, 1, 1] for Range-Rate
sigma_innov = np.sqrt(np.array([np.diag(S_k) for S_k in S_arr]))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- Range Innovations ---
ax1.scatter(times, innov_arr[:, 0]*1e3, s=3, c='blue', label='Range Innovation (Pre-fit Residuals)')
# ax1.plot(times, 3*sigma_innov[:, 0]*1e3, 'r--', alpha=0.8, label=r'$\pm3\sigma$ Bounds')
# ax1.plot(times, -3*sigma_innov[:, 0]*1e3, 'r--', alpha=0.8)
ax1.set_ylabel('Range Error (m)')
ax1.set_title('Innovations (Pre-fit Residuals) vs. Time')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- Range-Rate Innovations ---
ax2.scatter(times, innov_arr[:, 1]*1e3, s=3, c='green', label='Range-Rate Innovation (Pre-fit Residuals)')
# ax2.plot(times, 3*sigma_innov[:, 1]*1e3, 'r--', alpha=0.8)
# ax2.plot(times, -3*sigma_innov[:, 1]*1e3, 'r--', alpha=0.8)
ax2.set_ylabel('Range-Rate Error (m/s)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

report_filter_metrics(
    times=sol.t,
    state_errors=state_errors,
    postfit_residuals=postfit_residuals,
    filter_name="Linearized Kalman Filter"
)