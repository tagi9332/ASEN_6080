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
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6
from utils.filters.lkf_class import LKF

# ============================================================
# Main Execution
# ============================================================
# Initial State Deviation & Covariances
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
P0 = np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6])
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
# # Debug: Save the state solution to a csv for verification
# np.savetxt('HW_2/lkf_reference_trajectory.csv', 
#            np.column_stack((sol.t, sol.y.T[:, :6])), delimiter=',')

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
    axes[i].scatter(times, dx_hist[:, i], c='b', label='Estimate', s=2)
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
postfit_residuals = np.array(postfit_residuals)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# --- Range Residuals ---
ax1.scatter(times, postfit_residuals[:, 0], s=2, c='black', label='Post-fit Residual')
ax1.plot(times, 3*res_sigmas[:, 0], 'r--', alpha=0.8, label=fr'3$\sigma$ Threshold')
ax1.plot(times, -3*res_sigmas[:, 0], 'r--', alpha=0.8)
ax1.set_ylabel('Range Residual (km)')
ax1.set_title('Post-fit Residuals for Range and Range-Rate Measurements')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- Range-Rate Residuals ---
ax2.scatter(times, postfit_residuals[:, 1], s=2, c='black')
ax2.plot(times, 3*res_sigmas[:, 1], 'r--', alpha=0.8)
ax2.plot(times, -3*res_sigmas[:, 1], 'r--', alpha=0.8)
ax2.set_ylabel('Range-Rate Residual (km/s)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Print Filter Statistics ---
rms_range = np.sqrt(np.mean(postfit_residuals[:, 0]**2))
rms_range_rate = np.sqrt(np.mean(postfit_residuals[:, 1]**2))
print(f"--- Filter Performance Metrics ---")
print(f"RMS Range Residual:      {rms_range:.6e} km")
print(f"RMS Range-Rate Residual: {rms_range_rate:.6e} km/s")

# ============================================================
# Histogram of Residuals (Gaussian Distribution Check)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Range Residual Histogram ---
range_res = postfit_residuals[:, 0]
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
rr_res = postfit_residuals[:, 1]
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
ax1.scatter(times, innov_arr[:, 0], s=3, c='blue', label='Range Innovation (Pre-fit Residuals)')
# ax1.plot(times, 3*sigma_innov[:, 0], 'r--', alpha=0.8, label=r'$\pm3\sigma$ Bounds')
# ax1.plot(times, -3*sigma_innov[:, 0], 'r--', alpha=0.8)
ax1.set_ylabel('Range Error (km)')
ax1.set_title('Innovations (Pre-fit Residuals) vs. Time')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- Range-Rate Innovations ---
ax2.scatter(times, innov_arr[:, 1], s=3, c='green', label='Range-Rate Innovation (Pre-fit Residuals)')
# ax2.plot(times, 3*sigma_innov[:, 1], 'r--', alpha=0.8)
# ax2.plot(times, -3*sigma_innov[:, 1], 'r--', alpha=0.8)
ax2.set_ylabel('Range-Rate Error (km/s)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()