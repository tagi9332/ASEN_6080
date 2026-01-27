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
from utils.filters.ekf_class import EKF

# ============================================================
# Main Execution
# ============================================================
# 1. Initial State Deviation & Covariances
# ----------------------------------------
# Deviation added to truth to create the initial estimate
x_0_dev = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])

# Initial Covariance (P0)
P0 = np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6])

# Measurement Noise (Rk): [Range (km^2), Range-Rate (km/s)^2]
Rk = np.diag([1e-6, 1e-12])

# Process Noise (Q)
# Set to zero for this problem (deterministic dynamics assumed)
Q = np.diag([0, 0, 0, 0, 0, 0]) 

# 2. Setup Reference Trajectory (The "Truth" or Linearization Point)
# -----------------------------------------------------------------
# Initial Truth State (orbital elements -> inertial)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()

# Initial State Vector for Integrator: [Pos, Vel, STM]
# We integrate the "Truth" + Deviation to get a reference, 
# or strictly the truth if x_0_dev was 0. Here we strictly integrate the
# initial estimate as the reference for the LKF portion.
state0_ref = np.concatenate([r0 + x_0_dev[:3], v0 + x_0_dev[3:], Phi0])

# Load Measurements
df_meas = pd.read_csv(r'HW_2/measurements_noisy.csv')
time_eval = df_meas['Time(s)'].values

# ODE arguments
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for consistency with problem statement

print("Integrating Reference Trajectory for LKF (bootstrap phase)...")
sol = solve_ivp(
    zonal_sph_ode_6x6, 
    (0, time_eval[-1]), 
    state0_ref, 
    t_eval=time_eval, 
    args=(coeffs,),
    rtol=1e-10, 
    atol=1e-10
)

# 3. Run the Filter (Hybrid LKF -> EKF)
# -------------------------------------
ekf_filter = EKF(n_states=6)

# Key Fix: Using keyword arguments to avoid order mismatch
results = ekf_filter.run(
    meas_df=df_meas,
    x_0_dev=x_0_dev,
    P0=P0,
    Rk=Rk,
    Q=Q,
    coeffs=coeffs,
    sol_ref_lkf=sol,
    bootstrap_steps=101  # Optional: Define how many steps to stay in LKF mode
)
# --- Unpack results ---
# Create a mask to slice off the bootstrap period
start_idx = 1000  # Must match your bootstrap_steps=100

# Apply mask to all historical data
dx_hist = results.dx_hist[start_idx:]
P_hist = results.P_hist[start_idx:]
corrected_state_hist = results.state_hist[start_idx:]
innovations = results.innovations[start_idx:]
postfit_residuals = results.postfit_residuals[start_idx:]
S_hist = results.S_hist[start_idx:]
times = results.times[start_idx:]

# ============================================================
# Plotting (State Deviations - Post Bootstrap)
# ============================================================
sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
state_labels = ['x (km)', 'y (km)', 'z (km)', 'vx (km/s)', 'vy (km/s)', 'vz (km/s)']

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

for i in range(6):
    axes[i].scatter(times, dx_hist[:, i], c='b', label='Estimate Error', s=2)
    axes[i].plot(times, 3*sigmas[:, i], 'r--', alpha=0.7, label=r'3$\sigma$')
    axes[i].plot(times, -3*sigmas[:, i], 'r--', alpha=0.7)
    
    axes[i].set_ylabel(state_labels[i])
    axes[i].set_title(f'State Deviation: {state_labels[i]} (Post-Bootstrap)')
    axes[i].grid(True, linestyle=':', alpha=0.6)
    if i == 0:
        axes[i].legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# ============================================================
# Plotting (Residuals & Statistics - Post Bootstrap)
# ============================================================
# Calculate 3-sigma bounds for the residuals
res_sigmas = np.sqrt(np.array([np.diag(S) for S in S_hist]))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# --- Range Residuals ---
ax1.scatter(times, postfit_residuals[:, 0], s=2, c='black', label='Post-fit Residual')
ax1.plot(times, 3*res_sigmas[:, 0], 'r--', alpha=0.8, label=r'3$\sigma$ Bound')
ax1.plot(times, -3*res_sigmas[:, 0], 'r--', alpha=0.8)
ax1.set_ylabel('Range Residual (km)')
ax1.set_title('Post-fit Residuals (Post-Bootstrap)')
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

# --- Print Filter Statistics (Post-Bootstrap Only) ---
rms_range = np.sqrt(np.mean(postfit_residuals[:, 0]**2))
rms_range_rate = np.sqrt(np.mean(postfit_residuals[:, 1]**2))
print(f"--- Filter Performance Metrics (Post-Bootstrap) ---")
print(f"RMS Range Residual:      {rms_range:.6e} km")
print(f"RMS Range-Rate Residual: {rms_range_rate:.6e} km/s")

# ============================================================
# Histogram of Residuals (Post Bootstrap)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Range Residual Histogram ---
range_res = postfit_residuals[:, 0]
mu_r, std_r = np.mean(range_res), np.std(range_res)

count, bins, ignored = ax1.hist(range_res, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
x_axis = np.linspace(mu_r - 4*std_r, mu_r + 4*std_r, 100)
ax1.plot(x_axis, stats.norm.pdf(x_axis, mu_r, std_r), 'r-', lw=2, label='Fitted Normal')
ax1.set_title(f'Range Residuals (Post-Bootstrap)\n' + r'$\mu$={:.2e}, $\sigma$={:.2e}'.format(mu_r, std_r))
ax1.legend()
ax1.grid(alpha=0.3)

# --- Range-Rate Residual Histogram ---
rr_res = postfit_residuals[:, 1]
mu_rr, std_rr = np.mean(rr_res), np.std(rr_res)

count, bins, ignored = ax2.hist(rr_res, bins=50, density=True, alpha=0.6, color='salmon', edgecolor='black')
x_axis_rr = np.linspace(mu_rr - 4*std_rr, mu_rr + 4*std_rr, 100)
ax2.plot(x_axis_rr, stats.norm.pdf(x_axis_rr, mu_rr, std_rr), 'b-', lw=2, label='Fitted Normal')
ax2.set_title(f'Range-Rate Residuals (Post-Bootstrap)\n' + r'$\mu$={:.2e}, $\sigma$={:.2e}'.format(mu_rr, std_rr))
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Plot of Innovations (Post Bootstrap)
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- Range Innovations ---
ax1.scatter(times, innovations[:, 0], s=3, c='blue', label='Range Innovation')
ax1.set_ylabel('Range Innov. (km)')
ax1.set_title('Innovations vs. Time (Post-Bootstrap)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- Range-Rate Innovations ---
ax2.scatter(times, innovations[:, 1], s=3, c='green', label='Range-Rate Innovation')
ax2.set_ylabel('Range-Rate Innov. (km/s)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()