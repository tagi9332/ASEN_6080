import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Note: Ensure these paths and constants match your local environment
from utils.plotting.report_RMS_error import report_filter_metrics
from utils.plotting.compute_state_error import compute_state_error
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from utils.plotting.compute_state_error import compute_state_error
from utils.filters.batch_class import IterativeBatch
from resources.constants import MU_EARTH, J2, J3, R_EARTH


# ============================================================
# Main Execution
# ============================================================
# Initial Reference State (Initial Guess)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
dx_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
state0_initial = np.concatenate([r0 + dx_0[0:3], v0 + dx_0[3:6], Phi0])


# Ground Stations (lat, lon) [rad]
stations_ll = np.deg2rad([
    [-35.398333, 148.981944], # Station 1 (Canberra, Australia)
    [ 40.427222, 355.749444], # Station 2 (Fort Davis, USA)
    [ 35.247164, 243.205000]  # Station 3 (Madrid, Spain)
])

# Load Measurements
df_meas = pd.read_csv(fr'HW_2\measurements_2a_noisy.csv')

# Initial Covariances & Weights
# P0: Confidence in your initial r0, v0 guess
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2
# P0 = np.diag([1e3, 1e3, 1e3, 1, 1, 1])**2
# Rk: Measurement noise floor (Range and Range-Rate)
Rk = np.diag([1e-6, 1e-12])

print("Starting Iterative Batch Filter (Differential Correction)...")

batch_filter = IterativeBatch(n_states=6)
batch_results = batch_filter.run(
    meas_df= df_meas,
    state0_initial=state0_initial,
    P0_prior=P0,
    Rk=Rk,
    coeffs=[MU_EARTH, J2, 0],  # MU_EARTH, J2, J3
    stations_ll=stations_ll,
    max_iterations=10,
    tolerance=1e-10
)

# Unpack results for plotting
dx_hist = batch_results.dx_hist
P_hist = batch_results.P_hist
x_corrected = batch_results.corrected_state_hist
postfit_residuals = batch_results.postfit_residuals

# Compute State Errors
state_errors = compute_state_error(x_corrected, df_meas, truth_file=r'HW_2\problem_2a_traj.csv')
state_errors_m = state_errors * 1e3  # Convert to meters and m/s

# ============================================================
# Plotting (State Deviations)
# ============================================================
times = df_meas['Time(s)'].values
sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
state_labels = ['x (m)', 'y (m)', 'z (m)', 'vx (m/s)', 'vy (m/s)', 'vz (m/s)']

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()
# Plot Title
fig.suptitle('Iterative Batch State Estimation Errors with 3-Sigma Bounds', fontsize=16)

for i, ax in enumerate(axes):
    # Plot the estimation error (State Deviation)
    ax.scatter(times, state_errors_m[:, i], label='Estimation Error', color='blue', s=2)
    
    # Plot the +/- 3-sigma bounds
    ax.plot(times, 3 * sigmas[:, i] * 1e3, 'r--', label=r'$\pm 3\sigma$')
    ax.plot(times, -3 * sigmas[:, i] * 1e3, 'r--')

    ax.set_ylabel(f'Error {state_labels[i]}')
    ax.grid(True, alpha=0.3)
    
    # Add legend only to the first subplot to avoid clutter
    if i == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel('Time (s)')
axes[-2].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
# ============================================================
# Residual Plotting & Gaussian Distribution Analysis
# ============================================================
postfit_residuals = np.array(postfit_residuals)*1e3
res_sigmas = np.sqrt(np.diag(Rk)) * 1000  # Convert to meters and m/s

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
fig.suptitle('Iterative Batch Post-fit Measurement Residuals and Histograms', fontsize=16)

# --- Range Residuals Time Series ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(np.array(times), np.array(postfit_residuals[:, 0], dtype=float), s=2, c='black', label='Post-fit Residual')
ax1.axhline(2*res_sigmas[0], color='r', linestyle='--', alpha=0.8, label=r'2$\sigma$ Noise Floor')
ax1.axhline(-2*res_sigmas[0], color='r', linestyle='--', alpha=0.8)
ax1.set_ylabel('Range Residual (m)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- Range Residual Histogram ---
ax2 = fig.add_subplot(gs[0, 1])
r_res = np.array(postfit_residuals[:, 0], dtype=float)
mu_r, std_r = np.mean(r_res), np.std(r_res)
count, bins, ignored = ax2.hist(r_res, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
x_range = np.linspace(mu_r - 4*std_r, mu_r + 4*std_r, 100)
ax2.plot(x_range, stats.norm.pdf(x_range, mu_r, std_r), 'r-', lw=2, label='Fitted Normal')
ax2.set_title(fr'Range PDF\n$\mu$={mu_r:.1e}, $\sigma$={std_r:.1e}')
ax2.legend(fontsize='small')

# --- Range-Rate Residual Time Series ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(np.array(times), np.array(postfit_residuals[:, 1], dtype=float)*1e3, s=2, c='black')
ax3.axhline(2*res_sigmas[1]*1e3, color='r', linestyle='--', alpha=0.8)
ax3.axhline(-2*res_sigmas[1]*1e3, color='r', linestyle='--', alpha=0.8)
ax3.set_ylabel('Range-Rate Residual (m/s)')
ax3.set_xlabel('Time (s)')
ax3.grid(True, alpha=0.3)

# --- Range-Rate Residual Histogram ---
ax4 = fig.add_subplot(gs[1, 1])
rr_res = np.array(postfit_residuals[:, 1], dtype=float)*1e3
mu_rr, std_rr = np.mean(rr_res), np.std(rr_res)
count, bins, ignored = ax4.hist(rr_res, bins=40, density=True, alpha=0.6, color='salmon', edgecolor='black')
x_rr = np.linspace(mu_rr - 4*std_rr, mu_rr + 4*std_rr, 100)
ax4.plot(x_rr, stats.norm.pdf(x_rr, mu_rr, std_rr), 'b-', lw=2, label='Fitted Normal')
ax4.set_title(fr'Range-Rate PDF\n$\mu$={mu_rr:.1e}, $\sigma$={std_rr:.1e}')
ax4.set_xlabel('Residual')
ax4.legend(fontsize='small')

plt.tight_layout()
plt.show()

# ============================================================
# Plot RMS Error Statistics over Iterations
# ============================================================

# dx_hist is a list of arrays, each containing the state adjustment for that iteration
# We calculate the RMS of the position adjustment and velocity adjustment separately
iterations = np.arange(len(dx_hist))
pos_rms_iter = []
vel_rms_iter = []

for dx in dx_hist:
    # dx is [dx, dy, dz, dvx, dvy, dvz]
    pos_rms = np.sqrt(np.mean(dx[0:3]**2)) * 1e3 # Convert to meters
    vel_rms = np.sqrt(np.mean(dx[3:6]**2)) * 1e3 # Convert to mm/s
    pos_rms_iter.append(pos_rms)
    vel_rms_iter.append(vel_rms)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Iterative Batch Convergence: RMS State Adjustment per Iteration', fontsize=16)

# --- Position Adjustment Plot ---
ax1.scatter(iterations, pos_rms_iter, color='blue', marker='o', linewidth=2, s=2)
ax1.set_yscale('log') # Log scale is best for viewing convergence
ax1.set_ylabel('RMS Position $\delta \mathbf{r}$ (m)')
ax1.grid(True, which="both", ls="-", alpha=0.3)
ax1.set_title('Convergence of Position State')

# --- Velocity Adjustment Plot ---
ax2.scatter(iterations, vel_rms_iter, color='green', marker='s', linewidth=2, s=2)
ax2.set_yscale('log')
ax2.set_ylabel('RMS Velocity $\delta \mathbf{v}$ (mm/s)')
ax2.set_xlabel('Iteration Number')
ax2.grid(True, which="both", ls="-", alpha=0.3)
ax2.set_title('Convergence of Velocity State')

plt.tight_layout()
plt.show()

print(f"Convergence Check: Final Position adjustment was {pos_rms_iter[-1]:.2e} m")

# --- Print Final Statistics ---
print(f"\n--- Final Batch Statistics ---")
print(f"RMS Range Residual:     {np.sqrt(np.mean(r_res**2)):.6e} m")
print(f"RMS Range-Rate Residual: {np.sqrt(np.mean(rr_res**2)):.6e} m/s")


report_filter_metrics(
    times=np.array(df_meas['Time(s)']),
    state_errors=state_errors,
    postfit_residuals=np.array(postfit_residuals),
    filter_name="Iterative Batch Filter",
    ignore_first_n_values=500
)