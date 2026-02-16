import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ============================================================
# Imports & Constants (Preserved from your snippet)
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.filters.lkf_class import LKF
from utils.plotting.post_process import post_process

# ============================================================
# Configuration
# ============================================================

# Output directory for plots
output_dir = 'HW_3/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Define the Sweep Range (Sigma for Process Noise)
sigmas_to_test = np.logspace(-15, -2, num=28) 

# 2. File Paths
meas_file = r'data/measurements_2a_noisy.csv'
truth_file = r'data/problem_2a_traj.csv'

# 3. Initial Setup
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
P0 = np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6])
Rk = np.diag([1e-6, 1e-12])

r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
X_0_ref = np.concatenate([r0 + x_0[:3], v0 + x_0[3:], Phi0])

# ============================================================
# LOAD AND CLEAN DATA
# ============================================================
obs = pd.read_csv(meas_file)

# CORRECTED TRUTH FILE LOADING
# 1. delim_whitespace=True: Handles the spaces between numbers
# 2. header=None: Tells pandas there is no header row
# 3. names=[...]: Manually assigns the column names you need
truth_df = pd.read_csv(truth_file, 
                       delim_whitespace=True, 
                       header=None, 
                       names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

# Clean up measurement columns just in case
obs.columns = obs.columns.str.strip()

# Prepare Truth Interpolator
# Now that column names are manually assigned, this will work
truth_interp = interp1d(truth_df['Time(s)'], 
                        truth_df[['x', 'y', 'z', 'vx', 'vy', 'vz']].values, 
                        axis=0, kind='cubic', fill_value="extrapolate") # type: ignore

# Prepare Truth Interpolator (ensures we compare state at exact timestamps)
# Assuming truth file has columns: 'Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'
truth_interp = interp1d(truth_df['Time(s)'], 
                        truth_df[['x', 'y', 'z', 'vx', 'vy', 'vz']].values, 
                        axis=0, kind='cubic', fill_value="extrapolate") # type: ignore

coeffs = [MU_EARTH, J2, 0] 
lkf_options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10,
    'SNC_frame': 'ECI' # ECI or RIC frame for SNC implementation

}

# ============================================================
# Storage for Sweep Metrics
# ============================================================
rms_pos_history = []
rms_vel_history = []
rms_res_range_history = []
rms_res_rr_history = []

# ============================================================
# Main Sweep Loop
# ============================================================
print(f"Starting SNC Parameter Sweep over {len(sigmas_to_test)} values...")

for i, sigma_acc in enumerate(sigmas_to_test):
    print(f"--- Run {i+1}/{len(sigmas_to_test)}: Sigma = {sigma_acc:.1e} m/s^2 ---")

    # 1. Define Process Noise Spectral Density
    Q_PSD = (sigma_acc**2) * np.eye(3) 

    # 2. Run Filter
    lkf_filter = LKF(n_states=6)
    results = lkf_filter.run(obs, X_0_ref, x_0, P0, Rk, Q_PSD, lkf_options)
    
    # --- CALCULATE METRICS FOR PLOTS (i) and (ii) ---
    
    # A. Get Truth State at the measurement times used in the filter
    # 'results.state_hist' typically aligns with the filter steps. 
    # We must ensure we grab the truth at the corresponding times.
    # Note: LKF usually outputs N-1 steps (skips t0), check your implementation.
    # Here we assume results.state_hist aligns with obs['Time(s)'][1:] or similar.
    
    # Extract times corresponding to the filter output
    # (Adjust '1:' based on whether your LKF outputs t0 or starts at t1)
    filter_times = obs['Time(s)'].values[1:len(results.state_hist)+1] 
    
    # Get Truth at these times
    truth_states = truth_interp(filter_times)
    
    # B. Compute State Errors (Estimated - Truth)
    # results.state_hist is (N, 6)
    state_errors = results.state_hist - truth_states
    
    pos_err_3d = np.linalg.norm(state_errors[:, 0:3], axis=1)
    vel_err_3d = np.linalg.norm(state_errors[:, 3:6], axis=1)
    
    # C. Compute RMS (Root Mean Square)
    rms_pos = np.sqrt(np.mean(pos_err_3d**2))
    rms_vel = np.sqrt(np.mean(vel_err_3d**2))
    
    rms_pos_history.append(rms_pos)
    rms_vel_history.append(rms_vel)

    # D. Compute Post-fit Residual RMS
    # results.postfit_residuals is (N, 2) where col 0 is Range, col 1 is Range-Rate
    res_range = results.postfit_residuals[:, 0]
    res_rr    = results.postfit_residuals[:, 1]
    
    rms_res_range = np.sqrt(np.mean(res_range**2))
    rms_res_rr    = np.sqrt(np.mean(res_rr**2))
    
    rms_res_range_history.append(rms_res_range)
    rms_res_rr_history.append(rms_res_rr)

    # (Optional) Run full post-processing if you still want the individual PDF reports
    # post_process(results, obs, post_options)

print("\nSweep Complete. Generating Summary Plots...")

# ============================================================
# PLOTTING
# ============================================================

# Plot (i): Post-fit Measurement Residual RMS vs Sigma
plt.figure(figsize=(10, 6))
plt.loglog(sigmas_to_test, rms_res_range_history, 'b-o', label='Range RMS (km)')
plt.loglog(sigmas_to_test, rms_res_rr_history, 'r-s', label='Range-Rate RMS (km/s)')
plt.xlabel(r'Process Noise Sigma $\sigma$ [$m/s^2$]')
plt.ylabel('Post-fit Residual RMS')
plt.title('i. Measurement Residual RMS vs. Process Noise Sigma')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.savefig(os.path.join(output_dir, 'LKF_SNC_sweep_residuals.png'))
# plt.show()

# Plot (ii): 3D RMS Position and Velocity Errors vs Sigma
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel(r'Process Noise Sigma $\sigma$ [$m/s^2$]')
ax1.set_ylabel('3D Position RMS [km]', color=color)
ax1.loglog(sigmas_to_test, rms_pos_history, 'o-', color=color, label='3D Position RMS')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which="both", ls="-", alpha=0.5)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax2.set_ylabel('3D Velocity RMS [km/s]', color=color)
ax2.loglog(sigmas_to_test, rms_vel_history, 's-', color=color, label='3D Velocity RMS')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('ii. 3D State Error RMS vs. Process Noise Sigma')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(os.path.join(output_dir, 'LKF_SNC_sweep_state_errors.png'))
# plt.show()