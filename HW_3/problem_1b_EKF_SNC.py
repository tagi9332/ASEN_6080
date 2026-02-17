import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.filters.ekf_class import EKF

# ============================================================
# Configuration & Data Loading
# ============================================================

# Output directory for plots
output_dir = 'HW_3/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Define Sweep Range
sigmas_to_test = np.logspace(-15, -2, num=20) / 1000 # Convert from m/s^2 to km/s^2 for consistency with state units

# 2. File Paths
meas_file = r'data/measurements_2a_noisy.csv'
truth_file = r'data/problem_2a_traj.csv'

# 3. Load & Clean Data (Robust Method)
obs = pd.read_csv(meas_file)
obs.columns = obs.columns.str.strip() # Clean measurement columns

# Load Truth (Handling whitespace and missing headers)
truth_df = pd.read_csv(truth_file, 
                       delim_whitespace=True, 
                       header=None, 
                       names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

# 4. Prepare Truth Interpolator
truth_interp = interp1d(truth_df['Time(s)'], 
                        truth_df[['x', 'y', 'z', 'vx', 'vy', 'vz']].values, 
                        axis=0, kind='cubic', fill_value="extrapolate") # type: ignore

# ============================================================
# Initial Setup (Constant for all runs)
# ============================================================
# Initial Deviation
x_0_dev = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])

# Initial Covariance & Noise
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2
Rk = np.diag([1e-6, 1e-12])

# Reference Trajectory (Initial Guess)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
X_0 = np.concatenate([r0 + x_0_dev[:3], v0 + x_0_dev[3:], Phi0])

# Filter Options
coeffs = np.array([MU_EARTH, J2, 0])
ekf_options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10,
    'bootstrap_steps': 50, # Run as LKF for first 50 steps
    'SNC_frame': 'ECI' # ECI or RIC frame for SNC implementation

}

# ============================================================
# Sweep Loop
# ============================================================
rms_pos_history = []
rms_vel_history = []
rms_res_range_history = []
rms_res_rr_history = []

print(f"Starting EKF Parameter Sweep over {len(sigmas_to_test)} values...")

for i, sigma_acc in enumerate(sigmas_to_test):
    print(f"--- Run {i+1}/{len(sigmas_to_test)}: Sigma = {sigma_acc*1000:.1e} m/s^2 ---")

    # 1. Update Process Noise (Q)
    Q = (sigma_acc**2) * np.eye(3)

    # 2. Run EKF
    ekf_filter = EKF(n_states=6)
    results = ekf_filter.run(
        obs=obs,
        X_0=X_0,
        x_0=x_0_dev,
        P0=P0,
        Rk=Rk,
        Q=Q,
        options=ekf_options
    )

    # 3. Calculate Metrics
    # A. Get Truth at Filter Times
    # (Assuming filter output aligns with measurements, starting from 2nd measurement usually)
    filter_times = obs['Time(s)'].values[1:len(results.state_hist)+1]
    truth_states = truth_interp(filter_times)

    # B. State Errors
    # Note: For EKF, state_hist is typically the Total State (Ref + Deviation),
    # unlike LKF where it might be just deviation. Ensure your EKF class returns Total State.
    # If results.state_hist is deviation only, add it to reference. 
    # Assuming here results.state_hist is the FINAL TOTAL ESTIMATE.
    state_errors = results.state_hist - truth_states
    
    pos_err_3d = np.linalg.norm(state_errors[:, 0:3], axis=1)
    vel_err_3d = np.linalg.norm(state_errors[:, 3:6], axis=1)

    rms_pos = np.sqrt(np.mean(pos_err_3d**2))
    rms_vel = np.sqrt(np.mean(vel_err_3d**2))
    
    rms_pos_history.append(rms_pos)
    rms_vel_history.append(rms_vel)

    # C. Residual Errors
    res_range = results.postfit_residuals[:, 0]
    res_rr    = results.postfit_residuals[:, 1]
    
    rms_res_range = np.sqrt(np.mean(res_range**2))
    rms_res_rr    = np.sqrt(np.mean(res_rr**2))
    
    rms_res_range_history.append(rms_res_range)
    rms_res_rr_history.append(rms_res_rr)

print("\nSweep Complete. Generating Plots...")

# ============================================================
# Plotting
# ============================================================

# Plot (i): Post-fit Measurement Residual RMS vs Sigma
plt.figure(figsize=(10, 6))
plt.loglog(sigmas_to_test*1e3, rms_res_range_history, 'b-o', label='Range RMS (km)')
plt.loglog(sigmas_to_test*1e3, rms_res_rr_history, 'r-s', label='Range-Rate RMS (km/s)')
plt.xlabel(r'Process Noise Sigma $\sigma$ [$m/s^2$]')
plt.ylabel('Post-fit Residual RMS')
plt.title('i. EKF Measurement Residual RMS vs. Process Noise Sigma')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'EKF_SNC_sweep_residuals.png'))

# Plot (ii): 3D State Error RMS vs Sigma
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel(r'Process Noise Sigma $\sigma$ [$m/s^2$]')
ax1.set_ylabel('3D Position RMS [km]', color=color)
ax1.loglog(sigmas_to_test*1e3, rms_pos_history, 'o-', color=color, label='3D Position RMS')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which="both", ls="-", alpha=0.5)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('3D Velocity RMS [km/s]', color=color)
ax2.loglog(sigmas_to_test*1e3, rms_vel_history, 's-', color=color, label='3D Velocity RMS')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('ii. EKF 3D State Error RMS vs. Process Noise Sigma')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'EKF_SNC_sweep_state_errors.png'))
