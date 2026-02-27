import os, sys
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.plotting.post_process import post_process

# Filter Imports
from utils.filters.srif_class import SRIF
from utils.filters.lkf_class import LKF

# ============================================================
# Main Execution: Shared Setup
# ============================================================
# Initial State Deviation & Covariances
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
P0 = np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6])
Rk = np.diag([1e-6, 1e-12])

# Initial Truth State (Nominal Reference)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')

# Load Measurements
obs = pd.read_csv(r'data\measurements_noisy.csv')
time_eval = obs['Time(s)'].values

# ODE arguments
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for dynamics
base_options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10
}

# ============================================================
# 1. Run SRIF
# ============================================================
print("--- Initializing and Running SRIF ---")
X_nom_srif = np.concatenate([r0+x_0[:3], v0+x_0[3:]])
Q_srif = np.diag([0, 0, 0, 0, 0, 0])  # No process noise
uBar = np.zeros(3)                    # Mean process noise vector

srif_filter = SRIF(n_states=6)
srif_results = srif_filter.run(
    obs=obs, 
    X_0=X_nom_srif, 
    x_0=x_0,    
    P0=P0, 
    Q0=Q_srif, 
    uBar=uBar, 
    R_meas=Rk, 
    options=base_options
)

# ============================================================
# 2. Run LKF
# ============================================================
print("--- Initializing and Running LKF ---")
Phi0 = np.eye(6).flatten()
X_0_lkf = np.concatenate([r0+x_0[:3], v0+x_0[3:], Phi0])
sigma_a = 0 # km/s^2
Q_psd = sigma_a**2 * np.eye(3)

lkf_options = base_options.copy()
lkf_options.update({
    'method': 'SNC',          # Chosen method (SNC or DMC)
    'frame_type': 'ECI',      # Frame to apply noise in (RIC or ECI)
    'Q_cont': Q_psd,          # Continuous PSD matrix
    'threshold': 10.0,        # Max dt to prevent instability
    'B': None,                 # Not needed for SNC
    'potter_form': True        # Toggle for Potter formulation
})

lkf_filter = LKF(n_states=6)
lkf_results = lkf_filter.run(
    obs, 
    X_0_lkf, 
    x_0, 
    P0, 
    Rk, 
    lkf_options
)

# ============================================================
# 3. Post-Process (Optional Standard Plots)
# ============================================================
print("--- Running Standard Post-Processing for SRIF and LKF ---")

mask_idx = 0  # Mask out initial transient data for better visualization

post_options = {
    'truth_traj_file': r'data\HW1_truth.csv',
    'save_to_timestamped_folder': True,
    'data_mask_idx': mask_idx,
    'plot_state_errors': True,
    'plot_state_deviation': True,
    'plot_postfit_residuals': True,
    'plot_prefit_residuals': True,
    'plot_covariance_ellipsoid': True,
    'plot_residual_comparison': True,
    'plot_covariance_trace': True,
    'plot_filter_consistency': True,
    'plot_nis_metric': True
}

# Run for SRIF
print("--> Generating SRIF Plots...")
post_process(srif_results, obs, post_options)

# Brief pause to ensure a new timestamped folder is created for the LKF

# Run for LKF
print("--> Generating LKF Plots...")
post_process(lkf_results, obs, post_options)
# ============================================================
# 4. Plot Differences in State Deviation
# ============================================================
print("--- Plotting SRIF vs LKF Deviation Differences ---")

# Extract state deviations over time.
# NOTE: Update 'x_hat' to whatever attribute your filter classes use to store the deviations.
srif_dev = srif_results.dx_hist  
lkf_dev = lkf_results.dx_hist


# Ensure shapes match and time aligns (assuming Nx6 format)
if srif_dev.shape != lkf_dev.shape:
    print(f"Warning: SRIF shape {srif_dev.shape} does not match LKF shape {lkf_dev.shape}.")
    # Transpose if one is 6xN and the other is Nx6
    if srif_dev.shape[::-1] == lkf_dev.shape:
        lkf_dev = lkf_dev.T

# Calculate difference
dev_diff = srif_dev - lkf_dev

# Convert from km and km/s to mm and mm/s for better visualization
dev_diff[:, :] *= 1e6  # Position differences in mm

# Apply a mask to filter out initial transient data if needed
if len(time_eval) > mask_idx:
    time_eval = time_eval[mask_idx:]
    
    # Mask appropriately whether it's Nx6 or 6xN
    if dev_diff.shape[0] > dev_diff.shape[1]: 
        dev_diff = dev_diff[mask_idx:, :] # Nx6 format
    else: 
        dev_diff = dev_diff[:, mask_idx:] # 6xN format

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
fig.suptitle('Difference in State Deviations (SRIF - LKF)', fontsize=16)

labels = ['X Position Diff (mm)', 'Y Position Diff (mm)', 'Z Position Diff (mm)',
          'X Velocity Diff (mm/s)', 'Y Velocity Diff (mm/s)', 'Z Velocity Diff (mm/s)']

for i in range(6):
    row = i % 3
    col = i // 3
    ax = axes[row, col]
    
    # Check if array is Nx6 or 6xN and plot accordingly
    if dev_diff.shape[0] == len(time_eval):
        ax.scatter(time_eval, dev_diff[:, i], label=labels[i], color='purple', s=1)
    else:
        ax.scatter(time_eval, dev_diff[i, :], label=labels[i], color='purple', s=1)
        
    ax.set_ylabel(labels[i])
    ax.grid(True, linestyle='--', alpha=0.6)
    
axes[2, 0].set_xlabel('Time (s)')
axes[2, 1].set_xlabel('Time (s)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()