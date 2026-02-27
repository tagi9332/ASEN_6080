import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Imports & Constants (Adjusted to your environment)
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, R_EARTH
from utils.filters.srif_class import SRIF

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

X_nom_srif = np.concatenate([r0+x_0[:3], v0+x_0[3:]])
Q_srif = np.diag([0, 0, 0, 0, 0, 0])  # No process noise
uBar = np.zeros(3)                    # Mean process noise vector

# ============================================================
# 1. Run SRIF (Forced Upper Triangular)
# ============================================================
print("--- Running SRIF (force_upper_triangular=True) ---")
srif_filter_forced = SRIF(n_states=6)
results_forced = srif_filter_forced.run(
    obs=obs, X_0=X_nom_srif, x_0=x_0, P0=P0, Q0=Q_srif, 
    uBar=uBar, R_meas=Rk, options=base_options, 
    force_upper_triangular=True
)

# ============================================================
# 2. Run SRIF (Not Forced)
# ============================================================
print("--- Running SRIF (force_upper_triangular=False) ---")
srif_filter_unforced = SRIF(n_states=6)
results_unforced = srif_filter_unforced.run(
    obs=obs, X_0=X_nom_srif, x_0=x_0, P0=P0, Q0=Q_srif, 
    uBar=uBar, R_meas=Rk, options=base_options, 
    force_upper_triangular=False
)

# ============================================================
# 3. Plot Differences in State Deviation
# ============================================================
print("--- Plotting SRIF (Forced) vs SRIF (Unforced) Differences ---")

mask_idx = 0  # Adjust if you want to skip initial transients

dev_forced = results_forced.dx_hist  
dev_unforced = results_unforced.dx_hist

# Calculate difference
dev_diff = dev_forced - dev_unforced

# Convert from km and km/s to mm and mm/s
dev_diff[:, :] *= 1e6  

if len(time_eval) > mask_idx:
    time_eval = time_eval[mask_idx:]
    if dev_diff.shape[0] > dev_diff.shape[1]: 
        dev_diff = dev_diff[mask_idx:, :] 
    else: 
        dev_diff = dev_diff[:, mask_idx:] 

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
fig.suptitle('Difference in State Deviations\n(Forced Triangular - Unforced Triangular)', fontsize=16)

labels = ['X Position Diff (mm)', 'Y Position Diff (mm)', 'Z Position Diff (mm)',
          'X Velocity Diff (mm/s)', 'Y Velocity Diff (mm/s)', 'Z Velocity Diff (mm/s)']

for i in range(6):
    row = i % 3
    col = i // 3
    ax = axes[row, col]
    
    if dev_diff.shape[0] == len(time_eval):
        ax.scatter(time_eval, dev_diff[:, i], label=labels[i], color='teal', s=1)
    else:
        ax.scatter(time_eval, dev_diff[i, :], label=labels[i], color='teal', s=1)
        
    ax.set_ylabel(labels[i])
    ax.grid(True, linestyle='--', alpha=0.6)
    
axes[2, 0].set_xlabel('Time (s)')
axes[2, 1].set_xlabel('Time (s)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()