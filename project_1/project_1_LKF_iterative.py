import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from resources.constants import MU_EARTH, J2, OMEGA_EARTH
from utils.filters.lkf_class_project_1 import LKF
from utils.plotting.post_process import post_process
from utils.ground_station_utils.get_initial_station_eci import get_initial_station_eci
from project_1.run_iterative_LKF import run_iterative_LKF

# ============================================================
# Main Execution Setup
# ============================================================

# GS state positions in ECEF frame
gs101_ecef = np.array([-5127510.0, -3794160.0,       0.0])  # Pacific Ocean
gs337_ecef = np.array([ 3860910.0,  3238490.0, 3898094.0])  # Pirinclik
gs394_ecef = np.array([  549505.0, -1380872.0, 6182197.0])  # Thule

# Initial State Deviation & Covariances
P0_diag = np.diag([
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6,   
    1e20,                           
    1e6, 1e6,                       
    1e-10, 1e-10, 1e-10,            
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6    
])

Rk = np.diag([1e-4, 1e-6])  # Measurement noise covariance
Q = np.zeros((18, 18))      # No process noise

# Initial Truth State Guess
r0 = np.array([757700.0, 5222607.0, 4851500.0])
v0 = np.array([2213.21, 4678.34, -5371.30])

# Construct A Priori Reference Trajectory Vector X_0
# We define this ONCE so we can reset to it before every run
X0_apriori = np.concatenate([
    r0,                          # [0:3] Position
    v0,                          # [3:6] Velocity
    [MU_EARTH, J2, 2.0],         # [6:9] Parameters (Mu, J2, Cd)
    get_initial_station_eci(gs101_ecef),    # [9:12] Station 1
    get_initial_station_eci(gs337_ecef),    # [12:15] Station 2
    get_initial_station_eci(gs394_ecef)     # [15:18] Station 3
])

# Load Measurements
obs = pd.read_csv(fr'data\project_1_obs.csv')

# LKF Options
options = {
    'coeffs': [MU_EARTH, J2, 0],
    'abs_tol': 1e-12,
    'rel_tol': 1e-12
}

lkf_filter = LKF(n_states=18, station_map={101:0, 337:1, 394:2})

# ============================================================
# Multi-Run Logic (Corrected)
# ============================================================

# Define the runs we want to perform
test_cases = [
    {'name': 'Baseline (No Iteration)', 'k_max': 1},
    {'name': '2 Iterations',            'k_max': 2},
    {'name': '3 Iterations',            'k_max': 3},
    {'name': '5 Iterations',            'k_max': 5}
]

# Storage for final comparison
comparison_stats = []

for case in test_cases:
    case_name = case['name']
    k_val = case['k_max']
    
    print(f"\n{'#'*80}")
    print(f"STARTING RUN: {case_name} (Max Iterations: {k_val})")
    print(f"{'#'*80}")

    # 1. Reset Initial State to A Priori
    current_X0 = X0_apriori.copy()

    # 2. Run LKF
    # returns: (LKFResults object, final_X0)
    results, final_X0_est = run_iterative_LKF(
        lkf_filter, obs, current_X0, P0_diag, Rk, Q, options,
        num_iterations_max=k_val, tol=1e-3
    )

    # 3. Calculate Means (Bias)
    # FIX: Access attributes directly using dot notation (.)
    # Do NOT convert to dictionary. Do NOT use results['key'].
    pre_res = np.array(results.prefit_residuals)
    post_res = np.array(results.postfit_residuals)

    stats = {
        'name': case_name,
        'pre_mean_range': np.nanmean(pre_res[:, 0]),
        'pre_mean_rate':  np.nanmean(pre_res[:, 1]),
        'post_mean_range': np.nanmean(post_res[:, 0]),
        'post_mean_rate':  np.nanmean(post_res[:, 1])
    }
    comparison_stats.append(stats)
    
    print(f"\n--- Stats for {case_name} ---")
    print(f"Mean Pre-fit Range:  {stats['pre_mean_range']:.4e} m")
    print(f"Mean Post-fit Range: {stats['post_mean_range']:.4e} m")

    # 4. Post-Process
    run_folder_name = f"output_run_k_{k_val}"
    os.makedirs(run_folder_name, exist_ok=True)
    
    # Inject save_folder attribute directly into the object
    # (Dataclasses allow adding attributes dynamically by default)

    post_options = {
        'save_to_timestamped_folder': True, 
        'data_mask_idx': 0,
        'results_units': 'm', 
        'plot_state_deviation': True,
        'plot_postfit_residuals': True,
        'plot_prefit_residuals': True,
        'plot_residual_comparison': True,
        'plot_covariance_trace': True,
        'plot_nis_metric': True,
        'plot_covariance_ellipsoid': True
    }
    
    print(f"Generating plots in: {run_folder_name}...")
    # Pass the OBJECT, not a dictionary
    post_process(results, obs, post_options)

# ============================================================
# Final Comparison Plot (Line Plot Version)
# ============================================================
print(f"\n{'='*60}")
print("All Runs Complete. Generating Comparison Plot...")
print(f"{'='*60}")

# Extract data for plotting
labels = [d['name'] for d in comparison_stats]
x_indices = np.arange(len(labels))

pre_range = [d['pre_mean_range'] for d in comparison_stats]
post_range = [d['post_mean_range'] for d in comparison_stats]
pre_rate = [d['pre_mean_rate'] for d in comparison_stats]
post_rate = [d['post_mean_rate'] for d in comparison_stats]

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Range Residual Means ---
axs[0].plot(x_indices, pre_range, marker='o', linestyle='--', color='darkgreen', label='Pre-fit Mean', linewidth=2, markersize=8)
axs[0].plot(x_indices, post_range, marker='s', linestyle='-', color='black', label='Post-fit Mean', linewidth=2, markersize=8)

axs[0].set_ylabel('Mean Residual (m)')
axs[0].set_title('Range Residual Mean Bias Convergence')
axs[0].set_xticks(x_indices)
axs[0].set_xticklabels(labels)
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].axhline(0, color='grey', linewidth=0.8)

# --- Plot 2: Range-Rate Residual Means ---
axs[1].plot(x_indices, pre_rate, marker='o', linestyle='--', color='darkgreen', label='Pre-fit Mean', linewidth=2, markersize=8)
axs[1].plot(x_indices, post_rate, marker='s', linestyle='-', color='black', label='Post-fit Mean', linewidth=2, markersize=8)

axs[1].set_ylabel('Mean Residual (m/s)')
axs[1].set_title('Range-Rate Residual Mean Bias Convergence')
axs[1].set_xticks(x_indices)
axs[1].set_xticklabels(labels)
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].axhline(0, color='grey', linewidth=0.8)

fig.suptitle('Iterative LKF Performance: Convergence of Residual Means', fontsize=16)
plt.tight_layout()

# Save comparison plot
plt.savefig('comparison_residual_means_line.png', dpi=300)
plt.show()

print("Comparison plot saved to 'comparison_residual_means_line.png'.")