import os, sys
import numpy as np
import pandas as pd

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from resources.constants import MU_EARTH, J2, OMEGA_EARTH
from utils.filters.ekf_class_project_1 import EKF
from utils.plotting.post_process import post_process

# ============================================================
# Helper Functions
# ============================================================
def get_initial_station_eci(station_ecef, t_offset=0):
    """
    Rotates ECEF coordinates to ECI based on the problem's theta formula.
    theta = omega_earth * t
    """
    theta = OMEGA_EARTH * t_offset
    c = np.cos(theta)
    s = np.sin(theta)
    
    R_ecef2eci = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    return R_ecef2eci @ station_ecef

# ============================================================
# Main Execution
# ============================================================

# GS state positions in ECEF frame
gs101_ecef = np.array([-5127510.0, -3794160.0,       0.0])  # Pacific Ocean
gs337_ecef = np.array([ 3860910.0,  3238490.0, 3898094.0])  # Pirinclik
gs394_ecef = np.array([  549505.0, -1380872.0, 6182197.0])  # Thule

# Initial Covariance Matrix (P0)
# State Order: [r(3), v(3), mu, J2, Cd, GS1(3), GS2(3), GS3(3)]
P0_diag = np.diag([
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6,   # r, v (Loose)
    1e20,                           # mu (Very Loose)
    1e6,                            # J2 (Loose)
    1e6,                            # Cd (Loose)
    1e-10, 1e-10, 1e-10,            # GS1 (Locked - assuming known)
    1e6, 1e6, 1e6,                  # GS2 (Loose - Estimating)
    1e6, 1e6, 1e6                   # GS3 (Loose - Estimating)
])

# Tuning Matrices
Rk = np.diag([1e-4, 1e-6])   # Measurement noise [Range (km^2), Range-Rate (km^2/s^2)]
Q = np.zeros((18, 18))       # Process noise (Deterministic dynamics)

# Initial Truth State Guess
r0 = np.array([757700.0, 5222607.0, 4851500.0])
v0 = np.array([2213.21, 4678.34, -5371.30])

# Construct Initial State Vector X_0 (18 Elements)
X_0 = np.concatenate([
    r0,                          # [0:3] Position
    v0,                          # [3:6] Velocity
    [MU_EARTH, J2, 2.0],         # [6:9] Parameters (Mu, J2, Cd)
    get_initial_station_eci(gs101_ecef),    # [9:12] Station 1
    get_initial_station_eci(gs337_ecef),    # [12:15] Station 2
    get_initial_station_eci(gs394_ecef)     # [15:18] Station 3
])

# Load Measurements
obs = pd.read_csv(fr'data\project_1_obs.csv')

# Integration Options
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for dynamics
options = {
    'coeffs': coeffs,
    'bootstrap_steps': 384,
    'abs_tol': 1e-12,
    'rel_tol': 1e-12
}

# ============================================================
# Initialize and Run EKF
# ============================================================
print(f"{'='*60}")
print(f"Starting Extended Kalman Filter (Single Pass)")
print(f"{'='*60}")

# Initialize EKF Class
ekf_filter = EKF(n_states=18, station_map={101:0, 337:1, 394:2})

# Run Filter (Note: EKF does not take x_0 deviation, only full state X_0)
results = ekf_filter.run(obs, X_0, P0_diag, Rk, Q, options)

# ============================================================
# Post Processing
# ============================================================
print(f"\n{'='*60}")
print("Filter Complete. Running Post-Processing...")
print(f"{'='*60}")

# Calculate Final Residual RMS
rms_range = np.sqrt(np.mean(results.postfit_residuals[:,0]**2))
rms_rr = np.sqrt(np.mean(results.postfit_residuals[:,1]**2))

print(f"Final RMS Statistics:")
print(f"  Range RMS:      {rms_range:.4f} m")
print(f"  Range-Rate RMS: {rms_rr:.4f} m/s")

# Display Final Estimated Parameters
X_final = results.state_hist[-1]
print("\nFinal State Estimate:")
print(f"  Position: {X_final[0:3]}")
print(f"  Velocity: {X_final[3:6]}")
print(f"  Mu:       {X_final[6]:.4e}")
print(f"  J2:       {X_final[7]:.4e}")
print(f"  Cd:       {X_final[8]:.4f}")

post_options = {
    'save_to_timestamped_folder': True,
    'data_mask_idx': 0,
    'results_units': 'm', # 'm' or 'km'
    'plot_state_deviation': False, # Less relevant for EKF as we don't have a ref trajectory to deviate FROM
    'plot_postfit_residuals': True,
    'plot_prefit_residuals': True,
    'plot_residual_comparison': True,
    'plot_covariance_trace': True,
    'plot_nis_metric': True
}

# Generate Plots
post_process(results, obs, post_options)