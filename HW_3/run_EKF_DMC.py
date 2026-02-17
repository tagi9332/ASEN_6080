import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os, sys

# Adjust path to find local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from your project ---
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, R_EARTH
# CHANGED: Import the EKF class instead of LKF
from utils.filters.ekf_class_dmc import EKF 
from utils.plotting.post_process import post_process

# ============================================================
# CONFIGURATION
# ============================================================
output_dir = 'HW_3/plots_ekf_dmc' # CHANGED: Update output folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

meas_file = r'data/measurements_2a_noisy.csv'
truth_file = r'data/problem_2a_traj.csv'

# ============================================================
# 1. INITIALIZE STATE & ORBITAL PARAMETERS
# ============================================================

# Define Truth Initial State (from Orbital Elements)
r0_true, v0_true = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
a0_true = np.zeros(3) # Truth has no bias

# Calculate Orbit Period (P) for DMC tuning
r_mag = np.linalg.norm(r0_true)
v_mag = np.linalg.norm(v0_true)
energy = (v_mag**2 / 2) - (MU_EARTH / r_mag)
a_sma = -MU_EARTH / (2 * energy)
P_period = 2 * np.pi * np.sqrt(a_sma**3 / MU_EARTH)

print(f"Orbital Period P: {P_period:.2f} s")

# ============================================================
# 2. CONFIGURE DMC CONSTANTS
# ============================================================
# Set Time Constant relative to Period
tau = P_period / 30.0
beta = 1.0 / tau
print(f"DMC Time Constant tau: {tau:.2f} s")

# Construct B Matrix (3x3 diagonal)
# This is required by the DMC ODE
B_mat = np.eye(3) * beta

# Pack coefficients (Mu, J2, J3, B)
coeffs = (MU_EARTH, J2, 0, B_mat)

ekf_options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10,
    
    'dt_max': 60.0 # Standardize propagation steps
}

# ============================================================
# 3. SETUP INITIAL ESTIMATE & COVARIANCE
# ============================================================

# Initial Deviation (Error) we apply to our initial estimate
# We purposely start the filter slightly off from truth
x_0_dev = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3, 0, 0, 0]) # No initial bias in acceleration

# Construct the Initial Estimate (X_0)
# EKF State is 9x1: [rx, ry, rz, vx, vy, vz, ax, ay, az]
r0_est = r0_true + x_0_dev[:3]
v0_est = v0_true + x_0_dev[3:6]
a0_est = a0_true + x_0_dev[6:9]

# NOTE: EKF does not need the flattened STM appended to the state
X_0_est = np.concatenate([r0_est, v0_est, a0_est])

# Initial Covariance (P0)
# 9x9 Matrix
P0 = np.diag([
    1, 1, 1,              # Position Variance (1 km^2)
    1e-6, 1e-6, 1e-6,     # Velocity Variance
    1e-9, 1e-9, 1e-9      # Acceleration Variance (Small initial guess)
])

# Measurement Noise Matrix (R)
Rk = np.diag([1e-6, 1e-12])

# Initial Deviation estimate (Usually 0 for EKF initialization)
x_hat_0 = np.zeros(9)

# ============================================================
# 4. PROCESS NOISE SETUP (DMC)
# ============================================================
sigma = 1e-6 / 1000 # Convert from m/s^2 to km/s^2 for consistency with state units

# Calculate PSD (q) based on steady-state Variance (sigma^2)
# Formula: q = 2 * sigma^2 * beta
q_driving_noise = 2 * (sigma**2) * beta
Q_PSD = q_driving_noise * np.eye(3)

print(f"DMC Sigma: {sigma}")
print(f"Calculated Q_PSD: {q_driving_noise}")

# ============================================================
# 5. LOAD DATA
# ============================================================
obs = pd.read_csv(meas_file)
obs.columns = obs.columns.str.strip()

# ============================================================
# 6. RUN EKF
# ============================================================

print("\n--- Starting EKF Run ---")
# Initialize EKF Class
ekf = EKF(n_states=9)

# Run Filter
# Note: X_0_est is the 9-element vector. 
# The EKF handles STM integration internally.
results = ekf.run(obs, X_0_est, x_hat_0, P0, Rk, Q_PSD, ekf_options)

# ============================================================
# 7. POST-PROCESSING
# ============================================================
post_options = {
    'truth_traj_file': r'data/problem_2a_traj.csv',
    'save_to_timestamped_folder': True,
    'data_mask_idx': 50,
    'plot_state_errors': True,
    'plot_state_deviation': True,
    'plot_postfit_residuals': True,
    'plot_prefit_residuals': True,
    'plot_residual_comparison': True,
    'plot_covariance_trace': True,
    'plot_filter_consistency': True,
    'plot_covariance_ellipsoid': True,
    'plot_nis_metric': True
}

# The results object structure is identical to LKF, so post_process works directly
post_process(results, obs, post_options)

print(f"\nProcessing complete. Plots saved to: {output_dir}")