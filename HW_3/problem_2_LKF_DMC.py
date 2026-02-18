import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from your project ---
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, R_EARTH
from utils.filters.lkf_class_dmc import LKF_DMC

# ============================================================
# CONFIGURATION
# ============================================================
output_dir = 'HW_3/plots_dmc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

meas_file = r'data/measurements_2a_noisy.csv'
truth_file = r'data/problem_2a_traj.csv'

# DMC Sigma Sweep (Steady State Acceleration Sigma)
sigmas_to_test = np.logspace(-15, -2, num=15) / 1000 # Convert from m/s^2 to km/s^2 for consistency with state units

# ============================================================
# 1. INITIALIZE STATE & ORBITAL PARAMETERS
# ============================================================

# Define Truth Initial State (from Orbital Elements)
r0_true, v0_true = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
a0_true = np.zeros(3) # Truth has no "DMC" acceleration bias

# Calculate Orbit Period (P) based on Truth
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
B_mat = np.eye(3) * beta

# Pack coefficients (Mu, J2, J3, B)
coeffs = (MU_EARTH, J2, 0, B_mat)

lkf_options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10
}

# ============================================================
# 3. SETUP REFERENCE TRAJECTORY & INITIAL COVARIANCE
# ============================================================

# Initial Deviation (Error) we apply to our estimate
x_0_dev = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3, 0, 0, 0])

# Construct the Initial Reference Trajectory (The "Best Estimate")
r0_ref = r0_true + x_0_dev[:3]
v0_ref = v0_true + x_0_dev[3:6]
a0_ref = a0_true + x_0_dev[6:9]
Phi0   = np.eye(9).flatten()

# This is the big vector passed to ODE45
X_0_ref = np.concatenate([r0_ref, v0_ref, a0_ref, Phi0])

# Initial Covariance (P0)
P0 = np.diag([
    1, 1, 1,              # Position Variance (1 km^2)
    1e-3, 1e-3, 1e-3,     # Velocity Variance
    1e-6, 1e-6, 1e-6      # Acceleration Variance (Small initial guess)
])**2

# Measurement Noise Matrix
Rk = np.diag([1e-3, 1e-6])

# The initial LKF deviation state (x_hat) must be ZERO.
x_hat_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3, 0, 0, 0])

# ============================================================
# 4. LOAD DATA (MODIFIED: NO INTERPOLATION)
# ============================================================
obs = pd.read_csv(meas_file)
obs.columns = obs.columns.str.strip()

# Load Truth Data
try:
    truth_df = pd.read_csv(truth_file, sep='\s+', header=None, 
                           names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
except:
    truth_df = pd.read_csv(truth_file, delim_whitespace=True, header=None, 
                           names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

# --- KEY CHANGE: Filter Truth to match Measurement Times ---
# Instead of interpolating, we select the rows in truth_df where 
# 'Time(s)' matches the timestamps in our observation file.
meas_times = obs['Time(s)'].values

# Filter the truth dataframe
truth_aligned = truth_df[truth_df['Time(s)'].isin(meas_times)].copy()

# Sort to ensure time alignment (just in case)
truth_aligned = truth_aligned.sort_values('Time(s)')

# Extract the raw state vectors (x, y, z, vx, vy, vz) as a numpy array
# Shape: (N_measurements, 6)
truth_states_all = truth_aligned[['x', 'y', 'z', 'vx', 'vy', 'vz']].values

# Validation check to ensure data exists
if len(truth_states_all) == 0:
    raise ValueError("No matching timestamps found between Truth and Measurements! Check your data files.")

# ============================================================
# 5. SWEEP LOOP
# ============================================================
rms_pos_history = []
rms_vel_history = []
rms_res_range_history = []
rms_res_rr_history = []  

print(f"Starting DMC Sweep...")

for i, sigma in enumerate(sigmas_to_test):
    print(f"--- Run {i+1}/{len(sigmas_to_test)}: Sigma = {sigma:.1e} ---")

    # Calculate PSD (q) based on steady-state Variance (sigma^2)
    q_driving_noise = 2 * (sigma**2) * beta
    
    Q_PSD = q_driving_noise * np.eye(3)

    # Initialize and Run Filter
    lkf = LKF_DMC(n_states=9)
    results = lkf.run(obs, X_0_ref, x_hat_0, P0, Rk, Q_PSD, lkf_options)

    # --- METRICS ---    
    truth_states_k = truth_states_all[1:, :] 
    
    # Check shape alignment
    if len(results.state_hist) != len(truth_states_k):
        # Fallback if filter output length differs (e.g. if filter didn't skip t0)
        # This aligns the end of the truth array to the end of the results
        truth_states_k = truth_states_all[-len(results.state_hist):, :]

    # State Errors
    est_states_6 = results.state_hist[:, 0:6]
    pos_err = np.linalg.norm(est_states_6[:, 0:3] - truth_states_k[:, 0:3], axis=1)
    vel_err = np.linalg.norm(est_states_6[:, 3:6] - truth_states_k[:, 3:6], axis=1)
    
    rms_pos_history.append(np.sqrt(np.mean(pos_err**2)))
    rms_vel_history.append(np.sqrt(np.mean(vel_err**2)))
    
    # Residual Errors
    res_rng = results.postfit_residuals[:, 0]
    res_rr  = results.postfit_residuals[:, 1] 
    
    rms_res_range_history.append(np.sqrt(np.mean(res_rng**2)))
    rms_res_rr_history.append(np.sqrt(np.mean(res_rr**2)))

# ============================================================
# 6. PLOT
# ============================================================

# Plot 1: Position and Velocity RMS
plt.figure(figsize=(10,6))
plt.loglog(sigmas_to_test *1e3, rms_pos_history, 'b-o', label='Pos RMS (km)')
plt.loglog(sigmas_to_test *1e3, rms_vel_history, 'r-s', label='Vel RMS (km/s)')
plt.xlabel(r'Steady State Acceleration $\sigma$ [$m/s^2$]')
plt.ylabel('State RMS Error')
plt.title(f'LKF DMC State Accuracy Sweep (Tau={tau:.1f}s)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.savefig(os.path.join(output_dir, 'LKF_DMC_sweep_state_errors.png'))
plt.show()

# Plot 2: Measurement Residual RMS
plt.figure(figsize=(10,6))
plt.loglog(sigmas_to_test *1e3, rms_res_range_history, 'b-o', label='Range RMS (km)')
plt.loglog(sigmas_to_test *1e3, rms_res_rr_history, 'r-s', label='Range-Rate RMS (km/s)')
plt.xlabel(r'Steady State Acceleration $\sigma$ [$m/s^2$]')
plt.ylabel('Post-fit Residual RMS')
plt.title(f'LKF DMC Measurement Residual Sweep (Tau={tau:.1f}s)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.savefig(os.path.join(output_dir, 'LKF_DMC_sweep_residuals.png'))
plt.show()