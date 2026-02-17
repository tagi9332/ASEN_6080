import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# Adjust path to find local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from your project ---
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, R_EARTH
from utils.filters.ekf_class_dmc import EKF 

# ============================================================
# CONFIGURATION
# ============================================================
output_dir = 'HW_3/plots_ekf_dmc_sweep'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

meas_file = r'data/measurements_2a_noisy.csv'
truth_file = r'data/problem_2a_traj.csv'

# 1. Define Sweep Range (Steady State Acceleration Sigma)
# Logspace from 1e-15 to 1e-2 m/s^2, converted to km/s^2
sigmas_test_m_s2 = np.logspace(-15, -2, num=15)
sigmas_test_km_s2 = sigmas_test_m_s2 / 1000.0 

# ============================================================
# 2. INITIALIZE STATE & ORBITAL PARAMETERS
# ============================================================

# Define Truth Initial State
r0_true, v0_true = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
a0_true = np.zeros(3) 

# Calculate Orbit Period (P) for DMC tuning
r_mag = np.linalg.norm(r0_true)
v_mag = np.linalg.norm(v0_true)
energy = (v_mag**2 / 2) - (MU_EARTH / r_mag)
a_sma = -MU_EARTH / (2 * energy)
P_period = 2 * np.pi * np.sqrt(a_sma**3 / MU_EARTH)

print(f"Orbital Period P: {P_period:.2f} s")

# ============================================================
# 3. CONFIGURE DMC CONSTANTS
# ============================================================
tau = P_period / 30.0
beta = 1.0 / tau
print(f"DMC Time Constant tau: {tau:.2f} s")

# B Matrix (3x3 diagonal)
B_mat = np.eye(3) * beta

# Pack coefficients (Mu, J2, J3, B)
coeffs = (MU_EARTH, J2, 0, B_mat)

ekf_options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10,
    'dt_max': 60.0,
    'bootstrap_steps': 100
}

# ============================================================
# 4. SETUP INITIAL ESTIMATE & COVARIANCE
# ============================================================

# Initial Deviation (Error)
x_0_dev = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3, 0, 0, 0])

# Construct Initial Estimate (X_0)
r0_est = r0_true + x_0_dev[:3]
v0_est = v0_true + x_0_dev[3:6]
a0_est = a0_true + x_0_dev[6:9]

# EKF State: [rx, ry, rz, vx, vy, vz, ax, ay, az]
X_0_est = np.concatenate([r0_est, v0_est, a0_est])

# Initial Covariance (P0)
P0 = np.diag([
    1, 1, 1,              # Pos (km^2)
    1e-6, 1e-6, 1e-6,     # Vel (km^2/s^2)
    1e-9, 1e-9, 1e-9      # Accel (km^2/s^4)
])

# Measurement Noise
Rk = np.diag([1e-6, 1e-12])

# Initial Deviation estimate (0 for EKF)
x_hat_0 = np.zeros(9)

# ============================================================
# 5. LOAD & ALIGN DATA
# ============================================================
obs = pd.read_csv(meas_file)
obs.columns = obs.columns.str.strip()

# Load Truth
try:
    truth_df = pd.read_csv(truth_file, sep='\s+', header=None, 
                           names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
except:
    truth_df = pd.read_csv(truth_file, delim_whitespace=True, header=None, 
                           names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

# The EKF implementation typically processes measurements starting from index k=1 
# (skipping the initial condition at k=0).
# We filter the truth data to match exactly the timestamps the filter will output.
filter_meas_times = obs['Time(s)'].iloc[1:].values
truth_aligned = truth_df[truth_df['Time(s)'].isin(filter_meas_times)].copy()
truth_aligned = truth_aligned.sort_values('Time(s)')
truth_states_k = truth_aligned[['x', 'y', 'z', 'vx', 'vy', 'vz']].values

# ============================================================
# 6. SWEEP LOOP
# ============================================================
rms_pos_history = []
rms_vel_history = []
rms_res_range_history = []
rms_res_rr_history = []

print(f"\nStarting EKF DMC Sweep over {len(sigmas_test_km_s2)} values...")

for i, sigma_km in enumerate(sigmas_test_km_s2):
    sigma_m = sigmas_test_m_s2[i]
    print(f"--- Run {i+1}/{len(sigmas_test_km_s2)}: Sigma = {sigma_m:.1e} m/s^2 ---")

    # 1. Update Process Noise (Q_PSD)
    # Formula: q = 2 * sigma^2 * beta
    q_driving_noise = 2 * (sigma_km**2) * beta
    Q_PSD = q_driving_noise * np.eye(3)

    # 2. Run EKF
    ekf = EKF(n_states=9)
    results = ekf.run(obs, X_0_est, x_hat_0, P0, Rk, Q_PSD, ekf_options)

    # 3. Calculate Metrics
    # Ensure we only compare against aligned truth data
    # (Defensive check: slice to min length in case filter crashed early)
    n_pts = min(len(results.state_hist), len(truth_states_k))
    
    # Extract Position/Velocity (first 6 cols) from filter output
    # Note: results.state_hist might be (N, 9) or (N, 6) depending on your class fix.
    # We slice [:, :6] to be safe.
    est_states = results.state_hist[:n_pts, :6]
    truth_subset = truth_states_k[:n_pts, :]

    # State Errors
    pos_err = np.linalg.norm(est_states[:, 0:3] - truth_subset[:, 0:3], axis=1)
    vel_err = np.linalg.norm(est_states[:, 3:6] - truth_subset[:, 3:6], axis=1)

    rms_pos = np.sqrt(np.mean(pos_err**2))
    rms_vel = np.sqrt(np.mean(vel_err**2))
    
    rms_pos_history.append(rms_pos)
    rms_vel_history.append(rms_vel)

    # Residual Errors
    res_range = results.postfit_residuals[:n_pts, 0]
    res_rr    = results.postfit_residuals[:n_pts, 1]
    
    rms_res_range = np.sqrt(np.mean(res_range**2))
    rms_res_rr    = np.sqrt(np.mean(res_rr**2))
    
    rms_res_range_history.append(rms_res_range)
    rms_res_rr_history.append(rms_res_rr)

print("\nSweep Complete. Generating Plots...")

# ============================================================
# 7. PLOTTING
# ============================================================

# Plot (i): Post-fit Measurement Residual RMS vs Sigma
plt.figure(figsize=(10, 6))
# X-axis: m/s^2 (for readability), Y-axis: km
plt.loglog(sigmas_test_m_s2, rms_res_range_history, 'b-o', label='Range RMS (km)')
plt.loglog(sigmas_test_m_s2, rms_res_rr_history, 'r-s', label='Range-Rate RMS (km/s)')
plt.xlabel(r'Steady State Acceleration $\sigma$ [$m/s^2$]')
plt.ylabel('Post-fit Residual RMS')
plt.title(f'i. EKF (DMC) Residual RMS vs. Process Noise Sigma\n(Tau={tau:.1f}s)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'EKF_DMC_sweep_residuals.png'))

# Plot (ii): 3D State Error RMS vs Sigma
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel(r'Steady State Acceleration $\sigma$ [$m/s^2$]')
ax1.set_ylabel('3D Position RMS [km]', color=color)
ax1.loglog(sigmas_test_m_s2, rms_pos_history, 'o-', color=color, label='3D Position RMS')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which="both", ls="-", alpha=0.5)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('3D Velocity RMS [km/s]', color=color)
ax2.loglog(sigmas_test_m_s2, rms_vel_history, 's-', color=color, label='3D Velocity RMS')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f'ii. EKF (DMC) State Error RMS vs. Process Noise Sigma\n(Tau={tau:.1f}s)')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'EKF_DMC_sweep_state_errors.png'))
plt.show()

print(f"Plots saved to {output_dir}")