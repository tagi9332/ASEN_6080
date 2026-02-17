import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from your project ---
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, R_EARTH
from utils.filters.lkf_class_dmc import LKF_DMC
from utils.plotting.post_process import post_process

# ============================================================
# CONFIGURATION
# ============================================================
output_dir = 'HW_3/plots_dmc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

meas_file = r'data/measurements_2a_noisy.csv'
truth_file = r'data/problem_2a_traj.csv'

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
    1e-6, 1e-6, 1e-6,     # Velocity Variance
    1e-9, 1e-9, 1e-9      # Acceleration Variance (Small initial guess)
])

# Measurement Noise Matrix
Rk = np.diag([1e-6, 1e-12])

x_hat_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3, 0, 0, 0])

# ============================================================
# 4. LOAD DATA
# ============================================================
obs = pd.read_csv(meas_file)
obs.columns = obs.columns.str.strip()

# Handle potential deprecated delim_whitespace
try:
    truth_df = pd.read_csv(truth_file, sep='\s+', header=None, 
                           names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
except:
    truth_df = pd.read_csv(truth_file, delim_whitespace=True, header=None, 
                           names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

truth_interp = interp1d(truth_df['Time(s)'], 
                        truth_df[['x', 'y', 'z', 'vx', 'vy', 'vz']].values, 
                        axis=0, kind='cubic', fill_value="extrapolate") # type: ignore

# ============================================================
# RUN LKF
# ============================================================
sigma = 1e-6 / 1000

# Calculate PSD (q) based on steady-state Variance (sigma^2)
q_driving_noise = 2 * (sigma**2) * beta

Q_PSD = q_driving_noise * np.eye(3)

# Initialize and Run Filter
lkf = LKF_DMC(n_states=9)
results = lkf.run(obs, X_0_ref, x_hat_0, P0, Rk, Q_PSD, lkf_options)

# Run post-processing
post_options = {
    'truth_traj_file': r'data\problem_2a_traj.csv',
    'save_to_timestamped_folder': True,
    'data_mask_idx': 0,
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

post_process(results,obs,post_options)





