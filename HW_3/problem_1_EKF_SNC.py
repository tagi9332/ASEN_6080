import os, sys
import numpy as np
import pandas as pd

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.filters.ekf_class import EKF
from utils.plotting.post_process import post_process

# Output directory for plots
output_dir = 'HW_3/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ============================================================
# Main Execution
# ============================================================
# 1. Initial State Deviation & Covariances
# ----------------------------------------
# Deviation added to truth to create the initial estimate
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])

# Initial Covariance (P0)
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2

# Measurement Noise (Rk): [Range (km^2), Range-Rate (km/s)^2]
Rk = np.diag([1e-6, 1e-12])

# Process noise (SNC implementation)
sigma_a = 1e-8 # convert to km/s^2 for consistency with state units
Q_psd = sigma_a**2 * np.eye(3)


# 2. Setup Reference Trajectory (The "Truth" or Linearization Point)
# -----------------------------------------------------------------
# Initial Truth State (orbital elements -> inertial)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()

# Initial State Vector for Integrator: [Pos, Vel, STM]
X_0 = np.concatenate([r0 + x_0[:3], v0 + x_0[3:], Phi0])

# Load Measurements
obs = pd.read_csv(fr'data\measurements_2a_noisy.csv')
time_eval = obs['Time(s)'].values

# Filter arguments
coeffs = np.array([MU_EARTH, J2, 0])

# We pack all Process Noise settings here for compute_q_discrete
options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10,
    'bootstrap_steps': 500,  # Number of initial steps to run in LKF mode
    
    # Process Noise Settings
    'method': 'SNC',          # Use State Noise Compensation
    'frame_type': 'RIC',      # Frame to apply noise (ECI or RIC)
    'Q_cont': Q_psd,          # Continuous PSD matrix
    'threshold': 10.0,       # Max dt threshold
    'B': None                 # Not used for SNC
}

# Run the Filter
ekf_filter = EKF(n_states=6)

# Key Fix: Q is now inside 'options', removed from function arguments
results = ekf_filter.run(
    obs=obs,
    X_0=X_0,
    x_0=x_0,
    P0=P0,
    Rk=Rk,
    options=options
)


# Run post-processing
post_options = {
    'truth_traj_file': r'data\problem_2a_traj.csv',
    'save_to_timestamped_folder': True,
    'data_mask_idx': 500,
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

post_process(results, obs, post_options)