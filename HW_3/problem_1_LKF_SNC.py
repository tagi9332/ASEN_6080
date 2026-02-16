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
from utils.filters.lkf_class import LKF
from utils.plotting.post_process import post_process

# ============================================================
# Main Execution
# ============================================================
# Initial State Deviation & Covariances
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
P0 = np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6])
# P0 = np.diag([1e3, 1e3, 1e3, 1, 1, 1])**2
Rk = np.diag([1e-6, 1e-12])

# Initial Truth State (without deviation)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()

# Propagate the reference trajectory (truth plus deviation)
X_0 = np.concatenate([r0+x_0[:3], v0+x_0[3:], Phi0])

# Load Measurements
obs = pd.read_csv(fr'data\measurements_2a_noisy.csv')
time_eval = obs['Time(s)'].values

# ODE arguments
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for dynamics

# Set LKF options
options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10,
    'SNC_frame': 'RIC' # ECI or RIC frame for SNC implementation

}

# Process noise (SNC implementation)
sigma_a = 1e-8 # m/s^2
Q = sigma_a**2 * np.eye(3)

lkf_filter = LKF(n_states=6)

results = lkf_filter.run(obs, X_0, x_0, P0, Rk, Q, options)

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
