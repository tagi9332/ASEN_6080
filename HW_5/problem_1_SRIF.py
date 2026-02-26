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
from utils.plotting.post_process import post_process

# Import the new SRIF class (Update this path if you saved it elsewhere!)
from utils.filters.srif_class import SRIF

# ============================================================
# Main Execution
# ============================================================
# Initial State Deviation & Covariances
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
P0 = np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6])
# P0 = np.diag([1e3, 1e3, 1e3, 1, 1, 1])**2
Rk = np.diag([1e-6, 1e-12])

# Initial Truth State (Nominal Reference)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')

# The SRIF class handles the STM (Phi) internally now, so just pass the 6x1 state!
X_nom = np.concatenate([r0, v0])

# Load Measurements
obs = pd.read_csv(r'data\measurements_noisy.csv')
time_eval = obs['Time(s)'].values

# ODE arguments
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for dynamics

# Set Filter options
options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10
}

# Process noise
# Q = np.diag([1e-10, 1e-10, 1e-10, 1e-8, 1e-8, 1e-8])
Q = np.diag([0, 0, 0, 0, 0, 0])  # No process noise
uBar = np.zeros(3)               # Mean process noise vector (ignored since Q=0)

# Instantiate and run the Iterative SRIF
srif_filter = SRIF(n_states=6)

# Run the wrapper. You can tweak max_iter and tol here.
# iter_results = srif_filter.run_iterative(
#     obs=obs, 
#     X0=X_nom, 
#     x0=x_0, 
#     P0=P0, 
#     Q0=Q, 
#     uBar=uBar, 
#     R_meas=Rk, 
#     options=options,
#     max_iter=1,
#     tol=1e-3
# )

# Run the non-iterative version for a single pass
srif_results = srif_filter.run(
    obs=obs, 
    X_0=X_nom, 
    x_0=x_0,    
    P0=P0, 
    Q0=Q, 
    uBar=uBar, 
    R_meas=Rk, 
    options=options
)

# Run post-processing
post_options = {
    'truth_traj_file': r'data\HW1_truth.csv',
    'save_to_timestamped_folder': True,
    'data_mask_idx': 300,
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

# Pass the inner srif_out object so the plotting function sees the expected arrays
post_process(srif_results, obs, post_options)