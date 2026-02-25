import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Note: Ensure these paths and constants match your local environment
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from utils.filters.batch_class import IterativeBatch
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.plotting.post_process import post_process


# ============================================================
# Main Execution
# ============================================================
# Initial Reference State (Initial Guess)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
state0_initial = np.concatenate([r0 + x_0[0:3], v0 + x_0[3:6], Phi0])


# Ground Stations (lat, lon) [rad]
stations_ll = np.deg2rad([
    [-35.398333, 148.981944], # Station 1 (Canberra, Australia)
    [ 40.427222, 355.749444], # Station 2 (Fort Davis, USA)
    [ 35.247164, 243.205000]  # Station 3 (Madrid, Spain)
])

# Load Measurements
df_meas = pd.read_csv(fr'data\measurements_2a_noisy.csv')

# Initial Covariances & Weights
# P0: Confidence in your initial r0, v0 guess
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2
# P0 = np.diag([1e3, 1e3, 1e3, 1, 1, 1])**2
# Rk: Measurement noise floor (Range and Range-Rate)
Rk = np.diag([1e-6, 1e-12])

# Filter options
coeffs = np.array([MU_EARTH, J2, 0])  # Ignore J3 for this problem
options = {
    'max_iterations': 10,
    'tolerance': 1e-10,
    'stations_ll': stations_ll
}

print("Starting Iterative Batch Filter (Differential Correction)...")

batch_filter = IterativeBatch(n_states=6)
results = batch_filter.run(
    obs= df_meas,
    X_0=state0_initial,
    x_0=x_0,
    P0=P0,
    Rk=Rk,
    Q=None,
    coeffs=coeffs,
    options=options
)



# Run post-processing
post_options = {
    'truth_traj_file': r'data\problem_2a_traj.csv',
    'save_to_timestamped_folder': True,
    'data_mask_idx': 300,
    'plot_state_errors': True,
    'plot_state_deviation': True,
    'plot_postfit_residuals': True,
    'plot_prefit_residuals': True,
    'plot_residual_comparison': True,
    'plot_covariance_trace': True,
    'plot_filter_consistency': True,
    'plot_nis_metric': True
}

post_process(results,df_meas,post_options)
