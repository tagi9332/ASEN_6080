import os, sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.filters.lkf_class import LKF
from utils.plotting.post_process import post_process

# --- ADDED: Import Smoother ---
# Note: Adjust 'utils.filters.rts_smoother' to wherever you saved the smoother code
from utils.filters.rts_smoother import RTSSmoother
from utils.filters.run_smoother import run_smoother 

# ============================================================
# Main Execution
# ============================================================
# Initial State Deviation & Covariances
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2
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

# Process noise settings (SNC implementation)
sigma_a = 1e-8 # km/s^2
Q_psd = sigma_a**2 * np.eye(3)

# Set LKF options
options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10,
    
    # --- Process Noise Settings ---
    'method': 'SNC',          # Chosen method (SNC or DMC)
    'frame_type': 'RIC',      # Frame to apply noise in (RIC or ECI)
    'Q_cont': Q_psd,          # Continuous PSD matrix
    'threshold': 10.0,        # Max dt to prevent instability (example value)
    'B': None                 # Not needed for SNC
}

lkf_filter = LKF(n_states=6)

# Run LKF 
results = lkf_filter.run(obs, X_0, x_0, P0, Rk, options)

# ============================================================
# Run Smoother
# ============================================================
print("\n--- Preparing Truth Data for Smoother Evaluation ---")
truth_file = r'data\problem_2a_traj.csv'

# Load data without a header and use '\s+' to handle spaces/tabs as separators
truth_data = pd.read_csv(truth_file, header=None, sep=r'\s+')

# Manually assign the column names
truth_data.columns = ['Time', 'X', 'Y', 'Z', 'Xdot', 'Ydot', 'Zdot']

# Now we can interpolate easily
state_cols = ['X', 'Y', 'Z', 'Xdot', 'Ydot', 'Zdot']
truth_interp_func = interp1d(
    truth_data['Time'], 
    truth_data[state_cols], 
    axis=0, 
    kind='cubic', 
    fill_value='extrapolate'
)
X_truth_interp = truth_interp_func(time_eval)

# Trim the truth data to match the LKF output times (in case of extrapolation issues)
time_eval_trimmed = time_eval[1:]
X_truth_interp_trimmed = X_truth_interp[1:]

# Run the wrapper script (no plot argument anymore)
smooth_out_lkf_format, state_error_smooth, rms_comp, rms_full = run_smoother(
    results, 
    X_truth_interp_trimmed
)

# ============================================================
# Post-Processing
# ============================================================
post_options = {
    'truth_traj_file': truth_file,
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

# 1. Post-Process LKF
print("\n--- Plotting LKF Results ---")
post_process(results, obs, post_options)

# 2. Post-Process Smoother
print("\n--- Plotting Smoother Results ---")
smooth_post_options = post_options.copy()

# Turn off residual/NIS plots since the smoother doesn't compute new measurements
smooth_post_options['plot_postfit_residuals'] = False
smooth_post_options['plot_prefit_residuals'] = False
smooth_post_options['plot_residual_comparison'] = False
smooth_post_options['plot_nis_metric'] = False

post_process(smooth_out_lkf_format, obs, smooth_post_options)