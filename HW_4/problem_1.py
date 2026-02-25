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
from utils.filters.rts_smoother import RTSSmoother  # <-- Updated Import

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
sigma_a = 1e-6 # km/s^2
Q_psd = sigma_a**2 * np.eye(3)
# Q_psd = np.zeros((3, 3)) # Set to zero for no process noise case

# Set LKF options
options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10,
    'potter_form': True,  # Use Potter formulation for stability
    
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

# Check for process noise in LKF run
has_q = np.any(Q_psd > 0)

# Instantiate and run the new RTSSmoother class
smoother = RTSSmoother(n_states=6, has_process_noise=has_q)
smooth_res = smoother.run(results)

# Calculate State Errors
state_error_smooth = smooth_res.state_smooth[:, 0:6] - X_truth_interp[:, 0:6]

# Calculate RMS (Root Mean Square) Error
rms_comp = np.sqrt(np.mean(state_error_smooth**2, axis=0))
rms_full = np.sqrt(np.mean(np.sum(state_error_smooth**2, axis=1)))

print("\nSmoother Performance:")
print(f"  Pos RMS (X, Y, Z): {rms_comp[0:3]} km")
print(f"  Vel RMS (Xdot, Ydot, Zdot): {rms_comp[3:6]} km/s")
print(f"  Full State 3D RMS: {rms_full:.6f}")

# ============================================================
# Overwrite LKF Results with Smoother Output
# ============================================================
print("\n--- Merging Smoothed States into Forward Filter Results ---")

# Overwrite the state and covariance histories with the smoothed versions.
# All other metrics (innovations, postfit_residuals, nis_hist) are left untouched!
results.dx_hist = smooth_res.dx_smooth
results.P_hist = smooth_res.P_smooth
results.state_hist = smooth_res.state_smooth
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

# 1. Post-Process Combined Results
print("\n--- Plotting Smoothed Trajectory & Forward Residuals ---")
post_process(results, obs, post_options)