import os, sys
import numpy as np
import pandas as pd

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.filters.lkf_class import LKF
from utils.plotting.post_process import post_process

# ============================================================
# Main Execution
# ============================================================

# State vector : [x,y,z,vx,vy,vz,           # 0:5 satellite state ECI
#                 mu,J2,C_D,                # 6:8 physical parameters
#                 gs101_x,gs101_y,gs101_z,  # 9:11 ground station 101 (Pacific Ocean) ECI
#                 gs337_x,gs337_y,gs337_z,  # 12:14 ground station 337 (Pirinclik) ECI
#                 gs394_x,gs394_y,gs394_z]  # 15:17 ground station 394 (Thule) ECI

# GS state positions in ECEF frame
gs101_ecef = np.array([-5127510.0, -3794160.0,       0.0])  # Pacific Ocean
gs337_ecef = np.array([ 3860910.0,  3238490.0, 3898094.0])  # Pirinclik
gs394_ecef = np.array([  549505.0, -1380872.0, 6182197.0])  # Thule

# Initial State Deviation & Covariances
x_0 = np.array([0, 0, 0, 0, 0, 0])
P0 = np.diag([
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6,   
    1e20,                           
    1e6, 1e6,                       
    1e-10, 1e-10, 1e-10,            
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6    
])


Rk = np.diag([1e-6, 1e-12])

# Initial Truth State (without deviation)
r0 = np.array([757700.0,5222607.0,4851500.0])
v0 = np.array([2213.21, 4678.34, -5371.30])
Phi0 = np.eye(6).flatten()

# Propagate the reference trajectory (truth plus deviation)
X_0 = np.concatenate([r0+x_0[:3], v0+x_0[3:], Phi0])


# Load Measurements
obs = pd.read_csv(fr'data\measurements_noisy.csv')
time_eval = obs['Time(s)'].values

# ODE arguments
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for dynamics

# Set LKF options
options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10
}

# Process noise
# Q = np.diag([1e-10, 1e-10, 1e-10, 1e-8, 1e-8, 1e-8])
Q = np.diag([0, 0, 0, 0, 0, 0])  # No process noise

lkf_filter = LKF(n_states=6)

results = lkf_filter.run(obs, X_0, x_0, P0, Rk, Q, options)

# Run post-processing
post_options = {
    'truth_traj_file': r'data\HW1_truth.csv',
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

post_process(results,obs,post_options)
