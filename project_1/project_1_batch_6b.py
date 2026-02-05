import os, sys
import numpy as np
import pandas as pd

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from resources.constants import MU_EARTH, J2, OMEGA_EARTH
# 
# CHANGE 1: Import the BatchLS class instead of LKF
from utils.filters.batch_class_project_1 import BatchLS 
from utils.plotting.post_process import post_process
from utils.ground_station_utils.get_initial_station_eci import get_initial_station_eci

# ============================================================
# Main Execution
# ============================================================

# 1. Setup Ground Stations (ECEF)
gs101_ecef = np.array([-5127510.0, -3794160.0,       0.0])
gs337_ecef = np.array([ 3860910.0,  3238490.0, 3898094.0])
gs394_ecef = np.array([  549505.0, -1380872.0, 6182197.0])

# Initial State Deviation & Covariances (No stations fixed)
P0 = np.diag([
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6,   
    1e20,                           
    1e6, 1e6,                       
    1e6, 1e6, 1e6,            
    1e-10, 1e-10, 1e-10, 1e6, 1e6, 1e6    
])

# Measurement Noise
Rk = np.diag([1e-4, 1e-6]) 

# Note: Q (Process Noise) is removed. Batch LS does not use it.

# 3. Initial State Guess (A Priori State X_bar)
r0 = np.array([757700.0, 5222607.0, 4851500.0])
v0 = np.array([2213.21, 4678.34, -5371.30])

X_0_apriori = np.concatenate([
    r0,                              # [0:3] Position
    v0,                              # [3:6] Velocity
    [MU_EARTH, J2, 2.0],             # [6:9] Parameters
    get_initial_station_eci(gs101_ecef),    # [9:12] Station 1
    get_initial_station_eci(gs337_ecef),    # [12:15] Station 2
    get_initial_station_eci(gs394_ecef)     # [15:18] Station 3
])

# 4. Load Data
obs = pd.read_csv(fr'data\project_1_obs.csv')

# 5. Configure Filter Options
# CHANGE 2: Batch-specific options
options = {
    'max_iterations': 3,     # The loop is now internal to the class
    'convergence_tol': 1e-3,  # Stop if dx_0 updates are smaller than this
    'abs_tol': 1e-12,         # Integrator tolerance
    'rel_tol': 1e-12
}

# 6. Instantiate and Run
# 
print(f"{'='*60}")
print(f"Starting Batch Least Squares Processing")
print(f"{'='*60}")

# Instantiate
batch_filter = BatchLS(n_states=18, station_map={101:0, 337:1, 394:2})

# CHANGE 3: Run once (The class handles the iteration loop)
results = batch_filter.run(obs, X_0_apriori, P0, Rk, options)

# ============================================================
# Post Processing
# ============================================================
print(f"\n{'='*60}")
print("Batch Complete. Running Post-Processing...")
print(f"{'='*60}")

# Print final state deviation estimates (Converged X0 - Original X0)
# The BatchLS class stores this in the first element of dx_hist if you want t0, 
# or we can just look at the last state deviation.
final_state_vector = results.state_hist[0] # State at t=0 after convergence
final_deviation = final_state_vector - X_0_apriori

state_labels = ['x (m)', 'y (m)', 'z (m)', 'vx (m/s)', 'vy (m/s)', 'vz (m/s)',
                'mu (m^3/s^2)', 'J2', 'Cd',
                'GS1_x (m)', 'GS1_y (m)', 'GS1_z (m)',
                'GS2_x (m)', 'GS2_y (m)', 'GS2_z (m)',
                'GS3_x (m)', 'GS3_y (m)', 'GS3_z (m)']

print("\nFinal Estimated State Deviation (at t0):")
for label, value in zip(state_labels, final_deviation):
    print(f"   {label}: {value:4.5f}")


# Compute r and v norm deviations
r_dev = final_deviation[0:3]
v_dev = final_deviation[3:6]
total_deviation_t0 = np.concatenate([r_dev, v_dev])
print(f"\nTotal Position Deviation Magnitude at t0: {np.linalg.norm(r_dev):.6f} m")
print(f"Total Velocity Deviation Magnitude at t0: {np.linalg.norm(v_dev):.6f} m/s")

post_options = {
    'save_to_timestamped_folder': True,
    'data_mask_idx': 0,
    'results_units': 'm', 
    'plot_state_deviation': True,   # Will plot (Converged_Trajectory - Initial_Guess)
    'plot_postfit_residuals': True,
    'plot_prefit_residuals': True,  # Will plot (Initial_Guess_Residuals)
    'plot_residual_comparison': True,
    'plot_covariance_trace': True,
    'plot_nis_metric': True,
    'plot_covariance_ellipsoid': True
}

post_process(results, obs, post_options)