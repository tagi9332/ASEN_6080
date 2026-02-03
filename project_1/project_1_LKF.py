import os, sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp # Needed for back-propagation

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from resources.constants import MU_EARTH, J2, OMEGA_EARTH
from utils.filters.lkf_class_project_1 import LKF
from utils.plotting.post_process import post_process
from utils.zonal_harmonics.zonal_harmonics import stm_eom_mu_j2_drag

# ============================================================
# Helper Functions
# ============================================================
def get_initial_station_eci(station_ecef, t_offset=0):
    """
    Rotates ECEF coordinates to ECI based on the problem's theta formula.
    theta = omega_earth * t
    """
    theta = OMEGA_EARTH * t_offset
    c = np.cos(theta)
    s = np.sin(theta)
    
    R_ecef2eci = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    return R_ecef2eci @ station_ecef

# ============================================================
# Main Execution
# ============================================================

# GS state positions in ECEF frame
gs101_ecef = np.array([-5127510.0, -3794160.0,       0.0])  # Pacific Ocean
gs337_ecef = np.array([ 3860910.0,  3238490.0, 3898094.0])  # Pirinclik
gs394_ecef = np.array([  549505.0, -1380872.0, 6182197.0])  # Thule

# Initial State Deviation & Covariances
# Note: x_0 is the deviation. We reset this to 0 at the start of every iteration
# because we update the reference X_0 instead.
P0_diag = np.diag([
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6,   
    1e20,                           
    1e6, 1e6,                       
    1e-10, 1e-10, 1e-10,            
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6    
])

Rk = np.diag([1e-4, 1e-6])  # Measurement noise covariance [Range, Range-Rate]
Q = np.zeros((18, 18))       # No process noise

# Initial Truth State Guess
r0 = np.array([757700.0, 5222607.0, 4851500.0])
v0 = np.array([2213.21, 4678.34, -5371.30])

# Construct Initial Reference Trajectory Vector X_0
current_X0 = np.concatenate([
    r0,                      # [0:3] Position
    v0,                      # [3:6] Velocity
    [MU_EARTH, J2, 2.0],     # [6:9] Parameters (Mu, J2, Cd)
    get_initial_station_eci(gs101_ecef),    # [9:12] Station 1
    get_initial_station_eci(gs337_ecef),    # [12:15] Station 2
    get_initial_station_eci(gs394_ecef)     # [15:18] Station 3
])

# Load Measurements
obs = pd.read_csv(fr'data\project_1_obs.csv')
time_eval = obs['Time(s)'].values

# ODE arguments
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for dynamics

# Set LKF options
options = {
    'coeffs': coeffs,
    'abs_tol': 1e-12,
    'rel_tol': 1e-12
}

lkf_filter = LKF(n_states=18, station_map={101:0, 337:1, 394:2})

# ============================================================
# Iterative LKF Loop
# ============================================================
num_iterations_max = 20
tol = 1e-3
results = None

# Initialize storage
rms_hist = []
results = None
current_x0_dev = np.zeros(18)

print(f"{'='*60}")
print(f"Starting Iterative LKF ({num_iterations_max} iterations)")
print(f"{'='*60}")

for i in range(num_iterations_max):
    print(f"\n--- Iteration {i+1} / {num_iterations_max} ---")

    # Run LKF
    results = lkf_filter.run(obs, current_X0, current_x0_dev, P0_diag, Rk, Q, options)
    
    # Compute RMS of Post-Fit Residuals
    postfit_residuals = results.postfit_residuals
    rms_range = np.sqrt(np.mean(postfit_residuals[:,0]**2))
    rms_range_rate = np.sqrt(np.mean(postfit_residuals[:,1]**2))
  
    # Store RMS
    rms_hist.append((rms_range, rms_range_rate))

    # Print RMS
    print(f"   Post-Fit Residuals RMS: Range: {rms_range:.6f} m | Range-Rate: {rms_range_rate:.6f} m/s")

    # Check for convergence
    if rms_range < tol and rms_range_rate < tol:
        print("   Convergence criteria met. Stopping iterations.")
        break

    # Set up for next iteration
    x_est_final = results.dx_hist[-1]
    X_state_final = results.state_hist[-1]

    # Extract final STM
    phi_final = results.phi_hist[-1].reshape((18, 18))

    # Back-propagate to get initial deviation and state
    x0_init_est = np.linalg.inv(phi_final) @ x_est_final
    current_X0 = current_X0 + x0_init_est

    # Print
    print(f"   Updating Initial State Deviation for Next Iteration.")


# ============================================================
# Post Processing (On the final iteration results)
# ============================================================
print(f"\n{'='*60}")
print("Iterations Complete. Running Post-Processing...")
print(f"{'='*60}")

# 1. Recover the "Best Estimate" at t0
# current_X0 has been updated in the loop to become the converged state.
X0_best_estimate = current_X0 

# 2. Re-construct the Original Guess (A Priori)
# (Ideally, save this to a variable before the loop starts, e.g., X0_apriori)
X0_apriori = np.concatenate([
    r0, v0, [MU_EARTH, J2, 2.0],
    get_initial_station_eci(gs101_ecef),
    get_initial_station_eci(gs337_ecef),
    get_initial_station_eci(gs394_ecef)
])

# 3. Compute Total Deviation (The "Batch" Equivalent Result)
total_deviation_t0 = X0_best_estimate - X0_apriori

print("\nFinal Estimated State Deviation (at t0):")
state_labels = ['x (m)', 'y (m)', 'z (m)', 'vx (m/s)', 'vy (m/s)', 'vz (m/s)',
                'mu (m^3/s^2)', 'J2', 'Cd',
                'GS1_x (m)', 'GS1_y (m)', 'GS1_z (m)',
                'GS2_x (m)', 'GS2_y (m)', 'GS2_z (m)',
                'GS3_x (m)', 'GS3_y (m)', 'GS3_z (m)']

for label, value in zip(state_labels, total_deviation_t0):
    print(f"   {label}: {value:.6e}")


post_options = {
    'save_to_timestamped_folder': True,
    'data_mask_idx': 0,
    'results_units': 'm', # 'm or 'km'
    'plot_state_deviation': True,
    'plot_postfit_residuals': True,
    'plot_prefit_residuals': True,
    'plot_residual_comparison': True,
    'plot_covariance_trace': True,
    'plot_nis_metric': True
}

post_process(results, obs, post_options)