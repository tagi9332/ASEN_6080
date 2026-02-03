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
    'abs_tol': 1e-10,
    'rel_tol': 1e-10
}

lkf_filter = LKF(n_states=18, station_map={101:0, 337:1, 394:2})

# ============================================================
# Iterative LKF Loop
# ============================================================
num_iterations = 15
results = None

print(f"{'='*60}")
print(f"Starting Iterative LKF ({num_iterations} iterations)")
print(f"{'='*60}")

for i in range(num_iterations):
    print(f"\n--- Iteration {i+1} / {num_iterations} ---")
    
    # 1. Reset Deviation Estimate to Zero
    # (Since we absorbed the previous deviation into current_X0)
    current_x0_dev = np.zeros(18)
    
    # 2. Reset Covariance (Optional: Keep it if you want to tighten, 
    # but strictly for relinearization we often reset P0 to the a priori 
    # covariance around the new linearization point)
    current_P0 = P0_diag.copy()

    # 3. Run LKF
    results = lkf_filter.run(obs, current_X0, current_x0_dev, current_P0, Rk, Q, options)
    
    # 4. Extract Final Estimated State (Reference + Deviation) at t_end
    # results.state_hist is shape (N, 18)
    X_end_est = results.state_hist[-1]
    t_end = time_eval[-1]
    
    # 5. Back-Propagate to t=0 to update Initial Conditions
    # We need to construct the augmented state for the integrator (18 state + 18x18 STM)
    # The STM part doesn't matter for the trajectory back-prop, so we just use identity.
    phi_dummy = np.eye(18).flatten()
    X_end_augmented = np.concatenate([X_end_est, phi_dummy])
    
    print("   Back-propagating final estimate to update X0...")
    
    sol_back = solve_ivp(
        stm_eom_mu_j2_drag,
        (t_end, 0),  # Integrate backwards from End to Start
        X_end_augmented,
        rtol=options['abs_tol'],
        atol=options['rel_tol']
    )
    
    # The new best estimate for X0 is the end of the backward integration
    new_X0 = sol_back.y[0:18, -1]
    
    # 6. Calculate and Print Convergence Delta (Position update magnitude)
    delta_pos = np.linalg.norm(new_X0[0:3] - current_X0[0:3])
    delta_vel = np.linalg.norm(new_X0[3:6] - current_X0[3:6])
    
    print(f"   Update delta: Pos: {delta_pos:.6f} m | Vel: {delta_vel:.6f} m/s")
    
    # Update for next loop
    current_X0 = new_X0

# ============================================================
# Post Processing (On the final iteration results)
# ============================================================
print(f"\n{'='*60}")
print("Iterations Complete. Running Post-Processing...")
print(f"{'='*60}")

post_options = {
    'save_to_timestamped_folder': True,
    'data_mask_idx': 50,
    'plot_state_deviation': True,
    'plot_postfit_residuals': True,
    'plot_prefit_residuals': True,
    'plot_residual_comparison': True,
    'plot_covariance_trace': True,
    'plot_nis_metric': True
}

post_process(results, obs, post_options)