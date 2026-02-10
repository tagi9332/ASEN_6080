import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from resources.constants import MU_EARTH, J2
from utils.filters.lkf_class_project_1 import LKF
from utils.filters.lkf_class_project_1_potter import LKF_potter
from utils.ground_station_utils.get_initial_station_eci import get_initial_station_eci
from project_1.run_iterative_LKF import run_iterative_LKF

# ============================================================
# Configuration & Setup (Common to both)
# ============================================================

# 1. Ground Stations
gs101_ecef = np.array([-5127510.0, -3794160.0,       0.0])
gs337_ecef = np.array([ 3860910.0,  3238490.0, 3898094.0])
gs394_ecef = np.array([  549505.0, -1380872.0, 6182197.0])

# 2. Initial Covariance
P0_diag = np.diag([
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6,   
    1e20,                           
    1e6, 1e6,                       
    1e-10, 1e-10, 1e-10,            
    1e6, 1e6, 1e6, 1e6, 1e6, 1e6    
])

# 3. Measurement Noise
Rk = np.diag([1e-4, 1e-6]) 
Q = np.zeros((18, 18))

# 4. Initial Truth State Guess
r0 = np.array([757700.0, 5222607.0, 4851500.0])
v0 = np.array([2213.21, 4678.34, -5371.30])

current_X0 = np.concatenate([
    r0, v0, [MU_EARTH, J2, 2.0],
    get_initial_station_eci(gs101_ecef),
    get_initial_station_eci(gs337_ecef),
    get_initial_station_eci(gs394_ecef)
])

# 5. Load Data
obs = pd.read_csv(fr'data\project_1_obs.csv')
coeffs = [MU_EARTH, J2, 0] 
options = {'coeffs': coeffs, 'abs_tol': 1e-12, 'rel_tol': 1e-12}

# ============================================================
# Execution
# ============================================================

print(f"{'='*60}")
print("Running STANDARD Linearized Kalman Filter...")
print(f"{'='*60}")
lkf_standard = LKF(n_states=18, station_map={101:0, 337:1, 394:2})
res_std, _ = run_iterative_LKF(lkf_standard, obs, current_X0, P0_diag, Rk, Q, options, num_iterations_max=1, tol=1e-3)

print(f"\n{'='*60}")
print("Running POTTER Square Root Filter...")
print(f"{'='*60}")
lkf_potter = LKF_potter(n_states=18, station_map={101:0, 337:1, 394:2})
res_pot, _ = run_iterative_LKF(lkf_potter, obs, current_X0, P0_diag, Rk, Q, options, num_iterations_max=1, tol=1e-3)

# ============================================================
# Analysis & Plotting
# ============================================================
output_dir = 'results/potter_comparison'
os.makedirs(output_dir, exist_ok=True)
print(f"\nGenerating Comparison Plots in {output_dir}...")

# Extract times (assuming identical time steps for both)
# Note: LKF usually outputs one more state than measurements (initial state), 
# so we check length against measurements
times = obs['Time(s)'].values
t_hours = (times - times[0]) / 3600.0

# Ensure we align arrays (truncate initial state if necessary to match time vector length)
# LKFResults usually stores [t0, t1, ... tn]. Measurements are [t1 ... tn].
# We will plot the whole history from t0.
t_hours_hist = (times - times[0]) / 3600.0

# --- Plot 1: Machine Precision Difference (Potter - Standard) ---
# We compute (State_Potter - State_Standard)
diff_state = res_pot.state_hist - res_std.state_hist
pos_diff_norm = np.linalg.norm(diff_state[:, 0:3], axis=1)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t_hours_hist, diff_state[:, 0], label='X', linewidth=1)
plt.plot(t_hours_hist, diff_state[:, 1], label='Y', linewidth=1)
plt.plot(t_hours_hist, diff_state[:, 2], label='Z', linewidth=1)
plt.title('Numerical Difference: Potter vs Standard LKF (Position)')
plt.ylabel('Difference (m)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(t_hours_hist, pos_diff_norm, 'k-', label='Total Pos Magnitude', linewidth=1)
plt.ylabel('Norm Difference (m)')
plt.xlabel('Time (hours)')
plt.yscale('log') # Log scale helps see the floating point noise floor
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'potter_vs_standard_numerical_diff.png'))
print(" - Saved Numerical Difference Plot")


# --- Plot 2: Condition Number Evolution ---
# Comparing condition number of P vs condition number of S
cond_P_std = []
cond_S_theoretical = [] 

# Access P_hist instead of cov_hist
for P in res_std.P_hist:
    # Compute condition number of the covariance matrix P
    # We use singular values for stability: max(S)/min(S)
    # Using np.linalg.cond(P) is also valid but svd is explicit
    ev = np.linalg.svd(P, compute_uv=False)
    
    # Avoid division by zero if P is singular (rare in LKF but good practice)
    if ev.min() > 1e-20:
        c_num = ev.max() / ev.min()
    else:
        c_num = 1e20 # Cap it

    cond_P_std.append(c_num)
    cond_S_theoretical.append(np.sqrt(c_num))

plt.figure(figsize=(10, 6))
plt.semilogy(t_hours_hist, cond_P_std, 'r-', linewidth=2, label=r'Standard LKF: $\kappa(P)$')
plt.semilogy(t_hours_hist, cond_S_theoretical, 'b--', linewidth=2, label=r'Potter Equivalent: $\kappa(S) = \sqrt{\kappa(P)}$')

plt.title('Covariance Condition Number Analysis')
plt.ylabel('Condition Number (Log Scale)')
plt.xlabel('Time (hours)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'condition_number_comparison.png'))
print(" - Saved Condition Number Plot")

# plt.show() # Uncomment if running locally
print("Done.")