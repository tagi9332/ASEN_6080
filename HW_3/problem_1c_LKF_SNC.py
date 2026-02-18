import os, sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================
# Configuration & Imports
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2
from utils.filters.lkf_class import LKF

meas_file = r'data/measurements_2a_noisy.csv'
truth_file = r'data/problem_2a_traj.csv'

# Constants
sigma_acc = 1e-8 # km/s^2
Q_PSD = (sigma_acc**2) * np.eye(3)

# Initial State
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2
Rk = np.diag([1e-6, 1e-12])
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
X_0_ref = np.concatenate([r0 + x_0[:3], v0 + x_0[3:], Phi0])

# ============================================================
# 1. ROBUST DATA LOADING
# ============================================================
# Load Measurements
obs = pd.read_csv(meas_file)
obs.columns = obs.columns.str.strip()

# Load Truth (Restored whitespace delimiter settings)
truth_df = pd.read_csv(truth_file, 
                       delim_whitespace=True, 
                       header=None, 
                       names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

# Force columns to numeric to prevent string comparison errors
cols = ['x', 'y', 'z', 'vx', 'vy', 'vz']
for c in cols:
    truth_df[c] = pd.to_numeric(truth_df[c], errors='coerce')

# Create Interpolator
truth_interp = interp1d(truth_df['Time(s)'], 
                        truth_df[cols].values, 
                        axis=0, kind='cubic', fill_value="extrapolate")

# ============================================================
# 2. RUN CASES
# ============================================================
def run_case(frame_type):
    print(f"Running LKF with Process Noise in {frame_type} frame...")
    
    coeffs = [MU_EARTH, J2, 0]
    options = {
        'coeffs': coeffs,
        'abs_tol': 1e-10,
        'rel_tol': 1e-10,
        'method': 'SNC',
        'frame_type': frame_type,
        'Q_cont': Q_PSD,
        'threshold': 10.0,
        'B': None
    }
    
    lkf = LKF(n_states=6)
    results = lkf.run(obs, X_0_ref, x_0, P0, Rk, options)
    
    # Validation
    t_eval = obs['Time(s)'].values
    truth_states = truth_interp(t_eval)
    
    # Ensure shapes match
    n = min(len(results.state_hist), len(truth_states))
    est_states = np.array(results.state_hist)[:n]
    truth_states = truth_states[:n]
    
    state_errors = est_states - truth_states
    
    # Compute Float Metrics (Explicit Cast)
    pos_rms = float(np.sqrt(np.mean(np.linalg.norm(state_errors[:, :3], axis=1)**2)))
    vel_rms = float(np.sqrt(np.mean(np.linalg.norm(state_errors[:, 3:], axis=1)**2)))
    
    post_res = np.array(results.postfit_residuals)
    res_rng_rms = float(np.sqrt(np.mean(post_res[:, 0]**2)))
    res_rr_rms = float(np.sqrt(np.mean(post_res[:, 1]**2)))
    
    return {
        'Pos RMS (km)': pos_rms,
        'Vel RMS (km/s)': vel_rms,
        'Res Range (km)': res_rng_rms,
        'Res Rate (km/s)': res_rr_rms
    }

metrics_ric = run_case('RIC')
metrics_eci = run_case('ECI')

# ============================================================
# 3. COMPARISON & REPORTING
# ============================================================
# Create DataFrame fresh to avoid index duplicates from previous runs
df = pd.DataFrame([metrics_ric, metrics_eci], index=['RIC Frame', 'ECI Frame'])

# Calculate Delta
diff = df.loc['RIC Frame'] - df.loc['ECI Frame']
diff.name = 'Delta (RIC - ECI)'

# Append Delta row
df = pd.concat([df, diff.to_frame().T])

print("\n" + "="*60)
print(f"LKF SNC Comparison: Sigma = {sigma_acc} km/s^2")
print("="*60)
print(df.to_markdown(floatfmt=".6e"))
print("="*60)

print("\nInterpretation:")
# Explicit float casting for the comparison check
ric_val = float(df.loc['RIC Frame', 'Pos RMS (km)'])
eci_val = float(df.loc['ECI Frame', 'Pos RMS (km)'])

if ric_val < eci_val:
    print(f"--> RIC frame noise yielded lower Position Error.")
else:
    print(f"--> ECI frame noise yielded lower Position Error.")