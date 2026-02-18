import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os, sys

# Adjust path to find local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from your project ---
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, R_EARTH
from utils.filters.ekf_class_dmc import EKF 
from utils.plotting.post_process import post_process

# ============================================================
# HELPER: J3 ACCELERATION MODEL
# ============================================================
def compute_J3_acceleration(r_vec):
    """
    Computes inertial acceleration due to J3 zonal harmonic.
    
    Args:
        r_vec (np.array): Position vector [x, y, z] (km)
        
    Returns:
        np.array: Acceleration vector [ax, ay, az] (km/s^2)
    """
    # Standard J3 Constant (unnormalized)
    J3 = -2.5327e-6 
    
    r_mag = np.linalg.norm(r_vec)
    x, y, z = r_vec
    
    # Pre-compute repeated terms
    Re_r = R_EARTH / r_mag
    z_r = z / r_mag
    
    # J3 Acceleration Vector Formula (Cartesian)
    # Derived from gradient of Potential U_3
    # a_vec = (mu * Re^3 * J3 / r^5) * [ coeff_r * r_vec + coeff_k * k_hat ]
    
    factor = (MU_EARTH * (R_EARTH**3) * J3) / (r_mag**5)
    
    coeff_r = (5/2) * (7 * (z_r**3) - 3 * z_r)
    coeff_k = (3/2) * (1 - 5 * (z_r**2))
    
    a_x = factor * (coeff_r * (x / r_mag))
    a_y = factor * (coeff_r * (y / r_mag))
    a_z = factor * (coeff_r * (z / r_mag) + coeff_k * (1.0/r_mag) * r_mag) # mult by r_mag to handle units, effectively just adding k component
    
    # Simplified:
    # The term is factor * [ coeff_r * (r_vec/r) + coeff_k * k_hat ]
    # Note: factor has /r^5. 
    
    # Re-calculation for clarity/safety:
    term_common = (MU_EARTH * R_EARTH**3 * J3) / (2 * r_mag**7)
    
    a_x = term_common * x * (15 * z_r**2 - 3) * (5 * z) # Wait, let's stick to the vector form:
    
    # Vector form: a = - mu/r^2 * (Re/r)^3 * J3 * [ ... ]
    # Using formula from Montenbruck & Gill (Satellite Orbits):
    const = MU_EARTH * (R_EARTH**3) * J3 / (2 * r_mag**7)
    
    tmp1 = 30 * z**2 - 6 * r_mag**2 # Term related to 5(z/r)^2
    tmp2 = 35 * z**3 - 15 * z * r_mag**2 # Term related to 7(z/r)^3
    
    # Note: Formulas vary by sign convention of J3. 
    # Usually J3 is negative (~ -2.5e-6).
    # Correct Cartesian expansion:
    ax = const * x * (15 * z**2 - 3 * r_mag**2)
    ay = const * y * (15 * z**2 - 3 * r_mag**2)
    az = const * (z * (15 * z**2 - 3 * r_mag**2) + (10 * z**3 - 6 * z * r_mag**2) * -1) # This is getting messy
    
    # Let's use the explicit textbook expansion:
    # ax = (mu/r^3) * (Re/r)^3 * J3 * (5/2 * (7(z/r)^3 - 3(z/r)) * x) ... No.
    
    # Clean Implementation:
    coef = (MU_EARTH / r_mag**2) * (R_EARTH / r_mag)**3 * J3
    
    # a_vec = coef * [ (5/2 * (7*u^3 - 3*u) * r_vec/r) + (3/2 * (1 - 5*u^2) * e_z) ]
    # where u = z/r
    
    u = z / r_mag
    
    term_r = (5.0/2.0) * (7*u**3 - 3*u)
    term_z = (3.0/2.0) * (1 - 5*u**2)
    
    accel = np.zeros(3)
    accel[0] = coef * (term_r * (x/r_mag))
    accel[1] = coef * (term_r * (y/r_mag))
    accel[2] = coef * (term_r * (z/r_mag) + term_z)
    
    return accel

# ============================================================
# CONFIGURATION
# ============================================================
output_dir = 'HW_3/plots_ekf_dmc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

meas_file = r'data/measurements_2a_noisy.csv'
truth_file = r'data/problem_2a_traj.csv'

# ============================================================
# 1. INITIALIZE STATE
# ============================================================
r0_true, v0_true = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
a0_true = np.zeros(3)

# Period Calc
r_mag = np.linalg.norm(r0_true)
v_mag = np.linalg.norm(v0_true)
energy = (v_mag**2 / 2) - (MU_EARTH / r_mag)
a_sma = -MU_EARTH / (2 * energy)
P_period = 2 * np.pi * np.sqrt(a_sma**3 / MU_EARTH)

# ============================================================
# 2. CONFIGURE DMC 
# ============================================================
tau = P_period / 3.0
beta = 1.0 / tau
B_mat = np.eye(3) * beta
coeffs = (MU_EARTH, J2, 0, B_mat) # Pass J3=0 to filter, it must estimate it

ekf_options = {
    'coeffs': coeffs,
    'abs_tol': 1e-10,
    'rel_tol': 1e-10,
    'dt_max': 60.0
}

# ============================================================
# 3. SETUP ESTIMATE & COVARIANCE
# ============================================================
x_0_dev = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3, 0, 0, 0])

r0_est = r0_true + x_0_dev[:3]
v0_est = v0_true + x_0_dev[3:6]
a0_est = a0_true + x_0_dev[6:9]
X_0_est = np.concatenate([r0_est, v0_est, a0_est])

P0 = np.diag([1, 1, 1, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9])
Rk = np.diag([1e-6, 1e-12])
x_hat_0 = np.zeros(9)

# ============================================================
# 4. PROCESS NOISE (OPTIMAL SIGMA)
# ============================================================
sigma = 1e-8 # Optimal tuning from sweep
q_driving_noise = 2 * (sigma**2) * beta
Q_PSD = q_driving_noise * np.eye(3)

print(f"Running EKF with Sigma: {sigma}")

# ============================================================
# 5. LOAD & RUN
# ============================================================
obs = pd.read_csv(meas_file)
obs.columns = obs.columns.str.strip()

ekf = EKF(n_states=9)
results = ekf.run(obs, X_0_est, x_hat_0, P0, Rk, Q_PSD, ekf_options)

# Run standard post-processing first
post_options = {
    'truth_traj_file': truth_file,
    'save_to_timestamped_folder': False, # Save directly to output_dir
    'plot_state_errors': True,
    'plot_residual_comparison': True,
    'plot_nis_metric': True
}
# Note: Modified post_process call to support output_dir override if your utils support it
# Otherwise it saves to a timestamp folder. 

# ============================================================
# 6. J3 PERTURBATION vs. ESTIMATED W ANALYSIS (SCATTER)
# ============================================================
print("\nGenerating 'w' Term Analysis Scatter Plots...")

# --- 1. Load Truth Trajectory ---
try:
    truth_df = pd.read_csv(truth_file, sep='\s+', header=None, 
                           names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
except:
    truth_df = pd.read_csv(truth_file, delim_whitespace=True, header=None, 
                           names=['Time(s)', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

truth_interp = interp1d(truth_df['Time(s)'], truth_df[['x', 'y', 'z']].values, axis=0, kind='cubic')

# --- 2. Extract Data ---
meas_times = obs['Time(s)'].values[1:] 
w_estimated = np.array(results.accel_hist) 
cov_hist = np.array(results.P_hist)

# --- 3. Compute TRUE J3 Acceleration ---
w_true_J3 = []
for t in meas_times:
    r_t = truth_interp(t)
    a_j3 = compute_J3_acceleration(r_t) 
    w_true_J3.append(a_j3)
w_true_J3 = np.array(w_true_J3)

# --- 4. Define Mask ---
mask_idx = 500 

t_plot = meas_times[mask_idx:]
w_est_plot = w_estimated[mask_idx:]
w_true_plot = w_true_J3[mask_idx:]
w_error = w_est_plot - w_true_plot
w_sigmas = np.sqrt(np.diagonal(cov_hist[:, 6:9, 6:9], axis1=1, axis2=2))[mask_idx:]

# --- PLOT 1: Estimated w vs True J3 (Scatter) ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
coords = ['x', 'y', 'z']

for i in range(3):
    # Plot True J3 Perturbation as black scatter
    axes[i].scatter(t_plot, w_true_plot[:, i] * 1e6, color='black', s=2, label='True $J_3$ Perturbation', alpha=0.6)
    
    # Plot Filter Estimated 'w' as red scatter
    axes[i].scatter(t_plot, w_est_plot[:, i] * 1e6, color='blue', s=2, label='Estimated w (DMC)', alpha=0.6)
    
    axes[i].set_ylabel(f'Accel {coords[i]} [mm/$s^2$]') 
    axes[i].grid(True, alpha=0.3)
    
    if i == 0:
        axes[i].legend(loc='upper right', markerscale=5)
        axes[i].set_title(f"DMC Estimated 'w' vs Actual $J_3$ Perturbation (Scatter)")

axes[2].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'DMC_w_vs_J3_scatter.png'))
plt.show()

# --- PLOT 2: 'w' Estimation Error (Scatter) ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

for i in range(3):
    # Error = Estimated w - True J3 as blue scatter
    axes[i].scatter(t_plot, w_error[:, i] * 1e6, color='blue', s=2, label='Error (w - J3)', alpha=0.6)
    
    # 3-Sigma Bounds (Kept as fill_between for visual clarity of the envelope)
    axes[i].fill_between(t_plot, 
                         1 * w_sigmas[:, i] * 1e6, 
                         -1 * w_sigmas[:, i] * 1e6, 
                         color='red', alpha=0.15, label='1$\sigma$ Bound')
    
    axes[i].set_ylabel(f'Error {coords[i]} [mm/$s^2$]')
    axes[i].grid(True, alpha=0.3)
    
    # Stats
    rms = np.sqrt(np.mean(w_error[:, i]**2))
    axes[i].text(0.02, 0.85, f"RMS: {rms*1e6:.4f} mm/s^2", transform=axes[i].transAxes, 
                 bbox=dict(facecolor='white', alpha=0.5))

axes[0].set_title(f"Error in Estimating $J_3$ Perturbation (w) (Scatter)")
axes[0].legend(loc='upper right', markerscale=5)
axes[2].set_xlabel('Time [s]')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'DMC_w_Error_scatter.png'))
plt.show()

print(f"Analysis complete. Scatter plots saved to {output_dir}")