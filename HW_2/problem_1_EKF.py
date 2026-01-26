import os, sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.stats as stats

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6
from utils.filters.ekf_class import EKF
from utils.filters.lkf_class import LKF



# ============================================================
# Main Execution
# ============================================================
# Initial State Deviation & Covariances
x_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])
Rk = np.diag([1e-6, 1e-12])
Q = np.zeros((6, 6))

# Initial Reference State
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
state0 = np.concatenate([r0+x_0[:3], v0+x_0[3:], Phi0])

# Load Measurements
df_meas = pd.read_csv(r'HW_2\measurements_noisy.csv')
time_eval = df_meas['Time(s)'].values

# Set ODE arguments
coeffs = [MU_EARTH, J2, 0] # Ignoring J3 for dynamics

# Integrate Reference Trajectory for LKF bootstrap
print("Integrating 6x6 Reference Trajectory...")
sol = solve_ivp(
    zonal_sph_ode_6x6,
    (0, time_eval[-1]),
    state0,
    t_eval=time_eval,
    args=(coeffs,),
    rtol=1e-10, atol=1e-10)


EKF_filter = EKF(n_states=6)
print("Running EKF Filter...")
ekf_results = EKF_filter.run(
    sol_ref=sol, 
    meas_df=df_meas, 
    x_0=x_0, 
    P0=P0, 
    Rk=Rk, 
    Q=Q,
)

# Unpack results
dx_total_hist = ekf_results.dx_hist
P_hist = ekf_results.P_hist
innovations = ekf_results.innovations
postfit_residuals = ekf_results.postfit_residuals
S_hist = ekf_results.S_hist

# ============================================================
# Plotting (State Deviations)
# ============================================================
dx_plotting = dx_total_hist
times = sol.t
sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
state_labels = ['x (km)', 'y (km)', 'z (km)', 'vx (km/s)', 'vy (km/s)', 'vz (km/s)']

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

for i in range(6):
    axes[i].scatter(times, dx_plotting[:, i], c='b', label='Deviation', s=2)
    axes[i].plot(times, 3*sigmas[:, i], 'r--', alpha=0.7, label=fr'3$\sigma$')
    axes[i].plot(times, -3*sigmas[:, i], 'r--', alpha=0.7)
        
    axes[i].set_ylabel(state_labels[i])
    axes[i].set_title(f'State Deviation: {state_labels[i]}')
    axes[i].grid(True, linestyle=':', alpha=0.6)
    
    if i == 0:
        axes[i].legend(loc='upper right', fontsize='small')

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# ============================================================
# Plotting (Pre-fit vs Post-fit Residuals)
# ============================================================
innov_arr = np.array(innovations)
post_arr = np.array(postfit_residuals)
res_sigmas = np.sqrt(np.diag(Rk))

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

# Helper to plot residuals
def plot_residual_set(ax, data, sigma, title, ylabel, color):
    ax.scatter(times, data, s=2, c=color, alpha=0.5)
    ax.axhline(3*sigma, color='r', linestyle='--', alpha=0.8, label=r'3$\sigma$ Noise')
    ax.axhline(-3*sigma, color='r', linestyle='--')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

# Row 1: Range
plot_residual_set(axes[0,0], innov_arr[:,0], res_sigmas[0], 'Range Innovations (Pre-fit)', 'km', 'black')
plot_residual_set(axes[0,1], post_arr[:,0], res_sigmas[0], 'Range Residuals (Post-fit)', 'km', 'blue')

# Row 2: Range-Rate
plot_residual_set(axes[1,0], innov_arr[:,1], res_sigmas[1], 'Range-Rate Innovations (Pre-fit)', 'km/s', 'black')
plot_residual_set(axes[1,1], post_arr[:,1], res_sigmas[1], 'Range-Rate Residuals (Post-fit)', 'km/s', 'green')

axes[0,0].legend(loc='upper right')
plt.tight_layout()
plt.show()

# ============================================================
# Histogram (Cleaned Bell Curve)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

def plot_cleaned_hist(ax, data, label, color):
    mu, std = np.mean(data), np.std(data)
    
    # CUT TAILS: Only plot data within 4 sigma for visual clarity
    filtered_data = data[np.abs(data - mu) < 4 * std]
    
    ax.hist(filtered_data, bins=50, density=True, alpha=0.6, color=color, edgecolor='black')
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'k-', lw=2, label='Fitted Normal')
    ax.set_title(rf'{label} (Centered)')
    ax.set_xlabel('Residual Value')
    ax.legend()

# Using Post-fit residuals for the Gaussian check
plot_cleaned_hist(ax1, post_arr[100:, 0], 'Post-fit Range', 'skyblue')
plot_cleaned_hist(ax2, post_arr[100:, 1], 'Post-fit Range-Rate', 'salmon')

plt.tight_layout()
plt.show()

# ============================================================
# Requirement iii: RMS Metrics (Total vs. Settled)
# ============================================================
def get_rms(data): return np.sqrt(np.mean(data**2))

print(f"--- Filter Metrics (Post-fit) ---")
print(f"Total Range RMS:    {get_rms(post_arr[:,0]):.6e} km")
print(f"Settled Range RMS:  {get_rms(post_arr[100:,0]):.6e} km (Excluding 1st pass)")
print(f"Total R-Rate RMS:   {get_rms(post_arr[:,1]):.6e} km/s")
print(f"Settled R-Rate RMS: {get_rms(post_arr[100:,1]):.6e} km/s")