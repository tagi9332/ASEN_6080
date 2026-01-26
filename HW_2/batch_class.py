import os, sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.stats as stats
from dataclasses import dataclass
from typing import Any

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Note: Ensure these paths and constants match your local environment
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot

@dataclass
class IterativeBatchResults:
    dx_hist: Any
    P_hist: Any
    corrected_state_hist: Any
    postfit_residuals: Any

    def __post_init__(self):
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.corrected_state_hist = np.array(self.corrected_state_hist)
        self.postfit_residuals = np.array(self.postfit_residuals)

class IterativeBatch:
    def __init__(self, n_states: int = 6):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, state0_initial, df_meas, P0_prior, Rk, stations_ll, max_iterations=10, tolerance=1e-10) -> IterativeBatchResults:
        """
        Performs Differential Correction (Iterative Least Squares) to estimate 
        the initial state at epoch (t0).
        """
        # Initialize the state at t0 to be corrected
        curr_x0 = state0_initial[:6].copy()
        inv_Rk = np.linalg.inv(Rk)
        inv_P0 = np.linalg.inv(P0_prior)
        
        # Initialize variables to store final results
        dx0 = np.zeros(self.n) 
        final_sol = None
        final_P0 = np.linalg.inv(P0_prior)
        
        for i in range(max_iterations):
            print(f"--- Iteration {i+1} ---")
            
            # Reset STM for the new reference trajectory
            phi0 = self.I.flatten()
            state_to_integrate = np.concatenate([curr_x0, phi0])
            
            # 1. Integrate the NEW reference trajectory starting from curr_x0
            sol = solve_ivp(
                zonal_sph_ode_6x6, (0, df_meas['Time(s)'].values[-1]), 
                state_to_integrate, t_eval=df_meas['Time(s)'].values, 
                args=([MU_EARTH, J2, 0],), rtol=1e-10, atol=1e-10
            )
            
            # 2. Accumulate Normal Equations
            info_matrix = inv_P0.copy()
            normal_vector = np.zeros(self.n) 
            
            for k in range(len(sol.t)):
                Phi_tk_t0 = sol.y[6:, k].reshape(self.n, self.n)
                
                row = df_meas.iloc[k]
                station_idx = int(row['Station_ID']) - 1
                Rs, Vs = get_gs_eci_state(
                    stations_ll[station_idx][0], 
                    stations_ll[station_idx][1], 
                    sol.t[k], init_theta=np.deg2rad(122)
                )
                
                x_ref = sol.y[0:6, k]
                y_ref = compute_rho_rhodot(x_ref, np.concatenate([Rs, Vs]))
                y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])
                
                # Map H to epoch: H_epoch = H_tk * Phi(tk, t0)
                H = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs) @ Phi_tk_t0
                residual = y_obs - y_ref
                
                info_matrix += H.T @ inv_Rk @ H
                normal_vector += H.T @ inv_Rk @ residual

            # 3. Solve for initial state correction at epoch (t0)
            dx0 = np.linalg.solve(info_matrix, normal_vector)
            
            # 4. Apply Correction
            curr_x0 += dx0
            
            correction_norm = np.linalg.norm(dx0)
            print(f"Correction Norm: {correction_norm:.6e}")
            
            final_sol = sol
            final_P0 = np.linalg.inv(info_matrix)
            
            # 5. Check Convergence
            if correction_norm < tolerance:
                print("Converged!")
                break
        else:
            print("Warning: Max iterations reached.")

        return self._compute_final_stats(final_sol, df_meas, final_P0, dx0, stations_ll)

    def _compute_final_stats(self, sol, df_meas, P0_final, dx0_final, stations_ll):
        """
        Maps the converged initial correction forward in time to generate history.
        """
        _dx_hist, _P_hist, _states, _post_fits = [], [], [], []

        for k in range(len(sol.t)):
            Phi_tk_t0 = sol.y[6:, k].reshape(self.n, self.n)
            x_ref = sol.y[0:6, k]
            
            # Propagate state deviation and covariance to current time tk
            dx_k = Phi_tk_t0 @ dx0_final 
            x_corrected_k = sol.y[0:6, k] + dx_k
            P_k = Phi_tk_t0 @ P0_final @ Phi_tk_t0.T
        
            
            # Compute Post-fit Residuals
            row = df_meas.iloc[k]
            station_idx = int(row['Station_ID']) - 1
            Rs, Vs = get_gs_eci_state(
                stations_ll[station_idx][0], 
                stations_ll[station_idx][1], 
                sol.t[k], init_theta=np.deg2rad(122)
            )
            
            # Compute H for the reference (if needed for stats) and z_hat for residuals
            y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])

            # Use compute_rho_rhodot to get the predicted measurement (y)
            # This returns exactly [range, range_rate] which is shape (2,)
            X_station = np.concatenate([Rs, Vs])
            z_hat_post = compute_rho_rhodot(x_corrected_k, X_station)

            # Now the subtraction works perfectly (2,) - (2,)
            postfit_res = y_obs - z_hat_post
            
            _dx_hist.append(dx_k)
            _P_hist.append(P_k)
            _states.append(x_ref + dx_k)
            _post_fits.append(postfit_res)

        return IterativeBatchResults(_dx_hist, _P_hist, _states, _post_fits)

# ============================================================
# Main Execution
# ============================================================
# Initial Reference State (Initial Guess)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
dx_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3])  # Initial guess error
state0_initial = np.concatenate([r0 + dx_0[0:3], v0 + dx_0[3:6], Phi0])

# Ground Stations (lat, lon) [rad]
stations_ll = np.deg2rad([
    [-35.398333, 148.981944], # Station 1 (Canberra, Australia)
    [ 40.427222, 355.749444], # Station 2 (Fort Davis, USA)
    [ 35.247164, 243.205000]  # Station 3 (Madrid, Spain)
])

# Load Measurements
df_meas = pd.read_csv(r'HW_2\measurements_noisy.csv')

# Initial Covariances & Weights
# P0: Confidence in your initial r0, v0 guess
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])
# Rk: Measurement noise floor (Range and Range-Rate)
Rk = np.diag([1e-6, 1e-12])

print("Starting Iterative Batch Filter (Differential Correction)...")

batch_filter = IterativeBatch(n_states=6)
batch_results = batch_filter.run(
    state0_initial=state0_initial, 
    df_meas=df_meas, 
    P0_prior=P0, 
    Rk=Rk, 
    stations_ll=stations_ll,
    max_iterations=10, 
    tolerance=1e-10
)

# Unpack results for plotting
dx_hist = batch_results.dx_hist
P_hist = batch_results.P_hist
x_corrected = batch_results.corrected_state_hist
postfit_residuals = batch_results.postfit_residuals

# ============================================================
# Plotting (State Deviations)
# ============================================================
times = df_meas['Time(s)'].values
sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
state_labels = ['x (m)', 'y (m)', 'z (m)', 'vx (m/s)', 'vy (m/s)', 'vz (m/s)']

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axes = axes.flatten()

for i in range(6):
    # Remove the [1:, i] and use [:, i] to match the full length of 'times'
    axes[i].scatter(np.array(times), np.array(dx_hist)[:, i] * 1000, c='b', label='Estimate', s=2)  # Convert km to m for position states
    axes[i].plot(np.array(times), 3*np.array(sigmas)[:, i], 'r--', alpha=0.7, label=fr'3$\sigma$')
    axes[i].plot(np.array(times), -3*np.array(sigmas)[:, i], 'r--', alpha=0.7)
    
    axes[i].set_ylabel(state_labels[i])
    axes[i].set_title(f'State Deviation: {state_labels[i]}')
    axes[i].grid(True, linestyle=':', alpha=0.6)
    if i == 0:
        axes[i].legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# ============================================================
# Residual Plotting & Gaussian Distribution Analysis
# ============================================================
postfit_residuals = np.array(postfit_residuals)*1e3
res_sigmas = np.sqrt(np.diag(Rk)) * 1000  # Convert to meters and m/s

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

# --- Range Residuals Time Series ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(np.array(times), np.array(postfit_residuals[:, 0], dtype=float), s=2, c='black', label='Post-fit Residual')
ax1.axhline(2*res_sigmas[0], color='r', linestyle='--', alpha=0.8, label=r'2$\sigma$ Noise Floor')
ax1.axhline(-2*res_sigmas[0], color='r', linestyle='--', alpha=0.8)
ax1.set_ylabel('Range Residual (m)')
ax1.set_title('Post-fit Measurement Residuals')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- Range Residual Histogram ---
ax2 = fig.add_subplot(gs[0, 1])
r_res = np.array(postfit_residuals[:, 0], dtype=float)
mu_r, std_r = np.mean(r_res), np.std(r_res)
count, bins, ignored = ax2.hist(r_res, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
x_range = np.linspace(mu_r - 4*std_r, mu_r + 4*std_r, 100)
ax2.plot(x_range, stats.norm.pdf(x_range, mu_r, std_r), 'r-', lw=2, label='Fitted Normal')
ax2.set_title(rf'Range PDF\n$\mu$={mu_r:.1e}, $\sigma$={std_r:.1e}')
ax2.legend(fontsize='small')

# --- Range-Rate Residual Time Series ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(np.array(times), np.array(postfit_residuals[:, 1], dtype=float), s=2, c='black')
ax3.axhline(2*res_sigmas[1], color='r', linestyle='--', alpha=0.8)
ax3.axhline(-2*res_sigmas[1], color='r', linestyle='--', alpha=0.8)
ax3.set_ylabel('Range-Rate Residual (m/s)')
ax3.set_xlabel('Time (s)')
ax3.grid(True, alpha=0.3)

# --- Range-Rate Residual Histogram ---
ax4 = fig.add_subplot(gs[1, 1])
rr_res = np.array(postfit_residuals[:, 1], dtype=float)
mu_rr, std_rr = np.mean(rr_res), np.std(rr_res)
count, bins, ignored = ax4.hist(rr_res, bins=40, density=True, alpha=0.6, color='salmon', edgecolor='black')
x_rr = np.linspace(mu_rr - 4*std_rr, mu_rr + 4*std_rr, 100)
ax4.plot(x_rr, stats.norm.pdf(x_rr, mu_rr, std_rr), 'b-', lw=2, label='Fitted Normal')
ax4.set_title(rf'Range-Rate PDF\n$\mu$={mu_rr:.1e}, $\sigma$={std_rr:.1e}')
ax4.set_xlabel('Residual')
ax4.legend(fontsize='small')

plt.tight_layout()
plt.show()

# --- Print Final Statistics ---
print(f"\n--- Final Batch Statistics ---")
print(f"RMS Range Residual:     {np.sqrt(np.mean(r_res**2)):.6e} m")
print(f"RMS Range-Rate Residual: {np.sqrt(np.mean(rr_res**2)):.6e} m/s")