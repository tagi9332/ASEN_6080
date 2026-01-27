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

def H_range_rangerate(R, V, Rs, Vs, eps=1e-12):
    """
    Measurement partials for simplified range and range-rate:
        rho  = ||R - Rs||
        rhod = (R - Rs)·(V - Vs) / rho

    Inputs:  R,V,Rs,Vs are length-3 arrays (or 3x1 vectors)
    Output:  H is 2x6 Jacobian wrt X = [R; V]
             rows: [rho, rhod], cols: [R (3), V (3)]
    """
    R  = np.asarray(R,  dtype=float).reshape(3,)
    V  = np.asarray(V,  dtype=float).reshape(3,)
    Rs = np.asarray(Rs, dtype=float).reshape(3,)
    Vs = np.asarray(Vs, dtype=float).reshape(3,)

    r = R - Rs
    v = V - Vs

    rho = np.linalg.norm(r)
    rho_hat = r / rho    
    rhod = np.dot(r, v) / rho

    # Partials
    drho_dR = rho_hat.reshape(1, 3)      # 1x3
    drho_dV = np.zeros((1, 3))           # 1x3

    # d(rhod)/dR
    drhod_dR = (v / rho - (np.dot(r, v) / rho**3) * r).reshape(1, 3)

    # d(rhod)/dV
    drhod_dV = rho_hat.reshape(1, 3)

    H = np.block([
        [drho_dR,  drho_dV],
        [drhod_dR, drhod_dV]])  # 2x6

    return H

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
    def __init__(self, n_states=6):
        self.n = n_states

    def run(self, state0_initial, df_meas, P0_prior, Rk, stations_ll, max_iterations=10, tolerance=1e-10):
        inv_Rk = np.linalg.inv(Rk)
        inv_P0 = np.linalg.inv(P0_prior)
        
        # Set x0_bar to be the true initial state plus deviation
        x0_bar = state0_initial[:self.n].copy()
        x0_star = x0_bar.copy()
        t_meas = df_meas['Time(s)'].values

        # 1. Generate the Baseline (Initial Guess Trajectory)
        # We append an identity STM because zonal_sph_ode_6x6 requires 42 elements to reshape
        phi0_flat = np.eye(self.n).flatten()
        state_baseline = np.concatenate([x0_bar, phi0_flat])

        print("Propagating initial guess baseline...")
        initial_sol = solve_ivp(
            zonal_sph_ode_6x6, 
            (0, t_meas[-1]), 
            state_baseline, 
            t_eval=t_meas, 
            args=([MU_EARTH, J2, 0],), 
            rtol=1e-10, atol=1e-10,
            method="DOP853"
        )
        # We only need the r and v components (first 6) for the baseline comparison
        initial_guess_trajectory = initial_sol.y[:6, :].T 
        
        # Track these for the final output
        final_P0 = P0_prior
        sol = None

        # 2. Iterative Batch Loop (Differential Correction)
        for i in range(max_iterations):
            print(f"--- Iteration {i+1} ---")
            phi0 = np.eye(self.n).flatten()
            state_to_integrate = np.concatenate([x0_star, phi0])
            
            # Integrate current nominal trajectory and STM
            sol = solve_ivp(
                zonal_sph_ode_6x6, 
                (0, t_meas[-1]), 
                state_to_integrate, 
                t_eval=t_meas, 
                args=([MU_EARTH,J2,0],), 
                rtol=1e-10, atol=1e-10,
                method="DOP853"
            )

            Lambda = inv_P0.copy()
            N = inv_P0 @ (x0_bar - x0_star)
            
            # Accumulate measurements
            for k in range(len(sol.t)):
                Phi_tk_t0 = sol.y[6:, k].reshape(self.n, self.n)
                x_ref = sol.y[0:6, k]
                
                row = df_meas.iloc[k]
                station_idx = int(row['Station_ID']) - 1
                Rs, Vs = get_gs_eci_state(
                    stations_ll[station_idx][0], 
                    stations_ll[station_idx][1], 
                    sol.t[k], 
                    init_theta=np.deg2rad(122)
                )
                
                y_ref = compute_rho_rhodot(x_ref, np.concatenate([Rs, Vs]))
                y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])
                y_i = y_obs - y_ref
                
                # Global Jacobian mapping: H_epoch = H_local * Phi(tk, t0)
                H_k = H_range_rangerate(x_ref[0:3], x_ref[3:6], Rs, Vs)
                H = H_k @ Phi_tk_t0
                
                Lambda += H.T @ inv_Rk @ H
                N += H.T @ inv_Rk @ y_i

                # Debug prints for each measurement

            # Solve for correction dx0
            dx0 = np.linalg.solve(Lambda, N)
            x0_star += dx0
            
            final_P0 = np.linalg.inv(Lambda)
            norm_dx0 = np.linalg.norm(dx0)
            print(f"Correction Norm: {norm_dx0:.6e}")
            
            if norm_dx0 < tolerance:
                print("Converged!")
                break
        
        return self._generate_results(x0_star, sol, df_meas, final_P0, stations_ll, initial_guess_trajectory)

    def _generate_results(self, final_x0, sol, df_meas, P0_final, stations_ll, initial_guess_trajectory):
        """
        Final pass to populate results for plotting and statistics.
        """
        dx_hist = []
        P_hist = []
        postfit_res = []
        corrected_states = []

        for k in range(len(sol.t)):
            Phi_tk_t0 = sol.y[6:, k].reshape(self.n, self.n)
            x_ref = sol.y[0:6, k]
            
            # Map covariance forward in time: P(t) = Phi(t,t0) * P0 * Phi(t,t0)^T
            Pk = Phi_tk_t0 @ P0_final @ Phi_tk_t0.T
            
            # State deviation: Final corrected path minus original uncorrected guess path
            dx_hist.append(x_ref - initial_guess_trajectory[k])
            
            # Post-fit measurement calculation
            row = df_meas.iloc[k]
            station_idx = int(row['Station_ID']) - 1
            Rs, Vs = get_gs_eci_state(
                stations_ll[station_idx][0], 
                stations_ll[station_idx][1], 
                sol.t[k], 
                init_theta=np.deg2rad(122)
            )
            
            y_ref = compute_rho_rhodot(x_ref, np.concatenate([Rs, Vs]))
            y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])
            
            P_hist.append(Pk)
            postfit_res.append(y_obs - y_ref)
            corrected_states.append(x_ref)

        return IterativeBatchResults(dx_hist, P_hist, corrected_states, postfit_res)

# ============================================================
# Main Execution
# ============================================================
# Initial Reference State (Initial Guess)
r0, v0 = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
Phi0 = np.eye(6).flatten()
dx_0 = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)
state0_initial = np.concatenate([r0 + dx_0[0:3], v0 + dx_0[3:6], Phi0])




# Ground Stations (lat, lon) [rad]
stations_ll = np.deg2rad([
    [-35.398333, 148.981944], # Station 1 (Canberra, Australia)
    [ 40.427222, 355.749444], # Station 2 (Fort Davis, USA)
    [ 35.247164, 243.205000]  # Station 3 (Madrid, Spain)
])

# Load Measurements
df_meas = pd.read_csv(r'HW_2\measurements_noisy_3_compact.csv')

# Initial Covariances & Weights
# P0: Confidence in your initial r0, v0 guess
P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3])**2
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
    max_iterations=5, 
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