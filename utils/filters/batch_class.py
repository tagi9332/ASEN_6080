import os, sys
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Any

# ============================================================
# Imports & Constants
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Note: Ensure these paths and constants match your local environment
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_rho_rhodot, compute_H_matrix

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

    def run(self, meas_df, state0_initial, P0_prior, Rk, coeffs, stations_ll, max_iterations=10, tolerance=1e-10):
        inv_Rk = np.linalg.inv(Rk)
        inv_P0 = np.linalg.inv(P0_prior)
        
        # Set x0_bar to be the true initial state plus deviation
        x0_bar = state0_initial[:self.n].copy()
        x0_star = x0_bar.copy()
        t_meas = meas_df['Time(s)'].values

        # Generate the Baseline (Initial Guess Trajectory)
        phi0_flat = np.eye(self.n).flatten()
        state_baseline = np.concatenate([x0_bar, phi0_flat])

        print("Propagating initial guess baseline...")
        initial_sol = solve_ivp(
            zonal_sph_ode_6x6, 
            (0, t_meas[-1]), 
            state_baseline, 
            t_eval=t_meas, 
            args=(coeffs,), 
            rtol=1e-10, atol=1e-10,
            method="DOP853"
        )
        initial_guess_trajectory = initial_sol.y[:6, :].T 
        
        # Track these for the final output
        final_P0 = P0_prior
        sol = None

        # Iterative Batch Loop (Differential Correction)
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
                
                row = meas_df.iloc[k]
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
                H_k = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs)
                H = H_k @ Phi_tk_t0
                
                Lambda += H.T @ inv_Rk @ H
                N += H.T @ inv_Rk @ y_i


            # Solve for correction dx0
            dx0 = np.linalg.solve(Lambda, N)
            x0_star += dx0
            
            final_P0 = np.linalg.inv(Lambda)
            norm_dx0 = np.linalg.norm(dx0)
            print(f"Correction Norm: {norm_dx0:.6e}")
            
            if norm_dx0 < tolerance:
                print("Converged!")
                break
        
        return self._generate_results(x0_star, sol, meas_df, final_P0, stations_ll, initial_guess_trajectory)

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
