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
    state_hist: Any
    innovations: Any
    postfit_residuals: Any
    nis_hist: Any

    def __post_init__(self):
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.state_hist = np.array(self.state_hist)
        self.innovations = np.array(self.innovations)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.nis_hist = np.array(self.nis_hist)

class IterativeBatch:
    def __init__(self, n_states=6):
        self.n = n_states

    def run(self, obs, X_0, x_0, P0, Rk, Q, coeffs, options):
        inv_Rk = np.linalg.inv(Rk)
        inv_P0 = np.linalg.inv(P0)

        # Print initialization message
        print("Initializing Iterative Batch Filter...")

        # Set filter arguments
        max_iterations = options.get('max_iterations', 20)
        tolerance = options.get('tolerance', 1e-6)
        stations_ll = options['stations_ll']
        
        # Set x0_bar to be the true initial state plus deviation
        X_0 = X_0[:self.n].copy()
        x0_star = X_0.copy()
        t_meas = obs['Time(s)'].values

        # Generate the Baseline (Initial Guess Trajectory)
        phi0_flat = np.eye(self.n).flatten()
        state_baseline = np.concatenate([X_0, phi0_flat])

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
        final_P0 = P0
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
                args=(coeffs,), 
                rtol=1e-10, atol=1e-10,
                method="DOP853"
            )

            Lambda = inv_P0.copy()
            N = inv_P0 @ (X_0 - x0_star)
            
            # Accumulate measurements
            for k in range(len(sol.t)):
                Phi_tk_t0 = sol.y[6:, k].reshape(self.n, self.n)
                x_ref = sol.y[0:6, k]
                
                row = obs.iloc[k]
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
        
        return self._generate_results(x0_star, sol, obs, final_P0, stations_ll, initial_guess_trajectory,Rk)

    def _generate_results(self, final_x0, sol, df_meas, P0_final, stations_ll, initial_guess_trajectory, Rk):
            """
            Final pass to populate results.
            Computes Pre-fits (Initial Guess) and Post-fits (Converged Solution).
            """
            _x = []
            _P = []
            _state = []
            _nis = []

            # Initialize lists for residuals
            prefit_res_list = []
            postfit_res_list = []

            # We assume sol.t matches t_meas exactly because t_eval was used
            for k in range(len(sol.t)):
                # 1. Get States
                x_ref_final = sol.y[0:6, k]         # Converged State
                x_ref_initial = initial_guess_trajectory[k] # Initial Guess State
                
                # 2. Ground Station State (Valid for both)
                row = df_meas.iloc[k]
                station_idx = int(row['Station_ID']) - 1
                Rs, Vs = get_gs_eci_state(
                    stations_ll[station_idx][0], 
                    stations_ll[station_idx][1], 
                    sol.t[k], 
                    init_theta=np.deg2rad(122)
                )
                gs_state = np.concatenate([Rs, Vs])
                y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])

                # 3. Compute Measurement Models
                # Model relative to Initial Guess (Pre-fit)
                y_model_initial = compute_rho_rhodot(x_ref_initial, gs_state)
                
                # Model relative to Converged Solution (Post-fit)
                y_model_final = compute_rho_rhodot(x_ref_final, gs_state)

                # 4. Calculate Residuals (Observation - Model)
                # Matches LKF/EKF "dy" logic
                prefit_res_list.append(y_obs - y_model_initial)   
                postfit_res_list.append(y_obs - y_model_final)

                # 5. Covariance & State Deviation
                Phi_tk_t0 = sol.y[6:, k].reshape(self.n, self.n)
                Pk = Phi_tk_t0 @ P0_final @ Phi_tk_t0.T

                # Compute NIS for measurement
                H_k = compute_H_matrix(x_ref_final[0:3], x_ref_final[3:6], Rs, Vs)
                S_k = H_k @ Pk @ H_k.T + Rk
                innovation = postfit_res_list[-1]
                nis = innovation.T @ np.linalg.solve(S_k, innovation)
                _nis.append(nis)
                
                _x.append(x_ref_final - x_ref_initial) # Deviation: Final - Initial
                _P.append(Pk)
                _state.append(x_ref_final)

            # Convert to numpy arrays
            prefit_res = np.array(prefit_res_list)
            postfit_res = np.array(postfit_res_list)
            nis_hist = np.array(_nis)

            return IterativeBatchResults(_x, _P, _state, prefit_res, postfit_res, nis_hist)