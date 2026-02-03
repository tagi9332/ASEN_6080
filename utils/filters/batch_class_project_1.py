import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Any

# Local Imports (Assumed same structure as your LKF)
from utils.zonal_harmonics.zonal_harmonics import stm_eom_mu_j2_drag
from utils.ground_station_utils.gs_meas_model_H import compute_H_tilde_18_state, compute_rho_rhodot
from utils.misc.print_progress import print_progress
from resources.constants import OMEGA_EARTH

@dataclass
class BatchResults:
    dx_hist: Any
    P_hist: Any
    state_hist: Any
    phi_hist: Any
    prefit_residuals: Any
    postfit_residuals: Any
    nis_hist: Any

    def __post_init__(self):
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.state_hist = np.array(self.state_hist)
        self.phi_hist = np.array(self.phi_hist)
        self.prefit_residuals = np.array(self.prefit_residuals)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.nis_hist = np.array(self.nis_hist)

class BatchLS:
    def __init__(self, n_states: int = 18, station_map: dict = None):
        """
        Initializes the Batch Least Squares Filter for the 18-element State.
        State: [r_vec(3), v_vec(3), mu, J2, Cd, GS1(3), GS2(3), GS3(3)]
        """
        self.n = n_states
        self.I = np.eye(n_states)
        
        if station_map is None:
            self.station_map = {101: 0, 337: 1, 394: 2}
        else:
            self.station_map = station_map

    def run(self, obs, X_0_guess, P0_apriori, Rk, options) -> BatchResults:
        """
        Runs the Iterative Batch Least Squares (Differential Correction).
        
        Args:
            obs (DataFrame): Measurements
            X_0_guess (np.array): Initial Guess for Reference State at t0 (18x1)
            P0_apriori (np.array): A priori covariance matrix (18x18)
            Rk (np.array): Measurement noise covariance (2x2)
            options (dict): 'max_iterations', 'abs_tol', 'rel_tol', 'convergence_tol'
        """
        
        print(f"{'='*60}")
        print("Initializing Batch Least Squares Filter...")
        print(f"{'='*60}")

        # Extract Options
        max_iters = options.get('max_iterations', 10)
        convergence_tol = options.get('convergence_tol', 1e-3)
        abs_tol = options.get('abs_tol', 1e-12)
        rel_tol = options.get('rel_tol', 1e-12)

        # Setup
        time_eval = obs['Time(s)'].values
        t_span = (0, time_eval[-1])
        
        # In Batch, we update the Reference State directly.
        # X_ref_0 is our current best estimate of the epoch state.
        X_ref_0 = X_0_guess.copy()
        
        # Inversion of Weight Matrices
        W = np.linalg.inv(Rk)
        inv_P_bar = np.linalg.inv(P0_apriori)

        # Storage for the "Previous Best" to calculate convergence
        X_ref_prev = X_ref_0.copy()
        
        # Initial trajectory (for pre-fit residuals calculation only)
        # We integrate the initial guess once to store it as the "Baseline"
        print("Computing Initial Baseline Trajectory...")
        phi_0_flat = np.eye(self.n).flatten()
        aug_X0_initial = np.concatenate([X_0_guess, phi_0_flat])
        sol_initial = solve_ivp(
            stm_eom_mu_j2_drag, t_span, aug_X0_initial, 
            t_eval=time_eval, rtol=abs_tol, atol=rel_tol
        )
        initial_trajectory = sol_initial.y[:self.n, :].T

        # --- ITERATION LOOP ---
        final_P0 = None
        final_sol = None
        
        for i in range(max_iters):
            print(f"\n--- Batch Iteration {i+1} / {max_iters} ---")
            
            # 1. Integrate Reference Trajectory & STM
            #    We integrate the CURRENT best estimate X_ref_0
            aug_X_ref = np.concatenate([X_ref_0, phi_0_flat])
            
            sol = solve_ivp(
                stm_eom_mu_j2_drag, t_span, aug_X_ref, 
                t_eval=time_eval, rtol=abs_tol, atol=rel_tol
            )
            
            # 2. Accumulate Normal Equations
            #    Lambda = inv(P_bar) + sum( H^T * W * H )
            #    N      = inv(P_bar)*dx_bar + sum( H^T * W * y )
            #    Note: Since we updated X_ref_0, the deviation dx_bar = (X_apriori - X_ref_curr).
            
            Lambda = inv_P_bar.copy()
            
            # Deviation from A Priori (x_bar = X_apriori - X_curr)
            # This "pulls" the solution back towards the a priori guess based on P0 confidence
            dx_bar = X_0_guess - X_ref_0 
            N = inv_P_bar @ dx_bar
            
            rms_acum_range = 0.0
            rms_acum_rr = 0.0
            
            for k in range(len(sol.t)):
                # Retrieve Reference State at tk
                X_k = sol.y[:self.n, k]
                Phi_tk_t0 = sol.y[self.n:, k].reshape((self.n, self.n))
                
                # Retrieve Measurement
                meas_row = obs.iloc[k]
                station_id = int(meas_row['Station_ID'])
                st_idx = self.station_map[station_id]
                y_obs = np.array([meas_row['Range(m)'], meas_row['Range_Rate(m/s)']])
                
                # Computed Measurement (y = h(X_k))
                # Note: X_k includes the station position [9:18] integrated dynamically
                # We need to extract SC and GS states for the helper function
                r_sc = X_k[0:3]
                v_sc = X_k[3:6]
                
                # Extract specific station state from the 18-element vector
                st_start = 9 + (3 * st_idx)
                r_gs = X_k[st_start : st_start+3]
                
                # Station velocity must be computed as w x r (Earth Fixed assumption in Inertial Frame)
                # (Or if your state vector already tracks GS velocity, use that. 
                #  But based on your previous code, state only tracks Position, velocity is derived).
                #  Looking at stm_eom_mu_j2_drag logic: d_state[9:18] = cross(w, r).
                #  So velocity IS implicit, but we need to compute it for the RangeRate calc.
                w_earth = np.array([0, 0, OMEGA_EARTH]) # Ensure this matches your constants file
                v_gs = np.cross(w_earth, r_gs)
                
                # Assemble 6-element vectors for helper
                sc_state_6 = np.concatenate([r_sc, v_sc])
                gs_state_6 = np.concatenate([r_gs, v_gs])
                
                y_comp = compute_rho_rhodot(sc_state_6, gs_state_6)
                
                # Residuals (y - h(x))
                y_res = y_obs - y_comp
                
                # RMS accumulation
                rms_acum_range += y_res[0]**2
                rms_acum_rr += y_res[1]**2
                
                # H-Matrix (Local) 
                # We pass the full X_k because compute_H_tilde handles the 18-element logic
                H_local = compute_H_tilde_18_state(X_k, st_idx)
                
                # Map H to Epoch: H_tilde = H_local * Phi(tk, t0)
                H_tilde = H_local @ Phi_tk_t0
                
                # Accumulate Information
                Lambda += H_tilde.T @ W @ H_tilde
                N += H_tilde.T @ W @ y_res

            # 3. Solve Normal Equations
            #    delta_x0 = inv(Lambda) * N
            delta_x0 = np.linalg.solve(Lambda, N)
            
            # 4. Update Reference Trajectory
            X_ref_0 += delta_x0
            
            # Statistics for Iteration
            rms_range = np.sqrt(rms_acum_range / len(sol.t))
            rms_rr = np.sqrt(rms_acum_rr / len(sol.t))
            dx_norm = np.linalg.norm(delta_x0)
            
            print(f"  Correction Norm: {dx_norm:.6e}")
            print(f"  RMS Range: {rms_range:.4f} m | RMS Range-Rate: {rms_rr:.5f} m/s")
            
            # Convergence Check
            if dx_norm < convergence_tol:
                print("  Convergence Criteria Met!")
                final_P0 = np.linalg.inv(Lambda) # Covariance at t0
                final_sol = sol
                break
            
            if i == max_iters - 1:
                print("  Max iterations reached without strict convergence.")
                final_P0 = np.linalg.inv(Lambda)
                final_sol = sol

        # --- FINAL PASS / POST-PROCESSING ---
        print("\nGenerating Final Statistics...")
        return self._generate_results(
            final_sol, initial_trajectory, obs, final_P0, Rk, X_0_guess
        )

    def _generate_results(self, sol, initial_traj, obs, P0_final, Rk, X_0_apriori):
        """Generates the results object for plotting."""
        
        n_steps = len(sol.t)
        
        # Lists
        dx_hist = []
        P_hist = []
        state_hist = []
        phi_hist = []
        prefit_res = []
        postfit_res = []
        nis_hist = []
        
        # Re-compute everything one last time purely for logging
        # (This is fast, just loop through the solution we already have)
        
        w_earth = np.array([0, 0, OMEGA_EARTH]) 

        for k in range(n_steps):
            # States
            X_k = sol.y[:self.n, k]            # Final Converged State
            X_k_init = initial_traj[k]         # Initial Guess State
            Phi_k = sol.y[self.n:, k].reshape((self.n, self.n))
            
            # Covariance Propagation: Pk = Phi * P0 * Phi^T
            Pk = Phi_k @ P0_final @ Phi_k.T
            
            # Measurement
            meas_row = obs.iloc[k]
            st_idx = self.station_map[int(meas_row['Station_ID'])]
            y_obs = np.array([meas_row['Range(m)'], meas_row['Range_Rate(m/s)']])
            
            # --- Computed Observations ---
            
            # 1. Post-Fit (Converged)
            r_gs = X_k[9 + 3*st_idx : 9 + 3*st_idx + 3]
            v_gs = np.cross(w_earth, r_gs)
            gs_state = np.concatenate([r_gs, v_gs])
            sc_state = np.concatenate([X_k[0:3], X_k[3:6]])
            y_post = compute_rho_rhodot(sc_state, gs_state)
            
            # 2. Pre-Fit (Initial Guess)
            r_gs_i = X_k_init[9 + 3*st_idx : 9 + 3*st_idx + 3]
            v_gs_i = np.cross(w_earth, r_gs_i)
            gs_state_i = np.concatenate([r_gs_i, v_gs_i])
            sc_state_i = np.concatenate([X_k_init[0:3], X_k_init[3:6]])
            y_pre = compute_rho_rhodot(sc_state_i, gs_state_i)
            
            # Residuals
            res_post = y_obs - y_post
            res_pre = y_obs - y_pre
            
            # NIS Calculation
            # S = H P H^T + R
            H_local = compute_H_tilde_18_state(X_k, st_idx)
            S = H_local @ Pk @ H_local.T + Rk
            nis = res_post.T @ np.linalg.solve(S, res_post)
            
            # Store
            dx_hist.append(X_k - X_k_init) # Deviation from initial guess
            P_hist.append(Pk)
            state_hist.append(X_k)
            phi_hist.append(Phi_k)
            prefit_res.append(res_pre)
            postfit_res.append(res_post)
            nis_hist.append(nis)

        return BatchResults(dx_hist, P_hist, state_hist, phi_hist, prefit_res, postfit_res, nis_hist)