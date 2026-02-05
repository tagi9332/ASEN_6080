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
    def __init__(self, n_states: int = 18, station_map: dict = None): # Type: ignore
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

    def run(self, obs, X_0_guess, P0_apriori, Rk_full, options) -> BatchResults:
        """
        Runs the Iterative Batch Least Squares (Differential Correction).
        
        Args:
            obs (DataFrame): Measurements. Must contain 'Range(m)' OR 'Range_Rate(m/s)' (or both).
            X_0_guess (np.array): Initial Guess for Reference State at t0 (18x1)
            P0_apriori (np.array): A priori covariance matrix (18x18)
            Rk_full (np.array): Full Measurement noise covariance (2x2). 
                                Expected format: [[var_range, 0], [0, var_rr]]
            options (dict): 'max_iterations', 'abs_tol', 'rel_tol', 'convergence_tol'
        """
        
        print(f"{'='*60}")
        print("Initializing Batch Least Squares Filter...")
        print(f"{'='*60}")

        # --- 1. Detect Available Measurements ---
        meas_indices = []
        if 'Range(m)' in obs.columns:
            meas_indices.append(0)
        if 'Range_Rate(m/s)' in obs.columns:
            meas_indices.append(1)
        
        if not meas_indices:
            raise ValueError("Observation Dataframe must contain 'Range(m)' or 'Range_Rate(m/s)' columns.")

        meas_indices = np.array(meas_indices)
        print(f"Detected Measurement Modes (Indices): {meas_indices} (0=Range, 1=RangeRate)")

        # --- 2. Slice R Matrix and Compute Weight Matrix ---
        # We slice Rk_full to match only the measurements we are using
        Rk_active = Rk_full[meas_indices][:, meas_indices]
        W = np.linalg.inv(Rk_active)

        # Extract Options
        max_iters = options.get('max_iterations', 10)
        convergence_tol = options.get('convergence_tol', 1e-3)
        abs_tol = options.get('abs_tol', 1e-12)
        rel_tol = options.get('rel_tol', 1e-12)

        # Setup
        time_eval = obs['Time(s)'].values
        t_span = (0, time_eval[-1])
        
        # In Batch, we update the Reference State directly.
        X_ref_0 = X_0_guess.copy()
        inv_P_bar = np.linalg.inv(P0_apriori)

        # Storage for the "Previous Best" to calculate convergence
        X_ref_prev = X_ref_0.copy()
        
        # Initial trajectory (for pre-fit residuals calculation only)
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
            aug_X_ref = np.concatenate([X_ref_0, phi_0_flat])
            
            sol = solve_ivp(
                stm_eom_mu_j2_drag, t_span, aug_X_ref, 
                t_eval=time_eval, rtol=abs_tol, atol=rel_tol
            )
            
            # 2. Accumulate Normal Equations
            Lambda = inv_P_bar.copy()
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
                
                # Construct Full Observation Vector (with NaNs for missing data if needed, 
                # but here we rely on column existence checks)
                y_obs_full = np.zeros(2)
                if 0 in meas_indices: y_obs_full[0] = meas_row['Range(m)']
                if 1 in meas_indices: y_obs_full[1] = meas_row['Range_Rate(m/s)']
                
                # Slice Observation
                y_obs = y_obs_full[meas_indices]
                
                # Computed Measurement (y = h(X_k))
                r_sc = X_k[0:3]
                v_sc = X_k[3:6]
                st_start = 9 + (3 * st_idx)
                r_gs = X_k[st_start : st_start+3]
                
                w_earth = np.array([0, 0, OMEGA_EARTH]) 
                v_gs = np.cross(w_earth, r_gs)
                
                sc_state_6 = np.concatenate([r_sc, v_sc])
                gs_state_6 = np.concatenate([r_gs, v_gs])
                
                y_comp_full = compute_rho_rhodot(sc_state_6, gs_state_6)
                
                # Slice Computed Measurement
                y_comp = y_comp_full[meas_indices]
                
                # Residuals (y - h(x))
                y_res = y_obs - y_comp
                
                # RMS accumulation (Conditional)
                if 0 in meas_indices:
                    # Find where 0 is in the sliced array
                    idx_in_res = np.where(meas_indices == 0)[0][0]
                    rms_acum_range += y_res[idx_in_res]**2
                if 1 in meas_indices:
                    idx_in_res = np.where(meas_indices == 1)[0][0]
                    rms_acum_rr += y_res[idx_in_res]**2
                
                # H-Matrix (Local) - Compute full, then slice rows
                H_local_full = compute_H_tilde_18_state(X_k, st_idx) # Returns 2x18
                H_local = H_local_full[meas_indices, :]              # Returns len(meas_indices)x18
                
                # Map H to Epoch: H_tilde = H_local * Phi(tk, t0)
                H_tilde = H_local @ Phi_tk_t0
                
                # Accumulate Information
                Lambda += H_tilde.T @ W @ H_tilde
                N += H_tilde.T @ W @ y_res

            # 3. Solve Normal Equations
            delta_x0 = np.linalg.solve(Lambda, N)
            
            # 4. Update Reference Trajectory
            X_ref_0 += delta_x0
            
            # Statistics for Iteration
            rms_range = np.sqrt(rms_acum_range / len(sol.t)) if 0 in meas_indices else 0.0
            rms_rr = np.sqrt(rms_acum_rr / len(sol.t)) if 1 in meas_indices else 0.0
            dx_norm = np.linalg.norm(delta_x0)
            
            print(f"  Correction Norm: {dx_norm:.6e}")
            if 0 in meas_indices: print(f"  RMS Range: {rms_range:.4f} m ", end="")
            if 1 in meas_indices: print(f"| RMS Range-Rate: {rms_rr:.5f} m/s", end="")
            print("") # Newline
            
            # Convergence Check
            if dx_norm < convergence_tol:
                print("  Convergence Criteria Met!")
                final_P0 = np.linalg.inv(Lambda) 
                final_sol = sol
                break
            
            if i == max_iters - 1:
                print("  Max iterations reached without strict convergence.")
                final_P0 = np.linalg.inv(Lambda)
                final_sol = sol

        # --- FINAL PASS / POST-PROCESSING ---
        print("\nGenerating Final Statistics...")
        return self._generate_results(
            final_sol, initial_trajectory, obs, final_P0, Rk_active, X_0_guess, meas_indices
        )

    def _generate_results(self, sol, initial_traj, obs, P0_final, Rk_active, X_0_apriori, meas_indices):
        """Generates the results object for plotting."""
        
        n_steps = len(sol.t)
        
        dx_hist = []
        P_hist = []
        state_hist = []
        phi_hist = []
        prefit_res = []
        postfit_res = []
        nis_hist = []
        
        w_earth = np.array([0, 0, OMEGA_EARTH]) 

        for k in range(n_steps):
            # States
            X_k = sol.y[:self.n, k]           
            X_k_init = initial_traj[k]        
            Phi_k = sol.y[self.n:, k].reshape((self.n, self.n))
            
            # Covariance Propagation
            Pk = Phi_k @ P0_final @ Phi_k.T
            
            # Measurement
            meas_row = obs.iloc[k]
            st_idx = self.station_map[int(meas_row['Station_ID'])]
            
            y_obs_full = np.zeros(2)
            if 0 in meas_indices: y_obs_full[0] = meas_row['Range(m)']
            if 1 in meas_indices: y_obs_full[1] = meas_row['Range_Rate(m/s)']
            y_obs = y_obs_full[meas_indices]
            
            # --- Computed Observations ---
            # 1. Post-Fit (Converged)
            r_gs = X_k[9 + 3*st_idx : 9 + 3*st_idx + 3]
            v_gs = np.cross(w_earth, r_gs)
            gs_state = np.concatenate([r_gs, v_gs])
            sc_state = np.concatenate([X_k[0:3], X_k[3:6]])
            
            y_post_full = compute_rho_rhodot(sc_state, gs_state)
            y_post = y_post_full[meas_indices]
            
            # 2. Pre-Fit (Initial Guess)
            r_gs_i = X_k_init[9 + 3*st_idx : 9 + 3*st_idx + 3]
            v_gs_i = np.cross(w_earth, r_gs_i)
            gs_state_i = np.concatenate([r_gs_i, v_gs_i])
            sc_state_i = np.concatenate([X_k_init[0:3], X_k_init[3:6]])
            
            y_pre_full = compute_rho_rhodot(sc_state_i, gs_state_i)
            y_pre = y_pre_full[meas_indices]
            
            # Residuals
            res_post = y_obs - y_post
            res_pre = y_obs - y_pre
            
            # NIS Calculation
            H_local_full = compute_H_tilde_18_state(X_k, st_idx)
            H_local = H_local_full[meas_indices, :]
            S = H_local @ Pk @ H_local.T + Rk_active
            nis = res_post.T @ np.linalg.solve(S, res_post)
            
            # Store
            dx_hist.append(X_k - X_k_init)
            P_hist.append(Pk)
            state_hist.append(X_k)
            phi_hist.append(Phi_k)
            
            # For residuals, we can store just the active ones, 
            # or pad back to size 2 for easier plotting later. 
            # Here I'm storing ONLY the active residuals to match matrix dims.
            prefit_res.append(res_pre)
            postfit_res.append(res_post)
            nis_hist.append(nis)

        return BatchResults(dx_hist, P_hist, state_hist, phi_hist, prefit_res, postfit_res, nis_hist)