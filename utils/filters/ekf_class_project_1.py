import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Any

# Local Imports
from resources.constants import OMEGA_EARTH
from utils.zonal_harmonics.zonal_harmonics import stm_eom_mu_j2_drag
from utils.ground_station_utils.gs_meas_model_H import compute_H_tilde_18_state, compute_rho_rhodot
from utils.misc.print_progress import print_progress

@dataclass
class EKFResults:
    dx_hist: Any             
    P_hist: Any              
    state_hist: Any          
    prefit_residuals: Any         
    postfit_residuals: Any  
    nis_hist: Any 

    def __post_init__(self):
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.state_hist = np.array(self.state_hist)
        self.prefit_residuals = np.array(self.prefit_residuals)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.nis_hist = np.array(self.nis_hist)

class EKF:
    def __init__(self, n_states: int = 18, station_map: dict = None):
        self.n = n_states
        self.I = np.eye(n_states)
        if station_map is None:
            self.station_map = {101: 0, 337: 1, 394: 2}
        else:
            self.station_map = station_map

    def run(self, obs, X_0, P0, Rk, Q, options):
        print(f"Initializing Hybrid EKF (LKF-matched Bootstrapping)...")
        
        abs_tol = options.get('abs_tol', 1e-12)
        rel_tol = options.get('rel_tol', 1e-12)
        bootstrap_steps = options.get('bootstrap_steps', 0)

        # =======================================================
        # STEP 1: PRE-INTEGRATE REFERENCE (LKF STYLE)
        # =======================================================
        # We integrate the entire trajectory ONCE at the start.
        # This matches the LKF architecture exactly.
        
        times = obs['Time(s)'].values
        # Handle case where measurements don't start at t=0
        if times[0] != 0.0:
            t_eval_full = np.insert(times, 0, 0.0)
            obs_start_idx = 1
        else:
            t_eval_full = times
            obs_start_idx = 0

        if bootstrap_steps > 0:
            print("  -> Pre-integrating reference trajectory for bootstrapping...")
            phi_0_flat = np.eye(self.n).flatten()
            aug_0 = np.concatenate([X_0, phi_0_flat])
            
            # Continuous integration (LKF style)
            sol_ref = solve_ivp(
                stm_eom_mu_j2_drag, 
                (0, times[-1]), 
                aug_0, 
                t_eval=t_eval_full,
                rtol=rel_tol, atol=abs_tol
            )
            
            # Store the Global STM history for inversion
            Phi_history_global = sol_ref.y[self.n:, :].reshape(self.n, self.n, -1)
            X_ref_history = sol_ref.y[0:self.n, :]
        
        # =======================================================
        # STEP 2: MEASUREMENT LOOP
        # =======================================================
        
        # Current State Init
        X_curr = X_0.copy()   # EKF State (Total)
        x_dev = np.zeros(self.n) # LKF Deviation State
        P = P0.copy()
        
        # For LKF Inversion Logic
        Phi_prev_global = np.eye(self.n)
        
        # Storage
        _P = [P0.copy()]
        _state = [X_0.copy()]
        _dx_updates, _innovations, _postfit_res, _nis_hist = [], [], [], []
        t_prev = 0.0
        
        w_vec = np.array([0, 0, OMEGA_EARTH])
        phi_flat_identity = np.eye(self.n).flatten()

        for k in range(len(obs)):
            meas_row = obs.iloc[k]
            t_curr = meas_row['Time(s)']
            dt = t_curr - t_prev
            
            print_progress(k, len(obs))

            # --- A. PROPAGATION ---
            
            if k < bootstrap_steps:
                # ===============================================
                # LKF MODE (Exact Match to LKF Script)
                # ===============================================
                # Get index in the pre-computed solution
                sol_idx = k + obs_start_idx
                
                # 1. Retrieve Pre-computed Reference
                X_ref_k = X_ref_history[:, sol_idx]
                
                # 2. Compute Step STM via INVERSION (Matches LKF)
                # Phi_step = Phi_global(t_k) * inv(Phi_global(t_k-1))
                Phi_global_k = Phi_history_global[:, :, sol_idx]
                Phi_step = Phi_global_k @ np.linalg.inv(Phi_prev_global)
                
                # Update global tracker
                Phi_prev_global = Phi_global_k
                
                # 3. Propagate Deviation & Covariance
                x_dev_minus = Phi_step @ x_dev
                P_minus = Phi_step @ P @ Phi_step.T + (Q * dt)
                
                # For Measurement Update, our "Best Guess" for linearization is the Reference
                X_linearize = X_ref_k
                x_dev_in = x_dev_minus
                
            else:
                # ===============================================
                # EKF MODE (Standard Step-by-Step)
                # ===============================================
                
                # CHECK FOR RECTIFICATION SWITCH
                if k == bootstrap_steps:
                    print("\n[SWITCH] Bootstrapping finished. Switching to EKF Mode.")
                    # We take the last known best estimate from the LKF loop
                    # and make it the 'Current State' for the EKF integrator.
                    # Note: X_curr was updated at end of loop k-1
                    pass 

                # 1. Integrate Step-by-Step
                augmented_state = np.concatenate([X_curr, phi_flat_identity])
                
                if dt > 0:
                    sol_step = solve_ivp(
                        stm_eom_mu_j2_drag, 
                        (t_prev, t_curr), 
                        augmented_state,
                        rtol=rel_tol, atol=abs_tol
                    )
                    X_linearize = sol_step.y[0:self.n, -1]
                    Phi_step = sol_step.y[self.n:, -1].reshape(self.n, self.n)
                    P_minus = Phi_step @ P @ Phi_step.T + (Q * dt)
                else:
                    X_linearize = X_curr
                    P_minus = P
                
                # In EKF, deviation is zero relative to the propagated state
                x_dev_in = np.zeros(self.n)

            # --- B. MEASUREMENT UPDATE ---
            
            # Ground Station Logic
            raw_station_id = int(meas_row['Station_ID'])
            station_idx = self.station_map[raw_station_id]
            st_idx_start = 9 + (3 * station_idx)
            
            # Rotate Station (Common to both)
            r_gs_inertial = X_linearize[st_idx_start : st_idx_start + 3]
            theta = OMEGA_EARTH * t_curr
            c, s = np.cos(theta), np.sin(theta)
            R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            r_gs_curr = R_z @ r_gs_inertial
            v_gs_curr = np.cross(w_vec, r_gs_curr)
            
            # Construct States for h(x)
            sc_state = X_linearize[0:6]
            gs_state = np.concatenate([r_gs_curr, v_gs_curr])
            
            # Prediction
            y_pred = compute_rho_rhodot(sc_state, gs_state)
            y_obs = np.array([meas_row['Range(m)'], meas_row['Range_Rate(m/s)']])
            
            # H Matrix
            X_for_H = X_linearize.copy()
            X_for_H[st_idx_start : st_idx_start+3] = r_gs_curr
            H = compute_H_tilde_18_state(X_for_H, station_idx)
            
            # Innovation
            prefit_res = y_obs - y_pred
            innovation = prefit_res - (H @ x_dev_in)
            
            # Update
            S = H @ P_minus @ H.T + Rk
            K = P_minus @ H.T @ np.linalg.inv(S)
            
            dx = K @ innovation
            P_plus = (self.I - K @ H) @ P_minus @ (self.I - K @ H).T + K @ Rk @ K.T
            
            nis = innovation.T @ np.linalg.solve(S, innovation)

            # --- C. POST-PROCESSING & STATE UPDATE ---
            
            if k < bootstrap_steps:
                # LKF Mode: Accumulate deviation
                x_dev = x_dev_minus + dx
                # Best Estimate = Reference + Deviation
                X_curr = X_linearize + x_dev
                dx_log = dx # The correction this step
            else:
                # EKF Mode: Apply directly to state
                X_curr = X_linearize + dx
                x_dev = np.zeros(self.n) # Reset deviation
                dx_log = dx

            # Logging
            postfit_res = prefit_res - H @ dx_log # Approx
            
            _state.append(X_curr.copy())
            _P.append(P_plus.copy())
            _innovations.append(innovation)
            _postfit_res.append(postfit_res)
            _nis_hist.append(nis)
            _dx_updates.append(dx_log)

            t_prev = t_curr
            P = P_plus

        return EKFResults(_dx_updates, _P, _state, _innovations, _postfit_res, _nis_hist)