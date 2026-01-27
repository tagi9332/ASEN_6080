import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Any

# Local Imports
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from resources.gs_locations_latlon import stations_ll
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6

@dataclass
class FilterResults:
    """Standardized container for filter output."""
    dx_hist: Any             # Deviation from reference (LKF) or N/A (EKF)
    P_hist: Any              # Covariance history
    state_hist: Any          # Full estimated state [x, y, z, vx, vy, vz]
    innovations: Any         # Pre-fit residuals
    postfit_residuals: Any   # Post-fit residuals
    S_hist: Any              # Innovation covariance
    times: Any               # Time vector

    def __post_init__(self):
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.state_hist = np.array(self.state_hist)
        self.innovations = np.array(self.innovations)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.S_hist = np.array(self.S_hist)
        self.times = np.array(self.times)

class EKF:
    def __init__(self, n_states: int = 6):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, meas_df, x_0_dev, P0, Rk, Q, coeffs, 
            sol_ref_lkf, bootstrap_steps=10):
        """
        Runs a hybrid LKF/EKF filter.
        
        Phase 1 (k < bootstrap_steps): LKF Mode
            - Relies on 'sol_ref_lkf' (pre-computed reference trajectory).
            - Updates the deviation 'dx'.
        
        Phase 2 (k >= bootstrap_steps): EKF Mode
            - Propagates the full state non-linearly using 'zonal_sph_ode_6x6'.
            - Linearizes measurement H around the current estimate.
        """
        
        # --- Initialization ---
        # Initial Reference State (from the provided solution object)
        x_ref_0 = sol_ref_lkf.y[0:6, 0]
        
        # Current Best Estimate (Full State)
        x_est = x_ref_0 + x_0_dev
        
        # Current Covariance
        P = P0.copy()
        
        # LKF specific: Track deviation separately during bootstrap
        dx = x_0_dev.copy()
        Phi_prev_lkf = np.eye(self.n) # STM accumulator for LKF
        
        # History lists
        hist_P, hist_state, hist_dx = [], [], []
        hist_innov, hist_post, hist_S = [], [], []
        hist_times = []

        # Time management
        t_prev = sol_ref_lkf.t[0]

        # Loop through measurements
        for k in range(len(meas_df)):
            meas_row = meas_df.iloc[k]
            t_curr = meas_row['Time(s)']
            dt = t_curr - t_prev
            
            # --- 1. PREDICTION STEP ---
            if k < bootstrap_steps:
                # === MODE: LKF (Linearized Kalman Filter) ===
                # Logic: Propagate deviation (dx) using pre-computed STM from reference
                
                # Fetch Reference State and Global STM at step k
                x_ref_k = sol_ref_lkf.y[0:6, k]
                Phi_global_k = sol_ref_lkf.y[6:, k].reshape(self.n, self.n)
                
                # Compute Incremental STM (k-1 to k)
                if k == 0:
                    Phi_step = np.eye(self.n)
                else:
                    Phi_step = Phi_global_k @ np.linalg.inv(Phi_prev_lkf)
                
                # Propagate Deviation
                dx_pred = Phi_step @ dx
                x_pred = x_ref_k + dx_pred
                
                # Propagate Covariance
                P_pred = Phi_step @ P @ Phi_step.T + (Q * dt)
                
                # Store for next loop
                Phi_prev_lkf = Phi_global_k
                
            else:
                # === MODE: EKF (Extended Kalman Filter) ===
                # Logic: Propagate full state (x_est) using nonlinear ODE
                
                # Initial condition for integration: [Pos, Vel, STM=Identity]
                state_integ_0 = np.concatenate([x_est, np.eye(self.n).flatten()])
                
                # Integrate from t_prev to t_curr
                sol_step = solve_ivp(
                    zonal_sph_ode_6x6, 
                    (t_prev, t_curr), 
                    state_integ_0,
                    args=(coeffs,),
                    rtol=1e-12, atol=1e-12
                )
                
                # Extract results at t_curr
                x_pred = sol_step.y[0:6, -1]
                Phi_step = sol_step.y[6:, -1].reshape(self.n, self.n)
                
                # Propagate Covariance
                P_pred = Phi_step @ P @ Phi_step.T + (Q * dt)
                
                # For logging consistency, dx is 0 in EKF mode (or difference from reference if available)
                dx_pred = np.zeros(6) 

            # --- 2. GROUND STATION GEOMETRY ---
            station_idx = int(meas_row['Station_ID']) - 1
            
            # --- FIX APPLIED HERE: Unpack only 2 values (lat, lon) ---
            lat, lon = stations_ll[station_idx]
            
            # Rotate Earth to t_curr to get Station ECI state
            Rs, Vs = get_gs_eci_state(lat, lon, t_curr, init_theta=np.deg2rad(122))
            
            # --- 3. MEASUREMENT UPDATE ---
            # Predicted Observation (Range, Range-Rate)
            y_pred_vec = compute_rho_rhodot(x_pred, np.concatenate([Rs, Vs]))
            y_obs_vec = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])
            
            # H Matrix (Linearized about predicted state)
            H = compute_H_matrix(x_pred[0:3], x_pred[3:6], Rs, Vs)
            
            # Innovations (Pre-fit Residuals)
            innov = y_obs_vec - y_pred_vec
            
            # Kalman Gain
            S = H @ P_pred @ H.T + Rk
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # State Correction
            update = K @ innov
            
            # Apply Correction
            if k < bootstrap_steps:
                dx = dx_pred + update
                x_est = x_ref_k + dx
            else:
                x_est = x_pred + update
            
            # Covariance Update (Joseph Form for stability)
            IKH = self.I - K @ H
            P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T
            
            # Post-fit Residuals (for analysis)
            y_post_vec = compute_rho_rhodot(x_est, np.concatenate([Rs, Vs]))
            post_res = y_obs_vec - y_post_vec

            # --- 4. LOGGING ---
            hist_state.append(x_est.copy())
            hist_P.append(P.copy())
            hist_dx.append(dx.copy() if k < bootstrap_steps else (x_est - sol_ref_lkf.y[0:6, k]))
            hist_innov.append(innov)
            hist_post.append(post_res)
            hist_S.append(S)
            hist_times.append(t_curr)
            
            t_prev = t_curr

        return FilterResults(hist_dx, hist_P, hist_state, hist_innov, hist_post, hist_S, hist_times)