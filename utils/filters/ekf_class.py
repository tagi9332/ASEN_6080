import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Any

# Local Imports
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from resources.gs_locations_latlon import stations_ll
from utils.misc.print_progress import print_progress
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6
from utils.process_noise.compute_q_discrete import compute_q_discrete 

@dataclass
class FilterResults:
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

class EKF:
    def __init__(self, n_states: int = 6):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, obs, X_0, x_0, P0, Rk, options):

        # Print filter start message
        print(f"Initializing EKF with n_states={self.n}...")
        
        # --- Initialization ---
        coeffs = options['coeffs']
        bootstrap_steps = options.get('bootstrap_steps', 0)
        abs_tol = options['abs_tol']
        rel_tol = options['rel_tol']
        
        # ODE Function: Default to 6x6 if not provided, but allow override for DMC (9x9)
        ode_func = options.get('ode_func', zonal_sph_ode_6x6)

        # --- COEFFICIENT SAFETY CHECK ---
        # If using the default 6x6 ODE but coeffs has 4 elements (mu, J2, J3, B),
        # slice it to prevent "too many values to unpack" error.
        if ode_func == zonal_sph_ode_6x6 and len(coeffs) > 3:
            coeffs_for_ode = coeffs[:3]
        else:
            coeffs_for_ode = coeffs

        # Initialize State: Generic slicing to support 6 (SNC) or 9 (DMC) states
        # X_curr is the full state vector (e.g., 9x1 for DMC)
        X_curr = X_0[0:self.n].copy()
        
        # Deviation Estimate (x_hat)
        x_dev = x_0.copy()
        
        P = P0.copy()
        
        _P, _state, _x = [], [], []
        _prefit_res, _postfit_res = [], []
        _nis_hist = []
        
        # Initialize time
        t_prev = obs.iloc[0]['Time(s)'] 

        # Loop starts at 1, propagating from t_prev (k-1) to t_curr (k)
        for k in range(1, len(obs)):
            meas_row = obs.iloc[k]
            t_curr = meas_row['Time(s)']
            dt = t_curr - t_prev

            # Print progress
            print_progress(k, len(obs))

            # -------------------------------------------------------
            # 1. PROPAGATION (Step-by-Step)
            # -------------------------------------------------------
            # Initial condition: Current Ref State (n) + Identity STM (n*n)
            # We reset STM to Identity at every step to ensure numerical stability.
            state_integ_0 = np.concatenate([X_curr, np.eye(self.n).flatten()])
            
            sol_step = solve_ivp(
                ode_func, 
                (t_prev, t_curr), 
                state_integ_0,
                args=(coeffs_for_ode,),
                rtol=rel_tol, atol=abs_tol
            )
            
            # Extract propagated Reference (first n) and Step-STM (last n*n)
            X_ref_pred = sol_step.y[0:self.n, -1]
            Phi_step = sol_step.y[self.n:, -1].reshape(self.n, self.n)
            
            # Propagate Deviation: dx_k = Phi * dx_k-1
            x_dev_pred = Phi_step @ x_dev

            # --- PROCESS NOISE STEP ---
            # Use the new helper function to compute Q_k
            # It handles the rotation from RIC to ECI using position/velocity from X_ref_pred
            Q_k = compute_q_discrete(dt, X_ref_pred, options)
            
            # Propagate Covariance: P = Phi * P * Phi' + Q
            P_pred = Phi_step @ P @ Phi_step.T + Q_k
            
            # -------------------------------------------------------
            # 2. MEASUREMENT UPDATE
            # -------------------------------------------------------
            station_idx = int(meas_row['Station_ID']) - 1
            lat, lon = stations_ll[station_idx]
            Rs, Vs = get_gs_eci_state(lat, lon, t_curr, init_theta=np.deg2rad(122))
            
            # Predicted Observation (based on Reference)
            # compute_rho_rhodot expects 6-element state [r, v]
            y_pred_ref = compute_rho_rhodot(X_ref_pred[0:6], np.concatenate([Rs, Vs]))
            y_obs = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])
            
            # Residual (Observation - Reference)
            prefit_res = y_obs - y_pred_ref
            
            # H Matrix evaluated at Reference
            # compute_H_matrix returns 2x6. If n_states=9, we pad with zeros.
            H_6 = compute_H_matrix(X_ref_pred[0:3], X_ref_pred[3:6], Rs, Vs)
            
            if self.n > 6:
                # Pad H for extra states (e.g. DMC acceleration parameters)
                H = np.hstack([H_6, np.zeros((2, self.n - 6))])
            else:
                H = H_6
            
            # Innovation: dy - H * deviation_prediction
            innovation = prefit_res - H @ x_dev_pred 

            S = H @ P_pred @ H.T + Rk
            K = P_pred @ H.T @ np.linalg.inv(S)

            # NIS Calculation
            nis = innovation.T @ np.linalg.solve(S, innovation)
            
            # Update Deviation
            x_dev = x_dev_pred + K @ innovation
            x_dev_log = x_dev.copy()

            # Update Covariance (Joseph form)
            IKH = self.I - K @ H
            P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T
            
            # -------------------------------------------------------
            # 3. RECTIFICATION (The Switch)
            # -------------------------------------------------------
            if k < bootstrap_steps:
                # === LKF Mode ===
                # Reference trajectory evolves naturally (X_curr = X_ref_pred)
                # Deviation grows large.
                X_curr = X_ref_pred
                
                # Best Estimate = Ref + Dev
                X_best_est = X_curr + x_dev
                
            else:
                # === EKF Mode ===
                # Absorb deviation into the reference at every step
                X_curr = X_ref_pred + x_dev
                
                # Reset deviation to zero
                x_dev = np.zeros(self.n)
                
                X_best_est = X_curr

            # -------------------------------------------------------
            # 4. LOGGING
            # -------------------------------------------------------
            postfit_res = prefit_res - H @ x_dev_log

            _x.append(x_dev_log.copy())
            _state.append(X_best_est.copy())
            _P.append(P.copy())
            _prefit_res.append(prefit_res.copy())
            _postfit_res.append(postfit_res.copy())
            _nis_hist.append(nis)
            
            t_prev = t_curr

        return FilterResults(_x, _P, _state, _prefit_res, _postfit_res, _nis_hist)