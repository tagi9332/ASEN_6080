import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Any

# Local Imports
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from resources.gs_locations_latlon import stations_ll
from utils.misc.print_progress import print_progress
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_dmc

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

    def run(self, obs, X_0, x_0, P0, Rk, Q, options):

        # Print filter start message
        print("Initializing EKF...")
        
        # --- Initialization ---
        coeffs = options['coeffs']
        bootstrap_steps = options.get('bootstrap_steps', 0)
        abs_tol = options['abs_tol']
        rel_tol = options['rel_tol']
        SNC_frame = options.get('SNC_frame', 'ECI')
        
        # Force X_curr to be only the 6-element Cartesian state.
        X_curr = X_0[0:6].copy()
        
        # Deviation Estimate (x_hat)
        x_dev = x_0.copy()
        
        P = P0.copy()
        
        _P, _state, _x = [], [], []
        _prefit_res, _postfit_res = [], []
        _nis_hist = []
        t_prev = 0 # Assuming start time is 0

        for k in range(1,len(obs)):
            meas_row = obs.iloc[k]
            t_curr = meas_row['Time(s)']
            dt = t_curr - t_prev

            # Print progress
            print_progress(k,len(obs))

            # -------------------------------------------------------
            # 1. PROPAGATION
            # -------------------------------------------------------
            # Initial condition: Current Ref State (6) + Identity STM (36) = 42 elements
            state_integ_0 = np.concatenate([X_curr, np.eye(self.n).flatten()])
            
            sol_step = solve_ivp(
                zonal_sph_ode_6x6, 
                (t_prev, t_curr), 
                state_integ_0,
                args=(coeffs,),
                rtol=rel_tol, atol=abs_tol
            )
            
            # Extract propagated Reference (first 6) and STM (last 36)
            X_ref_pred = sol_step.y[0:6, -1]
            Phi_step = sol_step.y[6:, -1].reshape(self.n, self.n)
            
            # Propagate Deviation (LKF Prediction)
            x_dev_pred = Phi_step @ x_dev

            # SNC process noise
            Q_step_PSD = Q.copy() # Default if no frame transformation needed
            # If SNC is defined in RIC, rotate Q into ECI frame
            if SNC_frame == 'RIC':
                # Use current reference state to define the frame
                r_vec = X_ref_pred[0:3]
                v_vec = X_ref_pred[3:6]
                
                # --- Compute Rotation Matrix (RIC -> ECI) ---
                # Radial: Unit vector in direction of position
                u_r = r_vec / np.linalg.norm(r_vec)
                
                # Cross-track: Unit vector normal to orbital plane
                h_vec = np.cross(r_vec, v_vec)
                u_c = h_vec / np.linalg.norm(h_vec)
                
                # In-track: Completes the triad
                u_i = np.cross(u_c, u_r)
                
                # Rotation Matrix R = [u_r, u_i, u_c]
                R_RIC2ECI = np.column_stack((u_r, u_i, u_c))
                
                # Rotate the RIC diagonal covariance into ECI
                # Q_ECI = R * Q_RIC * R.T
                Q_step_PSD = R_RIC2ECI @ Q @ R_RIC2ECI.T

            # Calculate Discrete Process Noise Matrix (Q_k)
            # using the PSD for this specific step
            Q_rr = (dt**3 / 3) * Q_step_PSD
            Q_rv = (dt**2 / 2) * Q_step_PSD
            Q_vv = dt * Q_step_PSD
            
            Q_k  = np.block([
                [Q_rr, Q_rv],
                [Q_rv.T, Q_vv]
            ])
            
            # Propagate Covariance
            P_pred = Phi_step @ P @ Phi_step.T + Q_k
            
            # -------------------------------------------------------
            # 2. MEASUREMENT UPDATE
            # -------------------------------------------------------
            station_idx = int(meas_row['Station_ID']) - 1
            lat, lon = stations_ll[station_idx]
            Rs, Vs = get_gs_eci_state(lat, lon, t_curr, init_theta=np.deg2rad(122))
            
            # Predicted Observation (based on Reference)
            y_pred_ref = compute_rho_rhodot(X_ref_pred, np.concatenate([Rs, Vs]))
            y_obs = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])
            
            # Residual (Observation - Reference)
            prefit_res = y_obs - y_pred_ref
            
            # H Matrix evaluated at Reference
            H = compute_H_matrix(X_ref_pred[0:3], X_ref_pred[3:6], Rs, Vs)
            
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
                X_curr = X_ref_pred
                
                # Best Estimate = Ref + Dev
                X_best_est = X_curr + x_dev
                
            else:
                # === EKF Mode ===
                X_curr = X_ref_pred + x_dev
                
                # Reset deviation
                x_dev = np.zeros(6)
                
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