import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Any
import os, sys

# set the path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Imports
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from resources.gs_locations_latlon import stations_ll


@dataclass
class EKFResults:
    dx_hist: Any
    P_hist: Any
    corrected_state_hist: Any
    innovations: Any
    postfit_residuals: Any
    S_hist: Any

    def __post_init__(self):
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.corrected_state_hist = np.array(self.corrected_state_hist)
        self.innovations = np.array(self.innovations)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.S_hist = np.array(self.S_hist)

# ============================================================
# Extended Kalman Filter
# ============================================================
class EKF:
    def __init__(self, n_states: int = 6):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, sol_ref, meas_df, x_0, P0, Rk, Q, switch_idx=100) -> EKFResults:
            # Initial Filter State
            x = x_0.copy()       # Deviation for LKF phase
            P = P0.copy()          # Covariance
            current_state = (sol_ref.y[0:6, 0] + x_0).copy()   # Total state for EKF phase
            
            # Storage
            _x, _P, _states, _innov, _post, _S = [], [], [], [], [], []
            
            Phi_prev = self.I.copy()

            for k in range(len(meas_df)):
                row = meas_df.iloc[k]
                t_curr = row['Time(s)']
                t_prev = meas_df.iloc[k-1]['Time(s)'] if k > 0 else 0
                dt = t_curr - t_prev

                # Ground Station ECI State
                Rs, Vs = get_gs_eci_state(
                    stations_ll[int(row['Station_ID'])-1][0], 
                    stations_ll[int(row['Station_ID'])-1][1], 
                    t_curr, init_theta=np.deg2rad(122)
                )
                y_obs = np.array([row['Range(km)'], row['Range_Rate(km/s)']])

                # ========================================================
                # LKF BOOTSTRAP (k < switch_idx)
                # ========================================================
                if k < switch_idx:
                    # Prediction using Reference Trajectory
                    Phi_global = sol_ref.y[6:, k].reshape(self.n, self.n)
                    Phi_incr = Phi_global @ np.linalg.inv(Phi_prev)
                    x_ref = sol_ref.y[0:6, k]
                    
                    dx_pred = Phi_incr @ x
                    P_pred = Phi_incr @ P @ Phi_incr.T + (Q * dt)
                    
                    # Observation (Linearized around reference)
                    H = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs)
                    y_nom = compute_rho_rhodot(x_ref, np.concatenate([Rs, Vs]))
                    innovation = y_obs - (y_nom + H @ dx_pred)
                    
                    # Update
                    S = H @ P_pred @ H.T + Rk
                    K = P_pred @ H.T @ np.linalg.inv(S)
                    
                    x = dx_pred + K @ innovation
                    IKH = self.I - K @ H
                    P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T
                    
                    current_state = x_ref + x
                    y_post = y_obs - (y_nom + H @ x)
                    
                    Phi_prev = Phi_global # Update for next LKF step

                # ========================================================
                # EKF TRANSITION (k >= switch_idx)
                # ========================================================
                else:
                    # Prediction using Nonlinear Integration
                    phi_init = self.I.flatten()
                    ode_init = np.concatenate([current_state, phi_init])
                    
                    ekf_sol = solve_ivp(
                        zonal_sph_ode_6x6, (t_prev, t_curr), ode_init, 
                        args=([MU_EARTH, J2, 0],), rtol=1e-10, atol=1e-10
                    )
                    state_pred = ekf_sol.y[0:6, -1]
                    Phi_incr = ekf_sol.y[6:, -1].reshape(self.n, self.n)
                    
                    P_pred = Phi_incr @ P @ Phi_incr.T + (Q * dt)
                    
                    # Observation (Linearized around PREDICTED state)
                    y_pred_ekf, H = compute_H_matrix(state_pred[0:3], state_pred[3:6], Rs, Vs)
                    innovation = y_obs - y_pred_ekf
                    
                    # Update & Rectification
                    S = H @ P_pred @ H.T + Rk
                    K = P_pred @ H.T @ np.linalg.inv(S)
                    
                    current_state = state_pred + K @ innovation
                    IKH = self.I - K @ H
                    P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T
                    
                    # Post-fit residual (nonlinear)
                    y_post = y_obs - compute_H_matrix(current_state[0:3], current_state[3:6], Rs, Vs)[0]

                # Store results
                _x.append(current_state - sol_ref.y[0:6, k])
                _P.append(P.copy())
                _states.append(current_state.copy())
                _innov.append(innovation.copy())
                _post.append(y_post.copy())
                _S.append(S.copy())

            return EKFResults(_x, _P, _states, _innov, _post, _S)