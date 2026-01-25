import numpy as np
from dataclasses import dataclass
from typing import Any

# Local Imports
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix
from resources.gs_locations_latlon import stations_ll


@dataclass
class LKFResults:
    dx_hist: Any
    P_hist: Any
    corrected_state_hist: Any
    innovations: Any
    postfit_residuals: Any
    S_hist: Any

    def __post_init__(self):
        """Automatically converts lists to numpy arrays after initialization."""
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.corrected_state_hist = np.array(self.corrected_state_hist)
        self.innovations = np.array(self.innovations)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.S_hist = np.array(self.S_hist)

# ============================================================
# Linearized Kalman Filter
# ============================================================
class LKF:
    def __init__(self, n_states: int = 6):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, sol_ref, meas_df, dx_0, P0, Rk, Q) -> LKFResults:
        # Local state variables
        dx = dx_0.copy()
        P = P0.copy()
        Phi_prev = np.eye(self.n)

        # Temporary lists for collection
        _dx, _P, _state, _innov, _post, _S = [], [], [], [], [], []

        for k in range(len(sol_ref.t)):
            # --- 1. Dynamics & Prediction ---
            Phi_global = sol_ref.y[6:, k].reshape(self.n, self.n)
            Phi_incr = Phi_global @ np.linalg.inv(Phi_prev)
            dt = (sol_ref.t[k] - sol_ref.t[k-1]) if k > 0 else 0
            
            dx_pred = Phi_incr @ dx
            P_pred = Phi_incr @ P @ Phi_incr.T + (Q * dt)
            
            # --- 2. Ground Station & Measurement ---
            meas_row = meas_df.iloc[k]
            station_idx = int(meas_row['Station_ID']) - 1
            Rs, Vs = get_gs_eci_state(
                stations_ll[station_idx][0], 
                stations_ll[station_idx][1], 
                sol_ref.t[k], 
                init_theta=np.deg2rad(122)
            )
            
            x_ref = sol_ref.y[0:6, k]
            y_pred, H = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs)
            y_obs = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])

            # --- 3. Filter Update ---
            prefit_res = y_obs - (y_pred + H @ dx_pred)
            S = H @ P_pred @ H.T + Rk
            # Solving for K is more stable than np.linalg.inv(S)
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            dx = dx_pred + K @ prefit_res
            postfit_res = y_obs - (y_pred + H @ dx)
            
            # Joseph Form Covariance Update
            IKH = self.I - K @ H
            P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T

            # --- 4. Append Copies to Lists ---
            _dx.append(dx.copy())
            _P.append(P.copy())
            _state.append((x_ref + dx).copy())
            _innov.append(prefit_res.copy())
            _post.append(postfit_res.copy())
            _S.append(S.copy())
            
            Phi_prev = Phi_global.copy()

        # Hand off lists to the Dataclass, which converts them to arrays
        return LKFResults(_dx, _P, _state, _innov, _post, _S)