import numpy as np
from dataclasses import dataclass
from typing import Any
from scipy.integrate import solve_ivp

# Local Imports
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from resources.gs_locations_latlon import stations_ll
from utils.misc.print_progress import print_progress
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6

@dataclass
class LKFResults:
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

# ============================================================
# Linearized Kalman Filter
# ============================================================
class LKF:
    def __init__(self, n_states: int = 6):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, obs, X_0, x_0, P0, Rk, Q, options) -> LKFResults:

        # Print filter start message
        print("Initializing LKF...")
        
        # Initialize filter arguments
        coeffs = options['coeffs']
        abs_tol = options['abs_tol']
        rel_tol = options['rel_tol']

        # Time vector from measurements
        time_eval = obs['Time(s)'].values

        # Solve initial reference trajectory
        sol_ref = solve_ivp(
                zonal_sph_ode_6x6, 
                (0, time_eval[-1]), 
                X_0, 
                t_eval=time_eval, 
                args=(coeffs,),
                rtol=abs_tol, 
                atol=rel_tol
        )

        # Initialization
        x = x_0.copy()
        P = P0.copy()
        Phi_prev = np.eye(self.n)

        # Temporary lists for collection
        _x, _P, _state, _prefit_res, _postfit_res, _nis = [], [], [], [], [], []

        for k in range(len(sol_ref.t)):
            # Print progress
            print_progress(k, len(sol_ref.t))

            # Dynamics & Prediction
            Phi_global = sol_ref.y[6:, k].reshape(self.n, self.n)
            Phi_incr = Phi_global @ np.linalg.inv(Phi_prev)
            dt = (sol_ref.t[k] - sol_ref.t[k-1]) if k > 0 else 0
            
            # Prediction Step
            x_pred = Phi_incr @ x
            P_pred = Phi_incr @ P @ Phi_incr.T + (Q * dt)
            
            # Ground Station & Measurement
            meas_row = obs.iloc[k]
            station_idx = int(meas_row['Station_ID']) - 1
            Rs, Vs = get_gs_eci_state(
                stations_ll[station_idx][0], 
                stations_ll[station_idx][1], 
                sol_ref.t[k], 
                init_theta=np.deg2rad(122)
            )
            
            x_ref = sol_ref.y[0:6, k]
            y_pred_ref = compute_rho_rhodot(x_ref, np.concatenate([Rs, Vs]))
            y_obs = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])

            # Filter Update
            prefit_res = y_obs - (y_pred_ref)

            # Prefit Residual
            H = compute_H_matrix(x_ref[0:3], x_ref[3:6], Rs, Vs)
            innovation = prefit_res - H @ x_pred

            S = H @ P_pred @ H.T + Rk
            K = P_pred @ H.T @ np.linalg.inv(S)

            # NIS Calculation
            nis = innovation.T @ np.linalg.solve(S,innovation)
            
            x = x_pred + K @ innovation

            # Joseph Form Covariance Update
            IKH = self.I - K @ H
            P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T

            # Postfit Residual
            postfit_res = prefit_res - H @ x
            
            # --- 4. Append Copies to Lists ---
            _x.append(x.copy())
            _P.append(P.copy())
            _state.append((x_ref + x).copy())
            _prefit_res.append(prefit_res.copy())
            _postfit_res.append(postfit_res.copy())
            _nis.append(nis.copy())
            
            Phi_prev = Phi_global.copy()

        # Hand off lists to the Dataclass, which converts them to arrays
        return LKFResults(_x, _P, _state, _prefit_res, _postfit_res, _nis)