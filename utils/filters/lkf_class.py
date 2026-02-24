import numpy as np
from dataclasses import dataclass
from typing import Any
from scipy.integrate import solve_ivp

# Local Imports
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from resources.gs_locations_latlon import stations_ll
from utils.misc.print_progress import print_progress
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6, zonal_sph_ode_dmc
from utils.process_noise.compute_q_discrete import compute_q_discrete

@dataclass
class LKFResults:
    dx_hist: Any
    P_hist: Any
    state_hist: Any
    innovations: Any
    postfit_residuals: Any
    nis_hist: Any
    P_pred_hist: Any = None  # Optional: Store predicted covariance before update
    Phi_step_hist: Any = None  # Optional: Store step STMs for analysis
    X_ref_hist: Any = None  # Optional: Store reference trajectory history
    times: Any = None  # Optional: Store time history for alignment with measurements

    def __post_init__(self):
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.state_hist = np.array(self.state_hist)
        self.innovations = np.array(self.innovations)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.nis_hist = np.array(self.nis_hist)
        self.P_pred_hist = np.array(self.P_pred_hist) if self.P_pred_hist is not None else None
        self.Phi_step_hist = np.array(self.Phi_step_hist) if self.Phi_step_hist is not None else None
        self.X_ref_hist = np.array(self.X_ref_hist) if self.X_ref_hist is not None else None
        self.times = np.array(self.times) if self.times is not None else None

# ============================================================
# Linearized Kalman Filter
# ============================================================
class LKF:
    def __init__(self, n_states: int = 6):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, obs, X_0, x_0, P0, Rk, options) -> LKFResults:
        print(f"Initializing Fast LKF (One-Shot Integration) with n={self.n}...")
        
        # 1. Setup
        coeffs = options['coeffs']
        abs_tol = options['abs_tol']
        rel_tol = options['rel_tol']
        
        if options.get('method') == 'DMC':
            ode_func = zonal_sph_ode_dmc
        else:
            ode_func = zonal_sph_ode_6x6

        # Initialize State Variables
        X_ref_init = X_0[0:self.n]
        x_dev = x_0.copy()
        P = P0.copy()
        
        # Extract all time points
        times = obs['Time(s)'].values
        t_start = times[0]
        t_end = times[-1]
        
        # 2. ONE-SHOT INTEGRATION
        # ------------------------------------------------------------------
        # We integrate the Reference State and the Cumulative STM (Phi) 
        # from start to finish in one call.
        
        print(f"   Propagating reference trajectory for {len(times)} steps...")
        
        # Initial State: [Reference, Identity_STM]
        state_integ_0 = np.concatenate([X_ref_init, np.eye(self.n).flatten()])
        
        sol = solve_ivp(
            ode_func, 
            (t_start, t_end), 
            state_integ_0, 
            t_eval=times,  # Force output exactly at measurement times
            args=(coeffs,), 
            rtol=abs_tol, 
            atol=rel_tol
        )
        
        # Unpack results: Shape is (n_states, n_times)
        X_ref_all = sol.y[0:self.n, :]
        Phi_all_flat = sol.y[self.n:, :]
        
        # 3. FILTER LOOP
        # ------------------------------------------------------------------
        _x, _P, _state, _prefit_res, _postfit_res, _nis = [], [], [], [], [], []
        _Phi_step,  _P_pred, _X_ref = [], [], []
        
        # The integration includes the start time at index 0.
        # We iterate starting from k=1 (the second point).
        
        for k in range(1, len(times)):
            # Print progress less frequently to save I/O time
            if k % 100 == 0: print_progress(k, len(times))

            t_curr = times[k]
            t_prev = times[k-1]
            dt = t_curr - t_prev
            meas_row = obs.iloc[k]

            # --- A. Extract Pre-Computed State & STM ---
            X_ref_curr = X_ref_all[:, k]
            
            # Get Cumulative STMs: Phi(t_k, t0) and Phi(t_{k-1}, t0)
            Phi_cum_curr = Phi_all_flat[:, k].reshape(self.n, self.n)
            Phi_cum_prev = Phi_all_flat[:, k-1].reshape(self.n, self.n)
            
            # Compute Step STM: Phi(t_k, t_{k-1}) = Phi(t_k, t0) @ inv(Phi(t_{k-1}, t0))
            # This is the "transition from previous step to now"
            Phi_step = Phi_cum_curr @ np.linalg.inv(Phi_cum_prev)

            # --- B. Time Update (Prediction) ---
            # Propagate deviation: x_dev_k = Phi_step * x_dev_{k-1}
            x_dev_pred = Phi_step @ x_dev
            
            # Process Noise
            Q_k = compute_q_discrete(dt, X_ref_curr, options)
            
            # Propagate Covariance
            P_pred = Phi_step @ P @ Phi_step.T + Q_k

            # --- C. Measurement Update ---
            station_idx = int(meas_row['Station_ID']) - 1
            Rs, Vs = get_gs_eci_state(
                stations_ll[station_idx][0], 
                stations_ll[station_idx][1], 
                t_curr
            )
            
            # Measurement Models
            y_pred_ref = compute_rho_rhodot(X_ref_curr[0:6], np.concatenate([Rs, Vs]))
            y_obs = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])
            
            # Prefit Residual
            prefit_res = y_obs - y_pred_ref
            
            # H Matrix
            H_6 = compute_H_matrix(X_ref_curr[0:3], X_ref_curr[3:6], Rs, Vs)
            if self.n > 6:
                H = np.hstack([H_6, np.zeros((2, self.n - 6))])
            else:
                H = H_6
                
            # Kalman Gain
            innovation = prefit_res - H @ x_dev_pred
            S = H @ P_pred @ H.T + Rk
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # State Update
            x_dev = x_dev_pred + K @ innovation
            
            # Covariance Update (Joseph Form)
            IKH = self.I - K @ H
            P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T
            
            # --- D. Storage ---
            nis = innovation.T @ np.linalg.solve(S, innovation)
            postfit_res = prefit_res - H @ x_dev
            X_total_est = X_ref_curr + x_dev

            _x.append(x_dev.copy())
            _P.append(P.copy())
            _state.append(X_total_est.copy())
            _prefit_res.append(prefit_res.copy())
            _postfit_res.append(postfit_res.copy())
            _nis.append(nis.copy())
            _Phi_step.append(Phi_step.copy())
            _P_pred.append(P_pred.copy())
            _X_ref.append(X_ref_curr.copy())

        # Final cumulative STM is simply the last extracted cumulative STM
        Phi_final = Phi_all_flat[:, -1].reshape(self.n, self.n)

        return LKFResults(_x, _P, _state, _prefit_res, _postfit_res, _nis, _Phi_step, _P_pred, _X_ref)
    

    