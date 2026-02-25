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
# Linearized Kalman Filter (Togglable Potter / Joseph Form)
# ============================================================
class LKF:
    def __init__(self, n_states: int = 6):
        self.n = n_states
        self.I = np.eye(n_states)

    @staticmethod
    def _potter_update(x, P, innovation_scalar, H_row, R_scalar):
        """
        Performs a single scalar Potter Square Root update.
        """
        # 1. Cholesky Factorization P = S * S.T
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # Fallback: Force symmetry and slight positivity if nearly singular
            P_sym = (P + P.T) / 2
            P_sym += np.eye(len(P)) * 1e-18
            S = np.linalg.cholesky(P_sym)

        # 2. Calculate F = S.T * H.T
        F = S.T @ H_row.T
        
        # 3. Innovation Variance: alpha = 1 / (F.T*F + R)
        inv_var = (F.T @ F) + R_scalar
        alpha = 1.0 / inv_var
        
        # 4. Gamma Factor
        gamma = 1.0 / (1.0 + np.sqrt(R_scalar * alpha))
        
        # 5. Kalman Gain: K = alpha * (S * F)
        K = alpha * (S @ F)
        
        # 6. Update Square Root Matrix S
        SF = S @ F
        S_new = S - (alpha * gamma) * np.outer(SF, F)
        
        # 7. Reconstruct P = S_new * S_new.T
        P_new = S_new @ S_new.T
        
        # 8. Update State
        x_new = x + K * innovation_scalar
        
        return x_new, P_new

    def run(self, obs, X_0, x_0, P0, Rk, options) -> LKFResults:
        # Toggle for Potter formulation (defaults to False / Joseph Form)
        use_potter = options.get('potter_form', False)
        
        print(f"Initializing Stable LKF with n={self.n}...")
        
        # 1. Setup
        coeffs = options['coeffs']
        abs_tol = options['abs_tol']
        rel_tol = options['rel_tol']
        
        if options.get('method') == 'DMC':
            ode_func = zonal_sph_ode_dmc
        else:
            ode_func = zonal_sph_ode_6x6

        # Extract all time points
        times = obs['Time(s)'].values
        
        # Initialize State Variables
        X_ref_curr = X_0[0:self.n].copy()
        
        # Initialize history lists with t0 conditions (properly padded for smoother)
        _x = [x_0.copy()]
        _P = [P0.copy()]
        _state = [(X_ref_curr + x_0).copy()]
        _prefit_res = [np.zeros(2)]  # Dummy zero residuals for t0
        _postfit_res = [np.zeros(2)] 
        _nis = [0.0]                 
        _Phi_step = [np.eye(self.n)] # Step STM at t0 is Identity
        _P_pred = [P0.copy()]        
        _X_ref = [X_ref_curr.copy()]
        
        # 2. FILTER LOOP (Step-by-step Integration)
        # ------------------------------------------------------------------
        print(f"   Filtering and propagating {len(times)-1} steps...")
        
        for k in range(1, len(times)):
            if k % 100 == 0: print_progress(k, len(times))

            t_curr = times[k]
            t_prev = times[k-1]
            dt = t_curr - t_prev
            meas_row = obs.iloc[k]

            # --- A. Propagate Reference Trajectory and Step STM ---
            state_integ_0 = np.concatenate([X_ref_curr, np.eye(self.n).flatten()])
            
            sol = solve_ivp(
                ode_func, 
                (t_prev, t_curr), 
                state_integ_0, 
                t_eval=[t_curr],  
                args=(coeffs,), 
                rtol=abs_tol, 
                atol=rel_tol
            )
            
            X_ref_curr = sol.y[0:self.n, 0]
            Phi_step = sol.y[self.n:, 0].reshape(self.n, self.n)

            # --- B. Time Update (Prediction) ---
            x_dev_pred = Phi_step @ _x[-1]
            Q_k = compute_q_discrete(dt, X_ref_curr, options)
            P_pred = Phi_step @ _P[-1] @ Phi_step.T + Q_k

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
                
            # Compute Innovation and NIS (Common to both methods)
            innovation = prefit_res - (H @ x_dev_pred)
            S_innov = H @ P_pred @ H.T + Rk
            try:
                nis = innovation.T @ np.linalg.solve(S_innov, innovation)
            except np.linalg.LinAlgError:
                nis = 0.0

            # Toggle update method based on options
            if use_potter:
                # --- C.1 POTTER MEASUREMENT UPDATE (Sequential Scalar Updates) ---
                x_dev_curr = x_dev_pred.copy()
                P_curr = P_pred.copy()
                
                for i in range(2):
                    # 1. Compute scalar innovation for THIS specific row based on current x
                    inn_scalar = prefit_res[i] - (H[i, :] @ x_dev_curr)
                    
                    # 2. Extract variances and H
                    R_scalar = Rk[i, i]
                    H_row = H[i, :]
                    
                    # 3. Perform Potter Update
                    x_dev_curr, P_curr = self._potter_update(x_dev_curr, P_curr, inn_scalar, H_row, R_scalar)

                x_dev = x_dev_curr
                P = P_curr
            else:
                # --- C.2 STANDARD MEASUREMENT UPDATE (Joseph Form) ---
                K = P_pred @ H.T @ np.linalg.inv(S_innov)
                
                # State Update
                x_dev = x_dev_pred + K @ innovation
                
                # Covariance Update (Joseph Form)
                IKH = self.I - K @ H
                P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T
            
            # Postfit Residuals
            postfit_res = prefit_res - H @ x_dev
            X_total_est = X_ref_curr + x_dev

            # --- D. Storage ---
            _x.append(x_dev.copy())
            _P.append(P.copy())
            _state.append(X_total_est.copy())
            _prefit_res.append(prefit_res.copy())
            _postfit_res.append(postfit_res.copy())
            _nis.append(nis.copy())
            _Phi_step.append(Phi_step.copy())
            _P_pred.append(P_pred.copy())
            _X_ref.append(X_ref_curr.copy())

        return LKFResults(_x, _P, _state, _prefit_res, _postfit_res, _nis, _P_pred, _Phi_step, _X_ref, times)