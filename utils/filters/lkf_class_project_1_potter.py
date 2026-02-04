import numpy as np
from dataclasses import dataclass
from typing import Any
from scipy.integrate import solve_ivp

# Local Imports
from resources.constants import OMEGA_EARTH
from utils.zonal_harmonics.zonal_harmonics import stm_eom_mu_j2_drag
from utils.ground_station_utils.gs_meas_model_H import compute_H_tilde_18_state, compute_rho_rhodot
from utils.misc.print_progress import print_progress

@dataclass
class LKFResults:
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

# ============================================================
# Linearized Kalman Filter (Potter Square Root Implementation)
# ============================================================
class LKF:
    def __init__(self, n_states: int = 18, station_map: dict = None):
        """
        Initializes the LKF for the 18-element Augmented State.
        State: [r_vec(3), v_vec(3), mu, J2, Cd, GS1(3), GS2(3), GS3(3)]
        """
        self.n = n_states
        self.I = np.eye(n_states)
        
        if station_map is None:
            self.station_map = {101: 0, 337: 1, 394: 2}
        else:
            self.station_map = station_map

    @staticmethod
    def _potter_update(x, P, innovation_scalar, H_row, R_scalar):
        """
        Performs a single scalar Potter Square Root update.
        Inputs:
            x: State vector (n,)
            P: Covariance matrix (n,n)
            innovation_scalar: Scalar residual (z - Hx)
            H_row: Measurement row vector (n,)
            R_scalar: Scalar measurement noise variance
        Returns:
            x_new, P_new
        """
        # 1. Cholesky Factorization P = S * S.T
        # If P is slightly non-positive definite due to noise, Cholesky will fail.
        # We can try to repair it or just let it fail (Potter prevents this accumulation long term).
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # Fallback: Force symmetry and slight positivity if nearly singular
            P_sym = (P + P.T) / 2
            # Add tiny epsilon to diagonal if needed
            P_sym += np.eye(len(P)) * 1e-18
            S = np.linalg.cholesky(P_sym)

        # 2. Calculate F = S.T * H.T
        F = S.T @ H_row.T
        
        # 3. Innovation Variance: alpha = 1 / (F.T*F + R)
        inv_var = (F.T @ F) + R_scalar
        alpha = 1.0 / inv_var
        
        # 4. Gamma Factor
        # gamma = 1 / (1 + sqrt(R * alpha))
        gamma = 1.0 / (1.0 + np.sqrt(R_scalar * alpha))
        
        # 5. Kalman Gain: K = alpha * (S * F)
        K = alpha * (S @ F)
        
        # 6. Update Square Root Matrix S
        # S_new = S - (alpha * gamma) * (S @ F) @ F.T
        # We perform the outer product update
        SF = S @ F
        S_new = S - (alpha * gamma) * np.outer(SF, F)
        
        # 7. Reconstruct P = S_new * S_new.T
        P_new = S_new @ S_new.T
        
        # 8. Update State
        x_new = x + K * innovation_scalar
        
        return x_new, P_new

    def run(self, obs, X_0, x_0, P0, Rk, Q, options) -> LKFResults:
        print("Initializing LKF (Potter Square Root Formulation)...")
        
        abs_tol = options['abs_tol']
        rel_tol = options['rel_tol']

        time_eval = obs['Time(s)'].values

        # Pre-calculate flattened identity matrix for STM
        phi_0_flat = np.eye(self.n).flatten()
        augmented_X0 = np.concatenate([X_0, phi_0_flat])
        
        print("Solving reference trajectory...")
        # Note: Corrected tolerance order (rtol=rel_tol, atol=abs_tol)
        sol_ref = solve_ivp(
                stm_eom_mu_j2_drag, 
                (0, time_eval[-1]), 
                augmented_X0, 
                t_eval=time_eval, 
                rtol=rel_tol, 
                atol=abs_tol
        )

        # Initialization
        x = x_0.copy()
        P = P0.copy()
        Phi_prev = np.eye(self.n)

        # Storage
        _x = [x_0.copy()]
        _P = [P0.copy()]
        _state = [X_0.copy()]
        _phi_hist = [np.eye(self.n)]
        _prefit_res = [np.zeros(2)]
        _postfit_res = [np.zeros(2)]
        _nis = [0.0]

        w_vec = np.array([0, 0, OMEGA_EARTH])

        for k in range(1, len(sol_ref.t)):
            print_progress(k, len(sol_ref.t))
            
            t_curr = sol_ref.t[k]

            # --- A. Propagation ---
            X_ref_k = sol_ref.y[0:self.n, k]
            Phi_global = sol_ref.y[self.n:, k].reshape(self.n, self.n)
            
            Phi_incr = Phi_global @ np.linalg.inv(Phi_prev)
            dt = (t_curr - sol_ref.t[k-1]) if k > 0 else 0
            
            x_pred = Phi_incr @ x
            P_pred = Phi_incr @ P @ Phi_incr.T + (Q * dt)
            
            # --- B. Measurement Prep ---
            meas_row = obs.iloc[k]
            raw_station_id = int(meas_row['Station_ID'])
            station_idx = self.station_map[raw_station_id]
            
            # Reference Geometry
            r_sc_ref = X_ref_k[0:3]
            v_sc_ref = X_ref_k[3:6]
            sc_state_full = np.concatenate([r_sc_ref, v_sc_ref])
            
            st_idx_start = 9 + (3 * station_idx)
            r_gs_0 = X_0[st_idx_start : st_idx_start + 3]

            theta = OMEGA_EARTH * t_curr
            c, s = np.cos(theta), np.sin(theta)
            R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            r_gs_ref = R_z @ r_gs_0
            v_gs_ref = np.cross(w_vec, r_gs_ref)
            gs_state_full = np.concatenate([r_gs_ref, v_gs_ref])

            # Prediction
            y_pred_ref = compute_rho_rhodot(sc_state_full, gs_state_full)
            y_obs = np.array([meas_row['Range(m)'], meas_row['Range_Rate(m/s)']])
            
            # H Matrix
            X_ref_k_analytical = X_ref_k.copy()
            X_ref_k_analytical[st_idx_start : st_idx_start+3] = r_gs_ref
            H = compute_H_tilde_18_state(X_ref_k_analytical, station_idx)
            
            # Prefit Residual (Linearized Observation)
            prefit_res_vec = y_obs - y_pred_ref

            # --- C. NIS Calculation (Standard Matrix form for logging) ---
            # We calculate NIS *before* Potter updates for consistency with standard logs
            S_innov = H @ P_pred @ H.T + Rk
            # innovation for NIS is (z - h(ref) - H*x_pred)
            total_innovation = prefit_res_vec - (H @ x_pred)
            try:
                nis = total_innovation.T @ np.linalg.solve(S_innov, total_innovation)
            except np.linalg.LinAlgError:
                nis = 0.0

            # --- D. POTTER MEASUREMENT UPDATE (Sequential Scalar Updates) ---
            x_curr = x_pred.copy()
            P_curr = P_pred.copy()
            
            # Loop through rows (0: Range, 1: Range-Rate)
            for i in range(2):
                # 1. Compute scalar innovation for THIS specific row based on current x
                # Innovation = (y_obs - y_pred_ref)[i] - H_row @ x_deviation
                inn_scalar = prefit_res_vec[i] - (H[i, :] @ x_curr)
                
                # 2. Extract variances
                R_scalar = Rk[i, i]
                H_row = H[i, :]
                
                # 3. Perform Potter Update
                x_curr, P_curr = self._potter_update(x_curr, P_curr, inn_scalar, H_row, R_scalar)

            x = x_curr
            P = P_curr

            # Postfit Residuals (Approximate linear)
            postfit_res = prefit_res_vec - (H @ x)

            # --- E. Store Data ---
            _x.append(x.copy())
            _P.append(P.copy())
            _state.append((X_ref_k_analytical + x).copy())
            _phi_hist.append(Phi_global.copy())
            _prefit_res.append(prefit_res_vec.copy())
            _postfit_res.append(postfit_res.copy())
            _nis.append(nis)

            Phi_prev = Phi_global.copy()

        return LKFResults(_x, _P, _state, _phi_hist, _prefit_res, _postfit_res, _nis)