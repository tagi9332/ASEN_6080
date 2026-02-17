import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from dataclasses import dataclass
from typing import Any, Optional

# Local Imports
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from resources.gs_locations_latlon import stations_ll
from utils.misc.print_progress import print_progress
from resources.constants import R_EARTH 

@dataclass
class FilterResults:
    dx_hist: Any             
    P_hist: Any              
    state_hist: Any          
    innovations: Any         
    postfit_residuals: Any   
    nis_hist: Any
    accel_hist: Optional[Any] = None

    def __post_init__(self):
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.state_hist = np.array(self.state_hist)
        self.innovations = np.array(self.innovations)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.nis_hist = np.array(self.nis_hist)
        
        if self.accel_hist is not None:
            self.accel_hist = np.array(self.accel_hist)

# --- HELPER FUNCTIONS ---

def compute_van_loan(A_mat, G_mat, Q_cont, dt):
    """Computes Discrete Q_k and STM Phi_k via Van Loan."""
    n = A_mat.shape[0]
    zeros_n = np.zeros((n, n))
    GQGt = G_mat @ Q_cont @ G_mat.T
    
    top = np.hstack((-A_mat, GQGt))
    bot = np.hstack((zeros_n, A_mat.T))
    M = np.vstack((top, bot)) * dt
    
    # Pad computation for stability
    M_exp = expm(M)
    
    Phi_T_block = M_exp[n:, n:]
    Phi_k = Phi_T_block.T  
    
    Phi_inv_Qk = M_exp[0:n, n:]
    Q_k = Phi_k @ Phi_inv_Qk
    
    return Phi_k, Q_k

def dmc_ode_wrapper(t, y, coeffs):
    """
    Computes derivative for EKF-DMC.
    Handles 9-element (State) or 90-element (State + STM) inputs.
    Expects coeffs = (mu, J2, unused, B_mat)
    """
    r = y[0:3]
    v = y[3:6]
    a = y[6:9] 
    
    mu, J2, _, B_mat = coeffs 

    norm_r = np.linalg.norm(r)
    r2 = norm_r**2
    
    # Acceleration due to Gravity
    a_grav = -(mu / norm_r**3) * r
    
    # Acceleration due to J2
    z2 = r[2]**2
    factor_J2 = 1.5 * J2 * mu * (R_EARTH**2 / r2**2)
    tx = (r[0] / norm_r) * (5 * z2 / r2 - 1)
    ty = (r[1] / norm_r) * (5 * z2 / r2 - 1)
    tz = (r[2] / norm_r) * (5 * z2 / r2 - 3)
    a_J2 = factor_J2 * np.array([tx, ty, tz])

    # State Derivatives
    dr = v
    dv = a_grav + a_J2 + a 
    da = -B_mat @ a         

    dy_state = np.concatenate([dr, dv, da])

    # Handle STM Propagation if needed
    if len(y) == 90:
        Phi = y[9:].reshape((9, 9))
        
        A = np.zeros((9, 9))
        A[0:3, 3:6] = np.eye(3) 
        A[3:6, 6:9] = np.eye(3) 
        
        R3 = norm_r**3
        R5 = norm_r**5
        G = (3 * mu / R5) * np.outer(r, r) - (mu / R3) * np.eye(3)
        A[3:6, 0:3] = G 
        A[6:9, 6:9] = -B_mat

        dPhi = A @ Phi
        return np.concatenate([dy_state, dPhi.flatten()])

    return dy_state

class EKF:
    def __init__(self, n_states: int = 9):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, obs, X_0, x_0, P0, Rk, Q_PSD, options):

        print("Initializing EKF with DMC...")
        
        # --- Initialization ---
        coeffs = options['coeffs']
        abs_tol = options['abs_tol']
        rel_tol = options['rel_tol']
        dt_max = options.get('dt_max', 60.0) 
        
        # --- BOOTSTRAP CONFIGURATION ---
        # Default to 0 if not provided (Standard EKF)
        bootstrap_steps = options.get('bootstrap_steps', 0)
        
        mu, _, _, B_mat = coeffs

        # 1. Setup State Vector
        X_curr = X_0.copy()
        if len(X_curr) == 6:
            X_curr = np.concatenate([X_curr, np.zeros(3)])

        # 2. Setup Deviation Vector
        x_dev = np.zeros(self.n)
        if len(x_0) == self.n:
            x_dev = x_0.copy()

        # 3. Setup Covariance
        P = P0.copy()

        B_noise = np.zeros((9, 3))
        B_noise[6:9, :] = np.eye(3)

        _P, _state, _x = [], [], []
        _accel = [] 
        _prefit_res, _postfit_res = [], []
        _nis_hist = []
        
        t_prev = 0 
        
        # --- Measurement Loop ---
        for k in range(1, len(obs)):
            meas_row = obs.iloc[k]
            t_curr = meas_row['Time(s)']
            
            print_progress(k, len(obs))

            # -------------------------------------------------------
            # 1. PROPAGATION
            # -------------------------------------------------------
            t_internal = t_prev
            
            while t_internal < t_curr:
                h = min(dt_max, t_curr - t_internal)
                
                # A. Propagate Reference
                sol_step = solve_ivp(
                    dmc_ode_wrapper, 
                    (t_internal, t_internal + h), 
                    X_curr,
                    args=(coeffs,),
                    rtol=rel_tol, atol=abs_tol
                )
                X_curr = sol_step.y[:, -1] 

                # B. Linearize
                r_vec = X_curr[0:3]
                norm_r = np.linalg.norm(r_vec)
                R3 = norm_r**3
                R5 = norm_r**5
                G_grav = (3 * mu / R5) * np.outer(r_vec, r_vec) - (mu / R3) * np.eye(3)
                
                A_mat = np.zeros((9, 9))
                A_mat[0:3, 3:6] = np.eye(3)
                A_mat[3:6, 6:9] = np.eye(3)
                A_mat[3:6, 0:3] = G_grav
                A_mat[6:9, 6:9] = -B_mat
                
                Phi_step, Q_step = compute_van_loan(A_mat, B_noise, Q_PSD, h)
                
                # C. Propagate Deviation & Covariance
                x_dev = Phi_step @ x_dev
                P = Phi_step @ P @ Phi_step.T + Q_step
                
                t_internal += h
            
            # -------------------------------------------------------
            # 2. MEASUREMENT UPDATE
            # -------------------------------------------------------
            station_idx = int(meas_row['Station_ID']) - 1
            lat, lon = stations_ll[station_idx]
            Rs, Vs = get_gs_eci_state(lat, lon, t_curr, init_theta=np.deg2rad(122))
            
            y_pred_ref = compute_rho_rhodot(X_curr[0:6], np.concatenate([Rs, Vs]))
            y_obs = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])
            
            prefit_res_ref = y_obs - y_pred_ref
            
            H_spatial = compute_H_matrix(X_curr[0:3], X_curr[3:6], Rs, Vs)
            H = np.hstack([H_spatial, np.zeros((2, 3))])
            
            innovation = prefit_res_ref - H @ x_dev
            
            S = H @ P @ H.T + Rk
            K = (np.linalg.solve(S, H @ P)).T
            
            dx = K @ innovation
            x_dev = x_dev + dx
            
            IKH = self.I - K @ H
            P = IKH @ P @ IKH.T + K @ Rk @ K.T
            
            nis = innovation.T @ np.linalg.solve(S, innovation)
            postfit_res = prefit_res_ref - H @ x_dev

            # -------------------------------------------------------
            # 3. RECTIFICATION (Bootstrap Logic)
            # -------------------------------------------------------
            # If we are past the bootstrap phase, we rectify (move dev to ref).
            # If k < bootstrap_steps, we skip this block, behaving like an LKF.
            if k >= bootstrap_steps:
                X_curr[0:9] = X_curr[0:9] + x_dev
                x_dev = np.zeros(self.n)

            # -------------------------------------------------------
            # 4. LOGGING
            # -------------------------------------------------------
            X_total = X_curr.copy()
            if len(X_total) > 9: X_total = X_total[0:9]
            
            # Total State = Ref + Deviation
            X_total = X_total + x_dev
            
            _x.append(x_dev.copy())
            
            # SAVE ONLY 6 STATES TO MATCH TRUTH
            _state.append(X_total[0:6].copy())
            # SAVE ACCEL SEPARATELY
            _accel.append(X_total[6:9].copy())
            
            _P.append(P.copy())
            _prefit_res.append(prefit_res_ref.copy())
            _postfit_res.append(postfit_res.copy())
            _nis_hist.append(nis)
            
            t_prev = t_curr

        return FilterResults(_x, _P, _state, _prefit_res, _postfit_res, _nis_hist, accel_hist=_accel)