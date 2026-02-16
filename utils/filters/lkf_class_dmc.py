import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from dataclasses import dataclass
from typing import Any

# --- LOCAL IMPORTS ---
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_dmc, get_zonal_jacobian_dmc
from resources.gs_locations_latlon import stations_ll

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

# --- HELPER FUNCTIONS ---

def compute_van_loan(A_mat, G_mat, Q_cont, dt):
    """
    Computes both the Discrete Process Noise (Q_k) AND the 
    Discrete State Transition Matrix (Phi_k) for the step dt.
    """
    n = A_mat.shape[0]
    
    # Construct the Van Loan Matrix M
    # [ -A    G*Q*G.T ]
    # [  0     A.T    ]
    zeros_n = np.zeros((n, n))
    GQGt = G_mat @ Q_cont @ G_mat.T
    
    top = np.hstack((-A_mat, GQGt))
    bot = np.hstack((zeros_n, A_mat.T))
    M = np.vstack((top, bot)) * dt
    
    # Matrix Exponential
    M_exp = expm(M)
    
    # 1. Extract Phi_k (The Incremental STM)
    Phi_T_block = M_exp[n:, n:]
    Phi_k = Phi_T_block.T  
    
    # 2. Extract Q_k
    Phi_inv_Qk = M_exp[0:n, n:]
    Q_k = Phi_k @ Phi_inv_Qk
    
    return Phi_k, Q_k


# --- MAIN CLASS ---
class LKF_DMC:
    def __init__(self, n_states: int = 9):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, obs, X_0, x_0, P0, Rk, Q_PSD, options) -> LKFResults:
        print("Initializing LKF with DMC...")
        
        coeffs = options['coeffs']
        abs_tol = options['abs_tol']
        rel_tol = options['rel_tol']
        
        # FIX: Define a maximum step size for propagation (e.g., 60 seconds)
        # This prevents linearization errors over large gaps
        dt_max = options.get('dt_max', 60.0) 
        
        # Mapping matrix G (9x3)
        G = np.zeros((9, 3))
        G[6:9, :] = np.eye(3)

        time_eval = obs['Time(s)'].values

        # 1. Integrate Reference Trajectory (Dense Output)
        # We perform one global integration but enable 'dense_output'
        # so we can query the state at any sub-step time.
        sol_ref = solve_ivp(
                zonal_sph_ode_dmc, 
                (0, time_eval[-1]), 
                X_0, 
                t_eval=None,      # Allow solver to choose its own steps for accuracy
                dense_output=True, # REQUIRED for sub-stepping interpolation
                args=(coeffs,),
                rtol=abs_tol, 
                atol=rel_tol
        )
        
        # Create a function to access reference state at any time t
        ref_traj = sol_ref.sol

        x = x_0.copy()
        P = P0.copy()
        
        _x, _P, _state, _prefit_res, _postfit_res, _nis = [], [], [], [], [], []

        # Iterate through measurement times
        for k in range(1, len(time_eval)):
            t_prev = time_eval[k-1]
            t_curr = time_eval[k]
            
            # --- SUB-STEPPING PROPAGATION ---
            # Instead of one giant jump, we loop from t_prev to t_curr
            # in small increments (dt_max).
            
            t_internal = t_prev
            
            while t_internal < t_curr:
                # Determine step size h (don't overshoot t_curr)
                h = min(dt_max, t_curr - t_internal)
                
                # 1. Get Reference State at start of sub-step
                ref_state_local = ref_traj(t_internal)[0:9]
                r_vec_local = ref_state_local[0:3]
                v_vec_local = ref_state_local[3:6]
                
                # 2. Compute Jacobian A(t) at this specific location
                A_mat = get_zonal_jacobian_dmc(r_vec_local, v_vec_local, coeffs)
                
                # 3. Compute Phi and Q for this small step h
                Phi_step, Q_step = compute_van_loan(A_mat, G, Q_PSD, h)
                
                # 4. Propagate Error State and Covariance
                x = Phi_step @ x
                P = Phi_step @ P @ Phi_step.T + Q_step
                
                # Advance internal time
                t_internal += h
            
            # --- MEASUREMENT UPDATE (At t_curr) ---
            
            # Get Reference State exactly at measurement time
            ref_state_k = ref_traj(t_curr)[0:9]
            r_vec = ref_state_k[0:3]
            v_vec = ref_state_k[3:6]
            
            meas_row = obs.iloc[k]
            station_idx = int(meas_row['Station_ID']) - 1
            Rs, Vs = get_gs_eci_state(
                stations_ll[station_idx][0], 
                stations_ll[station_idx][1], 
                t_curr, 
                init_theta=np.deg2rad(122)
            )
            
            # H Matrix
            H_spatial = compute_H_matrix(r_vec, v_vec, Rs, Vs)
            H = np.hstack([H_spatial, np.zeros((2, 3))])
            
            y_pred_ref = compute_rho_rhodot(ref_state_k[0:6], np.concatenate([Rs, Vs]))
            y_obs = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])
            
            prefit_res = y_obs - y_pred_ref
            innovation = prefit_res - H @ x

            S = H @ P @ H.T + Rk
            K = (np.linalg.solve(S, H @ P)).T
            
            x = x + K @ innovation
            
            # Joseph Form Update
            IKH = self.I - K @ H
            P = IKH @ P @ IKH.T + K @ Rk @ K.T

            postfit_res = prefit_res - H @ x
            nis = innovation.T @ np.linalg.solve(S, innovation)

            _x.append(x.copy())
            _P.append(P.copy())
            _state.append((ref_state_k + x).copy())
            _prefit_res.append(prefit_res.copy())
            _postfit_res.append(postfit_res.copy())
            _nis.append(nis.copy())

        return LKFResults(_x, _P, _state, _prefit_res, _postfit_res, _nis)