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
# Linearized Kalman Filter
# ============================================================
class LKF:
    def __init__(self, n_states: int = 18, station_map: dict = None): # type: ignore
        """
        Initializes the LKF for the 18-element Augmented State.
        State: [r_vec(3), v_vec(3), mu, J2, Cd, GS1(3), GS2(3), GS3(3)]
        
        station_map: Dictionary mapping CSV 'Station_ID' to State Index (0, 1, 2)
        """
        self.n = n_states
        self.I = np.eye(n_states)
        
        # Default map if none provided
        if station_map is None:
            self.station_map = {101: 0, 337: 1, 394: 2}
        else:
            self.station_map = station_map

    def run(self, obs, X_0, x_0, P0, Rk, Q, options) -> LKFResults:
        """
        Runs the Linearized Kalman Filter.
        """

        print("Initializing LKF...")
        
        # Initialize filter arguments
        abs_tol = options['abs_tol']
        rel_tol = options['rel_tol']

        # Time vector from measurements
        time_eval = obs['Time(s)'].values

        # Pre-calculate flattened identity matrix for STM
        phi_0_flat = np.eye(self.n).flatten()
        augmented_X0 = np.concatenate([X_0, phi_0_flat])
        
        print("Solving reference trajectory...")
        # Note: We still integrate the full 18 states so the STM is consistent,
        # but we will override the Station positions with the analytical solution below.
        sol_ref = solve_ivp(
                stm_eom_mu_j2_drag, 
                (0, time_eval[-1]), 
                augmented_X0, 
                t_eval=time_eval, 
                rtol=abs_tol, 
                atol=rel_tol
        )

        # Initialization
        x = x_0.copy()
        P = P0.copy()
        Phi_prev = np.eye(self.n)

        # Initialize lists for collection
        _x = [x_0.copy()]
        _P = [P0.copy()]
        _state = [X_0.copy()]
        _phi_hist = [np.eye(self.n)]
        _prefit_res = [np.zeros(2)]
        _postfit_res = [np.zeros(2)]
        _nis = [0.0]

        # Define Earth Rotation Vector for Cross Product
        w_vec = np.array([0, 0, OMEGA_EARTH])

        for k in range(1,len(sol_ref.t)):
            print_progress(k, len(sol_ref.t))
            
            # Current time
            t_curr = sol_ref.t[k]

            # A. Extract Dynamics from Integrator Result
            # Reference State (18 elements)
            X_ref_k = sol_ref.y[0:self.n, k]
            
            # STM (reshaped to 18x18)
            Phi_global = sol_ref.y[self.n:, k].reshape(self.n, self.n)
            
            # Calculate Incremental STM
            Phi_incr = Phi_global @ np.linalg.inv(Phi_prev)
            
            dt = (t_curr - sol_ref.t[k-1]) if k > 0 else 0
            
            # B. Time Update (Prediction)
            x_pred = Phi_incr @ x
            P_pred = Phi_incr @ P @ Phi_incr.T + (Q * dt)
            
            # C. Measurement Update setup
            meas_row = obs.iloc[k]
            
            raw_station_id = int(meas_row['Station_ID'])
            try:
                station_idx = self.station_map[raw_station_id]
            except KeyError:
                raise ValueError(f"Station ID {raw_station_id} not found in map.")
            
            # --- Reference States for Prediction ---
            
            # 1. Spacecraft Reference (Inertial) -> From Integrator
            r_sc_ref = X_ref_k[0:3]
            v_sc_ref = X_ref_k[3:6]
            sc_state_full = np.concatenate([r_sc_ref, v_sc_ref])
            
            # Get index of this station in the INITIAL state vector X_0
            st_idx_start = 9 + (3 * station_idx)
            r_gs_0 = X_0[st_idx_start : st_idx_start + 3] # Initial ECI position at t=0

            # Compute Rotation Angle theta (radians)
            theta = OMEGA_EARTH * t_curr

            # Rotation Matrix (Z-axis rotation)
            c, s = np.cos(theta), np.sin(theta)
            R_z = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])

            # Rotate Initial Position to Current Time
            r_gs_ref = R_z @ r_gs_0
            
            # Calculate Station Velocity (Inertial) = w x r
            v_gs_ref = np.cross(w_vec, r_gs_ref)
            
            gs_state_full = np.concatenate([r_gs_ref, v_gs_ref])

            # D. Predict Measurement (Non-linear h(x_ref))
            y_pred_ref = compute_rho_rhodot(sc_state_full, gs_state_full)
            y_obs = np.array([meas_row['Range(m)'], meas_row['Range_Rate(m/s)']])

            # E. Compute H Matrix (18x2)
            # Note: We pass the ANALYTICAL station pos into X_ref_k temporary copy
            # to ensure H is computed at the correct location.
            X_ref_k_analytical = X_ref_k.copy()
            X_ref_k_analytical[st_idx_start : st_idx_start+3] = r_gs_ref
            
            H = compute_H_tilde_18_state(X_ref_k_analytical, station_idx)
            
            # F. Kalman Gain & Update
            prefit_res = y_obs - y_pred_ref
            
            innovation = prefit_res - (H @ x_pred)

            S = H @ P_pred @ H.T + Rk
            K = P_pred @ H.T @ np.linalg.inv(S)

            # NIS
            nis = innovation.T @ np.linalg.solve(S, innovation)
            
            # State Measurement Update
            x = x_pred + K @ innovation

            # Covariance MeasurementUpdate
            IKH = self.I - K @ H
            P = IKH @ P_pred @ IKH.T + K @ Rk @ K.T

            # Postfit Residuals
            postfit_res = prefit_res - (H @ x)
            
            # --- 4. Store Data ---
            _x.append(x.copy())
            _P.append(P.copy())
            # Store the Analytical ref + estimated deviation
            _state.append((X_ref_k_analytical + x).copy())
            _phi_hist.append(Phi_global.copy())
            _prefit_res.append(prefit_res.copy()) # type: ignore
            _postfit_res.append(postfit_res.copy())
            _nis.append(nis.copy())

            # Update Previous STM
            Phi_prev = Phi_global.copy()

        return LKFResults(_x, _P, _state, _phi_hist, _prefit_res, _postfit_res, _nis)