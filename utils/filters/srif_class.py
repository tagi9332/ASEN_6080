import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Any

# Local Imports
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from resources.gs_locations_latlon import stations_ll
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6, zonal_sph_ode_dmc


def householder(mat: np.ndarray) -> np.ndarray:
    """Upper-triangularizes the first n columns via reflections."""
    A = np.array(mat, dtype=float, copy=True)
    cols = A.shape[1]
    n = cols - 1 
    
    for k in range(n):
        A_kk = A[k, k]
        A_ik = A[k:, k]
        
        sgn = np.sign(A_kk)
        if sgn == 0: sgn = 1.0
            
        sigma = sgn * np.linalg.norm(A_ik)
        if np.isclose(sigma, 0.0): continue
            
        u_k = A_kk + sigma
        A[k, k] = -sigma
        
        u_i = A[k:, k].copy()
        u_i[0] = u_k
        beta = 1.0 / (sigma * u_k)
        
        for j in range(k + 1, cols):
            A_ij = A[k:, j]
            gamma = beta * np.dot(u_i, A_ij)
            A[k:, j] = A_ij - (gamma * u_i)
            
        A[k+1:, k] = 0.0
        
    return A


@dataclass
class SRIFResults:
    """Structure mirroring LKFResults but with SRIF-specific additions"""
    dx_hist: Any
    P_hist: Any
    state_hist: Any
    innovations: Any
    postfit_residuals: Any
    P_pred_hist: Any
    Phi_step_hist: Any
    X_ref_hist: Any
    nis_hist: Any          
    times: Any
    
    # SRIF-specific variables
    prefit_res_whitened: Any
    postfit_res_whitened: Any
    Phi_total_hist: Any
    
    # Process Noise / Smoother specific variables
    Ru: Any
    Rux: Any
    bTildeu: Any
    uHat: Any

    def __post_init__(self):
        # Auto-convert lists to arrays to match LKF behavior
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.state_hist = np.array(self.state_hist)
        self.innovations = np.array(self.innovations)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.P_pred_hist = np.array(self.P_pred_hist)
        self.Phi_step_hist = np.array(self.Phi_step_hist)
        self.nis_hist = np.array(self.nis_hist)
        self.X_ref_hist = np.array(self.X_ref_hist)
        self.times = np.array(self.times)
        
        self.prefit_res_whitened = np.array(self.prefit_res_whitened)
        self.postfit_res_whitened = np.array(self.postfit_res_whitened)
        self.Phi_total_hist = np.array(self.Phi_total_hist)
        self.Ru = np.array(self.Ru)
        self.Rux = np.array(self.Rux)
        self.bTildeu = np.array(self.bTildeu)
        self.uHat = np.array(self.uHat)


@dataclass
class IterativeSRIFResults:
    srif_results: SRIFResults
    rms_prefit_whitened_hist: np.ndarray
    rms_postfit_whitened_hist: np.ndarray
    rms_prefit_hist: np.ndarray
    rms_postfit_hist: np.ndarray
    converged: bool
    total_runs: int


class SRIF:
    def __init__(self, n_states: int = 6):
        self.n = n_states
        self.I = np.eye(n_states)

    @staticmethod
    def _gamma_func(dt: float) -> np.ndarray:
        return np.vstack([(dt / 2.0) * np.eye(3), np.eye(3)])

    @staticmethod
    def _qr_transform(mat: np.ndarray) -> np.ndarray:
        return householder(mat)

    def run(self, obs, X_0, x_0, P0, Q0, uBar, R_meas, options, force_upper_triangular=True) -> SRIFResults:
        print(f"Running Standardized SRIF with n={self.n} states...")
        
        coeffs = options['coeffs']
        abs_tol = options.get('abs_tol', 1e-12)
        rel_tol = options.get('rel_tol', 1e-12)
        ode_func = zonal_sph_ode_dmc if options.get('method') == 'DMC' else zonal_sph_ode_6x6

        times = obs['Time(s)'].values
        n = self.n
        q = len(uBar)
        has_process_noise = np.any(Q0 > 0)

        # ---------------------------------------------------------
        # Initialization & t0 Padding
        # ---------------------------------------------------------
        X_ref_prev = X_0.copy()
        x_dev_prev = x_0.copy()
        
        _dx_hist = [x_dev_prev.copy()]
        _P_hist = [P0.copy()]
        _state_hist = [(X_ref_prev + x_dev_prev).copy()]
        _innovations = [np.zeros(2)] 
        _postfit_residuals = [np.zeros(2)]
        _prefit_res_whitened = [np.zeros(2)]
        _postfit_res_whitened = [np.zeros(2)]
        _P_pred_hist = [P0.copy()]
        _Phi_step_hist = [np.eye(n)]
        _Phi_total_hist = [np.eye(n)]
        _nis_hist = [0.0]
        _X_ref_hist = [X_ref_prev.copy()]

        _Ru = [np.zeros((q, q))]
        _Rux = [np.zeros((q, n))]
        _bTildeu = [np.zeros(q)]
        _uHat = [np.zeros(q)]

        # SRIF Initial Data Arrays
        P0_inv = np.linalg.inv(P0)
        R_im1 = np.linalg.cholesky(P0_inv).T 
        b_im1 = R_im1 @ x_dev_prev

        if has_process_noise:
            Q0_inv = np.linalg.inv(Q0)
            Ru_im1 = np.linalg.cholesky(Q0_inv).T
            bu_im1 = Ru_im1 @ uBar
        else:
            bu_im1 = None

        Phi_full = np.eye(n)
        t_prev = times[0]

        # ---------------------------------------------------------
        # Pre-Whiten Observations
        # ---------------------------------------------------------
        V_meas = np.linalg.cholesky(R_meas)
        y_obs_whitened = []
        for i in range(len(times)):
            meas_row = obs.iloc[i]
            y_raw = np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']])
            y_w = la.solve_triangular(V_meas, y_raw, lower=True)
            y_obs_whitened.append(y_w)

        # ---------------------------------------------------------
        # Main SRIF Loop
        # ---------------------------------------------------------
        for k in range(1, len(times)):
            t_curr = times[k]
            dt = t_curr - t_prev
            meas_row = obs.iloc[k]

            # Propagate Full STM
            state_integ_0 = np.concatenate([X_ref_prev, Phi_full.flatten()])
            sol_full = solve_ivp(
                ode_func, (t_prev, t_curr), state_integ_0, 
                t_eval=[t_curr], args=(coeffs,), rtol=rel_tol, atol=abs_tol
            )
            Phi_full = sol_full.y[n:, 0].reshape(n, n)
            _Phi_total_hist.append(Phi_full.copy())

            # Propagate Reference Trajectory and Step STM
            state_step_0 = np.concatenate([X_ref_prev, np.eye(n).flatten()])
            sol_step = solve_ivp(
                ode_func, (t_prev, t_curr), state_step_0, 
                t_eval=[t_curr], args=(coeffs,), rtol=rel_tol, atol=abs_tol
            )
            X_ref_curr = sol_step.y[0:n, 0]
            Phi_step = sol_step.y[n:, 0].reshape(n, n)
            _Phi_step_hist.append(Phi_step.copy())

            # TIME UPDATE
            Phi_inv = np.linalg.inv(Phi_step)
            
            if has_process_noise and dt <= 10.0: 
                Gamma_i = self._gamma_func(dt)
                
                # Process noise
                Q_discrete = Q0 * dt
                Ru_k = np.linalg.cholesky(np.linalg.inv(Q_discrete)).T
                
                Rtilde = R_im1 @ Phi_inv
                
                top_block = np.hstack([Ru_k, np.zeros((q, n)), bu_im1.reshape(-1, 1)])
                bot_block = np.hstack([-Rtilde @ Gamma_i, Rtilde, b_im1.reshape(-1, 1)])
                mat = np.vstack([top_block, bot_block])
                
                qr_out = self._qr_transform(mat)
                
                Ru_k_next = qr_out[0:q, 0:q]
                Rux_k = qr_out[0:q, q:q+n]
                bTildeu_k = qr_out[0:q, q+n]
                R_i = qr_out[q:q+n, q:q+n]
                b_i = qr_out[q:q+n, q+n]
                
                bu_im1 = Ru_k_next @ uBar
                
                _Ru.append(Ru_k_next)
                _Rux.append(Rux_k)
                _bTildeu.append(bTildeu_k)
            else:
                R_i = R_im1 @ Phi_inv
                b_i = b_im1.copy()
                
                if force_upper_triangular:
                    mat = np.hstack([R_i, b_i.reshape(-1, 1)])
                    qr_out = self._qr_transform(mat)
                    R_i = qr_out[:, :n]
                    b_i = qr_out[:, n]
                    
                _Ru.append(np.zeros((q, q)))
                _Rux.append(np.zeros((q, n)))
                _bTildeu.append(np.zeros(q))

            # Reconstruct P_pred
            R_i_inv = np.linalg.inv(R_i)
            P_pred = R_i_inv @ R_i_inv.T
            _P_pred_hist.append(P_pred)

            # Measurement processing
            station_idx = int(meas_row['Station_ID']) - 1
            Rs, Vs = get_gs_eci_state(
                stations_ll[station_idx][0], stations_ll[station_idx][1], t_curr
            )
            
            y_exp_raw = compute_rho_rhodot(X_ref_curr[0:6], np.concatenate([Rs, Vs]))
            y_exp_w = la.solve_triangular(V_meas, y_exp_raw, lower=True)
            
            y_i = y_obs_whitened[k] - y_exp_w
            
            prefit_res_raw = V_meas @ y_i
            
            H_raw = compute_H_matrix(X_ref_curr[0:3], X_ref_curr[3:6], Rs, Vs)
            if n > 6:
                H_raw = np.hstack([H_raw, np.zeros((2, n - 6))])
            Htilde_i = la.solve_triangular(V_meas, H_raw, lower=True)

            # NIS computation
            x_dev_pred = la.solve_triangular(R_i, b_i, lower=False)
            innovation_pred = prefit_res_raw - (H_raw @ x_dev_pred)
            S_innov = H_raw @ P_pred @ H_raw.T + R_meas
            try:
                nis = innovation_pred.T @ la.solve(S_innov, innovation_pred)
            except np.linalg.LinAlgError:
                nis = 0.0

            # Measurement update via QR
            mat = np.vstack([
                np.hstack([R_i, b_i.reshape(-1, 1)]),
                np.hstack([Htilde_i, y_i.reshape(-1, 1)])
            ])
            qr_out = self._qr_transform(mat)
            
            R_i = qr_out[:n, :n]
            b_i = qr_out[:n, n]
            e = qr_out[n:, n] 

            x_dev = la.solve_triangular(R_i, b_i, lower=False)
            
            # Process noise smoother (if enabled)
            if has_process_noise and dt <= 10.0:
                u_im1 = la.solve_triangular(Ru_k_next, (bTildeu_k - Rux_k @ x_dev), lower=False)
                _uHat.append(u_im1)
            else:
                _uHat.append(np.zeros(q))

            # Store results for this step
            R_inv = np.linalg.inv(R_i)
            P_curr = R_inv @ R_inv.T
            
            innovation = V_meas @ y_i
            postfit_res = V_meas @ e

            _P_hist.append(P_curr)
            _dx_hist.append(x_dev)
            _prefit_res_whitened.append(y_i)
            _postfit_res_whitened.append(e)
            _innovations.append(innovation)
            _postfit_residuals.append(postfit_res)
            _nis_hist.append(nis)
            _X_ref_hist.append(X_ref_curr)
            _state_hist.append(X_ref_curr + x_dev)

            # Update variables for next iteration
            t_prev = t_curr
            X_ref_prev = X_ref_curr
            R_im1 = R_i
            b_im1 = b_i

        return SRIFResults(
            dx_hist=_dx_hist, P_hist=_P_hist, state_hist=_state_hist,
            innovations=_innovations, postfit_residuals=_postfit_residuals,
            P_pred_hist=_P_pred_hist, Phi_step_hist=_Phi_step_hist,
            X_ref_hist=_X_ref_hist, times=times,
            prefit_res_whitened=_prefit_res_whitened, postfit_res_whitened=_postfit_res_whitened,
            Phi_total_hist=_Phi_total_hist, Ru=_Ru, Rux=_Rux, bTildeu=_bTildeu, uHat=_uHat, nis_hist=_nis_hist
        )

    def run_iterative(self, obs, X0, x0, P0, Q0, uBar, R_meas, options, max_iter=5, tol=1e-3, force_upper_triangular=True) -> IterativeSRIFResults:
        print(f"\n\tRunning Iterative SRIF (Max Runs: {max_iter}):")
        
        X_ref_srif = X0.copy()
        x_dev_srif = x0.copy()
        
        rms_pre_w_hist, rms_post_w_hist = [], []
        rms_pre_hist, rms_post_hist = [], []
        
        runs = 0
        prev_rms_postfit = 1e99
        converged = False
        
        while runs < max_iter:
            out = self.run(
                obs, X_ref_srif, x_dev_srif, P0, Q0, uBar, R_meas, 
                options, force_upper_triangular
            )
            
            pre_w = np.concatenate(out.prefit_res_whitened)
            post_w = np.concatenate(out.postfit_res_whitened)
            pre = np.concatenate(out.innovations)
            post = np.concatenate(out.postfit_residuals)
            
            rms_pre_w = np.sqrt(np.mean(pre_w**2))
            rms_post_w = np.sqrt(np.mean(post_w**2))
            rms_pre = np.sqrt(np.mean(pre**2))
            rms_post = np.sqrt(np.mean(post**2))
            
            rms_pre_w_hist.append(rms_pre_w)
            rms_post_w_hist.append(rms_post_w)
            rms_pre_hist.append(rms_pre)
            rms_post_hist.append(rms_post)
            
            perc_change = abs((rms_post - prev_rms_postfit) / prev_rms_postfit)
            runs += 1
            
            if perc_change > tol:
                Phi_total_final = out.Phi_total_hist[-1]
                x_dev_final = out.dx_hist[-1]
                
                x_dev_srif = np.linalg.solve(Phi_total_final, x_dev_final)
                X_ref_srif = X_ref_srif + x_dev_srif
                
                prev_rms_postfit = rms_post
                
                if runs < max_iter:
                    print(f"Prefit RMS: {rms_pre:.4f}, Postfit RMS: {rms_post:.4f}. Iterating SRIF. Runs so far: {runs}")
                else:
                    print(f"Prefit RMS: {rms_pre:.4f}, Postfit RMS: {rms_post:.4f}. Hit max iterations.")
            else:
                converged = True
                break
                
        if converged:
            print(f"Final postfit RMS: {rms_post_hist[-1]:.4f}. Converged after {runs} runs")
        else:
            print(f"Final postfit RMS: {rms_post_hist[-1]:.4f}. Hit max {max_iter} runs")

        return IterativeSRIFResults(
            srif_results=out,
            rms_prefit_whitened_hist=np.array(rms_pre_w_hist),
            rms_postfit_whitened_hist=np.array(rms_post_w_hist),
            rms_prefit_hist=np.array(rms_pre_hist),
            rms_postfit_hist=np.array(rms_post_hist),
            converged=converged,
            total_runs=runs
        )