import numpy as np
import copy
from dataclasses import dataclass
from utils.filters.lkf_class import LKFResults

@dataclass
class SmootherResults:
    t_smooth: np.ndarray
    dx_smooth: np.ndarray
    P_smooth: np.ndarray
    state_smooth: np.ndarray

class RTSSmoother:
    def __init__(self, n_states: int = 6, has_process_noise: bool = True):
        self.n = n_states
        self.has_process_noise = has_process_noise

    def run(self, lkf_out: LKFResults) -> SmootherResults:
        print(f"Running RTS Smoother (n={self.n}, Process Noise={self.has_process_noise})...")
        
        # Extract forward filter data
        times = lkf_out.times
        x_filt = lkf_out.dx_hist          # x_{k|k}
        P_filt = lkf_out.P_hist           # P_{k|k}
        P_pred = lkf_out.P_pred_hist      # P_{k|k-1}
        Phi_step = lkf_out.Phi_step_hist  # Phi_{k+1, k}
        X_ref = lkf_out.X_ref_hist        # X*_{k}
        
        n_steps = len(x_filt)
        
        # Initialize smoother arrays
        x_smooth = np.zeros_like(x_filt)
        P_smooth = np.zeros_like(P_filt)
        state_smooth = np.zeros_like(lkf_out.state_hist)
        
        # Initialize smoother with the last filter step
        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]
        state_smooth[-1] = X_ref[-1] + x_smooth[-1]

        # Small numerical stability term for covariance inversion
        epsilon = 1e-12
        eye_n = np.eye(self.n)
        
        # Backward Pass
        for k in range(n_steps - 2, -1, -1):
            x_kk = x_filt[k]
            P_kk = P_filt[k]
            
            # k+1 predicted values from the forward filter
            P_kp1_k = P_pred[k+1] 
            Phi_kp1_k = Phi_step[k+1]
            
            # k+1 smoothed values from the previous smoother loop iteration
            x_kp1_n = x_smooth[k+1]
            P_kp1_n = P_smooth[k+1]

            # Debug assertion to check PD of covariance matrices
            assert np.all(np.linalg.eigvals(P_kk) > 0), f"P_kk at step {k} is not positive definite!"
            assert np.all(np.linalg.eigvals(P_kp1_k) > 0), f"P_kp1_k at step {k+1} is not positive definite!"
            
            if self.has_process_noise:
                # Smoothing gain with SNC
                S_k = P_kk @ Phi_kp1_k.T @ np.linalg.inv(P_kp1_k)
                x_smooth[k] = x_kk + S_k @ (x_kp1_n - Phi_kp1_k @ x_kk)
                P_smooth[k] = P_kk + S_k @ (P_kp1_n - P_kp1_k) @ S_k.T
            else:
                # Smoothing gain with no process noise
                Phi_inv = np.linalg.inv(Phi_kp1_k)
                x_smooth[k] = Phi_inv @ x_kp1_n
                P_smooth[k] = Phi_inv @ P_kp1_n @ Phi_inv.T
            
            # Reconstruct full state
            state_smooth[k] = X_ref[k] + x_smooth[k]
            
        return SmootherResults(times, x_smooth, P_smooth, state_smooth)
    
    def to_lkf_format(self, smoother_out: SmootherResults) -> LKFResults:
    # Convert SmootherResults back to LKFResults format for consistency
        return LKFResults(
            times=smoother_out.t_smooth,
            dx_hist=smoother_out.dx_smooth,
            P_hist=smoother_out.P_smooth,
            state_hist=smoother_out.state_smooth,
            postfit_residuals=None,  # Not needed for smoother output,
            nis_hist=None,  # Not needed for smoother output
            innovations=None,  # Not needed for smoother output
            P_pred_hist=None,  # Not needed for smoother output
            Phi_step_hist=None,  # Not needed for smoother output
            X_ref_hist=None  # Not needed for smoother output
        )