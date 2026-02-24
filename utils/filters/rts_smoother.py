import numpy as np
from dataclasses import dataclass
from utils.filters.lkf_class import LKFResults

@dataclass
class SmootherResults:
    t_smooth: np.ndarray
    dx_smooth: np.ndarray
    P_smooth: np.ndarray
    state_smooth: np.ndarray

class RTSSmoother:
    def __init__(self, n_states: int = 6):
        self.n = n_states

    def run(self, lkf_out: LKFResults) -> SmootherResults:
        print(f"Running RTS Smoother (n={self.n})...")
        
        # Extract forward filter data
        times = lkf_out.times
        x_filt = lkf_out.dx_hist          # x_{k|k}
        P_filt = lkf_out.P_hist           # P_{k|k}
        P_pred = lkf_out.P_pred_hist      # P_{k|k-1}
        Phi_step = lkf_out.Phi_step_hist  # Phi_{k, k-1}
        X_ref = lkf_out.X_ref_hist        # X*_{k}
        
        n_steps = len(x_filt)
        
        # Initialize smoother arrays
        x_smooth = np.zeros_like(x_filt)
        P_smooth = np.zeros_like(P_filt)
        state_smooth = np.zeros_like(lkf_out.state_hist)
        
        # 1. Initialize smoother with the last filter step
        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]
        state_smooth[-1] = X_ref[-1] + x_smooth[-1]
        
        # 2. Backward Pass
        for k in range(n_steps - 2, -1, -1):
            x_kk = x_filt[k]
            P_kk = P_filt[k]
            
            # The "k+1" predicted values from the forward filter
            P_kp1_k = P_pred[k+1] 
            Phi_kp1_k = Phi_step[k+1]
            
            # The "k+1" smoothed values from the previous smoother loop iteration
            x_kp1_n = x_smooth[k+1]
            P_kp1_n = P_smooth[k+1]
            
            # Calculate Smoothing Gain: S_k = P_{k|k} * Phi_{k+1,k}^T * P_{k+1|k}^{-1}
            # Using np.linalg.solve is numerically more stable than taking the inverse
            S_k = P_kk @ Phi_kp1_k.T @ np.linalg.pinv(P_kp1_k)
            
            # Smoothed state deviation
            x_smooth[k] = x_kk + S_k @ (x_kp1_n - Phi_kp1_k @ x_kk)
            
            # Smoothed covariance
            P_smooth[k] = P_kk + S_k @ (P_kp1_n - P_kp1_k) @ S_k.T
            
            # Reconstruct full state
            state_smooth[k] = X_ref[k] + x_smooth[k]
            
        return SmootherResults(times, x_smooth, P_smooth, state_smooth)