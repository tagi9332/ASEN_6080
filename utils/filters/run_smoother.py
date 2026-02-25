import numpy as np
import copy
from utils.filters.rts_smoother import RTSSmoother

class SmootherLKFEquivalent:
    pass

def run_smoother(lkf_out, X_truth_interp, process_noise: bool = True):
    """
    Runs the RTS smoother, calculates RMS metrics, and formats the 
    output to perfectly match the LKFResults structure for post-processing.
    """
    # Run the smoother, passing the process noise flag
    smoother = RTSSmoother(n_states=lkf_out.dx_hist.shape[1], has_process_noise=process_noise)
    smooth_out = smoother.run(lkf_out)
    
    # Calculate State Errors
    state_error_smooth = smooth_out.state_smooth[:, 0:6] - X_truth_interp[:, 0:6]
    
    # Calculate RMS (Root Mean Square) Error
    rms_comp = np.sqrt(np.mean(state_error_smooth**2, axis=0))
    rms_full = np.sqrt(np.mean(np.sum(state_error_smooth**2, axis=1)))
    
    print("\nSmoother Performance:")
    print(f"  Pos RMS (X, Y, Z): {rms_comp[0:3]} km")
    print(f"  Vel RMS (Xdot, Ydot, Zdot): {rms_comp[3:6]} km/s")
    print(f"  Full State 3D RMS: {rms_full:.6f}")
    
    # Package into LKF-compatible format using a shallow copy
    smooth_lkf_match = copy.copy(lkf_out)
    
    # Overwrite the forward filter states with the smoothed states
    smooth_lkf_match.times = smooth_out.t_smooth
    smooth_lkf_match.state_hist = smooth_out.state_smooth
    smooth_lkf_match.dx_hist = smooth_out.dx_smooth
    smooth_lkf_match.P_hist = smooth_out.P_smooth
        
    return smooth_lkf_match, state_error_smooth, rms_comp, rms_full