import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import numpy as np

def batch_filter(trajectory_ref, sorted_measurements, P0):
    """
    Python version of the Batch Least Squares Filter.
    
    Args:
        trajectory_ref: List of dicts containing 'time', 'state', 'STM'
        sorted_measurements: List of dicts containing 'time', 'residual', 'partials', 'covariance'
        P0: Initial covariance matrix (n x n)
    """
    # Initialization
    T = len(sorted_measurements)
    n = P0.shape[0]
    
    # Calculate max residual length for preallocation
    max_m = max(len(m['residual']) for m in sorted_measurements)

    # Initialize information matrix (H^T R⁻¹ H) and normal vector (H^T R^-1 r)
    # Using np.linalg.inv for the initial prior information
    info_matrix = np.linalg.inv(P0)
    normal_vector = np.zeros((n, 1))

    # Extract trajectory times for reference mapping
    trajectory_times = np.array([pt['time'] for pt in trajectory_ref])
    
    # --- Part 1: Accumulate Information (Normal Equations) ---
    print("Batch Filter Progress: ", end="", flush=True)

    for t, meas in enumerate(sorted_measurements):
        # Find index of trajectory point matching measurement time
        traj_idx = np.where(trajectory_times == meas['time'])[0][0]
        
        STM_t = trajectory_ref[traj_idx]['STM'] # Expecting (n, n)
        
        # Extract measurement data
        prefit_residual = np.array(meas['residual']).reshape(-1, 1)
        H_tilde = meas['partials']['wrt_X']
        R = np.diag(meas['covariance'])
        R_inv = np.linalg.inv(R)

        # Compute mapping to initial epoch: H = H_tilde * STM(t, t0)
        H = H_tilde @ STM_t
        
        # Accumulate: info += H.T * R_inv * H
        info_matrix += H.T @ R_inv @ H
        normal_vector += H.T @ R_inv @ prefit_residual

        if t % 10 == 0 or t == T-1:
            print(f"\rBatch Filter Progress: {round((t / T) * 100)}%", end="", flush=True)

    # Solve for state correction dx0 at epoch t0
    dx0 = np.linalg.solve(info_matrix, normal_vector)
    
    # Compute final initial covariance
    P0_updated = np.linalg.inv(info_matrix)

    # Propagation and post-fit residuals
    state_deviation_hist = np.zeros((n, T))
    state_corrected_hist = np.zeros((n, T))
    P_hist = np.zeros((n, n, T))
    postfit_residuals = np.full((max_m, T), np.nan)

    for t, meas in enumerate(sorted_measurements):
        traj_idx = np.where(trajectory_times == meas['time'])[0][0]
        
        STM_t = trajectory_ref[traj_idx]['STM']
        x_ref = trajectory_ref[traj_idx]['state'][:6].reshape(-1, 1)

        # Propagate state deviation and covariance to current time t
        dx_t = STM_t @ dx0
        P_t = STM_t @ P0_updated @ STM_t.T

        # Compute post-fit residual: y - H_tilde * dx_t
        prefit_residual = np.array(meas['residual']).reshape(-1, 1)
        H_tilde = meas['partials']['wrt_X']
        postfit_res = prefit_residual - (H_tilde @ dx_t)

        # Store results
        state_deviation_hist[:, t] = dx_t.flatten()
        state_corrected_hist[:, t] = (x_ref + dx_t).flatten()
        P_hist[:, :, t] = P_t
        
        m_len = len(postfit_res)
        postfit_residuals[:m_len, t] = postfit_res.flatten()

    print("\nBatch Filter Completed!")

    return {
        'state_corrected_hist': state_corrected_hist,
        'state_deviation_hist': state_deviation_hist,
        'P_hist': P_hist,
        'postfit_residuals': postfit_residuals
    }


