import numpy as np
import matplotlib.pyplot as plt
import os

def plot_filter_consistency(results_dict):
    """
    Plots State Error (Truth - Estimate) vs 3-Sigma Covariance Bounds.
    Rotates errors into the Radial-Transverse-Normal (RTN) frame.
    """
    save_folder = results_dict.get('save_folder', 'results')
    
    # Extract data
    errors_eci = results_dict['state_errors']  # Shape: (N, 6)
    P_hist = results_dict['P_hist']            # Shape: (N, 6, 6)
    states_est = results_dict['state_hist']    # Shape: (N, 6)
    
    n_steps = len(errors_eci)
    
    # Initialize arrays for RTN errors and sigmas
    errors_rtn = np.zeros((n_steps, 3))
    sigmas_rtn = np.zeros((n_steps, 3))
    
    for k in range(n_steps):
        r_vec = states_est[k, 0:3]
        v_vec = states_est[k, 3:6]
        
        # 1. Compute Rotation Matrix (ECI -> RTN)
        R_eci2rtn = compute_eci2rtn_matrix(r_vec, v_vec)
        
        # 2. Rotate Position Error
        errors_rtn[k, :] = R_eci2rtn @ errors_eci[k, 0:3]
        
        # 3. Rotate Covariance: P_rtn = R * P_eci * R^T
        P_pos_eci = P_hist[k, 0:3, 0:3]
        P_pos_rtn = R_eci2rtn @ P_pos_eci @ R_eci2rtn.T
        
        # 4. Extract 3-Sigma bounds (sqrt of diagonal variance)
        sigmas_rtn[k, :] = 3 * np.sqrt(np.diag(P_pos_rtn))
        
    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = ['Radial', 'Transverse (In-Track)', 'Normal (Cross-Track)']
    t = np.arange(n_steps) 

    for i in range(3):
        axs[i].plot(t, errors_rtn[:, i], 'b-', linewidth=1.5, label='Error')
        axs[i].plot(t, sigmas_rtn[:, i], 'r--', linewidth=1.5, label='3$\sigma$ Bound')
        axs[i].plot(t, -sigmas_rtn[:, i], 'r--', linewidth=1.5)
        
        axs[i].set_ylabel(f'{labels[i]} Error (km)')
        axs[i].grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Only add legend to the first plot
        if i == 0:
            axs[i].legend(loc='upper right')

    axs[-1].set_xlabel('Time Step')
    fig.suptitle('Consistency Check: State Errors vs 3$\sigma$ Covariance (RTN Frame)')
    
    fname = os.path.join(save_folder, 'consistency_check_plot.png')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def compute_eci2rtn_matrix(r, v):
    """Computes rotation matrix from ECI to Radial-Transverse-Normal."""
    r_norm = np.linalg.norm(r)
    r_u = r / r_norm                  # Radial (R)
    
    h = np.cross(r, v)
    h_u = h / np.linalg.norm(h)       # Normal (N) / Cross-track
    
    t_u = np.cross(h_u, r_u)          # Transverse (T) / In-track
    
    # R_eci2rtn rows are the new basis vectors
    return np.vstack((r_u, t_u, h_u))