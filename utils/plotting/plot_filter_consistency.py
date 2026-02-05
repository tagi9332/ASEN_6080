import numpy as np
import matplotlib.pyplot as plt
import os

def compute_eci2rtn_matrix(r, v):
    """
    Computes rotation matrix from ECI to Radial-Transverse-Normal.
    Inputs r, v can be in any consistent unit (m or km).
    """
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return np.eye(3) # Handle singularity safely
        
    r_u = r / r_norm                  # Radial (R)
    
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    
    if h_norm == 0:
        return np.eye(3) # Handle singularity (e.g. zero velocity)

    h_u = h / h_norm                  # Normal (N) / Cross-track
    
    t_u = np.cross(h_u, r_u)          # Transverse (T) / In-track
    
    # R_eci2rtn rows are the new basis vectors
    return np.vstack((r_u, t_u, h_u))

def plot_filter_consistency(results_dict):
    """
    Plots State Error (Truth - Estimate) vs 3-Sigma Covariance Bounds.
    Rotates errors into the Radial-Transverse-Normal (RTN) frame.
    
    - Handles 'results_units' flag (m vs km).
    - Uses actual simulation time for x-axis.
    """
    
    # --- 1. Setup Units & Scaling ---
    # Get user preference ('m' or 'km')
    unit_pref = results_dict.get('options', {}).get('results_units', 'm')

    if unit_pref == 'km':
        scale = 1
        unit_label = 'km'
    else:
        scale = 1.0
        unit_label = 'm'

    save_folder = results_dict.get('save_folder', 'results')
    
    # --- 2. Extract Data (Assumed SI Units: Meters) ---
    errors_eci = results_dict.get('state_errors') # Shape: (N, 6)
    
    # Safety Check: If no truth data, we can't plot consistency
    if errors_eci is None or errors_eci.size == 0:
        print("[Plotting] Skipping consistency check (no truth data available).")
        return

    P_hist = results_dict['P_hist']            # Shape: (N, 6, 6)
    states_est = results_dict['state_hist']    # Shape: (N, 6)
    times = results_dict['times']
    
    n_steps = len(errors_eci)
    
    # Initialize arrays for RTN errors and sigmas
    errors_rtn = np.zeros((n_steps, 3))
    sigmas_rtn = np.zeros((n_steps, 3))
    
    # --- 3. Rotate Errors & Covariance to RTN ---
    for k in range(n_steps):
        r_vec = states_est[k, 0:3]
        v_vec = states_est[k, 3:6]
        
        # A. Compute Rotation Matrix (ECI -> RTN)
        R_eci2rtn = compute_eci2rtn_matrix(r_vec, v_vec)
        
        # B. Rotate Position Error (ECI -> RTN)
        # We assume errors_eci is in meters. Rotation preserves units.
        errors_rtn[k, :] = R_eci2rtn @ errors_eci[k, 0:3]
        
        # C. Rotate Covariance: P_rtn = R * P_eci * R^T
        P_pos_eci = P_hist[k, 0:3, 0:3]
        P_pos_rtn = R_eci2rtn @ P_pos_eci @ R_eci2rtn.T
        
        # D. Extract 3-Sigma bounds (sqrt of diagonal variance)
        # Result is in meters (since P is m^2)
        sigmas_rtn[k, :] = 3 * np.sqrt(np.diag(P_pos_rtn))
        
    # --- 4. Apply Unit Scaling ---
    errors_rtn_scaled = errors_rtn * scale
    sigmas_rtn_scaled = sigmas_rtn * scale
        
    # --- 5. Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = ['Radial', 'Transverse (In-Track)', 'Normal (Cross-Track)']
    
    for i in range(3):
        axs[i].plot(times, errors_rtn_scaled[:, i], 'b-', linewidth=1.5, label='Error')
        axs[i].plot(times, sigmas_rtn_scaled[:, i], 'r--', linewidth=1.5, label='3$\sigma$ Bound')
        axs[i].plot(times, -sigmas_rtn_scaled[:, i], 'r--', linewidth=1.5)
        
        axs[i].set_ylabel(f'{labels[i]} Error ({unit_label})')
        axs[i].grid(True, which='both', linestyle=':', alpha=0.6)
        axs[i].set_title(f"{labels[i]} Consistency")
        
        # Only add legend to the first plot
        if i == 0:
            axs[i].legend(loc='upper right')

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Consistency Check: State Errors vs 3$\sigma$ Covariance (RTN Frame)', fontsize=16)
    
    fname = os.path.join(save_folder, 'consistency_check_plot.png')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(fname, dpi=300)
    plt.close()
