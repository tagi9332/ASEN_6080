import os
import sys
import numpy as np
import pandas as pd

# Add your project root to the path if needed so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your post_processing function
from utils.plotting.post_process import post_process

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_state_errors(results_dict, n_sigma=3):
    """
    Plot state estimation errors with n-sigma bounds.
    Automatically scales y-axis to km, m, or mm based on error magnitude.
    Assumes input units are km and km/s.
    """
    # 1. Safely extract state errors to avoid KeyError
    state_errors = results_dict.get('state_errors', np.array([]))
    if state_errors.size == 0:
        print("Error: 'state_errors' not found in results_dict or is empty.")
        print("Make sure you are manually adding it to results_dict before plotting!")
        return
        
    times = results_dict.get('times', np.arange(len(state_errors)))
    
    # 2. Safely extract Sigmas (Prefer 'sigma_hist' over 'P_hist' if available)
    if 'sigma_hist' in results_dict and len(results_dict['sigma_hist']) > 0:
        sigmas = np.array(results_dict['sigma_hist'])
    elif 'P_hist' in results_dict and len(results_dict['P_hist']) > 0:
        # Fallback: compute from covariance if it's a valid 3D array
        P_hist = results_dict['P_hist']
        if P_hist.ndim == 3:
            sigmas = np.array([np.sqrt(np.diag(P)) for P in P_hist])
        else:
            print("Error: P_hist is not 3D. Cannot compute sigmas.")
            return
    else:
        print("Error: Neither 'sigma_hist' nor 'P_hist' found in results_dict.")
        return

    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    fig.suptitle(f'State Estimation Errors ({n_sigma}$\\sigma$)', fontsize=16)

    for i in range(6):
        # Extract raw data (assumed km or km/s)
        raw_error = state_errors[:, i]
        raw_bound = n_sigma * sigmas[:, i]
        
        # Determine max value to select scale
        max_val = np.max(raw_bound) if np.any(raw_bound) else np.max(np.abs(raw_error))
        
        # Determine Unit and Scale Factor
        is_velocity = i >= 3
        base_unit = "km/s" if is_velocity else "km"
        
        if max_val >= 1.0:
            scale_factor = 1.0
            unit_label = base_unit
        elif max_val >= 1e-3:
            scale_factor = 1e3
            unit_label = base_unit.replace("km", "m") # m or m/s
        else:
            scale_factor = 1e6
            unit_label = base_unit.replace("km", "mm") # mm or mm/s

        # Apply Scale
        scaled_error = raw_error * scale_factor
        scaled_bound = raw_bound * scale_factor
        
        # Plotting
        axes[i].scatter(times, scaled_error, c='b', label='Error', s=2, zorder=3)
        axes[i].plot(times, scaled_bound, 'r--', alpha=0.7, label=fr'{n_sigma}$\sigma$')
        axes[i].plot(times, -scaled_bound, 'r--', alpha=0.7)
        
        axes[i].set_ylabel(f'{state_names[i]} ({unit_label})')
        axes[i].set_title(f'State Error: {state_names[i]}')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        
        # Only add legend to the first plot to avoid clutter
        if i == 0:
            axes[i].legend(loc='upper right')

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Make room for suptitle

    # Safely save to file
    save_folder = results_dict.get('save_folder', 'results')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    save_path = os.path.join(save_folder, "state_errors.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Plot successfully saved to: {save_path}")

# ============================================================
# Helper Functions for Unpacking
# ============================================================
class LoadedSmootherResults:
    """Dummy class to hold loaded data, mimicking LKFResults."""
    pass

def extract_2d_array(df, prefix, N):
    cols = [c for c in df.columns if c.startswith(prefix + '_') and not 'flat' in c]
    cols.sort(key=lambda x: int(x.split('_')[-1]))
    
    if not cols:
        return np.zeros((N, 0))
    
    # Force conversion to numeric, turning strings into NaNs
    data = df[cols].apply(pd.to_numeric, errors='coerce').values
    return data

def extract_3d_array(df, prefix, N):
    """Extracts flattened components and reshapes them into [N x dim x dim] arrays."""
    cols = [c for c in df.columns if c.startswith(prefix + '_flat_')]
    cols.sort(key=lambda x: int(x.split('_')[-1]))
    
    if not cols:
        return None
    
    flat_size = len(cols)
    dim = int(np.sqrt(flat_size)) # 36 elements -> 6x6 matrix
    
    flat_arr = np.zeros((N, flat_size))
    for i, col in enumerate(cols):
        flat_arr[:, i] = df[col].values
        
    arr_3d = np.zeros((N, dim, dim))
    for k in range(N):
        # MUST use order='F' (Fortran) to reverse MATLAB's column-wise flattening
        arr_3d[k] = flat_arr[k].reshape((dim, dim), order='F')
        
    return arr_3d

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    print("--- Loading MATLAB Smoother Results from CSV ---")
    
    # Define file paths
    smoother_csv_file = r'HW_4\Smoother_Results_Export.csv'
    obs_file = r'data\measurements_2a_noisy.csv'
    truth_file = r'HW_4\problem_2a_traj.csv'
        
    # --- Inside your Main Execution block ---
    df_smooth = pd.read_csv(smoother_csv_file)
    N = len(df_smooth)

    results = LoadedSmootherResults()
    results.times = df_smooth['t'].values
    # Extract states and covariances
    results.dx_hist = extract_2d_array(df_smooth, 'xSmoothed', N)
    results.state_hist   = extract_2d_array(df_smooth, 'X_Smooth', N)
    results.P_hist       = extract_3d_array(df_smooth, 'PSmoothed', N)
    # Inside your Python main execution block:
    results.state_hist = extract_2d_array(df_smooth, 'X_Smooth', N)
    results.state_errors = extract_2d_array(df_smooth, 'stateError', N)
    results.sigma_hist   = extract_2d_array(df_smooth, 'sigma', N)
    
    # Fill missing attributes with dummies to satisfy package_results()
    results.innovations = np.zeros((N, 2))
    results.postfit_residuals = np.zeros((N, 2))
    results.nis_hist = np.zeros(N)

    # 3. Load Observations (Required by package_results for time alignment)
    obs = pd.read_csv(obs_file)

    # 4. Configure Plotting Options
    post_options = {
        'truth_traj_file': truth_file,
        'save_to_timestamped_folder': True,
        'data_mask_idx': 0,
        
        # Turn OFF the deviation plot that is causing the crash
        'plot_state_deviation': False, 
        
        # Keep these ON
        'plot_state_errors': False, # We will call our custom one manually
        'plot_covariance_trace': True,
        'plot_covariance_ellipsoid': True,
        'plot_filter_consistency': True,
        
        # Turn OFF measurement-specific plots
        'plot_postfit_residuals': False,
        'plot_prefit_residuals': False,
        'plot_residual_comparison': False,
        'plot_nis_metric': False
    }

    # 5. Run Post-Processing
    print("\n--- Running Python Post-Processing ---")
    # This will return a dictionary containing aligned times and paths
    results_dict = post_process(results, obs, post_options)

    # 6. Manually call your custom State Error Plot
    # We inject the errors/sigmas into the results_dict after post_process finishes
    print("\n--- Generating Custom State Error Plots ---")
    results_dict['state_errors'] = results.state_errors
    results_dict['sigma_hist'] = results.sigma_hist
    
    # Trigger the function defined at the top of your script
    plot_state_errors(results_dict, n_sigma=3)