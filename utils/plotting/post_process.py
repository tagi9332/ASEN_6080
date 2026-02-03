import numpy as np
from datetime import datetime
import os 
from utils.plotting.plot_prefit_residuals import plot_prefit_residuals
from utils.plotting.plot_residual_comparison import plot_residual_comparison
from utils.plotting.plot_state_errors import plot_state_errors
from utils.plotting.plot_state_deviation import plot_state_deviation
from utils.plotting.plot_postfit_residuals import plot_postfit_residuals
from utils.plotting.package_results import package_results
from utils.plotting.compute_state_error import compute_state_error
from utils.plotting.plot_covariance_trace import plot_covariance_trace
from utils.plotting.plot_filter_consistency import plot_filter_consistency
from utils.plotting.plot_nis_metric import plot_nis_metric
from utils.plotting.report_RMS_error import report_filter_metrics



import os
from datetime import datetime
import numpy as np

def post_process(results, obs, options):
    """
    Post-process and visualize filter results.
    
    Logic:
    - If options['save_to_timestamped_folder'] is True: Saves to 'results/YYYY-MM-DD_HH-MM-SS/'
    - If False: Saves directly to 'results/'
    """
    
    # 1. Package results
    results_dict = package_results(results, obs, options)

    # 2. Determine Save Folder
    base_folder = 'results'
    
    if options.get('save_to_timestamped_folder', False):
        # Create a unique timestamped subfolder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_folder = os.path.join(base_folder, timestamp)
    else:
        # Save directly to the base results folder
        save_folder = base_folder

    # Create the directory (works for both cases)
    os.makedirs(save_folder, exist_ok=True)
    
    # Store the path in the dictionary for plotting functions to use
    results_dict['save_folder'] = save_folder

    # Store units flag in results_dict
    results_dict['results_units'] = options.get('results_units', 'km')  #

# 3. Compute state errors against truth (if file provided)
    if 'truth_traj_file' in options:
        try:
            # --- FIX: Apply the same mask to the observations ---
            # Retrieve mask (default to full slice if missing)
            mask_idx = options.get('data_mask_idx', slice(None))
            
            # If mask is an integer (start index), convert to slice
            if isinstance(mask_idx, int):
                mask_idx = slice(mask_idx, None)
                
            # Apply mask to the observations dataframe
            df_meas_masked = obs.iloc[mask_idx]

            state_errors = compute_state_error(
                x_corrected=results_dict['state_hist'], 
                df_meas=df_meas_masked,              # <--- PASS MASKED OBS
                truth_file=options['truth_traj_file']
            )
            results_dict['state_errors'] = np.array(state_errors)
        except Exception as e:
            print(f"Warning: Could not compute state errors. {e}")
            results_dict['state_errors'] = np.array([])

    # 4. Trigger Plots based on Options
    if options.get('plot_state_errors', False):
        plot_state_errors(results_dict)

    if options.get('plot_state_deviation', False):
        plot_state_deviation(results_dict)

    if options.get('plot_prefit_residuals', False):
        plot_prefit_residuals(results_dict)

    if options.get('plot_postfit_residuals', False):
        plot_postfit_residuals(results_dict)

    if options.get('plot_residual_comparison', False):
        plot_residual_comparison(results_dict)

    if options.get('plot_covariance_trace', False):
        plot_covariance_trace(results_dict)

    if options.get('plot_filter_consistency', False):
        plot_filter_consistency(results_dict)

    if options.get('plot_nis_metric', False):
        plot_nis_metric(results_dict)

    # Print Completion Message
    print("Post-processing and plotting completed.")
    print(f"Output directory saved to: {save_folder}")


    return results_dict

