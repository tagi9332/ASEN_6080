import numpy as np
from datetime import datetime
import os 
from utils.plotting.plot_covariance_ellipsoid import plot_covariance_ellipsoid
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
from utils.plotting.save_log_file import save_run_log



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
    print(f"Output directory set to: {save_folder}")

# 3. Compute state errors against truth
    if 'truth_traj_file' in options:
        try:
            # --- FIX: Ensure Observation Dataframe Matches Filter Results ---
            
            # Step A: Slice off initial state from obs if package_results did it
            # We can check this by comparing lengths.
            # results_dict['times'] is the "truth" for what the filter kept.
            
            filter_times = results_dict['times']
            
            # Filter the original observations to match the timestamps kept by package_results
            # This is safer than index slicing because it guarantees time alignment
            df_meas_masked = obs[obs['Time(s)'].isin(filter_times)]

            # Double check lengths to be safe
            if len(df_meas_masked) != len(results_dict['state_hist']):
                print(f"Warning: Length mismatch in post_process. "
                      f"Obs: {len(df_meas_masked)}, States: {len(results_dict['state_hist'])}")

            state_errors = compute_state_error(
                x_corrected=results_dict['state_hist'], 
                df_meas=df_meas_masked,              # <--- PASS TIME-MATCHED OBS
                truth_file=options['truth_traj_file']
            )
            results_dict['state_errors'] = np.array(state_errors)
            
        except Exception as e:
            print(f"Warning: Could not compute state errors. {e}")
            # Initialize empty array to prevent plotting crashes
            results_dict['state_errors'] = np.array([])

    # 4. Trigger Plots based on Options
    if options.get('plot_state_errors', False) and len(results_dict['state_errors']) > 0:
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

    if options.get('plot_covariance_ellipsoid', False):
        plot_covariance_ellipsoid(results_dict)

    # Print Completion Message
    print("Post-processing and plotting completed.")

    return results_dict

