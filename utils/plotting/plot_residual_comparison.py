import matplotlib.pyplot as plt
import numpy as np
import os

def plot_residual_comparison(results_dict):
    """
    Comparison plot of Pre-fit Innovations vs Post-fit Residuals.
    (Sigma bounds removed)
    
    - Handles 'results_units' flag (m vs km).
    - Assumes raw residuals in results_dict are in METERS and M/S.
    """
    # --- 1. Setup Units & Scaling ---
    # Get user preference ('m' or 'km')
    unit_pref = results_dict.get('options', {}).get('results_units', 'm')

    if unit_pref == 'km':
        scale = 1e-3
        dist_unit = 'km'
        rate_unit = 'km/s'
    else:
        scale = 1.0
        dist_unit = 'm'
        rate_unit = 'm/s'

    # --- 2. Extract Data ---
    times = np.array(results_dict['times'])
    
    # Residuals (N, 2)
    pre_res = np.array(results_dict['prefit_residuals'])
    post_res = np.array(results_dict['postfit_residuals'])
    
    # --- 3. Setup Plot (2x2 Grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    
    titles = [["Pre-fit Range Innovations", "Post-fit Range Residuals"],
              ["Pre-fit Range-Rate Innovations", "Post-fit Range-Rate Residuals"]]
    
    ylabels = [f"Range Error ({dist_unit})", f"Range-Rate Error ({rate_unit})"]
    colors = ['darkgreen', 'black'] # Green for pre, Black for post
    
    for row in range(2): # 0: Range, 1: RR
        for col in range(2): # 0: Prefit, 1: Postfit
            ax = axes[row, col]
            
            # Select Data & Scale
            if col == 0:
                # Pre-fit
                data = pre_res[:, row] * scale
                label_name = "Innovation"
            else:
                # Post-fit
                data = post_res[:, row] * scale
                label_name = "Residual"
            
            # Calculate RMS for the Title
            rms = np.sqrt(np.mean(data**2))
            
            # --- PLOTTING ---
            ax.scatter(times, data, s=2, c=colors[col], label=label_name, alpha=0.7)
            
            # Formatting
            ax.set_title(f"{titles[row][col]}\nRMS: {rms:.3e}")
            ax.grid(True, which='both', linestyle=':', alpha=0.5)
            
            # Set Y-Label only on the left column
            if col == 0:
                ax.set_ylabel(ylabels[row])
            
            # Legend only on the very first plot
            if row == 0 and col == 0:
                ax.legend(loc='upper right')

    # Common X-label
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 1].set_xlabel('Time (s)')
    
    fig.suptitle(f'LKF Performance: Pre-fit vs. Post-fit Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Make space for main title
    
    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    save_path = os.path.join(save_folder, "residual_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
