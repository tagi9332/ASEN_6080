import matplotlib.pyplot as plt
import numpy as np
import os

def plot_residual_comparison(results_dict):
    """
    Comparison plot of Pre-fit Innovations vs Post-fit Residuals.
    (Sigma bounds removed)
    
    Layout:
    -------
    Left Column : Pre-fit Innovations (Range, Range-Rate)
    Right Column: Post-fit Residuals (Range, Range-Rate)
    """
    # 1. Extract Data
    times = np.array(results_dict['times'])
    
    # Residuals (N, 2)
    pre_res = np.array(results_dict['prefit_residuals'])
    post_res = np.array(results_dict['postfit_residuals'])
    
    # 2. Setup Plot (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    
    titles = [["Pre-fit Range Innovations", "Post-fit Range Residuals"],
              ["Pre-fit Range-Rate Innovations", "Post-fit Range-Rate Residuals"]]
    
    ylabels = ["Range Error (m)", "Range-Rate Error (m/s)"]
    colors = ['darkgreen', 'black'] # Green for pre, Black for post
    
    for row in range(2): # 0: Range, 1: RR
        
        # Determine Unit Scaling (km -> m or m/s)
        scale = 1e3 
        
        for col in range(2): # 0: Prefit, 1: Postfit
            ax = axes[row, col]
            
            # Select Data
            if col == 0:
                data = pre_res[:, row] * scale
                label_name = "Innovation"
            else:
                data = post_res[:, row] * scale
                label_name = "Residual"
            
            # Calculate RMS for the Title
            rms = np.sqrt(np.mean(data**2))
            
            # --- PLOTTING ---
            ax.scatter(times, data, s=2, c=colors[col], label=label_name, alpha=0.7)
            
            # Formatting
            ax.set_title(f"{titles[row][col]}\nRMS: {rms:.3f}")
            ax.grid(True, which='both', linestyle=':', alpha=0.5)
            
            if col == 0:
                ax.set_ylabel(ylabels[row])
            
            if row == 0 and col == 0:
                ax.legend(loc='upper right')

    # Common X-label
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 1].set_xlabel('Time (s)')
    
    fig.suptitle('Filter Performance: Pre-fit vs. Post-fit Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Make space for main title
    
    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    save_path = os.path.join(save_folder, "residual_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()