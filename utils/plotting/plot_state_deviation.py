import matplotlib.pyplot as plt
import numpy as np
import os

def plot_state_deviation(results_dict):
    """
    Plot the estimated state deviation (dx) from the reference trajectory.
    (Sigma bounds removed)
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    
    base_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    # 1. Extract the Deviation History (dx)
    times = np.array(results_dict['times'])
    dx_hist = np.array(results_dict['x_hist']) 

    for i in range(6):
        # Get raw data for this state index
        raw_dev = dx_hist[:, i]
        
        # --- Dynamic Unit Scaling ---
        # Determine scale based on the maximum deviation value
        max_val = np.max(np.abs(raw_dev))
        
        # Default to meters if max_val is 0 (e.g. initial step)
        if max_val == 0:
            scale = 1e3
            unit = "m"
        elif max_val < 1e-4:  # < 0.1 mm -> use mm (scale up)
            scale = 1e6
            unit = "mm"
        elif max_val < 1.0:   # < 1.0 km -> use m (scale up)
            scale = 1e3
            unit = "m"
        else:                 # > 1.0 km -> keep as km
            scale = 1.0
            unit = "km"

        # Apply Scaling
        y_dev = raw_dev * scale
        
        # Construct Label
        if i >= 3: 
            label_str = f"{base_names[i]} ({unit}/s)"
        else:      
            label_str = f"{base_names[i]} ({unit})"

        # --- Plotting ---
        # Plot only the deviation (dx)
        axes[i].scatter(times, y_dev, s=2, c='blue', label=r'Estimated Deviation ($\delta x$)', alpha=0.6)
        
        # Formatting
        axes[i].set_ylabel(label_str)
        axes[i].set_title(f'Perturbation: {base_names[i]}')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        
        # Add legend only to the first plot
        if i == 0:
            axes[i].legend(loc='upper right')

    fig.suptitle('Trajectory Deviation from Reference', fontsize=16)
    
    # Set common X label
    axes[4].set_xlabel('Time (s)')
    axes[5].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Adjust for suptitle
    
    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    save_path = os.path.join(save_folder, "state_deviation.png")
    plt.savefig(save_path, dpi=300)
    plt.close()