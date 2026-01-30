import numpy as np
import matplotlib.pyplot as plt

def plot_state_errors(results_dict, n_sigma=3):
    """
    Plot state estimation errors with n-sigma bounds.
    Automatically scales y-axis to km, m, or mm based on error magnitude.
    Assumes input units are km and km/s.
    """
    times = results_dict['times']
    
    # Calculate sigmas from Covariance (assumed km^2 and km^2/s^2)
    sigmas = np.array([np.sqrt(np.diag(P)) for P in results_dict['P_hist']])
    
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    fig.suptitle(f'Linearized Kalman Filter State Estimation Errors ({n_sigma}$\sigma$)', fontsize=16)

    for i in range(6):
        # Extract raw data (assumed km or km/s)
        raw_error = results_dict['state_errors'][:, i]
        raw_bound = n_sigma * sigmas[:, i]
        
        # Determine max value to select scale
        # We look at the bounds max to ensure the red lines fit in the unit choice
        max_val = np.max(raw_bound)
        
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

    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    plt.savefig(f"{save_folder}/state_errors.png", dpi=300)
    plt.close()