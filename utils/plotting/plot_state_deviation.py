import matplotlib.pyplot as plt
import numpy as np
import os

def plot_state_deviation(results_dict):
    """
    Plot the estimated state deviation (dx) from the reference trajectory.
    (Sigma bounds removed)
    
    - Handles 'results_units' flag (m vs km).
    - Auto-scales y-axis for readability (e.g., switches km -> m if deviation is small).
    """
    
    # --- 1. Setup ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    
    base_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    # Extract Deviation History
    times = np.array(results_dict['times'])
    dx_hist = np.array(results_dict['x_hist']) # Shape: (N, 6)

    # --- 2. Check Input Unit Preference ---
    # We assume the raw filter data in dx_hist is in METERS (SI standard).
    unit_pref = results_dict.get('options', {}).get('results_units', 'm')

    # Define Base Scaling
    if unit_pref == 'km':
        base_scale = 1.0
        pos_base_unit = 'km'
        vel_base_unit = 'km/s'
    else:
        base_scale = 1.0
        pos_base_unit = 'm'
        vel_base_unit = 'm/s'

    # --- 3. Plotting Loop ---
    for i in range(6):
        # Get raw data (Meters) and convert to User's Base Unit
        # raw_dev_base is now in 'm' or 'km' depending on preference
        raw_dev_base = dx_hist[:, i] * base_scale
        
        # Determine labels
        is_velocity = i >= 3
        current_unit = vel_base_unit if is_velocity else pos_base_unit

        # --- Dynamic Unit Scaling ---
        # Check the magnitude of the deviation to see if we need to scale down (e.g. km -> m)
        max_val = np.max(np.abs(raw_dev_base))
        
        scale_display = 1.0
        unit_label = current_unit

        if max_val > 0:
            if max_val < 1e-3: 
                # e.g., have km, need mm OR have m, need mm
                # logic: if base is km, 1e-6 km = 1 mm. If base is m, 1e-3 m = 1 mm.
                scale_display = 1e6 if unit_pref == 'km' else 1e3
                unit_label = current_unit.replace('km', 'mm').replace('m', 'mm')
            elif max_val < 1.0:
                # e.g. have km, need m
                if unit_pref == 'km':
                    scale_display = 1e3
                    unit_label = current_unit.replace('km', 'm')
        
        # Apply Final Display Scaling
        y_dev = raw_dev_base * scale_display

        # --- Plotting ---
        axes[i].scatter(times, y_dev, s=2, c='blue', label=r'Estimated Deviation ($\delta x$)', alpha=0.6)
        
        # Formatting
        axes[i].set_ylabel(f"{base_names[i]} ({unit_label})")
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
