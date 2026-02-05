import numpy as np
import matplotlib.pyplot as plt

def plot_state_errors(results_dict, n_sigma=3):
    """
    Plot state estimation errors with n-sigma bounds.
    
    - Handles 'results_units' flag (m vs km).
    - Auto-scales y-axis (e.g., if user asks for 'km' but errors are 'mm', it switches to 'mm').
    - Assumes raw input data in results_dict is in METERS and METERS/S.
    """
    
    # --- 1. Safety Check: Ensure data exists ---
    state_errors = results_dict.get('state_errors')
    if state_errors is None or state_errors.size == 0:
        print("[Plotting] No state errors available (truth data might be missing). Skipping plot.")
        return

    # --- 2. Setup Data & Units ---
    times = results_dict['times']
    P_hist = results_dict['cov_hist']  # Ensure key matches your dict ('cov_hist' or 'P_hist')
    
    # Calculate sigmas (Sqrt of diagonal variances)
    # Assumed input: Meters and Meters/s
    sigmas = np.sqrt(np.diagonal(P_hist, axis1=1, axis2=2))
    
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    
    # Get user preference ('m' or 'km')
    unit_pref = results_dict.get('options', {}).get('results_units', 'm')
    
    # Define Base Scaling based on stored data
    if unit_pref == 'km':
        base_scale = 1.0
        pos_base_unit = 'km'
        vel_base_unit = 'km/s'
    else:
        base_scale = 1.0
        pos_base_unit = 'm'
        vel_base_unit = 'm/s'

    # --- 3. Plotting Loop ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    fig.suptitle(f'Linearized Kalman Filter State Estimation Errors ({n_sigma}$\sigma$)', fontsize=16)

    for i in range(6):
        # Extract raw data (Meters) and apply User Preference (m or km)
        # raw_error_base is now in the user's preferred unit
        raw_error_base = state_errors[:, i] * base_scale
        raw_bound_base = n_sigma * sigmas[:, i] * base_scale
        
        # Determine unit label for this specific axis
        is_velocity = i >= 3
        current_unit = vel_base_unit if is_velocity else pos_base_unit
        
        # --- 4. Auto-Scaling Logic ---
        # If the max error is tiny in the chosen unit, scale down further for readability
        max_val = np.max(np.abs(raw_bound_base))
        
        scale_display = 1.0
        unit_label = current_unit
        
        # Logic: If max value is < 0.1 of the base unit, drop down a metric prefix
        if max_val > 0: # Avoid div by zero
            if max_val < 1e-3: 
                # e.g., have km, need mm OR have m, need mm
                scale_display = 1e6 if unit_pref == 'km' else 1e3
                unit_label = current_unit.replace('km', 'mm').replace('m', 'mm')
            elif max_val < 1.0:
                # e.g., have km, need m
                if unit_pref == 'km':
                    scale_display = 1e3
                    unit_label = current_unit.replace('km', 'm')

        # Apply final display scaling
        final_error = raw_error_base * scale_display
        final_bound = raw_bound_base * scale_display

        # --- 5. Draw Plot ---
        axes[i].scatter(times, final_error, c='b', label='Error', s=2, zorder=3)
        axes[i].plot(times, final_bound, 'r--', alpha=0.7, label=fr'{n_sigma}$\sigma$')
        axes[i].plot(times, -final_bound, 'r--', alpha=0.7)
        
        axes[i].set_ylabel(f'{state_names[i]} ({unit_label})')
        axes[i].set_title(f'State Error: {state_names[i]}')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        
        if i == 1: # Legend on top-right plot
            axes[i].legend(loc='upper right')

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save
    save_folder = results_dict.get('save_folder', '.')
    plt.savefig(f"{save_folder}/state_errors.png", dpi=300)
    plt.close()
