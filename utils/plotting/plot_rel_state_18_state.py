import matplotlib.pyplot as plt
import numpy as np
import os

def plot_batch_relative_state_18(results_dict, save_folder=None):
    """
    Plots the difference between the A Priori (Initial Guess) trajectory 
    and the Final Estimated trajectory for all 18 states.
    
    Formula: Delta_x = x_apriori(t) - x_estimated(t)
    
    Args:
        results_dict (dict): Dictionary output from 'package_results'.
                             Must contain keys 'x_hist' and 'times'.
        save_folder (str, optional): Directory to save the image. Defaults to script dir.
    """
    
    # 1. Retrieve Time and State Difference
    # The BatchLS class stores dx = (Estimated - Apriori).
    # We want (Apriori - Estimated), so we multiply by -1.
    
    # Extract from dictionary
    dx_hist_arr = np.array(results_dict['x_hist']) 
    times = np.array(results_dict['times'])
    
    # Apply sign flip
    dx_data = -1 * dx_hist_arr 
    
    # 2. Setup Plot Grid (6 Rows x 3 Cols)
    fig, axes = plt.subplots(6, 3, figsize=(16, 18), sharex=True)
    
    # Define Labels and Units for the 18 states
    col_labels = [
        ['x (m)', 'y (m)', 'z (m)'],                     # Row 0: Sat Pos
        ['vx (m/s)', 'vy (m/s)', 'vz (m/s)'],             # Row 1: Sat Vel
        [r'$\mu$ ($m^3/s^2$)', r'$J_2$', r'$C_D$'],       # Row 2: Params
        ['GS101 x (m)', 'GS101 y (m)', 'GS101 z (m)'],    # Row 3: GS1
        ['GS337 x (m)', 'GS337 y (m)', 'GS337 z (m)'],    # Row 4: GS2
        ['GS394 x (m)', 'GS394 y (m)', 'GS394 z (m)']     # Row 5: GS3
    ]
    
    # Map grid indices (row, col) to State Vector indices (0-17)
    state_map = [
        [0, 1, 2],    # Sat Pos
        [3, 4, 5],    # Sat Vel
        [6, 7, 8],    # Params
        [9, 10, 11],  # GS1
        [12, 13, 14], # GS2
        [15, 16, 17]  # GS3
    ]

    # 3. Plotting Loop
    for row in range(6):
        for col in range(3):
            ax = axes[row, col]
            state_idx = state_map[row][col]
            
            # Extract data for this specific state
            y_vals = dx_data[:, state_idx]
            
            # Scatter plot
            ax.scatter(times, y_vals, s=5, c='dodgerblue', alpha=0.7)
            
            # Formatting
            ax.set_ylabel(f"$\Delta$ {col_labels[row][col]}", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Force scientific notation for very small values
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.yaxis.get_offset_text().set_fontsize(8)

            # Titles for the top of sections (Only on the middle column for cleanliness)
            if col == 1:
                if row == 0: ax.set_title("Satellite Position Deviation", fontsize=11, fontweight='bold')
                if row == 1: ax.set_title("Satellite Velocity Deviation", fontsize=11, fontweight='bold')
                if row == 2: ax.set_title("Dynamic Parameter Deviation", fontsize=11, fontweight='bold')
                if row == 3: ax.set_title("Station Coordinate Deviation", fontsize=11, fontweight='bold')

    # Common X-label at bottom
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (sec)', fontsize=10)

    # Global Title
    fig.suptitle(r'Filter Relative State: $\Delta x = \phi(t, x_{apriori}) - \hat{x}_{estimated}(t)$', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93) # Make room for suptitle

    # 4. Save Logic
    if save_folder is None:
        # Default to the directory where this script is running
        save_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure directory exists
    os.makedirs(save_folder, exist_ok=True)
    
    save_path = os.path.join(save_folder, "rel_state_deviations_state_18.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close()