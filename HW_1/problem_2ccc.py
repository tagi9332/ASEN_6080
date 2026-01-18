import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. Setup paths robustly
script_dir = os.path.dirname(os.path.abspath(__file__))
truth_file = os.path.join(script_dir, 'problem_2c_differences.csv')
stm_file = os.path.join(script_dir, 'problem_2cc_disturbances.csv')

# Check if files exist
if not os.path.exists(truth_file) or not os.path.exists(stm_file):
    print(f"Error: Could not find the CSV files in {script_dir}")
else:
    # 2. Load the data
    df_truth = pd.read_csv(truth_file)
    df_stm = pd.read_csv(stm_file)

    # 3. Calculate the Residuals (STM Prediction - Nonlinear Truth)
    time_hrs = df_truth['Time(s)'] / 3600.0
    
    # Position Differences (km)
    res_x = df_stm['Delta_X(km)'] - df_truth['Delta_X(km)']
    res_y = df_stm['Delta_Y(km)'] - df_truth['Delta_Y(km)']
    res_z = df_stm['Delta_Z(km)'] - df_truth['Delta_Z(km)']
    
    # Velocity Differences - Converted to m/s for better scale visibility
    res_vx = (df_stm['Delta_VX(km/s)'] - df_truth['Delta_VX(km/s)']) * 1000
    res_vy = (df_stm['Delta_VY(km/s)'] - df_truth['Delta_VY(km/s)']) * 1000
    res_vz = (df_stm['Delta_VZ(km/s)'] - df_truth['Delta_VZ(km/s)']) * 1000

    # 4. Create the 2x3 Plot Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)

    # --- TOP ROW: Position Errors (km) ---
    axes[0, 0].plot(time_hrs, res_x, 'r')
    axes[0, 0].set_title('Residual $\Delta$X (Position)')
    axes[0, 0].set_ylabel('Error (km)')

    axes[0, 1].plot(time_hrs, res_y, 'g')
    axes[0, 1].set_title('Residual $\Delta$Y (Position)')

    axes[0, 2].plot(time_hrs, res_z, 'b')
    axes[0, 2].set_title('Residual $\Delta$Z (Position)')

    # --- BOTTOM ROW: Velocity Errors (m/s) ---
    axes[1, 0].plot(time_hrs, res_vx, 'r')
    axes[1, 0].set_title('Residual $\Delta$Vx (Velocity)')
    axes[1, 0].set_ylabel('Error (m/s)')
    axes[1, 0].set_xlabel('Time (hours)')

    axes[1, 1].plot(time_hrs, res_vy, 'g')
    axes[1, 1].set_title('Residual $\Delta$Vy (Velocity)')
    axes[1, 1].set_xlabel('Time (hours)')

    axes[1, 2].plot(time_hrs, res_vz, 'b')
    axes[1, 2].set_title('Residual $\Delta$Vz (Velocity)')
    axes[1, 2].set_xlabel('Time (hours)')

    # 5. Global Formatting
    for ax in axes.flatten():
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)

    plt.suptitle('Linearization Validity: Difference Between STM and Nonlinear Truth (15 Orbits)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the comparison image
    plt.savefig(os.path.join(script_dir, 'stm_validity_residuals.png'))
    plt.show()

    # 6. Output numerical summary for your discussion
    final_pos_rss = np.sqrt(res_x.iloc[-1]**2 + res_y.iloc[-1]**2 + res_z.iloc[-1]**2)
    print(f"Final RSS Position Error after 15 orbits: {final_pos_rss:.6f} km")