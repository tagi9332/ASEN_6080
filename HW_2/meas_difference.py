import numpy as np
import matplotlib.pyplot as plt
import os

# ================= USER CONFIGURATION =================
# Replace these strings with the actual paths to your files
FILE_NEW = "HW_2\measurements_2a_noisy.csv"  # The file you just generated
FILE_REF = "HW_2\measurements_noisy.csv"     # The reference/truth file
# ======================================================

def read_measurements(filepath):
    """
    Reads a CSV with format: Time(s),Range(km),Range_Rate(km/s),Station_ID
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    # Load data: delimiter is comma, skip the first row (header)
    try:
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    print(f"Reading New File: {FILE_NEW}")
    data_new = read_measurements(FILE_NEW)
    
    print(f"Reading Ref File: {FILE_REF}")
    data_ref = read_measurements(FILE_REF)

    if data_new is None or data_ref is None:
        return

    # Safety Check: Ensure files are the same length
    min_len = min(len(data_new), len(data_ref))
    if len(data_new) != len(data_ref):
        print(f"WARNING: File lengths differ ({len(data_new)} vs {len(data_ref)}). Truncating to match.")
    
    # Truncate arrays to the shorter length to avoid errors
    data_new = data_new[:min_len]
    data_ref = data_ref[:min_len]

    # Extract Columns
    # Col 0: Time, Col 1: Range, Col 2: Range Rate, Col 3: Station ID
    time = data_new[:, 0]
    
    # Calculate Differences (New - Reference)
    diff_range = data_new[:, 1] - data_ref[:, 1]
    diff_rate  = data_new[:, 2] - data_ref[:, 2]

    # Calculate RMS for the title/legend
    rms_range = np.sqrt(np.mean(diff_range**2))
    rms_rate  = np.sqrt(np.mean(diff_rate**2))

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Range Residuals
    ax1.plot(time, diff_range, 'b.', markersize=4, label='Residuals')
    ax1.set_ylabel('Range Difference [km]')
    ax1.set_title(f'Measurement Residuals\n(RMS Range: {rms_range:.4e} km)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axhline(0, color='k', linewidth=0.8)

    # 2. Range Rate Residuals
    ax2.plot(time, diff_rate, 'r.', markersize=4, label='Residuals')
    ax2.set_ylabel('Range Rate Difference [km/s]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title(f'Range Rate Residuals\n(RMS Rate: {rms_rate:.4e} km/s)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.axhline(0, color='k', linewidth=0.8)

    plt.tight_layout()
    
    # Save the plot
    output_img = "measurement_residuals.png"
    plt.savefig(output_img)
    print(f"\nPlot saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    main()