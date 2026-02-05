import matplotlib.pyplot as plt
import numpy as np
import os

def plot_residual_comparison(results_dict):
    """
    Comparison plot of Pre-fit Innovations vs Post-fit Residuals.
    (Sigma bounds removed)
    
    - Handles 'results_units' flag (m vs km).
    - Dynamic Layout: Plots only the measurement types (Range/Rate) that exist.
    """
    # --- 1. Setup Units & Scaling ---
    unit_pref = results_dict.get('options', {}).get('results_units', 'm')

    if unit_pref == 'km':
        scale = 1
        dist_unit = 'km'
        rate_unit = 'km/s'
    else:
        scale = 1.0
        dist_unit = 'm'
        rate_unit = 'm/s'

    # --- 2. Extract Data ---
    times = np.array(results_dict['times'])
    
    # Residuals (N, 2) or (N, 1) depending on filter run
    pre_res = np.array(results_dict['prefit_residuals'])
    post_res = np.array(results_dict['postfit_residuals'])

    # Helper to check if a column contains valid data
    def has_data(arr, col_idx):
        if arr.ndim < 2: 
            # Handle 1D array case (if filter squeezed dimensions)
            if col_idx == 0 and arr.size > 0: return not np.all(np.isnan(arr))
            return False
        if col_idx >= arr.shape[1]: 
            return False
        return not np.all(np.isnan(arr[:, col_idx]))

    # --- 3. Determine Active Measurement Types ---
    # We define what we *want* to plot, then check if it exists
    plot_specs = []
    
    # Check Range (Index 0)
    if has_data(pre_res, 0) or has_data(post_res, 0):
        plot_specs.append({
            'idx': 0,
            'title_pre': "Pre-fit Range Innovations",
            'title_post': "Post-fit Range Residuals",
            'ylabel': f"Range Error ({dist_unit})",
            'color_pre': 'darkgreen',
            'color_post': 'black'
        })

    # Check Range-Rate (Index 1)
    if has_data(pre_res, 1) or has_data(post_res, 1):
        plot_specs.append({
            'idx': 1,
            'title_pre': "Pre-fit Range-Rate Innovations",
            'title_post': "Post-fit Range-Rate Residuals",
            'ylabel': f"Range-Rate Error ({rate_unit})",
            'color_pre': 'darkgreen',
            'color_post': 'black'
        })

    if not plot_specs:
        print("Warning: No valid residuals found to plot.")
        return

    # --- 4. Setup Plot Layout ---
    n_rows = len(plot_specs)
    # 2 Columns (Pre vs Post) are fixed. Rows depend on active data.
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows), sharex=True)
    
    # Ensure axes is always a 2D array [row, col] for consistent indexing
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    # --- 5. Plotting Loop ---
    for row_idx, spec in enumerate(plot_specs):
        
        # Data Index (0 for Range, 1 for Rate)
        d_idx = spec['idx']
        
        # --- Column 0: Pre-fit ---
        ax_pre = axes[row_idx, 0]
        # Handle 1D vs 2D array shape safely
        if pre_res.ndim == 2:
            data_pre = pre_res[:, d_idx] * scale
        else:
            data_pre = pre_res * scale # Fallback if 1D array implies the single active type
            
        # Clean NaNs for RMS calculation
        valid_pre = data_pre[~np.isnan(data_pre)]
        rms_pre = np.sqrt(np.mean(valid_pre**2)) if len(valid_pre) > 0 else 0.0
        
        ax_pre.scatter(times, data_pre, s=10, c=spec['color_pre'], label="Innovation", alpha=0.7)
        ax_pre.set_title(f"{spec['title_pre']}\nRMS: {rms_pre:.3e}")
        ax_pre.set_ylabel(spec['ylabel'])
        ax_pre.grid(True, which='both', linestyle=':', alpha=0.5)
        
        if row_idx == 0:
            ax_pre.legend(loc='upper right')

        # --- Column 1: Post-fit ---
        ax_post = axes[row_idx, 1]
        if post_res.ndim == 2:
            data_post = post_res[:, d_idx] * scale
        else:
            data_post = post_res * scale

        valid_post = data_post[~np.isnan(data_post)]
        rms_post = np.sqrt(np.mean(valid_post**2)) if len(valid_post) > 0 else 0.0

        ax_post.scatter(times, data_post, s=10, c=spec['color_post'], label="Residual", alpha=0.7)
        ax_post.set_title(f"{spec['title_post']}\nRMS: {rms_post:.3e}")
        ax_post.grid(True, which='both', linestyle=':', alpha=0.5)

    # --- 6. Formatting ---
    # Set X-Label only on the bottom row
    for col in range(2):
        axes[-1, col].set_xlabel('Time (s)')
    
    fig.suptitle(f'Filter Performance: Pre-fit vs. Post-fit Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90 if n_rows > 1 else 0.85) # Adjust based on height
    
    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    save_path = os.path.join(save_folder, "residual_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()