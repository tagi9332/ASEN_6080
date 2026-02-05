import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def plot_postfit_residuals(results_dict, n_sigma=3):
    """
    Plotting function for post-fit residuals and their Gaussian distributions.
    
    - Handles 'results_units' flag (m vs km).
    - Dynamic Layout: Plots only the measurement types (Range/Rate) that exist.
    """
    
    # --- 1. Setup Units & Scaling ---
    unit_pref = results_dict.get('options', {}).get('results_units', 'm')

    if unit_pref == 'km':
        scale = 1.0
        dist_unit = 'km'
        rate_unit = 'km/s'
    else:
        scale = 1.0
        dist_unit = 'm'
        rate_unit = 'm/s'

    # --- 2. Extract Data ---
    times = np.array(results_dict['times'])
    postfit_residuals = np.array(results_dict['postfit_residuals'])

    # Helper to safely check for valid data in a column
    def has_data(arr, col_idx):
        if arr.ndim < 2:
            if col_idx == 0 and arr.size > 0: return not np.all(np.isnan(arr))
            return False
        if col_idx >= arr.shape[1]: 
            return False
        return not np.all(np.isnan(arr[:, col_idx]))

    # --- 3. Determine Active Measurement Types ---
    plot_specs = []

    # Check Range (Index 0)
    if has_data(postfit_residuals, 0):
        # Extract and Scale
        if postfit_residuals.ndim == 2:
            data = postfit_residuals[:, 0] * scale
        else:
            data = postfit_residuals * scale
            
        plot_specs.append({
            'data': data,
            'title_time': 'Range Residuals vs Time',
            'title_hist': 'Range PDF',
            'ylabel': f'Range Residual ({dist_unit})',
            'xlabel_hist': f'Residual ({dist_unit})',
            'color_hist': 'skyblue',
            'fit_color': 'r'
        })

    # Check Range-Rate (Index 1)
    if has_data(postfit_residuals, 1):
        if postfit_residuals.ndim == 2:
            data = postfit_residuals[:, 1] * scale
        else:
            data = postfit_residuals * scale # Fallback if 1D array implies rate

        plot_specs.append({
            'data': data,
            'title_time': 'Range-Rate Residuals vs Time',
            'title_hist': 'Range-Rate PDF',
            'ylabel': f'Range-Rate Residual ({rate_unit})',
            'xlabel_hist': f'Residual ({rate_unit})',
            'color_hist': 'salmon',
            'fit_color': 'b'
        })

    if not plot_specs:
        print("Warning: No valid post-fit residuals found to plot.")
        return

    # --- 4. Setup Plot Layout ---
    n_rows = len(plot_specs)
    fig, axs = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
    
    # Ensure axs is always 2D [row, col]
    if n_rows == 1:
        axs = axs.reshape(1, 2)

    fig.suptitle('Filter Residual Analysis: Time History and Error Distributions', fontsize=18)

    # --- 5. Plotting Loop ---
    for row_idx, spec in enumerate(plot_specs):
        
        res_data = spec['data']
        # Filter NaNs for stats
        valid_res = res_data[~np.isnan(res_data)]
        
        if len(valid_res) == 0:
            continue

        mu, std = np.mean(valid_res), np.std(valid_res)

        # === Col 1: Time History ===
        ax_time = axs[row_idx, 0]
        ax_time.scatter(times, res_data, s=10, c='black', label='Post-fit Residual', alpha=0.7)
        
        # Empirical Sigma Bounds
        ax_time.axhline(y=n_sigma * std, color='r', linestyle='--', alpha=0.8, label=fr'{n_sigma}$\sigma$ (Empirical)')
        ax_time.axhline(y=-n_sigma * std, color='r', linestyle='--', alpha=0.8)
        
        ax_time.set_ylabel(spec['ylabel'])
        ax_time.set_title(spec['title_time'])
        ax_time.grid(True, alpha=0.3)
        
        if row_idx == 0:
            ax_time.legend(loc='upper right')
        
        # Only add time label to the bottom plot
        if row_idx == n_rows - 1:
            ax_time.set_xlabel('Time (s)')

        # === Col 2: Histogram ===
        ax_hist = axs[row_idx, 1]
        
        # Filter outliers for cleaner histogram visualization (keep within 6 sigma of mean)
        if std > 1e-12:
            hist_data = valid_res[np.abs(valid_res - mu) <= 6 * std]
        else:
            hist_data = valid_res

        ax_hist.hist(hist_data, bins=40, density=True, alpha=0.6, 
                     color=spec['color_hist'], edgecolor='black', label='Residual Dist')
        
        # Normal Fit Curve
        if std > 1e-12:
            x_fit = np.linspace(mu - 4*std, mu + 4*std, 100)
            ax_hist.plot(x_fit, stats.norm.pdf(x_fit, mu, std), 
                         color=spec['fit_color'], lw=2, label='Normal Fit')
        
        ax_hist.set_title(fr"{spec['title_hist']} ($\mu$={mu:.2e}, $\sigma$={std:.2e})")
        ax_hist.set_xlabel(spec['xlabel_hist'])
        ax_hist.grid(alpha=0.3)
        ax_hist.legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90 if n_rows > 1 else 0.85)

    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    save_path = os.path.join(save_folder, "postfit_residuals.png")
    plt.savefig(save_path, dpi=300)
    plt.close()