import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def plot_postfit_residuals(results_dict, n_sigma=3):
    """
    Plotting function for post-fit residuals and their Gaussian distributions.
    
    - Handles 'results_units' flag (m vs km).
    - Assumes raw residuals in results_dict are in METERS and M/S.
    """
    
    # --- 1. Setup Units & Scaling ---
    # Get user preference ('m' or 'km')
    unit_pref = results_dict.get('options', {}).get('results_units', 'm')

    if unit_pref == 'km':
        scale = 1e-3
        dist_unit = 'km'
        rate_unit = 'km/s'
    else:
        scale = 1.0
        dist_unit = 'm'
        rate_unit = 'm/s'

    # --- 2. Extract & Scale Data ---
    times = np.array(results_dict['times'])
    postfit_residuals = np.array(results_dict['postfit_residuals'])

    # Range (Column 0) - Apply Scale
    range_res = postfit_residuals[:, 0] * scale
    mu_r, std_r = np.mean(range_res), np.std(range_res)

    # Range-Rate (Column 1) - Apply Scale
    rr_res = postfit_residuals[:, 1] * scale
    mu_rr, std_rr = np.mean(rr_res), np.std(rr_res)

    # --- 3. Setup Plot ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Filter Residual Analysis: Time History and Error Distributions', fontsize=18)

    # ==========================================
    # ROW 1: RANGE
    # ==========================================

    # Col 1: Time History
    axs[0, 0].scatter(times, range_res, s=2, c='black', label='Post-fit Residual')
    
    # Plot empirical sigma bounds
    axs[0, 0].axhline(y=n_sigma * std_r, color='r', linestyle='--', alpha=0.8, label=fr'{n_sigma}$\sigma$ (Empirical)')
    axs[0, 0].axhline(y=-n_sigma * std_r, color='r', linestyle='--', alpha=0.8)
    
    axs[0, 0].set_ylabel(f'Range Residual ({dist_unit})')
    axs[0, 0].set_title('Range Residuals vs Time')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(loc='upper right')

    # Col 2: Histogram
    if std_r > 1e-12:
        range_filt = range_res[np.abs(range_res - mu_r) <= 6 * std_r]
    else:
        range_filt = range_res

    axs[0, 1].hist(range_filt, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Residual Dist')
    
    # Normal Fit
    if std_r > 1e-12:
        x_r = np.linspace(mu_r - 4*std_r, mu_r + 4*std_r, 100)
        axs[0, 1].plot(x_r, stats.norm.pdf(x_r, mu_r, std_r), 'r-', lw=2, label='Normal Fit')
    
    axs[0, 1].set_title(fr'Range PDF ($\mu$={mu_r:.2e}, $\sigma$={std_r:.2e})')
    axs[0, 1].set_xlabel(f'Residual ({dist_unit})')
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend(loc='upper right')

    # ==========================================
    # ROW 2: RANGE-RATE
    # ==========================================

    # Col 1: Time History
    axs[1, 0].scatter(times, rr_res, s=2, c='black', label='Post-fit Residual')
    
    # Plot empirical sigma bounds
    axs[1, 0].axhline(y=n_sigma * std_rr, color='r', linestyle='--', alpha=0.8)
    axs[1, 0].axhline(y=-n_sigma * std_rr, color='r', linestyle='--', alpha=0.8)
    
    axs[1, 0].set_ylabel(f'Range-Rate Residual ({rate_unit})')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_title('Range-Rate Residuals vs Time')
    axs[1, 0].grid(True, alpha=0.3)

    # Col 2: Histogram
    if std_rr > 1e-12:
        rr_filt = rr_res[np.abs(rr_res - mu_rr) <= 6 * std_rr]
    else:
        rr_filt = rr_res

    axs[1, 1].hist(rr_filt, bins=40, density=True, alpha=0.6, color='salmon', edgecolor='black', label='Residual Dist')
    
    # Normal Fit
    if std_rr > 1e-12:
        x_rr = np.linspace(mu_rr - 4*std_rr, mu_rr + 4*std_rr, 100)
        axs[1, 1].plot(x_rr, stats.norm.pdf(x_rr, mu_rr, std_rr), 'b-', lw=2, label='Normal Fit')
    
    axs[1, 1].set_title(fr'Range-Rate PDF ($\mu$={mu_rr:.2e}, $\sigma$={std_rr:.2e})')
    axs[1, 1].set_xlabel(f'Residual ({rate_unit})')
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend(loc='upper right')

    plt.tight_layout()
    
    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    save_path = os.path.join(save_folder, "postfit_residuals.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
