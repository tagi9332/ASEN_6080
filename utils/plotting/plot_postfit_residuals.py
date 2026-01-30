import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def plot_postfit_residuals(results_dict, n_sigma=3):
    """
    Plotting function for post-fit residuals and their Gaussian distributions.
    Fixed: Added labels to histograms to prevent legend warnings.
    """
    # 1. Extract Data
    times = np.array(results_dict['times'])
    postfit_residuals = np.array(results_dict['postfit_residuals'])

    # 2. Setup Plot
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Filter Residual Analysis: Time History and Error Distributions', fontsize=18)

    # --- Pre-calculations ---
    
    # Range (Column 0)
    range_res = postfit_residuals[:, 0] * 1e3  # km -> m
    mu_r = np.mean(range_res)
    std_r = np.std(range_res) 

    # Range-Rate (Column 1)
    rr_res = postfit_residuals[:, 1] * 1e3     # km/s -> m/s
    mu_rr = np.mean(rr_res)
    std_rr = np.std(rr_res)   

    # --- ROW 1, COL 1: Range Residual Time History ---
    axs[0, 0].scatter(times, range_res, s=2, c='black', label='Post-fit Residual')
    
    # Plot empirical sigma bounds
    axs[0, 0].axhline(y=n_sigma * std_r, color='r', linestyle='--', alpha=0.8, label=fr'{n_sigma}$\sigma$ (Empirical)')
    axs[0, 0].axhline(y=-n_sigma * std_r, color='r', linestyle='--', alpha=0.8)
    
    axs[0, 0].set_ylabel('Range Residual (m)')
    axs[0, 0].set_title('Range Residuals vs Time')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(loc='upper right')

    # --- ROW 1, COL 2: Range Residual Histogram ---
    if std_r > 1e-9:
        range_filt = range_res[np.abs(range_res - mu_r) <= 6 * std_r]
    else:
        range_filt = range_res

    # [FIX] Added label='Residual Dist'
    axs[0, 1].hist(range_filt, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Residual Dist')
    
    # Normal Fit
    if std_r > 1e-9:
        x_r = np.linspace(mu_r - 4*std_r, mu_r + 4*std_r, 100)
        axs[0, 1].plot(x_r, stats.norm.pdf(x_r, mu_r, std_r), 'r-', lw=2, label='Normal Fit')
    
    axs[0, 1].set_title(fr'Range PDF ($\mu$={mu_r:.2e}, $\sigma$={std_r:.2e})')
    axs[0, 1].set_xlabel('Residual (m)')
    axs[0, 1].grid(alpha=0.3)
    axs[0, 1].legend(loc='upper right')

    # --- ROW 2, COL 1: Range-Rate Residual Time History ---
    axs[1, 0].scatter(times, rr_res, s=2, c='black', label='Post-fit Residual')
    
    # Plot empirical sigma bounds
    axs[1, 0].axhline(y=n_sigma * std_rr, color='r', linestyle='--', alpha=0.8)
    axs[1, 0].axhline(y=-n_sigma * std_rr, color='r', linestyle='--', alpha=0.8)
    
    axs[1, 0].set_ylabel('Range-Rate Residual (m/s)')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_title('Range-Rate Residuals vs Time')
    axs[1, 0].grid(True, alpha=0.3)

    # --- ROW 2, COL 2: Range-Rate Residual Histogram ---
    if std_rr > 1e-9:
        rr_filt = rr_res[np.abs(rr_res - mu_rr) <= 6 * std_rr]
    else:
        rr_filt = rr_res

    # [FIX] Added label='Residual Dist'
    axs[1, 1].hist(rr_filt, bins=40, density=True, alpha=0.6, color='salmon', edgecolor='black', label='Residual Dist')
    
    # Normal Fit
    if std_rr > 1e-9:
        x_rr = np.linspace(mu_rr - 4*std_rr, mu_rr + 4*std_rr, 100)
        axs[1, 1].plot(x_rr, stats.norm.pdf(x_rr, mu_rr, std_rr), 'b-', lw=2, label='Normal Fit')
    
    axs[1, 1].set_title(fr'Range-Rate PDF ($\mu$={mu_rr:.2e}, $\sigma$={std_rr:.2e})')
    axs[1, 1].set_xlabel('Residual (m/s)')
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend(loc='upper right')

    plt.tight_layout()
    
    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    save_path = os.path.join(save_folder, "postfit_residuals.png")
    plt.savefig(save_path, dpi=300)
    plt.close()