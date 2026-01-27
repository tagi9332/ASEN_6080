import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats

def plot_state_errors(times, state_errors, sigmas, 
                      state_labels=None, 
                      sigma_mult=3, 
                      scale_factor=1.0, 
                      title="State Estimation Errors"):
    """
    Plots state errors (Estimation - Truth) alongside covariance bounds.
    
    Parameters:
    - times: Array of time stamps.
    - state_errors: (N, 6) Array of state errors.
    - sigmas: (N, 6) Array of standard deviations (sqrt(diag(P))).
    - state_labels: List of 6 strings for subplot titles.
    - sigma_mult: Multiplier for covariance bounds (default 3 for 3-sigma).
    - scale_factor: float to convert units (e.g., 1e3 for km -> m).
    """
    if state_labels is None:
        state_labels = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    # Apply scaling
    errors_scaled = state_errors * scale_factor
    sigmas_scaled = sigmas * scale_factor

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Plot Error Scatter
        ax.scatter(times, errors_scaled[:, i], label='Est. Error', 
                   color='blue', s=2, alpha=0.7)
        
        # Plot Sigma Bounds
        bound = sigma_mult * sigmas_scaled[:, i]
        ax.plot(times, bound, 'r--', label=f'$\pm {sigma_mult}\sigma$')
        ax.plot(times, -bound, 'r--')

        # Styling
        ax.set_ylabel(f'Error {state_labels[i]}')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', framealpha=0.9)

    axes[-1].set_xlabel('Time (s)')
    axes[-2].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_residual_analysis(times, residuals, meas_noise_std, 
                           meas_labels=None, 
                           scale_factors=None):
    """
    Plots time-series residuals and their histograms with Gaussian fits.
    
    Parameters:
    - times: Array of timestamps.
    - residuals: (N, M) Array of post-fit residuals (M = num measurement types).
    - meas_noise_std: List/Array of length M with 1-sigma noise values (for bounds).
    - meas_labels: List of strings (e.g., ['Range (m)', 'Range-Rate (m/s)']).
    - scale_factors: List of floats to scale data (e.g., [1e3, 1e3]).
    """
    num_meas = residuals.shape[1]
    
    # Defaults
    if meas_labels is None:
        meas_labels = [f'Meas {i}' for i in range(num_meas)]
    if scale_factors is None:
        scale_factors = [1.0] * num_meas

    fig = plt.figure(figsize=(14, 5 * num_meas))
    gs = gridspec.GridSpec(num_meas, 2, width_ratios=[2, 1])

    print("\n--- Residual Statistics ---")
    
    for i in range(num_meas):
        # Apply scaling
        res_data = residuals[:, i] * scale_factors[i]
        sigma_bound = meas_noise_std[i] * scale_factors[i]
        
        # --- Time Series Plot (Left Column) ---
        ax_ts = fig.add_subplot(gs[i, 0])
        ax_ts.scatter(times, res_data, s=2, c='black', alpha=0.6, label='Residual')
        ax_ts.axhline(2 * sigma_bound, color='r', linestyle='--', label=r'$2\sigma$ Bound')
        ax_ts.axhline(-2 * sigma_bound, color='r', linestyle='--')
        
        ax_ts.set_ylabel(f'{meas_labels[i]}')
        ax_ts.grid(True, alpha=0.3)
        if i == 0:
            ax_ts.legend(loc='upper right')
        if i == num_meas - 1:
            ax_ts.set_xlabel('Time (s)')

        # --- Histogram Plot (Right Column) ---
        ax_hist = fig.add_subplot(gs[i, 1])
        
        # Stats
        mu, std = np.mean(res_data), np.std(res_data)
        rms = np.sqrt(np.mean(res_data**2))
        
        # Plot Hist
        ax_hist.hist(res_data, bins=40, density=True, alpha=0.6, 
                     color='skyblue' if i==0 else 'salmon', edgecolor='black')
        
        # Plot Fitted Gaussian
        x_range = np.linspace(mu - 4*std, mu + 4*std, 100)
        ax_hist.plot(x_range, stats.norm.pdf(x_range, mu, std), 
                     'b-', lw=2, label='Fit')
        
        ax_hist.set_title(rf'$\mu$={mu:.2e}, $\sigma$={std:.2e}')
        ax_hist.grid(True, alpha=0.3)
        if i == num_meas - 1:
            ax_hist.set_xlabel('Residual Magnitude')

        # Print stats to console
        print(f"{meas_labels[i]}: RMS = {rms:.6e}, Mean = {mu:.6e}, Std = {std:.6e}")

    plt.tight_layout()
    plt.show()