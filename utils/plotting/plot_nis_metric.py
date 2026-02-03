import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2

def plot_nis_metric(results_dict):
    """
    Plots the Normalized Innovation Squared (NIS).
    Includes 95% Chi-Squared confidence bounds.
    
    - NIS is unitless, so no 'results_units' scaling is applied.
    - Updated to use actual Simulation Time on x-axis.
    """
    save_folder = results_dict.get('save_folder', 'results')
    
    # --- 1. Safety Check ---
    if 'nis_hist' not in results_dict:
        print("[Plotting] Warning: 'nis_hist' missing. Cannot plot NIS.")
        return

    nis = np.array(results_dict['nis_hist'])
    times = np.array(results_dict['times'])
    
    # --- 2. Setup Statistics ---
    # Degrees of Freedom = Number of Measurements per step
    # Assuming Range + Range-Rate = 2 DOF
    k = 2 
    
    # Chi-Square 95% Confidence Interval bounds
    # We want the probability mass to be 0.95 inside the bounds.
    # Usually for NIS we care about the upper bound (consistency check).
    upper_bound = chi2.ppf(0.975, df=k)
    lower_bound = chi2.ppf(0.025, df=k)
    
    # --- 3. Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Plot NIS
    plt.scatter(times, nis, s=15, c='b', alpha=0.6, label='NIS Sample')
    
    # Plot Bounds
    plt.axhline(y=upper_bound, color='r', linestyle='--', label='95% Bound') # type: ignore
    plt.axhline(y=lower_bound, color='r', linestyle='--') # type: ignore
    plt.axhline(y=k, color='g', linestyle='-', alpha=0.5, label=f'Expected Mean (k={k})')

    plt.yscale('log') # Log scale handles outliers better
    plt.xlabel('Time (s)')
    plt.ylabel(r'NIS ($\epsilon_{\nu}$)')
    plt.title('Normalized Innovation Squared (Filter Health)')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)

    fname = os.path.join(save_folder, 'nis_metric.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
