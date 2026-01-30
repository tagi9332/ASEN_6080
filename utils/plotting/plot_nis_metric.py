import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2

def plot_nis_metric(results_dict):
    """
    Plots the Normalized Innovation Squared (NIS).
    Includes 95% Chi-Squared confidence bounds.
    """
    save_folder = results_dict.get('save_folder', 'results')
    
    if 'nis_hist' not in results_dict:
        print("Warning: 'nis_hist' missing. Cannot plot NIS.")
        return

    nis = results_dict['nis_hist']
    n_steps = len(nis)
    t = np.arange(n_steps)
    
    # Degrees of Freedom = Number of Measurements per step
    # Assuming Range + Range-Rate = 2 DOF
    k = 2 
    
    # Chi-Square 95% Confidence Interval bounds
    upper_bound = chi2.ppf(0.975, df=k)
    lower_bound = chi2.ppf(0.025, df=k)
    
    plt.figure(figsize=(10, 6))
    
    # Plot NIS
    plt.scatter(t, nis, s=15, c='b', alpha=0.6, label='NIS Sample')
    
    # Plot Bounds
    plt.axhline(y=upper_bound.item(), color='r', linestyle='--', label='95% Bound')
    plt.axhline(y=lower_bound.item(), color='r', linestyle='--')
    plt.axhline(y=k, color='g', linestyle='-', alpha=0.5, label=f'Expected Mean (k={k})')

    plt.yscale('log') # Log scale handles outliers better
    plt.xlabel('Measurement Step')
    plt.ylabel('NIS ($\epsilon_{v}$)')
    plt.title('Normalized Innovation Squared (Filter Health)')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    fname = os.path.join(save_folder, 'nis_test.png')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()