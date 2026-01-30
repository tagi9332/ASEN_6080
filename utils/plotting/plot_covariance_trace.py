import matplotlib.pyplot as plt
import numpy as np

def plot_covariance_trace(results_dict):
    """
    Plots the trace of the Position and Velocity covariance sub-matrices.
    Uses a log scale to visualize filter convergence.
    
    Parameters:
    -----------
    results_dict : dict
        Must contain 'P_hist' (Covariance history) and 'times'.
    """
    # 1. Extract Data
    times = np.array(results_dict['times'])
    P_hist = np.array(results_dict['P_hist'])
    
    # 2. Compute Traces
    # Trace of Position (upper left 3x3) = Var(x) + Var(y) + Var(z)
    trace_pos = np.trace(P_hist[:, 0:3, 0:3], axis1=1, axis2=2)
    
    # Trace of Velocity (lower right 3x3) = Var(vx) + Var(vy) + Var(vz)
    trace_vel = np.trace(P_hist[:, 3:6, 3:6], axis1=1, axis2=2)

    # 3. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- Position Trace ---
    ax1.semilogy(times, trace_pos, 'b-', linewidth=2, label='Trace($P_{pos}$)')
    ax1.set_ylabel(r'Position Variance ($km^2$)')
    ax1.set_title('Covariance Trace Evolution: Position')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend()
    
    # --- Velocity Trace ---
    ax2.semilogy(times, trace_vel, 'g-', linewidth=2, label='Trace($P_{vel}$)')
    ax2.set_ylabel(r'Velocity Variance ($km^2/s^2$)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Covariance Trace Evolution: Velocity')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    plt.savefig(f"{save_folder}/covariance_trace.png", dpi=300)
    plt.close()
