import matplotlib.pyplot as plt
import numpy as np
import os

def plot_covariance_trace(results_dict):
    """
    Plots the trace of the Position and Velocity covariance sub-matrices.
    Uses a log scale to visualize filter convergence.
    
    - Handles 'results_units' flag (m vs km).
    - Correctly squares the scale factor for Variance (m^2 -> km^2).
    """
    
    # --- 1. Setup Units & Scaling ---
    # Get user preference ('m' or 'km')
    unit_pref = results_dict.get('options', {}).get('results_units', 'm')

    if unit_pref == 'km':
        linear_scale = 1
        # Variance scales as length squared!
        var_scale = linear_scale ** 2 
        
        pos_unit = r'km^2'
        vel_unit = r'km^2/s^2'
    else:
        var_scale = 1.0
        pos_unit = r'm^2'
        vel_unit = r'm^2/s^2'

    # --- 2. Extract Data ---
    times = np.array(results_dict['times'])
    P_hist = np.array(results_dict['P_hist']) # Shape (N, 6, 6) in SI units (m^2)
    
    # --- 3. Compute Traces & Scale ---
    # Trace of Position (upper left 3x3) = Var(x) + Var(y) + Var(z)
    raw_trace_pos = np.trace(P_hist[:, 0:3, 0:3], axis1=1, axis2=2)
    trace_pos = raw_trace_pos * var_scale
    
    # Trace of Velocity (lower right 3x3) = Var(vx) + Var(vy) + Var(vz)
    raw_trace_vel = np.trace(P_hist[:, 3:6, 3:6], axis1=1, axis2=2)
    trace_vel = raw_trace_vel * var_scale

    # --- 4. Setup Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Position Trace
    ax1.semilogy(times, trace_pos, 'b.', linewidth=2, label=r'Trace($P_{pos}$)')
    ax1.set_ylabel(f'Position Variance ({pos_unit})')
    ax1.set_title('Covariance Trace Evolution: Position')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)
    ax1.legend()
    
    # Velocity Trace
    ax2.semilogy(times, trace_vel, 'g.', linewidth=2, label=r'Trace($P_{vel}$)')
    ax2.set_ylabel(f'Velocity Variance ({vel_unit})')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Covariance Trace Evolution: Velocity')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save to file
    save_folder = results_dict.get('save_folder', '.')
    save_path = os.path.join(save_folder, "covariance_trace.png")
    plt.savefig(save_path, dpi=300)
    plt.close()