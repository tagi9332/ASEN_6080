import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_covariance_ellipsoid(results_dict, sigma=3):
    """
    Plots the 3D position covariance ellipsoid for the FINAL state estimate.
    
    - Uses 'cov_hist' (or 'P_hist') from results_dict.
    - Adapts to 'results_units' flag (scales m -> km if needed).
    - Saves to 'save_folder'.
    
    Args:
        results_dict (dict): Dictionary containing filter results.
        sigma (int): Sigma level for the ellipsoid (default=3).
    """
    
    # --- 1. Setup Units & Scaling ---
    unit_pref = results_dict.get('results_units', 'km')

    if unit_pref == 'km':
        scale_factor = 1e-3  # Convert meters to km
    else:
        scale_factor = 1.0   # Keep in meters

    # --- 2. Extract Data ---
    # Handle potential naming differences
    if 'cov_hist' in results_dict:
        P_hist = np.array(results_dict['cov_hist'])
    elif 'P_hist' in results_dict:
        P_hist = np.array(results_dict['P_hist'])
    else:
        print("Warning: No covariance history found. Skipping ellipsoid plot.")
        return

    if len(P_hist) == 0:
        return

    # Get the FINAL covariance matrix
    P_final = P_hist[-1]
    
    # Extract 3x3 Position Covariance (Top-Left block) in [m^2]
    P_pos_m2 = P_final[0:3, 0:3]
    
    # Convert Covariance to Output Units:
    # Var_new = Var_old * (scale)^2
    P_pos = P_pos_m2 * (scale_factor ** 2)

    # --- 3. Compute Eigenvalues and Eigenvectors ---
    # eig_vals are variances along principal axes
    eig_vals, eig_vecs = np.linalg.eigh(P_pos)

    # Sort descending for consistency
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Calculate radii (sigma * std_dev)
    # std_dev = sqrt(variance)
    radii = sigma * np.sqrt(np.abs(eig_vals))

    # --- 4. Generate Ellipsoid Surface Data ---
    # Parametric equations for a unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    # Reshape for transformation (N_points x 3)
    sphere_points = np.stack([x_sphere, y_sphere, z_sphere], axis=2)
    
    # Transform: Rotate and Scale
    # Point_new = R * diag(radii) * Point_old
    ellipsoid_points = np.zeros_like(sphere_points)
    
    for i in range(len(u)):
        for j in range(len(v)):
            point = sphere_points[i, j, :]
            # Scale by radii then Rotate
            transformed = eig_vecs @ (point * radii)
            ellipsoid_points[i, j, :] = transformed

    X = ellipsoid_points[:, :, 0]
    Y = ellipsoid_points[:, :, 1]
    Z = ellipsoid_points[:, :, 2]

    # --- 5. Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, color='c', 
                           alpha=0.3, linewidth=0.1, edgecolors='k')

    # Plot Principal Axes
    for i in range(3):
        axis_vec = eig_vecs[:, i] * radii[i]
        # Draw line from center to surface
        ax.plot([0, axis_vec[0]], [0, axis_vec[1]], [0, axis_vec[2]], 
                'r-', lw=2, label=f'Eigenvector {i+1}' if i==0 else "")
        ax.plot([0, -axis_vec[0]], [0, -axis_vec[1]], [0, -axis_vec[2]], 
                'r-', lw=2)

    # Formatting
    ax.set_xlabel(f'X Error ({unit_pref})')
    ax.set_ylabel(f'Y Error ({unit_pref})')
    ax.set_zlabel(f'Z Error ({unit_pref})')
    ax.set_title(f'Final Position Covariance Ellipsoid ({sigma}$\sigma$)')

    # Enforce Equal Axis Limits for proper aspect ratio
    max_radius = radii.max()
    limit = max_radius * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    # Clean background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # --- 6. Save ---
    save_folder = results_dict.get('save_folder', '.')
    filename = f'final_covariance_ellipsoid_{sigma}sigma.png'
    file_path = os.path.join(save_folder, filename)
    
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()
    
