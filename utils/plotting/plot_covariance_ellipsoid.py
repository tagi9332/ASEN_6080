import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_covariance_ellipsoid(results_dict):
    """
    Plots nested 1, 2, and 3-sigma 3D Position AND Velocity covariance ellipsoids.
    
    - 1 and 2-sigma: Smooth, translucent surfaces.
    - 3-sigma: Translucent surface with a GREY POINT CLOUD boundary.
    - Titles: Include statistics (max sigma).
    """
    
    # --- 1. Setup Units & Scaling ---
    unit_pref = results_dict.get('results_units', 'km')

    if unit_pref == 'km':
        scale_factor = 1
        pos_unit = 'km'
        vel_unit = 'km/s'
    else:
        scale_factor = 1.0
        pos_unit = 'm'
        vel_unit = 'm/s'

    # --- 2. Extract Data ---
    if 'cov_hist' in results_dict:
        P_hist = np.array(results_dict['cov_hist'])
    elif 'P_hist' in results_dict:
        P_hist = np.array(results_dict['P_hist'])
    else:
        print("Warning: No covariance history found. Skipping ellipsoid plot.")
        return

    if len(P_hist) == 0:
        return

    P_final = P_hist[-1]
    
    # Extract and Scale
    P_pos = P_final[0:3, 0:3] * (scale_factor ** 2)
    P_vel = P_final[3:6, 3:6] * (scale_factor ** 2)

    # --- Helper: Generate Ellipsoid Data ---
    def get_ellipsoid_points(P_sub, sigma_val, num_points=1000):
        """
        Returns scattered XYZ points on the surface of the ellipsoid 
        and the principal axes vectors.
        """
        # Eigen decomposition
        eig_vals, eig_vecs = np.linalg.eigh(P_sub)
        
        # Sort descending
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        # Radii
        radii = sigma_val * np.sqrt(np.abs(eig_vals))
        
        # Generate Random Points on Unit Sphere
        vecs = np.random.randn(3, num_points)
        norms = np.linalg.norm(vecs, axis=0)
        vecs /= norms
        
        # Transform to Ellipsoid
        scaled_vecs = vecs * radii[:, np.newaxis] 
        ellipsoid_points = eig_vecs @ scaled_vecs
        
        return ellipsoid_points, eig_vecs, radii, np.sqrt(np.abs(eig_vals))

    def get_ellipsoid_surface(P_sub, sigma_val):
        """Returns structured grid for plotting smooth surfaces."""
        eig_vals, eig_vecs = np.linalg.eigh(P_sub)
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        radii = sigma_val * np.sqrt(np.abs(eig_vals))
        
        u = np.linspace(0.0, 2.0 * np.pi, 25)
        v = np.linspace(0.0, np.pi, 25)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones_like(u), np.cos(v))
        
        ellipsoid_points = np.zeros((len(u), len(v), 3))
        for i in range(len(u)):
            for j in range(len(v)):
                point = np.array([x_sphere[i,j], y_sphere[i,j], z_sphere[i,j]])
                ellipsoid_points[i,j,:] = eig_vecs @ (point * radii)
        return ellipsoid_points

    # --- 3. Plotting Setup ---
    fig = plt.figure(figsize=(16, 7))
    sigmas = [1, 2, 3]
    alphas = {1: 0.3, 2: 0.15, 3: 0.05} 

    # --- Subplot 1: Position ---
    ax1 = fig.add_subplot(121, projection='3d')
    max_pos_r = 0
    pos_std_devs = []

    for s in sigmas:
        surf = get_ellipsoid_surface(P_pos, s)
        ax1.plot_surface(surf[:,:,0], surf[:,:,1], surf[:,:,2], 
                         rstride=1, cstride=1, color='blue', 
                         alpha=alphas[s], linewidth=0, shade=True)
        
        if s == 3:
            pts, vecs, radii, std_devs = get_ellipsoid_points(P_pos, s, num_points=800)
            max_pos_r = radii.max()
            pos_std_devs = std_devs # 1-sigma standard deviations along principal axes
            
            ax1.scatter(pts[0,:], pts[1,:], pts[2,:], c='grey', s=2, alpha=0.6, depthshade=False)
            
            for i in range(3):
                v = vecs[:, i] * radii[i]
                ax1.plot([-v[0], v[0]], [-v[1], v[1]], [-v[2], v[2]], 'b-', lw=2, zorder=10)

    ax1.set_xlabel(f'X ({pos_unit})')
    ax1.set_ylabel(f'Y ({pos_unit})')
    ax1.set_zlabel(f'Z ({pos_unit})')
    
    # Title with Stats
    title_str_pos = (f"Position Covariance (1, 2, 3$\sigma$)\n"
                     f"$\sigma_{{max}}$: {pos_std_devs[0]:.4f} {pos_unit} | "
                     f"$\sigma_{{min}}$: {pos_std_devs[2]:.4f} {pos_unit}")
    ax1.set_title(title_str_pos)
    
    limit_p = max_pos_r * 1.2
    ax1.set_xlim(-limit_p, limit_p)
    ax1.set_ylim(-limit_p, limit_p)
    ax1.set_zlim(-limit_p, limit_p)

    # --- Subplot 2: Velocity ---
    ax2 = fig.add_subplot(122, projection='3d')
    max_vel_r = 0
    vel_std_devs = []

    for s in sigmas:
        surf = get_ellipsoid_surface(P_vel, s)
        ax2.plot_surface(surf[:,:,0], surf[:,:,1], surf[:,:,2], 
                         rstride=1, cstride=1, color='green', 
                         alpha=alphas[s], linewidth=0, shade=True)
        
        if s == 3:
            pts, vecs, radii, std_devs = get_ellipsoid_points(P_vel, s, num_points=800)
            max_vel_r = radii.max()
            vel_std_devs = std_devs
            
            ax2.scatter(pts[0,:], pts[1,:], pts[2,:], c='grey', s=2, alpha=0.6, depthshade=False)
            
            for i in range(3):
                v = vecs[:, i] * radii[i]
                ax2.plot([-v[0], v[0]], [-v[1], v[1]], [-v[2], v[2]], 'g-', lw=2, zorder=10)

    ax2.set_xlabel(f'Vx ({vel_unit})')
    ax2.set_ylabel(f'Vy ({vel_unit})')
    ax2.set_zlabel(f'Vz ({vel_unit})')
    
    # Title with Stats
    title_str_vel = (f"Velocity Covariance (1, 2, 3$\sigma$)\n"
                     f"$\sigma_{{max}}$: {vel_std_devs[0]:.4e} {vel_unit} | "
                     f"$\sigma_{{min}}$: {vel_std_devs[2]:.4e} {vel_unit}")
    ax2.set_title(title_str_vel)

    limit_v = max_vel_r * 1.2
    ax2.set_xlim(-limit_v, limit_v)
    ax2.set_ylim(-limit_v, limit_v)
    ax2.set_zlim(-limit_v, limit_v)

    # --- 4. Save ---
    save_folder = results_dict.get('save_folder', '.')
    filename = 'final_covariance_ellipsoids_stats.png'
    file_path = os.path.join(save_folder, filename)
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()