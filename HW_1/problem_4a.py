import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Helper functions
def compute_gs_visibility(sc_positions, gs_positions, elevation_mask):
    """
    Compute visibility of ground stations from spacecraft positions based on elevation mask.
    
    Inputs:
        sc_positions : (N, 3) np.array - Spacecraft position vectors
        gs_positions : (N, 3) np.array - Ground station position vectors
        elevation_mask : float - Minimum elevation angle in radians
    Outputs:
        is_visible : (N,) np.array of bool - Visibility status for each ground station
        elevation_angles : (N,) np.array - Elevation angles in radians
    """

    # Allocate output visibility array
    is_visible = np.zeros(gs_positions.shape[0], dtype=bool)

    # Compute visibility and elevation angles for each ground station
    for i in range(gs_positions.shape[0]):
        rho_vec = sc_positions[i, :] - gs_positions[i, :]
        rho = np.linalg.norm(rho_vec)
        rho_hat = rho_vec / rho

        gs_z_hat = gs_positions[i, :] / np.linalg.norm(gs_positions[i, :])
        elevation_angle = np.arcsin(np.dot(rho_hat, gs_z_hat))

        if elevation_angle >= elevation_mask:
            is_visible[i] = True

    return is_visible, elevation_angle

def compute_eci_position(lat, lon, Re=6378.0):
    """
    Convert ground station latitude and longitude to ECI position vector.
    
    Inputs:
        lat : float - Latitude in radians
        lon : float - Longitude in radians
        Re : float - Earth's radius in km (default 6378.0 km)
    Outputs:
        gs_position : (3,) np.array - Ground station position vector in ECI frame
    """
    gs_x = Re * np.cos(lat) * np.cos(lon)
    gs_y = Re * np.cos(lat) * np.sin(lon)
    gs_z = Re * np.sin(lat)
    return np.array([gs_x, gs_y, gs_z])

def compute_gs_location_inertial(gs_positions_latlon, init_theta, times):
    """
    Compute ground station positions in inertial frame over time.
    
    Inputs:
        gs_positions : (M, 3) np.array - Ground station position vectors in ECEF frame
        init_theta : float - Initial rotation angle of the Earth in radians
        times : (N,) np.array - Time instances in seconds
    Outputs:
        gs_inertial_positions : (N, M, 3) np.array - Ground station positions in inertial frame
        gs_inertial_velocities : (N, M, 3) np.array - Ground station velocities in inertial frame

    """

    # Constant parameters
    omega_earth = 2 * np.pi / 86400.0  # rad/s
    Re = 6378.0  # km

    # Convert lat/lon to ECEF positions
    gs_ecef_positions = np.array([compute_eci_position(lat, lon, Re) for lat, lon in gs_positions_latlon])
    num_times = len(times)
    num_stations = gs_ecef_positions.shape[0]
    gs_inertial_positions = np.zeros((num_times, num_stations, 3))
    gs_inertial_velocities = np.zeros((num_times, num_stations, 3))

    for i, t in enumerate(times):
        theta = init_theta + omega_earth * t
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        R_ie = np.array([[cos_theta, -sin_theta, 0],
                         [sin_theta,  cos_theta, 0],
                         [0,          0,         1]])

        for j in range(num_stations):
            gs_inertial_positions[i, j, :] = R_ie @ gs_ecef_positions[j, :]

            # Velocity due to Earth's rotation
            omega_vec = np.array([0, 0, omega_earth])
            gs_inertial_velocities[i, j, :] = np.cross(omega_vec, gs_inertial_positions[i, j, :])
    return gs_inertial_positions, gs_inertial_velocities

def range_obs_partials(R, V, Rs, Vs):
    """
    Compute the observation partials for range and range-rate measurements.
    
    Inputs:
        R : (3,) np.array - Spacecraft position vector
        V : (3,) np.array - Spacecraft velocity vector
        Rs : (3,) np.array - Ground station position vector
        Vs : (3,) np.array - Ground station velocity vector
    """

    # Relative position and velocity
    rho_vec = R - Rs
    v_rel = V - Vs
    
    
    # Range and range rate
    rho = np.linalg.norm(rho_vec)
    rho_dot = np.dot(rho_vec, v_rel) / rho
    
    # Range partials
    d_rho_dR = rho_vec / rho
    d_rho_dV = np.zeros(3)

    # Range rate partials
    d_rhodot_dV = d_rho_dR  
    d_rhodot_dR = (rho * v_rel - rho_dot * rho_vec) / (rho**2)

    # Partials w.r.t SC position and velocity (2x6 measurement jacobian)
    obs_matrix_dR = np.zeros((2, 6))
    obs_matrix_dR[0, 0:3] = d_rho_dR
    obs_matrix_dR[1, 0:3] = d_rhodot_dR
    obs_matrix_dR[1, 3:6] = d_rhodot_dV

    # Partials w.r.t. GS position and velocity (2x3 measurement jacobian)
    obs_matrix_dRs = np.zeros((2, 3))
    obs_matrix_dRs[0, 0:3] = -d_rho_dR
    obs_matrix_dRs[1, 0:3] = -d_rhodot_dR\
    
    # Output Dictionary of observation partials
    obs_measurements = np.array([rho, rho_dot])
    obs_matrix = {
        'wrt_R': obs_matrix_dR,
        'wrt_Rs': obs_matrix_dRs
    }

    return obs_measurements, obs_matrix

if __name__ == "__main__":
    # Constants
    Re = 6378.0  
    mu = 398600.4418  
    omega_earth = 2 * np.pi / 86400.0  
    theta_0 = np.deg2rad(122) 
    elevation_mask = np.deg2rad(10)  

    # Ground station locations (lat, lon) in radians
    stations = np.deg2rad(np.array([
        [-35.398333, 148.981944],  # Station 1
        [40.427222, 355.749444],   # Station 2 
        [35.247164, 243.205]       # Station 3 
    ]))

    # Load spacecraft trajectory data
    data = np.loadtxt('HW_1/HW1_truth.csv', delimiter=' ')
    times = data[:, 0]
    sc_positions = data[:, 1:4]
    sc_velocities = data[:, 4:7]

    # Compute ground station positions and velocities in inertial frame
    gs_inertial_positions, gs_inertial_velocities = compute_gs_location_inertial(
        stations, theta_0, times)
    
    # Process measurements
    num_times = len(times)
    num_stations = stations.shape[0]

    # Measurement indices (every 10 seconds)
    indices = list(range(0, num_times, 10))
    if (num_times - 1) not in indices:
        indices.append(num_times - 1)

    # Lists to store data for plotting
    times_visible = []
    ranges = []
    range_rates = []
    station_ids = []

    print("--- Processing Measurements ---")

    # Loop through each time and ground station
    for i in indices:
        for j in range(num_stations):
            sc_pos = sc_positions[i, :]
            sc_vel = sc_velocities[i, :]
            gs_pos = gs_inertial_positions[i, j, :]
            gs_vel = gs_inertial_velocities[i, j, :]

            is_visible, el_angle = compute_gs_visibility(
                sc_pos.reshape(1, 3), gs_pos.reshape(1, 3), elevation_mask)

            if is_visible[0]:
                obs_measurements, obs_matrix = range_obs_partials(
                    sc_pos, sc_vel, gs_pos, gs_vel)
                
                
                # Store for plotting
                times_visible.append(times[i])
                ranges.append(obs_measurements[0])
                range_rates.append(obs_measurements[1])
                station_ids.append(j + 1)

    # Fixed Print Statements
    if times_visible:
        print(f"First measurement at: {times_visible[0]} s (Station {station_ids[0]})")
        print(f"Range: {ranges[0]:.4f} km, Range-rate: {range_rates[0]:.6f} km/s, Elevation: {np.rad2deg(el_angle):.4f} deg, Station ID: {station_ids[0]}")
        print(f"Last measurement at: {times_visible[-1]} s (Station {station_ids[-1]})")
        print(f"Range: {ranges[-1]:.4f} km, Range-rate: {range_rates[-1]:.6f} km/s, Elevation: {np.rad2deg(el_angle):.4f} deg, Station ID: {station_ids[-1]}")
    else:
        print("No stations were visible during the pass.")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Define your specific color list: Orange, Blue, Green
    # These are high-contrast hex codes for a professional look
    custom_colors = [
        '#ff7f0e', # Orange
        '#1f77b4', # Blue
        '#2ca02c'  # Green
    ]
    custom_cmap = ListedColormap(custom_colors)

    # Plot Range
    # c=station_ids maps Station 1 -> Orange, 2 -> Blue, 3 -> Green
    scatter1 = ax1.scatter(times_visible, ranges, c=station_ids, cmap=custom_cmap, s=10)
    ax1.set_ylabel('Range (km)')
    ax1.set_title('Station Observations over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot Range-Rate
    scatter2 = ax2.scatter(times_visible, range_rates, c=station_ids, cmap=custom_cmap, s=10)
    ax2.set_ylabel('Range-Rate (km/s)')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add the legend
    # The legend_elements() handles the mapping of the discrete colors back to the IDs
    handles, labels = scatter1.legend_elements()
    ax1.legend(handles, ["Station 1", "Station 2", "Station 3"], 
            title="Ground Stations", loc='best', frameon=True)

    plt.tight_layout()

    # Save the figure (mimicking MATLAB's exportgraphics)
    plt.savefig('Observation_Plots.pdf', dpi=300)

    plt.show()

    # Plot elevation angles wrt each station at measurement times
    plt.figure(figsize=(10, 5))
    for j in range(num_stations):
        el_angles = []
        for i in indices:
            sc_pos = sc_positions[i, :]
            gs_pos = gs_inertial_positions[i, j, :]
            _, el_angle = compute_gs_visibility(
                sc_pos.reshape(1, 3), gs_pos.reshape(1, 3), elevation_mask)
            el_angles.append(np.rad2deg(el_angle))
        plt.plot(times[indices], el_angles, label=f'Station {j+1}')
    plt.axhline(np.rad2deg(elevation_mask), color='r', linestyle='--', label='Elevation Mask')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Elevation Angle (degrees)')
    plt.title('Elevation Angles over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Elevation_Angles.pdf', dpi=300)
    plt.show()

    # Part D: Doppler shift data visualization
    c = 299792.458  # km/s
    fT_ref = 8.44e9  # Hz

    # Convert to Range Units (RU) and Doppler Shift

    ranges = np.array(ranges)
    range_rates = np.array(range_rates)

    range_ru = (221/749 * ranges) / (c / fT_ref)
    doppler_shifts = -2 * range_rates * fT_ref / c

    # Plot range RU and Doppler Shift
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.scatter(times_visible, range_ru, c=station_ids, cmap=custom_cmap, s=10)
    ax1.set_ylabel('Range Units (RU)')
    ax1.set_title('Range Units and Doppler Shift over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.scatter(times_visible, doppler_shifts/1e6, c=station_ids, cmap=custom_cmap, s=10)
    ax2.set_ylabel('Doppler Shift (MHz)')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Range_RU_Doppler_Shift.pdf', dpi=300)
    plt.show()

    # Part d: Add Gaussian noise to range-rate measurements
    sigma_noise = 0.5 / 1e6  # 0.5 mm/s in km/s
    noisy_range_rates = range_rates + sigma_noise * np.random.randn(len(range_rates)) 
    residuals = (noisy_range_rates - range_rates) * 1e6  # Convert to mm/s

    # --- Part D: Noise Analysis ---
    plt.figure(figsize=(10, 8))

    # Subplot 1: Comparing Original vs Noisy Range-Rate
    plt.subplot(2, 1, 1)
    plt.scatter(np.array(times_visible)/3600, range_rates, c='k', label='Original (Truth)', s=10, alpha=0.7)
    plt.scatter(np.array(times_visible)/3600, noisy_range_rates, c=station_ids, cmap=custom_cmap, s=5, label='Noisy Measurements', alpha=0.6)
    plt.ylabel('Range-Rate (km/s)')
    plt.title('Comparison of Truth vs. Noisy Range-Rate Measurements')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Residuals (Difference)
    plt.subplot(2, 1, 2)
    plt.scatter(np.array(times_visible)/3600, residuals, c=station_ids, cmap=custom_cmap, s=8, alpha=0.7)
    # Plotting the 3-sigma bounds
    plt.axhline(3 * (sigma_noise * 1e6), color='r', linestyle='--', label=r'$\pm 3\sigma$ Bound')
    plt.axhline(-3 * (sigma_noise * 1e6), color='r', linestyle='--')

    plt.ylabel('Residuals (mm/s)')
    plt.xlabel('Time (hours)')
    plt.title('Measurement Residuals (Noisy - Original)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('HW1_Noise_Comparison.pdf', dpi=300)
    plt.show()