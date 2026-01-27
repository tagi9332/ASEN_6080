import numpy as np

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
    obs_matrix_dRs[1, 0:3] = -d_rhodot_dR
    
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

    # Lists to store data for plotting
    times_visible = []
    ranges = []
    range_rates = []
    station_ids = []

    print("--- Processing Measurements ---")

    # Loop through each time and ground station
    for i in range(num_times):
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

    # Add Gaussian noise to measurements (sigma_range=1m, sigma_range_rate=1mm/s)
    sigma_range = 1e-3  # km
    sigma_range_rate = 1e-6  # km/s

    ranges_noisy = np.array(ranges) + np.random.normal(0, sigma_range, len(ranges))
    range_rates_noisy = np.array(range_rates) + np.random.normal(0, sigma_range_rate, len(range_rates))

    # Save measurements to csv file
    measurements = np.column_stack((times_visible, ranges_noisy, range_rates_noisy, station_ids))

    # fmt defines the format for each column:
    # %.1f -> 1 decimal place (Time)
    # %.14f -> 14 decimal places (Range)
    # %.14f -> 14 decimal places (Range Rate)
    # %d -> integer (Station ID)
    np.savetxt('HW_1/measurements_noisy.csv', measurements, delimiter=',',
            header='Time(s),Range(km),Range_Rate(km/s),Station_ID', comments='',
            fmt=['%.1f', '%.14f', '%.14f', '%d'])
    
    # Save truth measurements without noise for reference
    measurements_truth = np.column_stack((times_visible, ranges, range_rates, station_ids))
    np.savetxt('HW_1/measurements_truth.csv', measurements_truth, delimiter=',',
            header='Time(s),Range(km),Range_Rate(km/s),Station_ID', comments='',
            fmt=['%.1f', '%.14f', '%.14f', '%d'])

    print("Measurements saved")
    