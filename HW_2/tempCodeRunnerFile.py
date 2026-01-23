import numpy as np

def compute_gs_eci_state(lat, lon, time, init_theta=122, Re=6378.0):
    """
    Computes the ECI position and velocity of a ground station.
    """

    omega_earth = 7.2921159e-5  # rad/s

    # Convert intial theta to radians
    init_theta = np.deg2rad(init_theta)

    # -------------------------
    # 1. Ground station in ECEF
    # -------------------------
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)

    r_ecef = np.array([
        Re * cos_lat * np.cos(lon),
        Re * cos_lat * np.sin(lon),
        Re * sin_lat
    ])

    # -------------------------
    # 2. Rotate ECEF → ECI
    # -------------------------
    theta = init_theta + omega_earth * time
    c, s = np.cos(theta), np.sin(theta)

    R_ie = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])

    Rs = R_ie @ r_ecef

    # -------------------------
    # 3. Velocity: ω × r
    # -------------------------
    omega_vec = np.array([0.0, 0.0, omega_earth])
    Vs = np.cross(omega_vec, Rs)

    return Rs, Vs


if __name__ == "__main__":
    lat_rad = np.deg2rad(40.427222)
    lon_rad = np.deg2rad(355.749444)
    time_s = 0

    Rs, Vs = compute_gs_eci_state(lat_rad, lon_rad, time_s)

    print("Position (Rs):", Rs)
    print("Velocity (Vs):", Vs)