import numpy as np
from resources.constants import R_EARTH


def get_gs_eci_state(lat, lon, time, init_theta=np.deg2rad(122)):
    """
    Compute ground station ECI position and velocity from lat, lon, and time.

    :param lat: Latitude in radians
    :param lon: Longitude in radians
    :param time: Time in seconds
    :param init_theta: Initial Earth rotation angle in radians
    """

    OMEGA_EARTH = 2 * np.pi / 86400  # rad/s
      
    theta_total = init_theta + (OMEGA_EARTH * time) + lon
    cos_lat = np.cos(lat)
    
    Rs = np.array([
        R_EARTH * cos_lat * np.cos(theta_total),
        R_EARTH * cos_lat * np.sin(theta_total),
        R_EARTH * np.sin(lat)
    ])
    Vs = np.array([-OMEGA_EARTH * Rs[1], OMEGA_EARTH * Rs[0], 0.0])
    return Rs, Vs