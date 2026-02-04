import numpy as np
from resources.constants import OMEGA_EARTH

def get_initial_station_eci(station_ecef, t_offset=0):
    """
    Rotates ECEF coordinates to ECI based on the problem's theta formula.
    theta = omega_earth * t
    """
    theta = OMEGA_EARTH * t_offset
    c = np.cos(theta)
    s = np.sin(theta)
    
    R_ecef2eci = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    return R_ecef2eci @ station_ecef