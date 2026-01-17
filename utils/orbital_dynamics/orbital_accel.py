"""
 Utility functions for orbital dynamics calculations.
 """
import numpy as np

def grav_accel(r):
    MU_EARTH = 398600.4418  # km^3/s^2
    r_mag = np.linalg.norm(r)
    return -MU_EARTH * r / r_mag**3

