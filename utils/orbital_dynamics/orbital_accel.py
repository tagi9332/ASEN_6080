"""
 Utility functions for orbital dynamics calculations.
 """
import numpy as np
from resources.constants import MU_EARTH

def grav_accel(r):
    r_mag = np.linalg.norm(r)
    return -MU_EARTH * r / r_mag**3

