import numpy as np
from resources.constants import R_EARTH

def compute_atm_rho(position_state, r_earth=R_EARTH):
    """
    Calculates atmospheric density using the exponential model.
    Compatible with KILOMETER state vectors.
    
    Parameters:
    -----------
    position_state : array_like or float
        The satellite's position vector [x, y, z] in KM, 
        OR the scalar magnitude of the radius r in KM.
    r_earth : float, optional
        Radius of the Earth in KM (default from constants).
        
    Returns:
    --------
    rho : float
        Atmospheric density in kg/m^3.
    """
    
    # Constants (Converted to KM)
    rho0 = 3.614e-13      # kg/m^3 (Standard density)
    H = 88.667            # Scale height in KM (was 88667 m)
    r0 = 700.0 + r_earth  # Reference radius in KM (700 km altitude)

    # Handle input: determine if it is a vector or a scalar magnitude
    position_state = np.array(position_state)
    if position_state.size == 3:
        r_mag = np.linalg.norm(position_state)
    else:
        r_mag = float(position_state)

    # Calculate Density
    # rho = rho0 * exp(-(r - r0) / H)
    # Note: (r - r0) is in km, H is in km. The units cancel nicely.
    exponent = -(r_mag - r0) / H
    rho = rho0 * np.exp(exponent)

    return rho