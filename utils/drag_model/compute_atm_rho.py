import numpy as np

from resources.constants import R_EARTH

def compute_atm_rho(position_state, r_earth=R_EARTH):
    """
    Calculates atmospheric density using the exponential model.
    
    Parameters:
    -----------
    position_state : array_like or float
        The satellite's position vector [x, y, z] in meters, 
        OR the scalar magnitude of the radius r in meters.
    r_earth : float, optional
        Radius of the Earth in km
        
    Returns:
    --------
    rho : float
        Atmospheric density in kg/m^3.
    """
    
    # Constants from the model definition
    rho0 = 3.614e-13  # kg/m^3
    H = 88667.0       # Scale height in meters
    r_earth *= 1e3  # Convert Earth radius to meters
    r0 = 700000.0 + r_earth  # Reference radius (700 km altitude)

    # Handle input: determine if it is a vector or a scalar magnitude
    position_state = np.array(position_state)
    if position_state.size == 3:
        # Calculate norm if input is a 3D vector
        r_mag = np.linalg.norm(position_state)
    else:
        # Assume input is already scalar magnitude
        r_mag = float(position_state)

    # Calculate Density (Eq 5)
    # rho = rho0 * exp(-(r - r0) / H)
    exponent = -(r_mag - r0) / H
    rho = rho0 * np.exp(exponent)

    return rho
