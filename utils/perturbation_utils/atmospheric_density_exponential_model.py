import numpy as np

def rho_expo_model(r_vec, REarth=6378137.0):
    """
    Estimate mass density using a single-layer simplified exponential model.
    
    Inputs:
        r_vec  -> [float or array] Magnitude of satellite radius vector [m]
                  (or position vector if using np.linalg.norm)
        REarth -> [float] Earth radius [m], defaults to WGS84 value.

    Outputs:
        rho    -> [float or array] Density [kg/m^3]
    """
    # Constants provided
    rho0 = 3.614e-13  # [kg/m^3] Reference density
    r0 = 700000.0 + REarth  # [m] Reference radius
    H = 88667.0  # [m] Scale height
    
    # Ensure r is a numpy array for vectorization
    r = np.asanyarray(r_vec)
    
    # Calculate density: rho = rho0 * exp(-(r - r0) / H)
    rho = rho0 * np.exp(-(r - r0) / H)
    
    return rho