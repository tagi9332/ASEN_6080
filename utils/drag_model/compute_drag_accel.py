import numpy as np

# Local Imports
from utils.drag_model.compute_atm_rho import compute_atm_rho
from resources.constants import OMEGA_EARTH

def get_drag_acceleration(state, area=3.0, mass=970.0, Cd=2.0):
    """
    Computes the perturbation acceleration due to atmospheric drag.
    
    Parameters:
    -----------
    state : array_like
        The satellite state vector [x, y, z, vx, vy, vz] in KM and KM/S.
    area : float
        Cross-sectional area in m^2 (default 3.0).
    mass : float
        Satellite mass in kg (default 970.0).
    Cd : float
        Drag coefficient (default 2.0).
        
    Returns:
    --------
    a_drag : numpy.ndarray
        The drag acceleration vector [ax, ay, az] in KM/S^2.
    """
    
    # Unpack state (Expect KM and KM/S)
    r_vec = np.array(state[0:3]) 
    v_vec = np.array(state[3:6]) 
    
    # Calculate atmospheric density (kg/m^3)
    rho = compute_atm_rho(r_vec)
    
    # Calculate velocity relative to the rotating atmosphere
    omega_vec = np.array([0, 0, OMEGA_EARTH])  # Earth's rotation vector
    
    # v_rel = v_inertial - (omega_earth x r)
    # All inputs here are in KM and S, so v_rel is in KM/S
    v_atm = np.cross(omega_vec, r_vec)
    v_rel = v_vec - v_atm
    
    v_rel_mag = np.linalg.norm(v_rel)
    
    # --- Unit Analysis ---
    # Formula: a = -0.5 * rho * (Cd * A / m) * v_rel * |v_rel|
    # Units: [kg/m^3] * [m^2/kg] * [km/s] * [km/s]
    #      = [1/m] * [km^2/s^2]
    # We want output in [km/s^2].
    # Since 1 km = 1000 m, 1/m = 1000/km.
    # The raw result is currently 1000x too large (in units of milli-km/s^2).
    # We must multiply by 1e-3 (1/1000) to convert 1/m to 1/km.
    
    unit_conversion = 1e-3
    
    factor = -0.5 * rho * (Cd * area / mass) * v_rel_mag * unit_conversion
    a_drag = factor * v_rel
    
    return a_drag