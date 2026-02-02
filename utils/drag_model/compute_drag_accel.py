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
        The satellite state vector [x, y, z, vx, vy, vz] in meters and m/s.
    area : float
        Cross-sectional area in m^2 (default 3.0).
    mass : float
        Satellite mass in kg (default 970.0).
    Cd : float
        Drag coefficient (default 2.0).
        
    Returns:
    --------
    a_drag : numpy.ndarray
        The drag acceleration vector [ax, ay, az] in m/s^2.
    """
    
    # Unpack state
    r_vec = np.array(state[0:3]) # Position [x, y, z]
    v_vec = np.array(state[3:6]) # Inertial Velocity [vx, vy, vz]
    
    # Calculate atmospheric density using exponential model
    rho = compute_atm_rho(r_vec)
    
    # Calculate velocity relative to the rotating atmosphere
    omega_vec = np.array([0, 0, OMEGA_EARTH])  # Earth's rotation vector
    
    # v_rel = v_inertial - (omega_earth x r)
    v_atm = np.cross(omega_vec, r_vec)
    v_rel = v_vec - v_atm
    
    v_rel_mag = np.linalg.norm(v_rel)
    
    # Compute drag acceleration vector   
    factor = -0.5 * rho * (Cd * area / mass) * v_rel_mag
    a_drag = factor * v_rel
    
    return a_drag
