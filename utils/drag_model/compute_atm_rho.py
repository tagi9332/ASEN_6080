import numpy as np
from resources.constants import R_EARTH, RHO_0, R0, H

def compute_atm_rho(r):
    """
    Compute atmospheric density based on position state. (computed in m)
    """
    
    # Normalize radius if vector is given
    if len(r.shape) > 1:
        r = np.linalg.norm(r, axis=1) 
    rho = RHO_0 * np.exp(-(r-R0)/H)  # Compute atmospheric density

    return rho