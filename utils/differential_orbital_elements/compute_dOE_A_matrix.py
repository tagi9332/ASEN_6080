# Compute A transformation matrix from orbital elements
import numpy as np
from resources.constants import MU_EARTH

def compute_a_matrix(SMA, theta, inc, q1, q2):
    """
    Construct the inverse A-matrix (A^{-1}) for orbital element to state transformation. Returns A
    
    Parameters
    ----------
    SMA : float
        Semi-major axis [km]
    R : float
        Orbital radius [km]
    theta : float
        True latitude [rad]
    inc : float
        Inclination [rad]
    q1, q2 : float
        Equinoctial elements
    Vt : float
        Transverse velocity [km/s]
    Vr : float
        Radial velocity [km/s]
    
    Returns
    -------
    Ainv : ndarray
        6x6 transformation matrix
    """
    
    # Geometry
    R = SMA * (1 - q1**2 - q2**2) / (1 + q1 * np.cos(theta) + q2 * np.sin(theta))
    p = SMA * (1 - q1**2 - q2**2)
    h = np.sqrt(p * MU_EARTH)
    # Velocities
    Vr = h/p * (q1 * np.sin(theta) - q2 * np.cos(theta))
    Vt = h/p * (1 + q1 * np.cos(theta) + q2 * np.sin(theta))

    # Parameters
    alpha = SMA / R
    nu = Vr / Vt
    rho = R / p
    kappa1 = alpha * (1/rho - 1)
    kappa2 = alpha * nu**2 / rho
    
    # Trig
    cO, sO = np.cos(theta), np.sin(theta)
    c2O, s2O = np.cos(2*theta), np.sin(2*theta)
    cotI = 1 / np.tan(inc) 
    
    # Initialize
    Ainv = np.zeros((6,6))
    
    # Row 1
    Ainv[0,0] = 2*alpha*(2 + 3*kappa1 + 2*kappa2)
    Ainv[0,1] = -2*alpha*nu*(1 + 2*kappa1 + kappa2)
    Ainv[0,3] = 2*alpha**2*nu*p / Vt
    Ainv[0,4] = 2*SMA/Vt*(1 + 2*kappa1 + kappa2)
    
    # Row 2
    Ainv[1,1] = 1/R
    Ainv[1,2] = cotI/R*(cO + nu*sO)
    Ainv[1,5] = -sO*cotI / Vt
    
    # Row 3
    Ainv[2,2] = (sO - nu*cO)/R
    Ainv[2,5] = cO/Vt
    
    # Row 4
    Ainv[3,0] = (1/(rho*R))*(3*cO + 2*nu*sO)
    Ainv[3,1] = -(1/R)*(nu**2/rho*sO + q1*s2O - q2*c2O)
    Ainv[3,2] = -(q2*cotI/R)*(cO + nu*sO)
    Ainv[3,3] = sO/(rho*Vt)
    Ainv[3,4] = (1/(rho*Vt))*(2*cO + nu*sO)
    Ainv[3,5] = (q2*cotI*sO)/Vt
    
    # Row 5
    Ainv[4,0] = (1/(rho*R))*(3*sO - 2*nu*cO)
    Ainv[4,1] = (1/R)*(nu**2/rho*cO + q2*s2O + q1*c2O)
    Ainv[4,2] = (q1*cotI/R)*(cO + nu*sO)
    Ainv[4,3] = -cO/(rho*Vt)
    Ainv[4,4] = (1/(rho*Vt))*(2*sO - nu*cO)
    Ainv[4,5] = -(q1*cotI*sO)/Vt
    
    # Row 6
    Ainv[5,2] = -(cO + nu*sO)/(R*np.sin(inc))
    Ainv[5,5] = sO/(Vt*np.sin(inc))
    
    # Invert A
    A = np.linalg.inv(Ainv)

    return A


