import numpy as np
from resources.constants import MU_EARTH

def orbital_elements_to_inertial(a, e, i, raan, argp, ta, mu=MU_EARTH, units='rad'):
    """
    Classical Orbital Elements -> inertial position/velocity.
    """
    if units == 'deg':
        i = np.radians(i)
        raan = np.radians(raan)
        argp = np.radians(argp)
        ta = np.radians(ta)

    if np.isclose(e, 1.0, atol=1e-12):
        raise ValueError("Parabolic case (e close to 1) not supported.")
    
    p = a * (1.0 - e**2)

    cnu, snu = np.cos(ta), np.sin(ta)
    r_pf = (p / (1.0 + e * cnu)) * np.array([cnu, snu, 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-snu, e + cnu, 0.0])

    # Rotation PQW -> IJK
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(argp), np.sin(argp)

    R3_O = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    R1_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    R3_o = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])

    Q_pqw_to_ijk = R3_O @ R1_i @ R3_o

    r = Q_pqw_to_ijk @ r_pf
    v = Q_pqw_to_ijk @ v_pf
    return r, v
