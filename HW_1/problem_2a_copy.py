# Script to propagate initial conditions with a small perturbation
import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from scipy.integrate import solve_ivp
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_6x6



# Test case initial conditions
coes = {
    'a': 10000,          # Semi-major axis [km]
    'e': 0.001,           # Eccentricity
    'i': 40,            # Inclination [deg]
    'RAAN': 80,          # Right Ascension of Ascending Node
    'arg_periapsis': 40, # Argument of Periapsis [deg]
    'true_anomaly': 0    # True Anomaly [deg]
}

# Convert COEs to inertial state vector
r_vec, v_vec = orbital_elements_to_inertial(
    coes['a'], coes['e'], coes['i'], coes['RAAN'],
    coes['arg_periapsis'], coes['true_anomaly'], units='deg'
)


# Setup Propagation
# 15 Orbits calculation
period = 2 * np.pi * np.sqrt(10000**3 / MU_EARTH)
t_span = (0, 15 * period)

# timespan in even steps of 10 seconds
t_eval = np.arange(0, t_span[1], 10.0)

# Initial conditions
r_init, v_init = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
phi_init = np.eye(6).flatten()
y0 = np.concatenate([r_init, v_init, phi_init])
coeffs = [MU_EARTH, J2, J3]

# 3. Solve IVP
sol = solve_ivp(
    zonal_sph_ode_6x6, 
    t_span, 
    y0, 
    t_eval=t_eval, 
    args=(coeffs,),  # Note the comma to make it a tuple
    rtol=1e-10, 
    atol=1e-10, 
    method='DOP853'
)

# === EXPORT TO FILE ===

# 1. Stack Time, Position, and Velocity (Transpose to get rows of data)
# sol.y[0:6, :] grabs the first 6 rows (x, y, z, vx, vy, vz) and ignores the STM
data_to_save = np.column_stack((sol.t, sol.y[0:6, :].T))

# 2. Save to CSV
# fmt='%.14E' uses scientific notation with 14 decimal places
# delimiter=' ' uses a space to separate columns (as requested previously)
np.savetxt("problem_2a_traj.csv", data_to_save, delimiter=' ', fmt='%.14E')

print(f"Data saved to 'problem_2a_traj.csv'")