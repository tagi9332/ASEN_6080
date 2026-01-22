import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode, get_zonal_jacobian
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from scipy.integrate import solve_ivp

# Station Positions
stations = np.deg2rad(np.array([
    [-35.398333, 148.981944],  # Station 1
    [40.427222, 355.749444],   # Station 2 
    [35.247164, 243.205]       # Station 3 
]))

# Load measurements from measurements_noisy.csv
measurements = np.loadtxt(r'C:\Users\tagi9332\OneDrive - UCB-O365\Documents\ASEN_6080\HW_2\measurements_noisy.csv', delimiter=',', skiprows=1)
rho_meas = measurements[:, 1:4]
rho_dot_meas = measurements[:, 4:7]

# Extract time vector from measurements 
time_eval = measurements[:, 0]

# Initial reference trajectory
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

# Initial covariance
P0 = np.diag([1,1,1,1-3,1-3,1-3])

# Initial disturbance
delta_x0 = np.array([[1e-6], [1e-6], [1e-6], [1e-9], [1e-9], [1e-9]])

# Propagate full reference with STM

