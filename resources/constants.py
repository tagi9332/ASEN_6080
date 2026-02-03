# PROTOS/data/resources/constants.py
# ASEN 6080 provided values

import numpy as np

# Gravitational parameters [m^3/s^2]
MU_EARTH = 3.986004415e14

# Other constants
R_EARTH = 6378136.3  # m, Earth radius

# J2 perturbation constant
J2 = 1.082626925638815e-3
J3 = -2.5324105e-6

# Earth rotation rate [rad/s]
OMEGA_EARTH =  7.2921158553e-5  # rad/s

# Standard atmospheric model constants
RHO_0 = 3.614e-13  # kg/m^3
R0 = 700000.0  + R_EARTH  # m, reference altitude
H = 88667.0  # m, scale height

