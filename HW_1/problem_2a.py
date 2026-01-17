# Script to propagate initial conditions with a small perturbation
import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, J3, R_EARTH
from scipy.integrate import solve_ivp

def dfdx_wJ2J3(r_vec, mu, J2, J3, Re=6378.0,use_J2=True,use_J3=True):
    
    x, y, z = r_vec
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2) 
    r3, r5, r7, r9, r11 = r**3, r**5, r**7, r**9, r**11
    
    # Build the 3x3 Gravity Gradient
    # Point Mass
    G = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)
    
    # J2 Contribution if enabled
    if use_J2:
        j2_c = -1.5 * mu * J2 * Re**2
        G_j2 = np.array([
            [1/r5 - 5*x**2/r7 - 5*z**2/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9, -15*x*z/r7 + 35*x*z**3/r9],
            [-5*x*y/r7 + 35*x*y*z**2/r9, 1/r5 - 5*y**2/r7 - 5*z**2/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
            [-15*x*z/r7 + 35*x*z**3/r9, -15*y*z/r7 + 35*y*z**3/r9, 3/r5 - 30*z**2/r7 + 35*z**4/r9]
        ])
        G += j2_c * G_j2
    
    # J3 Contribution if enabled
    if use_J3:
        j3_c = -2.5 * mu * J3 * Re**3
        G_j3 = np.array([
            [3*z/r7 - 21*x**2*z/r9 - 7*z**3/r9 + 63*x**2*z**3/r11, -21*x*y*z/r9 + 63*x*y*z**3/r11, 3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11],
            [-21*x*y*z/r9 + 63*x*y*z**3/r11, 3*z/r7 - 21*y**2*z/r9 - 7*z**3/r9 + 63*y**2*z**3/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11],
            [3*x/r7 - 42*x*z**2/r9 + 63*x*z**4/r11, 3*y/r7 - 42*y*z**2/r9 + 63*y*z**4/r11, 15*z/r7 - 70*z**3/r9 + 63*z**5/r11]
        ])
        G += j3_c * G_j3

# Build the 3x3 Sensitivity Matrix (S = del_a / del_params)    
    j2_c = -1.5 * mu * J2 * Re**2
    j3_c = -2.5 * mu * J3 * Re**3

    # Point Mass Acceleration 
    a_pm = -(mu / r3) * r_vec
    
    # J2 Acceleration
    vec_j2 = np.array([
        x * (1/r5 - 5*z**2/r7),
        y * (1/r5 - 5*z**2/r7),
        z * (3/r5 - 5*z**2/r7)
    ])
    a_j2 = j2_c * vec_j2
    
    # J3 Acceleration
    vec_j3 = np.array([
        x * (3*z/r7 - 7*z**3/r9),
        y * (3*z/r7 - 7*z**3/r9),
        (6*z**2/r7 - 7*z**4/r9 - 0.6/r5) 
    ])
    a_j3 = j3_c * vec_j3

    # Assemble S Matrix
    S = np.zeros((3, 3))
    
    # Partial wrt mu (Total Accel / mu)
    S[:, 0] = (a_pm + a_j2 + a_j3) / mu
    
    # Partial wrt J2
    S[:, 1] = a_j2 / J2
    
    # Partial wrt J3
    S[:, 2] = a_j3 / J3

    # Assemble the Full 9x9 A-Matrix 
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3) # dr/dv block (zeros)
    A[3:6, 0:3] = G        # dv/dr block (Gravity Gradient)
    A[3:6, 6:9] = S        # dv/dp block (Sensitivity)
    
    return A

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


# Equations of Motion
def eom_with_stm(t, state_flat, mu, j2, j3, re):
    # Unpack state: [x, y, z, vx, vy, vz, Phi(81 elements flattened)]
    r_vec = state_flat[0:3]
    v_vec = state_flat[3:6]
    phi_flat = state_flat[6:]
    Phi = phi_flat.reshape((9, 9))
    
    r = np.linalg.norm(r_vec)
    
    # Calculate Accelerations
    # Point Mass
    a_vec = -(mu / r**3) * r_vec
    
    # J2 Perturbation
    x, y, z = r_vec
    z_r2 = (z/r)**2
    factor = 1.5 * j2 * mu * (re**2 / r**5)
    a_j2 = factor * np.array([
        x * (5*z_r2 - 1),
        y * (5*z_r2 - 1),
        z * (5*z_r2 - 3)
    ])
    
    # Total Acceleration
    dvdt = a_vec + a_j2
    
    # 2. STM Propagation: dPhi/dt = A * Phi
    A = dfdx_wJ2J3(r_vec, mu, j2, j3, Re=re, use_J3=False)
    Phi_dot = A @ Phi
    
    # Pack it back up
    return np.concatenate([v_vec, dvdt, Phi_dot.flatten()])

# Setup Propagation
# 15 Orbits calculation
period = 2 * np.pi * np.sqrt(10000**3 / MU_EARTH)
t_span = (0, 15 * period)

# timespan in even steps of 10 seconds
t_eval = np.arange(0, t_span[1], 10.0)

# Initial conditions
r_init, v_init = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
phi_init = np.eye(9).flatten()
y0 = np.concatenate([r_init, v_init, phi_init])

# Solve
sol = solve_ivp(eom_with_stm, t_span, y0, t_eval=t_eval, 
                args=(MU_EARTH, J2, J3, R_EARTH), rtol=1e-10, atol=1e-10, method='DOP853')

# Output / Verification
final_state = sol.y[:6, -1]
final_stm = sol.y[6:, -1].reshape((9, 9))

# Export results to CSV
header = "time(s),x(km),y(km),z(km),vx(km/s),vy(km/s),vz(km/s)"
for i in range(9):
    for j in range(9):
        header += f",Phi_{i+1}{j+1}"

output_data = np.vstack((sol.t, sol.y)).T
np.savetxt("problem_2a_propagation_results.csv", output_data, delimiter=",", header=header, comments='')

print("-" * 30)
print(f"Propagation Complete: 15 orbits ({sol.t[-1]/3600:.2f} hours)")
print(f"Final Position (km): {final_state[0:3]}")
print(f"Final Velocity (km/s): {final_state[3:6]}")

# Comparison against Truth Values
target_data = np.array([149280.0000, -3.74428318223291E+03, 8.06372447309221E+03, 4.55584929373468E+03, -4.31621106392489E+00, -3.62133680561135E+00, 2.86260874964455E+00])


target_time = target_data[0]
target_state = target_data[1:]

# Calculate errors
state_error = final_state - target_state
pos_error_mag = np.linalg.norm(state_error[0:3])
vel_error_mag = np.linalg.norm(state_error[3:6])

# Convert to cm and cm/s for reporting
state_error[0:3] *= 1e5  # km to cm
state_error[3:6] *= 1e5  # km/s to cm/s
pos_error_mag *= 1e5
vel_error_mag *= 1e5


print("-" * 30)
print(f"Comparison at t = {target_time}s:")
print(f"Position Error Vector (cm): {state_error[0:3]}")
print(f"Velocity Error Vector (cm/s): {state_error[3:6]}")
print(f"\nMagnitude of Position Error: {pos_error_mag:.6e} cm")
print(f"Magnitude of Velocity Error: {vel_error_mag:.6e} cm/s")

# Final STM Check
print("-" * 30)
print("Final State Transition Matrix (Full 9x9):")
print(np.array2string(final_stm, precision=4, suppress_small=True))
