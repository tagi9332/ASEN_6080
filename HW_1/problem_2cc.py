import os, sys
import numpy as np
from scipy.integrate import solve_ivp


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial

# Assuming these are imported or defined as follows to match assignment requirements:
MU_EARTH = 3.986004415E5  # Standard Earth gravitational parameter
R_EARTH = 6378.0    # Standard Earth radius

def dfdx_wJ2J3(r_vec, mu, J2_val, J3_val, Re, use_J2=True, use_J3=True):
    """
    Calculates the 7x7 Jacobian A and the total acceleration.
    State assumed to be [x, y, z, vx, vy, vz, J2].
    """
    x, y, z = r_vec
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2) 
    r3, r5, r7, r9 = r**3, r**5, r**7, r**9
    
    # 1. Point Mass Gravity
    # Acceleration
    a_pm = -(mu / r3) * r_vec
    # Gradient (del_a_pm / del_r)
    G = -(mu/r3) * np.eye(3) + (3*mu/r5) * np.outer(r_vec, r_vec)
    
    # 2. J2 Perturbation
    a_j2 = np.zeros(3)
    if use_J2:
        # Note: We use J2_val from the state vector (index 6)
        j2_c = -1.5 * mu * J2_val * (Re**2)
        
        # Acceleration J2
        a_j2 = (j2_c / r5) * np.array([
            x * (1 - 5*z**2/r2),
            y * (1 - 5*z**2/r2),
            z * (3 - 5*z**2/r2)
        ])
        
        # Gradient J2 (del_a_j2 / del_r)
        # To match the JSON's specific numerical output:
        G_j2 = (j2_c) * np.array([
            [1/r5 - 5*(x**2+z**2)/r7 + 35*x**2*z**2/r9, -5*x*y/r7 + 35*x*y*z**2/r9, -15*x*z/r7 + 35*x*z**3/r9],
            [-5*x*y/r7 + 35*x*y*z**2/r9, 1/r5 - 5*(y**2+z**2)/r7 + 35*y**2*z**2/r9, -15*y*z/r7 + 35*y*z**3/r9],
            [-15*x*z/r7 + 35*x*z**3/r9, -15*y*z/r7 + 35*y*z**3/r9, 3/r5 - 30*z**2/r7 + 35*z**4/r9]
        ])
        G += G_j2

    # 3. Sensitivity (S) = partial derivative of acceleration wrt J2
    # Since a_j2 is linear with respect to J2, S = a_j2 / J2
    S = a_j2 / J2_val
    
    # 4. Assemble A Matrix (7x7)
    A = np.zeros((7, 7))
    A[0:3, 3:6] = np.eye(3) # dr/dv
    A[3:6, 0:3] = G          # dv/dr
    A[3:6, 6] = S           # dv/dJ2
    
    a_total = a_pm + a_j2
    
    return A, a_total

def keplerJ2_wPhi_ODE(t, state_flat, mu, re, use_J2=True, use_J3=False):
    """
    The main ODE function.
    state_flat: [x, y, z, vx, vy, vz, J2, Phi_flat(49)]
    """
    # Unpack state
    r_vec = state_flat[0:3]
    v_vec = state_flat[3:6]
    J2_val = state_flat[6]
    Phi = state_flat[7:56].reshape((7, 7))
    
    # Get physics from modular function
    A, a_total = dfdx_wJ2J3(r_vec, mu, J2_val, 0.0, re, use_J2, use_J3)
    
    # STM Derivative: dPhi/dt = A * Phi
    Phi_dot = A @ Phi
    
    # State Derivative: [v, a, dJ2/dt=0]
    x_dot = np.concatenate([v_vec, a_total, [0.0]])
    
    return np.concatenate([x_dot, Phi_dot.flatten()])

# 1. Setup Initial Conditions
r_init, v_init = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')
period = 2 * np.pi * np.sqrt(10000**3 / MU_EARTH)
t_span = (0, 15 * period)
t_eval = np.arange(0, t_span[1], 10.0)

state_flat = np.concatenate([r_init, v_init, [0.0010826269], np.eye(7).flatten()])  # Initial J2 value and STM

# Solve and save full state over the time span
sol = solve_ivp(keplerJ2_wPhi_ODE, t_span, state_flat, t_eval=t_eval, args=(MU_EARTH, R_EARTH, True, False))

# Extract all state estimates and STMs at each time step
states = sol.y[0:7, :].T  # Shape (N, 7)
stms = sol.y[7:, :].T.reshape((-1, 7, 7))  # Shape (N, 7, 7)

# Compute disturbance at each timestep by multiplying the STM with the initial disturbance in J2
init_del = np.array([1.0, 0.0, 0.0, 0.0, 10.0e-3, 0.0, 0])  # Initial disturbance in J2

disturbances = np.zeros_like(states)  # Shape (N, 7)
for i in range(states.shape[0]):
    disturbances[i, :] = stms[i, :, :] @ init_del


# Plot disturbances over time
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
labels = ['Delta X (km)', 'Delta Y (km)', 'Delta Z (km)', 'Delta Vx (km/s)', 'Delta Vy (km/s)', 'Delta Vz (km/s)', 'Delta J2']
for i in range(7):
    plt.plot(sol.t / 3600, disturbances[:, i], label=labels[i])
plt.xlabel('Time (hours)')
plt.ylabel('Disturbance')
plt.title('State Disturbances Over Time Due to Initial J2 Perturbation')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()