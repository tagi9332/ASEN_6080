import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions ---

def radiometric_measurement(sc_pos, sc_vel, gs_pos, gs_vel):
    dR = sc_pos - gs_pos
    dV = sc_vel - gs_vel
    rho = np.linalg.norm(dR)
    rho_dot = np.dot(dR, dV) / rho

    # Spacecraft partials
    d_rho_dR = dR / rho
    d_rho_dV = np.zeros(3)
    d_rho_dot_dR = (rho * dV - rho_dot * dR) / (rho**2)
    d_rho_dot_dV = dR / rho

    # Combine into 2x6
    d_meas_dX = np.zeros((2, 6))
    d_meas_dX[0, 0:3], d_meas_dX[0, 3:6] = d_rho_dR, d_rho_dV
    d_meas_dX[1, 0:3], d_meas_dX[1, 3:6] = d_rho_dot_dR, d_rho_dot_dV

    # Ground Station position partials (2x3)
    d_meas_dRs = np.vstack([-d_rho_dR, -d_rho_dot_dR])

    return np.array([rho, rho_dot]), {'wrt_X': d_meas_dX, 'wrt_Rs': d_meas_dRs}

def compute_gs_eci(gs_location_deg, theta0_deg, times):
    re = 6378.0
    omega_e = 2 * np.pi / 86400.0
    lat = np.deg2rad(gs_location_deg[0])
    lon = np.deg2rad(gs_location_deg[1])
    theta0 = np.deg2rad(theta0_deg)

    N = len(times)
    gs_pos = np.zeros((3, N))
    gs_vel = np.zeros((3, N))

    for i, t in enumerate(times):
        theta = theta0 + omega_e * t
        gs_pos[:, i] = re * np.array([
            np.cos(lon + theta) * np.cos(lat),
            np.sin(lon + theta) * np.cos(lat),
            np.sin(lat)
        ])
        gs_vel[:, i] = np.cross([0, 0, omega_e], gs_pos[:, i])
    return gs_pos, gs_vel

def compute_visibility_mask(sc_pos_traj, gs_pos_traj, elev_mask_deg):
    num_steps = sc_pos_traj.shape[1]
    elev_angles = np.zeros(num_steps)
    is_visible = np.zeros(num_steps, dtype=bool)

    for i in range(num_steps):
        rel_pos = sc_pos_traj[:, i] - gs_pos_traj[:, i]
        zenith = gs_pos_traj[:, i] / np.linalg.norm(gs_pos_traj[:, i])
        sin_el = np.dot(rel_pos, zenith) / np.linalg.norm(rel_pos)
        el = np.degrees(np.arcsin(sin_el))
        elev_angles[i] = el
        if el > elev_mask_deg:
            is_visible[i] = True
    return is_visible, elev_angles

# --- Main Script ---

# Constants
fT_ref = 8.44e9  # Hz
c = 299792.458   # km/s
theta0 = 122.0
elevation_mask = 10.0

stations = np.array([
    [-35.398333, 148.981944],
    [40.427222, 355.749444],
    [35.247164, 243.205]
])

# Load Trajectory (Assumed space-delimited based on previous prompts)
data = np.loadtxt('HW_1/HW1_truth.csv', delimiter=' ')
t_ref = data[:, 0]
sc_pos = data[:, 1:4].T # Transpose to (3, N)
sc_vel = data[:, 4:7].T

# Pre-allocate for visibility and measurements
range_meas, rr_meas = [], []
stn_idx_list, time_list = [], []
vis_matrix = []
elev_matrix = []

for idx, stn_coords in enumerate(stations):
    gs_p, gs_v = compute_gs_eci(stn_coords, theta0, t_ref)
    visible, elevations = compute_visibility_mask(sc_pos, gs_p, elevation_mask)
    
    vis_matrix.append(visible)
    elev_matrix.append(elevations)
    
    for i in range(len(t_ref)):
        if visible[i]:
            meas, _ = radiometric_measurement(sc_pos[:, i], sc_vel[:, i], gs_p[:, i], gs_v[:, i])
            range_meas.append(meas[0])
            rr_meas.append(meas[1])
            stn_idx_list.append(idx + 1)
            time_list.append(t_ref[i])

# Convert to arrays
range_meas = np.array(range_meas)
rr_meas = np.array(rr_meas)
time_hrs = np.array(time_list) / 3600.0

# Exercise 4 Plotting
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
for i in range(1, 4):
    mask = np.array(stn_idx_list) == i
    plt.plot(time_hrs[mask], range_meas[mask], '.', label=f'GS {i}')
plt.ylabel(r'$\rho$ (km)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for i in range(1, 4):
    mask = np.array(stn_idx_list) == i
    plt.plot(time_hrs[mask], rr_meas[mask], '.', label=f'GS {i}')
plt.ylabel(r'$\dot{\rho}$ (km/s)')
plt.xlabel('Time (hours)')
plt.grid(True)
plt.tight_layout()

# Range RU and Doppler Shift
range_ru = (221/749 * range_meas) / (c / fT_ref)
doppler = -2 * rr_meas * fT_ref / c

# Noisy Range Rate
sigma = 0.5 / 1e6 # 0.5 mm/s to km/s
noisy_rr = rr_meas + sigma * np.random.randn(len(rr_meas))
residuals = (noisy_rr - rr_meas) * 1e6 # km/s to mm/s

plt.figure(figsize=(10, 5))
plt.plot(time_hrs, residuals, '.', alpha=0.5)
plt.axhline(3*sigma*1e6, color='r', linestyle='--', label=r'$3\sigma$')
plt.axhline(-3*sigma*1e6, color='r', linestyle='--')
plt.ylabel('Residuals (mm/s)')
plt.xlabel('Time (hours)')
plt.title('Range-Rate Residuals with Gaussian Noise')
plt.legend()
plt.grid(True)

plt.show()

print(f"First measurement: {min(time_list):.2f} s")
print(f"Last measurement: {max(time_list):.2f} s")