import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from dataclasses import dataclass
from typing import Any
import os, sys

# Adjust path to find local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from your project ---
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial
from resources.constants import MU_EARTH, J2, R_EARTH
from utils.plotting.post_process import post_process
from utils.ground_station_utils.gs_latlon import get_gs_eci_state
from utils.ground_station_utils.gs_meas_model_H import compute_H_matrix, compute_rho_rhodot
from utils.zonal_harmonics.zonal_harmonics import zonal_sph_ode_dmc, get_zonal_jacobian_dmc
from resources.gs_locations_latlon import stations_ll

# ============================================================
# DATA STRUCTURES & HELPERS
# ============================================================
@dataclass
class LKFResults:
    dx_hist: Any
    P_hist: Any
    state_hist: Any
    innovations: Any
    postfit_residuals: Any
    nis_hist: Any
    accel_hist: Any 

    def __post_init__(self):
        self.dx_hist = np.array(self.dx_hist)
        self.P_hist = np.array(self.P_hist)
        self.state_hist = np.array(self.state_hist)
        self.innovations = np.array(self.innovations)
        self.postfit_residuals = np.array(self.postfit_residuals)
        self.nis_hist = np.array(self.nis_hist)
        if self.accel_hist is not None:
            self.accel_hist = np.array(self.accel_hist)

def compute_van_loan(A_mat, G_mat, Q_cont, dt):
    n = A_mat.shape[0]
    GQGt = G_mat @ Q_cont @ G_mat.T
    top = np.hstack((-A_mat, GQGt))
    bot = np.hstack((np.zeros((n, n)), A_mat.T))
    M_exp = expm(np.vstack((top, bot)) * dt)
    Phi_k = M_exp[n:, n:].T  
    Q_k = Phi_k @ M_exp[0:n, n:]
    return Phi_k, Q_k

def compute_J3_acceleration(r_vec):
    """Analytical J3 acceleration for truth comparison."""
    J3 = -2.5327e-6 
    r_mag = np.linalg.norm(r_vec)
    x, y, z = r_vec
    u = z / r_mag
    coef = (MU_EARTH / r_mag**2) * (R_EARTH / r_mag)**3 * J3
    term_r = (5.0/2.0) * (7*u**3 - 3*u)
    term_z = (3.0/2.0) * (1 - 5*u**2)
    return np.array([
        coef * (term_r * (x/r_mag)),
        coef * (term_r * (y/r_mag)),
        coef * (term_r * (z/r_mag) + term_z)
    ])

# ============================================================
# LKF DMC CLASS
# ============================================================
class LKF_DMC:
    def __init__(self, n_states: int = 9):
        self.n = n_states
        self.I = np.eye(n_states)

    def run(self, obs, X_0, x_0, P0, Rk, Q_PSD, options) -> LKFResults:
        coeffs = options['coeffs']
        dt_max = options.get('dt_max', 60.0) 
        G = np.zeros((9, 3))
        G[6:9, :] = np.eye(3)
        time_eval = obs['Time(s)'].values

        sol_ref = solve_ivp(zonal_sph_ode_dmc, (0, time_eval[-1]), X_0, 
                            dense_output=True, args=(coeffs,), 
                            rtol=options['rel_tol'], atol=options['abs_tol'])
        ref_traj = sol_ref.sol
        x, P = x_0.copy(), P0.copy()
        _x, _P, _state, _prefit_res, _postfit_res, _nis, _w_terms = [], [], [], [], [], [], []

        for k in range(1, len(time_eval)):
            t_prev, t_curr = time_eval[k-1], time_eval[k]
            t_internal = t_prev
            
            while t_internal < t_curr:
                h = min(dt_max, t_curr - t_internal)
                ref_s = ref_traj(t_internal)[0:9]
                A = get_zonal_jacobian_dmc(ref_s[0:3], ref_s[3:6], coeffs)
                Phi_s, Q_s = compute_van_loan(A, G, Q_PSD, h)
                x, P = Phi_s @ x, Phi_s @ P @ Phi_s.T + Q_s
                t_internal += h
            
            ref_k = ref_traj(t_curr)[0:9]
            meas_row = obs.iloc[k]
            station_idx = int(meas_row['Station_ID']) - 1
            Rs, Vs = get_gs_eci_state(stations_ll[station_idx][0], stations_ll[station_idx][1], 
                                      t_curr, init_theta=np.deg2rad(122))
            
            H = np.hstack([compute_H_matrix(ref_k[0:3], ref_k[3:6], Rs, Vs), np.zeros((2, 3))])
            y_pred = compute_rho_rhodot(ref_k[0:6], np.concatenate([Rs, Vs]))
            innovation = (np.array([meas_row['Range(km)'], meas_row['Range_Rate(km/s)']]) - y_pred) - H @ x

            S = H @ P @ H.T + Rk
            K = (np.linalg.solve(S, H @ P)).T
            x = x + K @ innovation
            IKH = self.I - K @ H
            P = IKH @ P @ IKH.T + K @ Rk @ K.T

            # Save total estimated acceleration (Ref + Deviation)
            _w_terms.append((ref_k[6:9] + x[6:9]).copy())
            _x.append(x.copy()); _P.append(P.copy()); _nis.append(innovation.T @ np.linalg.solve(S, innovation))
            _state.append((ref_k + x).copy()); _postfit_res.append(innovation - H @ (K @ innovation))

        return LKFResults(_x, _P, _state, [], _postfit_res, _nis, accel_hist=_w_terms)

# ============================================================
# MAIN EXECUTION
# ============================================================
output_dir = 'HW_3/plots_dmc'
if not os.path.exists(output_dir): os.makedirs(output_dir)

meas_file, truth_file = r'data/measurements_2a_noisy.csv', r'data/problem_2a_traj.csv'
r0_true, v0_true = orbital_elements_to_inertial(10000, 0.001, 40, 80, 40, 0, units='deg')

# Period and DMC Config
a_sma = -MU_EARTH / (2 * ((np.linalg.norm(v0_true)**2 / 2) - (MU_EARTH / np.linalg.norm(r0_true))))
P_period = 2 * np.pi * np.sqrt(a_sma**3 / MU_EARTH)
beta = 1.0 / (P_period / 30.0)

obs = pd.read_csv(meas_file)
obs.columns = obs.columns.str.strip()
truth_df = pd.read_csv(truth_file, delim_whitespace=True, header=None, names=['T', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
truth_interp = interp1d(truth_df['T'], truth_df[['x', 'y', 'z']].values, axis=0, kind='cubic')

# Run LKF
x_0_dev = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3, 0, 0, 0])
X_0_ref = np.concatenate([r0_true + x_0_dev[:3], v0_true + x_0_dev[3:6], [0,0,0], np.eye(9).flatten()])
lkf = LKF_DMC()
results = lkf.run(obs, X_0_ref, x_0_dev, np.diag([1,1,1,1e-6,1e-6,1e-6,1e-9,1e-9,1e-9]), 
                  np.diag([1e-6, 1e-12]), (2*(1e-8**2)*beta)*np.eye(3), 
                  {'coeffs': (MU_EARTH, J2, 0, np.eye(3)*beta), 'abs_tol': 1e-10, 'rel_tol': 1e-10})

# ============================================================
# ANALYSIS & SCATTER PLOTS
# ============================================================
mask_idx = 500
t_plot = obs['Time(s)'].values[1:][mask_idx:]
w_est = results.accel_hist[mask_idx:]
w_true = np.array([compute_J3_acceleration(truth_interp(t)) for t in t_plot])
w_err = (w_est - w_true) * 1e6
w_sig = np.sqrt(np.diagonal(results.P_hist[:, 6:9, 6:9], axis1=1, axis2=2))[mask_idx:] * 1e6

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for i, c in enumerate(['x', 'y', 'z']):
    axes[i].scatter(t_plot, w_true[:, i]*1e6, color='black', s=2, alpha=0.5, label='True $J_3$')
    axes[i].scatter(t_plot, w_est[:, i]*1e6, color='blue', s=2, alpha=0.5, label='LKF DMC')
    axes[i].set_ylabel(f'{c} [mm/$s^2$]')
    axes[i].grid(True, alpha=0.3)
axes[0].set_title("LKF Estimated Acceleration vs Actual $J_3$ (Scatter)")
axes[0].legend(loc='upper right', markerscale=5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'LKF_DMC_accel_comparison.png'))
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for i, c in enumerate(['x', 'y', 'z']):
    axes[i].scatter(t_plot, w_err[:, i], color='blue', s=2, alpha=0.6)
    axes[i].fill_between(t_plot, w_sig[:, i], -w_sig[:, i], color='red', alpha=0.15, label='1$\sigma$')
    axes[i].set_ylabel(f'Err {c} [mm/$s^2$]')
    axes[i].grid(True, alpha=0.3)
axes[0].set_title("LKF DMC Estimation Error vs 1-Sigma Bounds")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'LKF_DMC_accel_error.png'))
plt.show()