import numpy as np
import sys
import json
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../")

# from src.Functions.stations import Stations
# from src.Functions.batch import batch_estimate_x0, postprocess_batch_for_plots


# import numpy as np
# from scipy.integrate import solve_ivp
# from .jacobians import stm
# from .range_rangerate import H_range_rangerate
 


def propagate_state_and_stm_history(
    x0_ref_6: np.ndarray,
    t_eval: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    Re: float = 6378.0,
    reltol: float = 1e-10,
    abstol: float = 1e-10,
    method: str = "DOP853",
):
    """
    One integration from t_eval[0] to t_eval[-1], returning state+STM at all t_eval
    """
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    t0 = float(t_eval[0])
    tf = float(t_eval[-1])

    remove = np.array([6, 7, 8], dtype=int)
    nx = 6

    x0_ref_6 = np.asarray(x0_ref_6, dtype=float).reshape(6,)
    X0_9 = np.hstack((x0_ref_6, mu, J2, J3))
    Phi0 = np.eye(nx)
    y0 = np.hstack((X0_9, Phi0.flatten()))

    fun = lambda t, y: stm(
        t,
        state9=y[:9],
        phi=y[9:].reshape(nx, nx),
        rows_col_to_remove=remove,
        Re=Re,
        j2=True,
        j3=False,
    )

    sol = solve_ivp(
        fun, (t0, tf), y0,
        t_eval=t_eval,
        rtol=reltol, atol=abstol,
        method=method,
    )
    if not sol.success:
        raise RuntimeError(f"STM integration failed: {sol.message}") #safety flag

    Y = sol.y.T                       # (N, 9+36)
    X_hist = Y[:, :6]                 # (N,6)
    Phi_hist = Y[:, 9:].reshape(-1, nx, nx)  # (N,6,6)

    return X_hist, Phi_hist



def predict_meas_vec(station, x_i_6, ti):
    d = station.measure(x_i_6[:3], x_i_6[3:], float(ti))
    return np.array([d["rho_km"], d["rho_dot_km_s"]], dtype=float)

#I am guessing the elevation mask made some esitmates fall just under 10 deg and leacve a measurement empyt causing matrix size errors, investigate further later

# def predict_meas_vec(station, x_i_6, ti):
#     """
#     Predict [rho; rhodot] from state at time ti for this station,
#     WITHOUT ELEVATION MASK
#     """
#     r_sc = np.asarray(x_i_6[:3], dtype=float)
#     v_sc = np.asarray(x_i_6[3:], dtype=float)

#     v_zero = np.zeros(3)
#     r_st, v_st, _ = station.ecef2eci(float(ti), station.r_ecef, v_zero)

#     rho_vec = r_sc - r_st
#     rho = np.linalg.norm(rho_vec)
#     rho_hat = rho_vec / rho
#     rho_dot = float(np.dot(rho_hat, (v_sc - v_st)))

#     return np.array([rho, rho_dot], dtype=float)


def batch_estimate_x0(
    all_meas,
    stations,
    x0_bar: np.ndarray,
    P0: np.ndarray,
    R: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    Re: float = 6378.0,
    max_iter: int = 10,
    tol: float = 1e-10,
    reltol: float = 1e-10,
    abstol: float = 1e-10,
):
    """
    Iterated batch least-squares for initial 6-state x0=[r0;v0].

    Saves:
      - dx0_hist: the full correction vector δx0 each iteration
      - prefit residuals (final iteration): y_pre = Y - yhat(x*(ti))
      - postfit residuals (final estimate): y_post = Y - yhat(xhat(ti))
      - measurement meta arrays for plotting (time, station)

    Returns
    -------
    x0_hat : (6,)
    P0_hat : (6,6)
    info   : dict with fields described above
    """

    station_map = {st.name: st for st in stations}

    t0 = float(all_meas[0]["t"])
    P0inv = np.linalg.inv(P0)
    Rinv = np.linalg.inv(R)

    x0_star = np.asarray(x0_bar, dtype=float).reshape(6,).copy()

    # Save iteration history
    dx0_hist = []          # list of (6,) vectors
    x0_star_hist = [x0_star.copy()]

    # Store final iteration prefit residuals
    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
    t0 = float(t_meas[0])
    st_meas = [m["station"] for m in all_meas]

    prefit_resids_final = None  

    for it in range(max_iter):

        # Normal equations initialization:
       
        Lambda = P0inv.copy()
        N = P0inv @ (x0_bar - x0_star)

        X_hist, Phi_hist = propagate_state_and_stm_history(
            x0_ref_6=x0_star,
            t_eval=t_meas,
            mu=mu, J2=J2, J3=J3,
            Re=Re,
            reltol=reltol,
            abstol=abstol,
        )
        # Build prefit residual


        res_list = []

        # Inner loop over measurements
        for j, m in enumerate(all_meas):
            ti = float(m["t"])
            st = station_map[m["station"]]

            # Reference propagation + STM
            x_i = X_hist[j, :]
            Phi_i0 = Phi_hist[j, :, :]

            # Predicted measurement at reference trajectory
            d = st.measure(x_i[:3], x_i[3:], float(ti))
            if d is None:
                continue
            yhat_i = np.array([d["rho_km"], d["rho_dot_km_s"]], dtype=float)

            Yi = np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)

            # Prefit residual
            y_i = Yi - yhat_i
            res_list.append(y_i)

            # Station ECI state for Jacobian
            v_zero = np.zeros(3)
            Rs, Vs, _ = st.ecef2eci(ti, st.r_ecef, v_zero)

            #H~
            r_i = x_i[:3]
            v_i = x_i[3:6]
            H_tilde = H_range_rangerate(r_i, v_i, Rs, Vs)  # (2,6)

            # H_i 
            H_i = H_tilde @ Phi_i0  # (2,6)

            # Accumulate
            Lambda += H_i.T @ Rinv @ H_i
            N += H_i.T @ Rinv @ y_i

        # Solve δx0
        dx0 = np.linalg.solve(Lambda, N)

        # Update x0* to x0* + δx0
        x0_star = x0_star + dx0

        dx0_hist.append(dx0.copy())
        x0_star_hist.append(x0_star.copy())

        # Save the final iteration's prefit residuals for plotting
        prefit_resids_final = np.vstack(res_list) if len(res_list) > 0 else np.zeros((0, 2))

        if float(np.linalg.norm(dx0)) < tol:
            break

    x0_hat = x0_star
    P0_hat = np.linalg.inv(Lambda)

    X_hat_hist, _ = propagate_state_and_stm_history(
        x0_ref_6=x0_hat,
        t_eval=t_meas,
        mu=mu, J2=J2, J3=J3,
        Re=Re,
        reltol=reltol,
        abstol=abstol,
    )

    postfit_resids = []
    for j, m in enumerate(all_meas):
        ti = float(m["t"])
        st = station_map[m["station"]]

        x_i_hat = X_hat_hist[j, :]

        d = st.measure(x_i_hat[:3], x_i_hat[3:], float(ti))
        if d is None:
            continue
        yhat_i_hat = np.array([d["rho_km"], d["rho_dot_km_s"]], dtype=float)

        Yi = np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)
        postfit_resids.append(Yi - yhat_i_hat)

    postfit_resids = np.vstack(postfit_resids) if len(postfit_resids) > 0 else np.zeros((0, 2))

    info = {
        "t0": t0,
        "t_meas": t_meas,
        "station_meas": st_meas,

        # iteration history
        "dx0_hist": dx0_hist,               # list of (6,) vectors
        "x0_star_hist": x0_star_hist,       # list of (6,) vectors
        "num_iters": len(dx0_hist),

        # residuals
        "prefit_resids_final": prefit_resids_final,  # (m,2)
        "postfit_resids": postfit_resids,            # (m,2)

        # final normal equation pieces (sometimes useful)
        "Lambda_final": Lambda,
        "N_final": N,
    }

    return x0_hat, P0_hat, info


def postprocess_batch_for_plots(
    all_meas,
    stations,
    x0_hat: np.ndarray,
    P0_hat: np.ndarray,
    mu: float,
    J2: float,
    J3: float,
    Re: float = 6378.0,
    truth_times: np.ndarray | None = None,
    truth_states_6: np.ndarray | None = None,
    reltol: float = 1e-10,
    abstol: float = 1e-10,
):
    """
    Post-processing helper to support:
      1) state error vs time with +/-2sigma bounds (if truth provided)
      2) postfit residual plots (also returned here for convenience)

    This propagates the FINAL batch estimate to each measurement time, computes:
      - xhat(ti)
      - Phi(ti,t0)
      - P(ti) = Phi P0_hat Phi^T
      - 2sigma(ti) from diag(P(ti))
      - state_error(ti) = xhat(ti) - xtrue(ti) if truth provided
      - postfit residuals: y_post(ti) = Y(ti) - yhat(xhat(ti))

    Parameters
    ----------
    all_meas : list[dict]
        Measurements (noisy), with keys: station, t, rho_km, rho_dot_km_s
    stations : list[Stations]
        Station objects
    x0_hat : (6,)
        Batch-estimated initial state
    P0_hat : (6,6)
        Batch-estimated covariance at t0
    truth_times : (N,) optional
        Times corresponding to truth_states_6
    truth_states_6 : (N,6) optional
        Truth trajectory [r;v] at truth_times
        If provided, state errors will be computed by interpolating truth onto measurement times.
        (Interpolation is component-wise linear.)
    Returns
    -------
    out : dict with arrays:
        t_meas               (m,)
        station_meas         list length m
        xhat_meas            (m,6)
        P_meas               (m,6,6)
        two_sigma_meas       (m,6)
        state_error_meas     (m,6) or None
        postfit_resids_meas  (m,2)
    """
    # Sort measurements and station lookup
    station_map = {st.name: st for st in stations}

    t_meas = np.array([float(m["t"]) for m in all_meas], dtype=float)
    X_hist, Phi_hist = propagate_state_and_stm_history(
        x0_ref_6=x0_hat,
        t_eval=t_meas,
        mu=mu, J2=J2, J3=J3,
        Re=Re,
        reltol=reltol,
        abstol=abstol,
    )
    st_meas = [m["station"] for m in all_meas]

    # Reference epoch for STM mapping
    t0 = float(all_meas[0]["t"])

    mcount = len(all_meas)
    xhat_meas = np.zeros((mcount, 6), dtype=float)
    P_meas = np.zeros((mcount, 6, 6), dtype=float)
    two_sigma = np.zeros((mcount, 6), dtype=float)

    # Keep fixed-size output; masked rows get NaN residuals
    postfit_resids = np.full((mcount, 2), np.nan, dtype=float)

    # If truth provided, interpolate truth to measurement times (component-wise)
    state_error = None
    xtrue_interp = None
    if truth_times is not None and truth_states_6 is not None:
        truth_times = np.asarray(truth_times, dtype=float).reshape(-1,)
        truth_states_6 = np.asarray(truth_states_6, dtype=float)
        if truth_states_6.shape[1] != 6:
            raise ValueError("truth_states_6 must be shape (N,6)")

        xtrue_interp = np.zeros((mcount, 6), dtype=float)
        for j in range(6):
            xtrue_interp[:, j] = np.interp(t_meas, truth_times, truth_states_6[:, j])
        state_error = np.zeros((mcount, 6), dtype=float)

    # Main loop over measurement times
    for i, m in enumerate(all_meas):
        ti = float(m["t"])
        st = station_map[m["station"]]

        # Propagate final estimate to ti, and get Phi(ti,t0)
        x_i = X_hist[i, :]
        Phi_i0 = Phi_hist[i, :, :]

        xhat_meas[i, :] = x_i

        # Covariance along the trajectory: P(ti) = Phi P0 Phi^T
        Pi = Phi_i0 @ P0_hat @ Phi_i0.T
        P_meas[i, :, :] = Pi

        # +/- 2 sigma bounds from diagonal
        two_sigma[i, :] = 2.0 * np.sqrt(np.maximum(np.diag(Pi), 0.0))

        # State error if truth provided
        if state_error is not None and xtrue_interp is not None:
            state_error[i, :] = x_i - xtrue_interp[i, :]

        # Postfit residuals at measurement times
        d = st.measure(x_i[:3], x_i[3:], float(ti))
        if d is None:
            continue
        yhat_i = np.array([d["rho_km"], d["rho_dot_km_s"]], dtype=float)

        Yi = np.array([m["rho_km"], m["rho_dot_km_s"]], dtype=float)
        postfit_resids[i, :] = Yi - yhat_i

    rms = {}

    # Helper: component-wise RMS
    def _rms_comp(A):
        return np.sqrt(np.mean(A**2, axis=0))

    # Helper: detect "first pass" as first contiguous block before a big time gap
    # (minimal assumption; works well for DSN-like pass scheduling)
    def _first_pass_mask(t, gap_factor=10.0):
        t = np.asarray(t, dtype=float).reshape(-1)
        if t.size < 2:
            return np.ones_like(t, dtype=bool)
        dt = np.diff(t)
        med = np.median(dt)
        if med <= 0:
            # fallback: no gap detection possible
            return np.ones_like(t, dtype=bool)
        # first gap much larger than typical spacing
        idx = np.where(dt > gap_factor * med)[0]
        if idx.size == 0:
            return np.ones_like(t, dtype=bool)
        cut = idx[0] + 1  # split after this index
        mask = np.zeros_like(t, dtype=bool)
        mask[:cut] = True
        return mask

    # State error RMS (requires truth)
    if state_error is not None:
        rms["state_comp_rms"] = _rms_comp(state_error)  # (6,)

        # 3D position + 3D velocity RMS
        pos_3d = np.linalg.norm(state_error[:, 0:3], axis=1)
        vel_3d = np.linalg.norm(state_error[:, 3:6], axis=1)
        rms["pos3d_rms"] = float(np.sqrt(np.mean(pos_3d**2)))
        rms["vel3d_rms"] = float(np.sqrt(np.mean(vel_3d**2)))

        # Ignore first pass (drop first pass, compute RMS on remaining)
        first_pass = _first_pass_mask(t_meas)
        keep = ~first_pass
        if np.any(keep):
            se2 = state_error[keep, :]
            rms["state_comp_rms_ignore_first_pass"] = _rms_comp(se2)
            pos_3d2 = np.linalg.norm(se2[:, 0:3], axis=1)
            vel_3d2 = np.linalg.norm(se2[:, 3:6], axis=1)
            rms["pos3d_rms_ignore_first_pass"] = float(np.sqrt(np.mean(pos_3d2**2)))
            rms["vel3d_rms_ignore_first_pass"] = float(np.sqrt(np.mean(vel_3d2**2)))
        else:
            rms["state_comp_rms_ignore_first_pass"] = None
            rms["pos3d_rms_ignore_first_pass"] = None
            rms["vel3d_rms_ignore_first_pass"] = None
    else:
        rms["state_comp_rms"] = None
        rms["pos3d_rms"] = None
        rms["vel3d_rms"] = None
        rms["state_comp_rms_ignore_first_pass"] = None
        rms["pos3d_rms_ignore_first_pass"] = None
        rms["vel3d_rms_ignore_first_pass"] = None

    # Postfit residual RMS (always available)
    rms["postfit_rms"] = _rms_comp(postfit_resids)  # (2,)

    # Ignore first pass for residual RMS
    first_pass = _first_pass_mask(t_meas)
    keep = ~first_pass
    if np.any(keep):
        rms["postfit_rms_ignore_first_pass"] = _rms_comp(postfit_resids[keep, :])
    else:
        rms["postfit_rms_ignore_first_pass"] = None

    



    return {
        "t_meas": t_meas,
        "station_meas": st_meas,
        "xhat_meas": xhat_meas,
        "P_meas": P_meas,
        "two_sigma_meas": two_sigma,
        "state_error_meas": state_error,            # None if no truth provided
        "postfit_resids_meas": postfit_resids,
        "rms": rms,
    }


meas_path = "meas_data/hw2_measurements_noisy.csv"  
df = pd.read_csv(meas_path)

# Convert to list-of-dicts like your batch expects
all_meas = df.to_dict(orient="records")

# -----------------------------
# 2) Stations (same as HW1)
# -----------------------------
stations = [
    Stations("Station 1", lat_deg=-35.398333, lon_deg=148.981944),
    Stations("Station 2", lat_deg=40.427222,  lon_deg=355.749444),
    Stations("Station 3", lat_deg=35.247164,  lon_deg=243.205000),
]

# 3) Define measurement noise R
sigma_rho_km = 1.0e-3          # 1 m = 1e-3 km
sigma_rhod_km_s = 1.0e-6       # 1 mm/s = 1e-6 km/s

R = np.diag([sigma_rho_km**2, sigma_rhod_km_s**2])

# -----------------------------
# 4) Define a priori covariance P0
#    σr = 1 km, σv = 1 m/s = 1e-3 km/s
# -----------------------------
sigma_r0_km = 1.0
sigma_v0_km_s = 1.0e-3

P0 = np.diag([
    sigma_r0_km**2, sigma_r0_km**2, sigma_r0_km**2,
    sigma_v0_km_s**2, sigma_v0_km_s**2, sigma_v0_km_s**2
])

# -----------------------------
# 5) Initial guess x0_bar
#    You said you have dx = [0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3]
#
#    IMPORTANT: x0_bar must be an actual initial state guess.
#    If you have a nominal truth/reference x0_true, you'd do:
#      x0_bar = x0_true + dx
#
#    If you *don't* have x0_true handy here yet, you must load it from
#    the truth CSV (first row of prob2a_traj.csv), or however HW specifies.
# -----------------------------
dx = np.array([0.1, -0.03, 0.25, 0.3e-3, -0.5e-3, 0.2e-3], dtype=float)

# Example: load truth IC from the truth trajectory CSV first row
truth_path = "../Homework1/prob2a_traj.csv"
truth_df = pd.read_csv(truth_path)
x0_true = np.array([
    truth_df.loc[0, "x_km"],
    truth_df.loc[0, "y_km"],
    truth_df.loc[0, "z_km"],
    truth_df.loc[0, "vx_km_s"],
    truth_df.loc[0, "vy_km_s"],
    truth_df.loc[0, "vz_km_s"],
], dtype=float)

x0_bar = x0_true + dx

# -----------------------------
# 6) Dynamics parameters (fill with your assignment values)
# -----------------------------
# NOTE: put the actual numbers your HW uses.
# If your HW1 had these in a JSON or constants section, reuse them.
mu = 398600.4418   # km^3/s^2 (example Earth)
J2 = 1.08262668e-3
J3 = -2.532153e-6  # example; set to whatever your file uses (or 0 if not used)

# -----------------------------
# 7) Run batch
# -----------------------------
x0_hat, P0_hat, info = batch_estimate_x0(
    all_meas=all_meas,
    stations=stations,
    x0_bar=x0_bar,
    P0=P0,
    R=R,
    mu=mu,
    J2=J2,
    J3=J3,
    max_iter=10,
    tol=1e-10,
    reltol=1e-10,
    abstol=1e-10,
)

print("\n=== Batch finished ===")
print("num_iters:", info["num_iters"])
print("x0_hat:", x0_hat)
print("diag(P0_hat):", np.diag(P0_hat))

# quick residual sanity
postfit = info["postfit_resids"]
print("postfit residuals shape:", postfit.shape)
print("postfit mean [rho, rhod]:", postfit.mean(axis=0))
print("postfit std  [rho, rhod]:", postfit.std(axis=0))

# -----------------------------
# 8) Postprocess (no plotting yet)
#    Pass truth if you want error vs 2sigma later
# -----------------------------
truth_times = truth_df["t_s"].to_numpy(float)
truth_states_6 = truth_df[["x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"]].to_numpy(float)

plot_data = postprocess_batch_for_plots(
    all_meas=all_meas,
    stations=stations,
    x0_hat=x0_hat,
    P0_hat=P0_hat,
    mu=mu,
    J2=J2,
    J3=J3,
    truth_times=truth_times,
    truth_states_6=truth_states_6,
    reltol=1e-10,
    abstol=1e-10,
    x0_star_hist=info["x0_star_hist"],   # <-- ADD THIS
    first_pass_gap_s=6*3600.0,           # optional; tune if you want
)


# print("\n=== Postprocess finished ===")
# print("xhat_meas shape:", plot_data["xhat_meas"].shape)
# print("P_meas shape:", plot_data["P_meas"].shape)
# print("two_sigma shape:", plot_data["two_sigma_meas"].shape)
# if plot_data["state_error_meas"] is None:
#     print("state_error_meas: None (truth not provided)")
# else:
#     print("state_error_meas shape:", plot_data["state_error_meas"].shape)

# print("postfit_resids_meas shape:", plot_data["postfit_resids_meas"].shape)




### Plot pretty please
t = plot_data["t_meas"]
err = plot_data["state_error_meas"]
two_sigma = plot_data["two_sigma_meas"]

if err is None:
    raise ValueError("plot_data['state_error_meas'] is None. Make sure you passed truth_times and truth_states_6.")

three_sigma =  (3.0/2.0) * two_sigma  # stored 2σ -> convert to 3σ

labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]

fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()

for i in range(6):
    ax = axs[i]

    # Smaller scatter points
    ax.scatter(t, err[:, i], s=6, marker="o", label="state error")

    # Force BOTH bounds to be the same color
    sigma_color = "k"  # black (change if you want)
    ax.plot(t, +three_sigma[:, i], linestyle="--", linewidth=1.5, color=sigma_color, label="+3σ")
    ax.plot(t, -three_sigma[:, i], linestyle="--", linewidth=1.5, color=sigma_color, label="-3σ")

    ax.set_ylabel(labels[i])
    ax.grid(True)

axs[-2].set_xlabel("t (s)")
axs[-1].set_xlabel("t (s)")

# Legend once
handles, leglabels = axs[0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc="upper right")

fig.suptitle("Batch State Error vs Time with ±3σ Bounds", y=0.98)
plt.tight_layout()



t = plot_data["t_meas"]
post = plot_data["postfit_resids_meas"]  # (m,2): [rho_km, rho_dot_km_s]

# If you want to overlay sigma bounds, use the same sigmas you used to build R
sigma_rho_km = 1.0e-3     # 1 m
sigma_rhod_km_s = 1.0e-6  # 1 mm/s

fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# --- Range residuals ---
axs[0].scatter(t, post[:, 0], s=6, marker="o")
axs[0].axhline(0.0, linewidth=1)
axs[0].axhline(+3*sigma_rho_km, linestyle="--", linewidth=1.5, color="k")
axs[0].axhline(-3*sigma_rho_km, linestyle="--", linewidth=1.5, color="k")
axs[0].set_ylabel("Postfit ρ residual (km)")
axs[0].grid(True)

# --- Range-rate residuals ---
axs[1].scatter(t, post[:, 1], s=6, marker="o")
axs[1].axhline(0.0, linewidth=1)
axs[1].axhline(+3*sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k")
axs[1].axhline(-3*sigma_rhod_km_s, linestyle="--", linewidth=1.5, color="k")
axs[1].set_ylabel("Postfit ρ̇ residual (km/s)")
axs[1].set_xlabel("t (s)")
axs[1].grid(True)

fig.suptitle("Postfit Residuals vs Time (with ±3σ noise bounds)", y=0.98)
plt.tight_layout()

rms = plot_data["rms_final"]

rms = plot_data["rms_final"]

rows = []

# Postfit residual RMS (always available)
rows.append({
    "Metric": "Postfit RMS: rho (km)",
    "All meas": rms["postfit_all"][0],
    "Ignore first pass": rms["postfit_ignore_first"][0],
})
rows.append({
    "Metric": "Postfit RMS: rho_dot (km/s)",
    "All meas": rms["postfit_all"][1],
    "Ignore first pass": rms["postfit_ignore_first"][1],
})

# State RMS (only if truth provided)
if rms["state_comp_all"] is not None:
    labels_state = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]
    for i, lab in enumerate(labels_state):
        rows.append({
            "Metric": f"State RMS: {lab}",
            "All meas": rms["state_comp_all"][i],
            "Ignore first pass": rms["state_comp_ignore_first"][i],
        })

    rows.append({
        "Metric": "3D RMS: ||r|| (km)",
        "All meas": rms["pos3_all"],
        "Ignore first pass": rms["pos3_ignore_first"],
    })
    rows.append({
        "Metric": "3D RMS: ||v|| (km/s)",
        "All meas": rms["vel3_all"],
        "Ignore first pass": rms["vel3_ignore_first"],
    })

df_rms = pd.DataFrame(rows)

pd.set_option("display.float_format", lambda x: f"{x:.3e}")
print("\n=== RMS Summary (Final Batch) ===")
print(df_rms.to_string(index=False))

# --- (iv) RMS vs iteration ---
rms_iter = plot_data["rms_by_iter"]
if rms_iter is None:
    print("No rms_by_iter returned. Did you pass x0_star_hist=info['x0_star_hist'] to postprocess?")
else:
    K = rms_iter["postfit_all"].shape[0]
    it = np.arange(K)  # iter index (0 = initial guess)

    # Postfit RMS vs iteration
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(it, rms_iter["postfit_all"][:, 0], marker="o", label="rho RMS (all)")
    ax.plot(it, rms_iter["postfit_all"][:, 1], marker="o", label="rho_dot RMS (all)")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.set_xlabel("Iteration index")
    ax.set_ylabel("RMS")
    ax.set_title("Postfit Residual RMS vs Iteration (Batch)")
    ax.legend()
    plt.tight_layout()

    # State RMS vs iteration (component-wise + 3D) only if truth exists
    if rms_iter["state_comp_all"] is not None:
        # 3D RMS vs iteration
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(it, rms_iter["pos3_all"], marker="o", label="||r|| RMS (all)")
        ax.plot(it, rms_iter["vel3_all"], marker="o", label="||v|| RMS (all)")
        ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.set_xlabel("Iteration index")
        ax.set_ylabel("RMS")
        ax.set_title("3D State RMS vs Iteration (Batch)")
        ax.legend()
        plt.tight_layout()

        # Component RMS vs iteration (6 lines)
        labels_state = ["x", "y", "z", "vx", "vy", "vz"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(6):
            ax.plot(it, rms_iter["state_comp_all"][:, i], marker="o", label=labels_state[i])
        ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.set_xlabel("Iteration index")
        ax.set_ylabel("RMS")
        ax.set_title("State Component RMS vs Iteration (Batch)")
        ax.legend(ncol=3)
        plt.tight_layout()

plt.show()