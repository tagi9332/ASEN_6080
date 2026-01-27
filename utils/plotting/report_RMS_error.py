import numpy as np

def report_filter_metrics(times, state_errors, postfit_residuals, filter_name="Filter", ignore_first_value=True):
    """
    Reports State RMS errors (component-wise & 3D) and Post-fit Residual RMS.
    
    Parameters:
    times (ndarray): Array of simulation timestamps.
    state_errors (ndarray): Nx6 array (x, y, z, vx, vy, vz) in km and km/s.
    postfit_residuals (ndarray): Nx2 array (range, range-rate) in km and km/s.
    filter_name (str): Name of the filter for the print header.
    ignore_first_value (bool): If True, excludes the first data point from RMS calculations.
    """
    
    # Apply slicing if ignore_first_value is True
    if ignore_first_value:
        state_errors = state_errors[1:]
        postfit_residuals = postfit_residuals[1:]
        # times = times[1:] # Optional: slice times if you use them for calculations

    # 1. State RMS Errors (Component-wise)
    rms_state = np.sqrt(np.mean(state_errors**2, axis=0))
    
    # 2. 3D State RMS Errors (RSS of the components)
    rms_3d_pos = np.sqrt(np.mean(np.sum(state_errors[:, :3]**2, axis=1)))
    rms_3d_vel = np.sqrt(np.mean(np.sum(state_errors[:, 3:]**2, axis=1)))
    
    # 3. Post-fit Residual RMS
    rms_residuals = np.sqrt(np.mean(postfit_residuals**2, axis=0))

    print(f"\n" + "="*50)
    print(f"REPORT: {filter_name}")
    if ignore_first_value:
        print(" (Note: Initial state/residual excluded from metrics)")
    print("="*50)
    
    print(f"--- State RMS Errors (Component-wise) ---")
    print(f"X:  {rms_state[0]:.6e} km")
    print(f"Y:  {rms_state[1]:.6e} km")
    print(f"Z:  {rms_state[2]:.6e} km")
    print(f"VX: {rms_state[3]:.6e} km/s")
    print(f"VY: {rms_state[4]:.6e} km/s")
    print(f"VZ: {rms_state[5]:.6e} km/s")
    
    print(f"\n--- State RMS Errors (3D) ---")
    print(f"Position (3D): {rms_3d_pos:.6e} km")
    print(f"Velocity (3D): {rms_3d_vel:.6e} km/s")
    
    print(f"\n--- Post-fit Residual RMS ---")
    print(f"Range:      {rms_residuals[0]*1e3:.6f} m")
    print(f"Range-Rate: {rms_residuals[1]*1e3:.6f} mm/s")
    print("="*50 + "\n")

    return {
        "rms_state": rms_state,
        "rms_3d_pos": rms_3d_pos,
        "rms_3d_vel": rms_3d_vel,
        "rms_residuals": rms_residuals
    }