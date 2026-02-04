import numpy as np

def run_iterative_LKF(lkf_filter, obs, current_X0, P0_diag, Rk, Q, options,num_iterations_max=1, tol=1e-3):
    """
    Runs the Iterative Linearized Kalman Filter for Project 1.
    """
    # ============================================================
    # Iterative LKF Loop
    # ============================================================
    num_iterations_max = num_iterations_max
    tol = tol
    results = None

    # Initialize storage
    rms_hist = []
    results = None
    current_x0_dev = np.zeros(18)

    print(f"{'='*60}")
    print(f"Starting Iterative LKF ({num_iterations_max} iterations)")
    print(f"{'='*60}")

    for i in range(num_iterations_max):
        print(f"\n--- Iteration {i+1} / {num_iterations_max} ---")

        # Run LKF
        results = lkf_filter.run(obs, current_X0, current_x0_dev, P0_diag, Rk, Q, options)
        
        # Compute RMS of Post-Fit Residuals
        postfit_residuals = results.postfit_residuals
        rms_range = np.sqrt(np.mean(postfit_residuals[:,0]**2))
        rms_range_rate = np.sqrt(np.mean(postfit_residuals[:,1]**2))
    
        # Store RMS
        rms_hist.append((rms_range, rms_range_rate))

        # Print RMS
        print(f"   Post-Fit Residuals RMS: Range: {rms_range:.6f} m | Range-Rate: {rms_range_rate:.6f} m/s")

        # Check for convergence
        if rms_range < tol and rms_range_rate < tol:
            print("   Convergence criteria met. Stopping iterations.")
            break

        # Set up for next iteration
        x_est_final = results.dx_hist[-1]
        X_state_final = results.state_hist[-1]

        # Extract final STM
        phi_final = results.phi_hist[-1].reshape((18, 18))

        # Back-propagate to get initial deviation and state
        x0_init_est = np.linalg.inv(phi_final) @ x_est_final
        current_X0 = current_X0 + x0_init_est

        # Print
        print(f"   Updating Initial State Deviation for Next Iteration.")

    return results, current_X0,