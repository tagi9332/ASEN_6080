import numpy as np

def package_results(results, obs, options={}):
    """
    Packages results. 
    Handles 'data_mask_idx' logic:
      - If None: returns everything.
      - If int (e.g., 300): returns slice(300, None) -> skips first 300.
      - If slice object: uses it directly.
    """
    # 1. Determine the Mask
    raw_mask = options.get('data_mask_idx')

    if raw_mask is None:
        idx_mask = slice(None) # Select all
    elif isinstance(raw_mask, int):
        # Interpret integer as "Skip the first N elements"
        idx_mask = slice(raw_mask, None)
    else:
        # Assume it's already a slice or list of indices
        idx_mask = raw_mask

    # 2. Extract Raw Data
    # Convert to numpy arrays first to ensure slicing works
    x_hist = np.array(results.dx_hist)
    state_hist = np.array(results.state_hist)
    P_hist = np.array(results.P_hist)
    prefit_residuals = np.array(results.prefit_residuals)
    postfit_residuals = np.array(results.postfit_residuals)
    nis_hist = np.array(results.nis_hist)
    times = obs['Time(s)'].values

    # 3. Apply Mask
    # We apply the mask to all arrays so they stay synchronized
    return {
        'x_hist': x_hist[idx_mask],
        'state_hist': state_hist[idx_mask],
        'P_hist': P_hist[idx_mask],
        'prefit_residuals': prefit_residuals[idx_mask],
        'postfit_residuals': postfit_residuals[idx_mask],
        'nis_hist': nis_hist[idx_mask],
        'times': times[idx_mask]
    }