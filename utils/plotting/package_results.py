import numpy as np

def package_results(results, obs, options={}):
    """
    Packages results and ensures all arrays are the exact same length.
    Solves the "Obs: 6126, States: 6125" mismatch globally.
    """
    # 1. Extract Raw Filter Data
    x_hist = np.array(results.dx_hist)
    state_hist = np.array(results.state_hist)
    P_hist = np.array(results.P_hist)
    prefit_residuals = np.array(results.innovations)
    postfit_residuals = np.array(results.postfit_residuals)
    nis_hist = np.array(results.nis_hist)
    
    # 2. Extract Raw Times
    times = obs['Time(s)'].values

    # 3. GLOBAL ALIGNMENT (The Fix)
    # We trust the filter output length (n_filter) as the "truth" of what was processed.
    n_filter = len(x_hist)
    n_times = len(times)

    if n_times > n_filter:
        # Case: Obs has 6126, Filter has 6125.
        # This usually means the filter treated index 0 as t0 and started estimating at index 1.
        # We slice off the START of the times vector to align with the filter.
        diff = n_times - n_filter
        times = times[diff:]
        
    elif n_filter > n_times:
        # Case: Filter includes Initial Guess (t0), but Times does not.
        # We slice off the START of the filter vectors.
        diff = n_filter - n_times
        print(f"Aligning data: Trimming first {diff} filter states (initial guess).")
        x_hist = x_hist[diff:]
        state_hist = state_hist[diff:]
        P_hist = P_hist[diff:]
        prefit_residuals = prefit_residuals[diff:] if len(prefit_residuals) > n_times else prefit_residuals
        postfit_residuals = postfit_residuals[diff:] if len(postfit_residuals) > n_times else postfit_residuals
        nis_hist = nis_hist[diff:] if len(nis_hist) > n_times else nis_hist

    # 4. Double Check Lengths (Safety Net)
    # If there is still a mismatch (e.g., jagged arrays), truncate to minimum common length
    min_len = min(len(times), len(x_hist))
    times = times[:min_len]
    x_hist = x_hist[:min_len]
    state_hist = state_hist[:min_len]
    P_hist = P_hist[:min_len]
    # Handle residuals separately as they might be empty or different
    if len(prefit_residuals) > min_len: prefit_residuals = prefit_residuals[:min_len]
    if len(postfit_residuals) > min_len: postfit_residuals = postfit_residuals[:min_len]
    if len(nis_hist) > min_len: nis_hist = nis_hist[:min_len]

    # 5. Apply User Mask (data_mask_idx)
    raw_mask = options.get('data_mask_idx')
    if raw_mask is None:
        idx_mask = slice(None)
    elif isinstance(raw_mask, int):
        idx_mask = slice(raw_mask, None)
    else:
        idx_mask = raw_mask

    return {
        'x_hist': x_hist[idx_mask],
        'state_hist': state_hist[idx_mask],
        'P_hist': P_hist[idx_mask],
        'prefit_residuals': prefit_residuals[idx_mask],
        'postfit_residuals': postfit_residuals[idx_mask],
        'nis_hist': nis_hist[idx_mask],
        'times': times[idx_mask]
    }