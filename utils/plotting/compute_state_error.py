import pandas as pd
import numpy as np

def compute_state_error(x_corrected, df_meas, truth_file):
    """
    Compute state estimation error with robust time-alignment.

    Args:
        x_corrected (np.ndarray): (N, 6) array of state estimates.
        df_meas (pd.DataFrame): Dataframe containing 'Time(s)' column.
        truth_file (str): Path to the truth data file.
    """

    # 1. Internal length validation
    if len(x_corrected) != len(df_meas):
        # Truncate to the shorter length to avoid one-off index errors
        min_len = min(len(x_corrected), len(df_meas))
        x_corrected = x_corrected[:min_len]
        df_meas = df_meas.iloc[:min_len]

    # 2. Load Truth Data
    df_truth = pd.read_csv(truth_file, sep=r'\s+', header=None)
    df_truth.columns = ['Time', 'x', 'y', 'z', 'vx', 'vy', 'vz']

    # 3. Create a DataFrame for the filter estimates
    df_est = pd.DataFrame(x_corrected, columns=['x_e', 'y_e', 'z_e', 'vx_e', 'vy_e', 'vz_e'])
    # Assign the times from the filtered observation dataframe
    df_est['Time_Est'] = df_meas['Time(s)'].values

    # 4. Align Truth with Estimation Times using an Inner Join
    # This handles any remaining gaps or offsets in the time series
    df_aligned = pd.merge(
        df_est, 
        df_truth, 
        left_on='Time_Est', 
        right_on='Time', 
        how='inner'
    )

    if len(df_aligned) == 0:
        raise ValueError("Alignment failed: No matching timestamps found between estimates and truth.")

    # 5. Extract aligned components and compute error
    est_state = df_aligned[['x_e', 'y_e', 'z_e', 'vx_e', 'vy_e', 'vz_e']].values
    truth_state = df_aligned[['x', 'y', 'z', 'vx', 'vy', 'vz']].values

    return est_state - truth_state