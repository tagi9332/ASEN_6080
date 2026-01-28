import pandas as pd
import numpy as np

def compute_state_error(x_corrected, df_meas, truth_file):
    """
    Compute the state estimation error by comparing the corrected state estimates
    against the ground truth data.

    :param x_corrected: numpy array of shape (N, 6) containing the corrected state estimates
                        [x, y, z, vx, vy, vz] at each measurement time.
    :param df_meas: pandas DataFrame containing the measurement data with a 'Time(s)' column.
                    The ground truth data is expected to be in 'HW_2/HW1_truth.csv'.
    :return: numpy array of shape (N, 6) containing the state estimation errors.
    """

    # 1. Load Truth Data with the correct delimiter
    df_truth = pd.read_csv(truth_file, sep=r'\s+', header=None)

    # 2. Assign column names for clarity
    df_truth.columns = ['Time', 'x', 'y', 'z', 'vx', 'vy', 'vz']

    # 3. Align Truth with Estimation Times
    df_merged = pd.merge(df_meas, df_truth, left_on='Time(s)', right_on='Time', suffixes=('_meas', '_truth'))

    # Extract the aligned truth state as a numpy array (N x 6)
    truth_state = df_merged[['x', 'y', 'z', 'vx', 'vy', 'vz']].values

    # 4. Calculate State Errors (Estimation - Truth)
    state_errors = x_corrected - truth_state
    state_errors_m = state_errors
    return state_errors_m