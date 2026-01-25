import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
# Assuming lkf_reference has columns: [Time, x, y, z, vx, vy, vz]
# Assuming HW1_truth has columns: [Time, x, y, z, vx, vy, vz]
ref_df = pd.read_csv('HW_2/lkf_reference_trajectory.csv', delimiter=',')
truth_df = pd.read_csv('HW_2/HW1_truth.csv', delimiter=' ')

# Save both to numpy arrays for easier manipulation
ref_states = ref_df.to_numpy()
truth_states = truth_df.to_numpy()

# Match times
common_times = np.intersect1d(ref_states[:, 0], truth_states[:, 0])
ref_common = ref_states[np.isin(ref_states[:, 0], common_times)]

truth_common = truth_states[np.isin(truth_states[:, 0], common_times)]

# Compute errors
errors = ref_common[:, 1:] - truth_common[:, 1:]
position_errors = np.linalg.norm(errors[:, :3], axis=1)
velocity_errors = np.linalg.norm(errors[:, 3:], axis=1)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(common_times, position_errors, label='Position Error (m)')
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Reference Trajectory Position Error Over Time')
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(common_times, velocity_errors, label='Velocity Error (m/s)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Error (m/s)')
plt.title('Reference Trajectory Velocity Error Over Time')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()