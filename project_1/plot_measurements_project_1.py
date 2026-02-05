import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Measurements
obs = pd.read_csv(fr'data\project_1_obs.csv')

# Plot Measurements
time_eval = obs['Time(s)'].values
range_meas = obs['Range(m)'].values
range_rate_meas = obs['Range_Rate(m/s)'].values

# Set station colors
station_colors = {
    101: 'r',
    337: 'g',
    394: 'b'
}

# Plot Range and Range Rate Measurements
plt.figure(figsize=(12, 6))
for station_id, group in obs.groupby('Station_ID'):
    plt.subplot(2, 1, 1)
    plt.scatter(group['Time(s)'], group['Range(m)'], label=f'Station {station_id}', color=station_colors[station_id], s=3)
    plt.title('Range Measurements')
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.scatter(group['Time(s)'], group['Range_Rate(m/s)'], label=f'Station {station_id}', color=station_colors[station_id], s=3)
    plt.title('Range Rate Measurements')
    plt.xlabel('Time (s)')
    plt.ylabel('Range Rate (m/s)')
plt.tight_layout()
plt.show()
