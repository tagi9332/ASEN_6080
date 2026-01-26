import pandas as pd

# 1. Load the original noisy measurements
# 'sep=None' helps pandas detect if it's strictly comma or space-separated
df = pd.read_csv('HW_2/measurements_noisy_reece.csv', sep=None, engine='python')

# 2. Rename the columns to match the target format
# Mapping: t -> Time(s), rho -> Range(km), rhodot -> Range_Rate(km/s), station_id -> Station_ID
mapping = {
    't': 'Time(s)',
    'rho': 'Range(km)',
    'rhodot': 'Range_Rate(km/s)',
    'station_id': 'Station_ID'
}

# 3. Select only the columns we need and rename them
df_converted = df[['t', 'rho', 'rhodot', 'station_id']].rename(columns=mapping)

# 4. Clean up the data types
# Ensure Station_ID is an integer and Time/Range/Rate are floats
df_converted['Station_ID'] = df_converted['Station_ID'].astype(int)
df_converted['Time(s)'] = df_converted['Time(s)'].round(1)

# 5. Save to the new CSV format
output_filename = 'measurements_formatted.csv'
df_converted.to_csv(output_filename, index=False)

print(f"Successfully converted data. Saved to: {output_filename}")

# --- Quick Preview ---
print("\nFirst 5 rows of formatted data:")
print(df_converted.head())