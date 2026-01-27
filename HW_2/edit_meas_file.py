import pandas as pd

def convert_to_compact_format(input_csv, output_csv):
    # Load the "verbose" format: (station, t, rho_km, rho_dot_km_s, elev_rad)
    df = pd.read_csv(input_csv)

    # 1. Map columns to the "reference" format
    # Target: Time(s), Range(km), Range_Rate(km/s), Station_ID
    mapping = {
        't': 'Time(s)',
        'rho_km': 'Range(km)',
        'rho_dot_km_s': 'Range_Rate(km/s)'
    }
    
    df_converted = df.rename(columns=mapping)

    # 2. Convert Station Name string (e.g., "Station 2") to ID (e.g., 2)
    # This splits the string and takes the last element as an integer
    df_converted['Station_ID'] = df_converted['station'].apply(
        lambda x: int(str(x).split()[-1])
    )

    # 3. Select and reorder the specific columns required
    cols = ['Time(s)', 'Range(km)', 'Range_Rate(km/s)', 'Station_ID']
    df_final = df_converted[cols]

    # 4. Save to CSV
    df_final.to_csv(output_csv, index=False)
    print(f"Successfully converted to compact format: {output_csv}")

# Example usage:
convert_to_compact_format(r'HW_2/measurements_noisy_3.csv', 'HW_2/measurements_noisy_3_compact.csv')