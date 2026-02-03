import csv

def convert_txt_to_csv(input_filename, output_filename):
    try:
        with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
            
            # Initialize CSV writer
            writer = csv.writer(outfile)
            
            # Write the specific header requested
            header = ['Time(s)', 'Range(m)', 'Range_Rate(m/s)', 'Station_ID']
            writer.writerow(header)
            
            # Process the text file line by line
            for line in infile:
                # remove leading/trailing whitespace and split by spaces
                parts = line.strip().split()
                
                # Skip empty lines
                if not parts:
                    continue
                
                try:
                    # Parse columns based on your example:
                    # Col 0: Time
                    # Col 1: Station ID
                    # Col 2: Range (Meters)
                    # Col 3: Range Rate (Meters/Second)
                    
                    time_val = float(parts[0])
                    station_id = parts[1] # Keep as string or int
                    range_m = float(parts[2])
                    range_rate_ms = float(parts[3])
                                        
                    # Write row in the new order: Time, Range, Range_Rate, Station_ID
                    writer.writerow([time_val, range_m, range_rate_ms, station_id])
                    
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
                    
        print(f"Success! Converted '{input_filename}' to '{output_filename}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")

# --- Execution ---
if __name__ == "__main__":
    convert_txt_to_csv(r"D:\tanne\Documents\ASEN_6080\data\project.txt", r"D:\tanne\Documents\ASEN_6080\data\project_1_obs.csv")