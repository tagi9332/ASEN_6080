import os
import numpy as np
import pprint
from datetime import datetime

def save_run_log(save_folder, config):
    """
    Writes a dictionary of configuration settings to a text file.
    Handles Numpy arrays and nested dictionaries gracefully.
    """
    log_path = os.path.join(save_folder, "run_log.txt")
    
    try:
        with open(log_path, "w") as f:
            f.write(f"Run Log Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for key, val in config.items():
                f.write(f"[{key}]\n")
                
                if isinstance(val, np.ndarray):
                    # Pretty print numpy arrays with precision
                    val_str = np.array2string(val, separator=', ', precision=8, suppress_small=True)
                    f.write(f"{val_str}\n")
                elif isinstance(val, dict):
                    # Pretty print nested dictionaries
                    f.write(pprint.pformat(val, indent=4) + "\n")
                else:
                    f.write(f"{val}\n")
                
                f.write("\n" + "-"*30 + "\n\n")
                
        print(f"Configuration log saved to: {log_path}")
    except Exception as e:
        print(f"Warning: Failed to save run log. {e}")