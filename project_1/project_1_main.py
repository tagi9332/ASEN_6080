import sys, os
import subprocess
# ============================================================
# Main Execution Script for Project 1
#     Author: Tanner Gill
#     Date: Feb 2026
#     All plots generated are saved in timestamped folders within 'results' folder
# ============================================================

# ============================================================
# Part 2/3: Run batch and LKF scripts with full observation set with original covariance matrix)
# ============================================================
# Run batch script
subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_1_batch.py'))])

# Run LKF script
subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_1_LKF.py'))])
breakpoint()

# ============================================================
# Part 5: Range and range-rate exclusion tests
# ============================================================
subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_1_batch_5.py'))])
breakpoint()

# ============================================================
# Part 6a: Un-fix station 101
# ============================================================
# Run batch script for part 6a
subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_1_batch_6a.py'))])

# Run LKF script for part 6a
subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_1_LKF_6a.py'))])
breakpoint()

# ============================================================
# Part 6b: Fix station 337
# ============================================================
# Run batch script for part 6b
subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_1_batch_6b.py'))])

# Run LKF script for part 6b
subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_1_LKF_6b.py'))])
breakpoint()


# ============================================================
# Part 7: Potter Square Root Implementation of LKF
# ============================================================
# Run Potter LKF script
subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_1_LKF_potter.py'))])
breakpoint()

# ============================================================
# Part 9: Iterative LKF implementation
# ============================================================
subprocess.run([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), 'project_1_LKF_iterative.py'))])
