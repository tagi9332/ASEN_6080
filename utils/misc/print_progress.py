def print_progress(k: int, total_steps: int):
    # Calculate checkpoints
    q1 = total_steps // 4
    q2 = total_steps // 2
    q3 = (3 * total_steps) // 4
    
    if k == 0:
        print("Progress: 0%")
    elif k == q1:
        print("Progress: 25%")
    elif k == q2:
        print("Progress: 50%")
    elif k == q3:
        print("Progress: 75%")
    elif k == total_steps - 1:
        print("Progress: 100%")