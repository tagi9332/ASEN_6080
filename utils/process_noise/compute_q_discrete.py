import numpy as np
from scipy.linalg import expm

def compute_q_discrete(dt: float, state: np.ndarray, options: dict) -> np.ndarray:
    """
    Calculate the discrete process noise covariance matrix (Q_disc) numerically.
    
    Method:
        Q_k = Integral[ Phi(tau) * Q_continuous * Phi(tau)^T d_tau ] from 0 to dt
        Approximated via Trapezoidal integration or Van Loan.
    
    Args:
        dt (float): Time step in seconds.
        state (np.ndarray): State vector [rx, ry, rz, vx, vy, vz, ...] (6+ elements).
        options (dict): Dictionary containing configuration:
            - 'Q_cont' (np.ndarray): Continuous process noise PSD (3x3).
            - 'threshold' (float): Maximum allowed dt.
            - 'frame_type' (str): 'ECI' or 'RIC'.
            - 'method' (str): 'SNC' or 'DMC'.
            - 'B' (np.ndarray): Time-constant matrix (required for DMC, 3x3).
            
    Returns:
        np.ndarray: Discrete process noise covariance matrix (6x6 or 9x9).
    """
    
    # 1. Setup & Checks
    Q_cont_3x3 = options['Q_cont'].copy()
    threshold = options['threshold']
    dt = min(dt, threshold)
    
    method = options['method']
    frame_type = options.get('frame_type', 'ECI')
    
    # 2. Construct the Continuous Process Noise Matrix (Q_c_full) for the full state
    #    and the System Dynamics Matrix (A_continuous)
    
    if method == 'SNC':
        n = 6
        # A matrix for SNC (Double Integrator): 
        # r_dot = v, v_dot = noise
        # [ 0 I ]
        # [ 0 0 ]
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        
        # Q continuous for full state (6x6)
        # Noise only enters at Velocity level (indices 3:6)
        Q_c_full = np.zeros((6, 6))
        
        # Handle Frame Rotation
        if frame_type == 'RIC':
            R_eci_to_ric = _transform_to_ric_frame(state)
            # Rotate noise: Q_eci = R.T * Q_ric * R
            Q_noise_eci = R_eci_to_ric.T @ Q_cont_3x3 @ R_eci_to_ric
            Q_c_full[3:6, 3:6] = Q_noise_eci
        else:
            Q_c_full[3:6, 3:6] = Q_cont_3x3
            
    elif method == 'DMC':
        n = 9
        B = options.get('B')
        if B is None:
             raise ValueError("Matrix 'B' is required for DMC.")
             
        # A matrix for DMC (9x9):
        # r_dot = v
        # v_dot = a_grav + a_dmc (Linearized: 0*r + 0*v + I*a_dmc)
        # a_dmc_dot = -B * a_dmc + noise
        
        A = np.zeros((9, 9))
        A[0:3, 3:6] = np.eye(3) # v = dr/dt
        A[3:6, 6:9] = np.eye(3) # a = dv/dt (connection from DMC acc to velocity)
        A[6:9, 6:9] = -B        # Decay of DMC correlation
        
        # Q continuous for full state (9x9)
        # Noise enters at Acceleration level (indices 6:9)
        Q_c_full = np.zeros((9, 9))
        
        # Handle Frame Rotation
        if frame_type == 'RIC':
            R_eci_to_ric = _transform_to_ric_frame(state)
            # Rotate noise: Q_eci = R.T * Q_ric * R
            Q_noise_eci = R_eci_to_ric.T @ Q_cont_3x3 @ R_eci_to_ric
            Q_c_full[6:9, 6:9] = Q_noise_eci
        else:
            Q_c_full[6:9, 6:9] = Q_cont_3x3
            
    else:
        raise ValueError(f"Unknown method: {method}")

    # 3. Numerical Integration (Trapezoidal Rule)
    # This is often safer and cleaner than Van Loan for simple linear systems
    # Formula: Q_k approx (Q_c + Phi*Q_c*Phi.T) * dt / 2
    
    # Calculate STM for this step (Phi = expm(A * dt))
    # Since A is constant over the small step dt:
    Phi = expm(A * dt)
    
    # Trapezoidal integration of Q(t) = Phi(t) * Q_c * Phi(t).T
    # We approximate the integral as area of trapezoid formed by t=0 and t=dt
    
    # At t=0: Phi(0) = I, so integrand is Q_c_full
    Integrand_0 = Q_c_full
    
    # At t=dt: Integrand is Phi * Q_c_full * Phi.T
    Integrand_dt = Phi @ Q_c_full @ Phi.T
    
    # Area = (Height_0 + Height_dt) * dt / 2
    Q_discrete = (Integrand_0 + Integrand_dt) * dt * 0.5
    
    return Q_discrete

def _transform_to_ric_frame(state: np.ndarray) -> np.ndarray:
    """
    Computes the rotation matrix from Inertial (ECI) to Radial-InTrack-CrossTrack (RIC).
    """
    r = state[0:3]
    v = state[3:6]
    
    # Radial unit vector
    u_r = r / np.linalg.norm(r)
    
    # Cross-track (Normal) unit vector (Angular Momentum direction)
    h = np.cross(r, v)
    u_c = h / np.linalg.norm(h)
    
    # In-track unit vector (completes the triad)
    u_i = np.cross(u_c, u_r)
    
    # Rotation Matrix (Rows are the new basis vectors)
    # Transform v_RIC = R * v_ECI
    R_eci_to_ric = np.vstack([u_r, u_i, u_c])
    
    # We want ECI->RIC? Or RIC->ECI?
    # Usually we want to map RIC noise -> ECI frame.
    # Q_eci = R.T @ Q_ric @ R  (where R is ECI->RIC)
    # The calling code expects R_eci_to_ric for the R.T part? 
    # Yes, see usage above: R_eci_to_ric.T @ Q @ R_eci_to_ric
    
    return R_eci_to_ric