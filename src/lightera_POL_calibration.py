# src/lightera_POL_calibration.py
import numpy as np
from scipy.optimize import least_squares

def fit_power_row(D: np.ndarray, target_power: np.ndarray = None) -> np.ndarray:
    """
    Solves for the top row (C0) of the calibration matrix using linear least squares.
    
    The top row maps the 4 detector voltages (D) to the total optical power (S0).
    If target_power is None, assumes constant power = 1.0 for all calibration points.
    
    Args:
        D: (4, N) matrix of averaged detector voltages for N calibration SOPs.
        target_power: (N,) array of measured optical powers. If None, uses ones.
    
    Returns:
        C0: (4,) array containing the top row coefficients [C00, C01, C02, C03].
    """
    if target_power is None:
        target_power = np.ones(D.shape[1])
    
    # We want to solve: D.T @ C0 = target_power
    # Using least squares: C0 = argmin || D.T @ C0 - target_power ||^2
    C0, _, _, _ = np.linalg.lstsq(D.T, target_power, rcond=None)
    return C0

def generate_tetrahedral_guess(C0: np.ndarray) -> np.ndarray:
    """
    Generates a 4x4 initial guess for the full calibration matrix based on an ideal tetrahedron.
    
    This follows Equation (9) from the paper. The top row is fixed to the fitted C0.
    The bottom three rows assume the 4 detector projections point to the vertices
    of a tetrahedron on the Poincare sphere.
    
    Args:
        C0: (4,) fitted top row coefficients.
    
    Returns:
        C_guess: (4, 4) initial calibration matrix.
    """
    eta = np.mean(C0)  # Average scale factor from the top row
    
    # Row 0: directly use the fitted power coefficients
    row0 = C0.copy()
    
    # Row 1: [3η, -η, -η, -η] / 4
    row1 = np.array([3.0 * eta, -eta, -eta, -eta]) / 4.0
    
    # Row 2: [0, 2η√2, -η√2, -η√2] / 4
    row2 = np.array([0.0, 2.0 * eta * np.sqrt(2), -eta * np.sqrt(2), -eta * np.sqrt(2)]) / 4.0
    
    # Row 3: [0, 0, η√6, -η√6] / 4
    row3 = np.array([0.0, 0.0, eta * np.sqrt(6), -eta * np.sqrt(6)]) / 4.0
    
    C_guess = np.vstack([row0, row1, row2, row3])
    return C_guess

def dop_residuals(params_flat: np.ndarray, D: np.ndarray, C0: np.ndarray) -> np.ndarray:
    """
    Residual function for the non-linear least squares optimizer.
    
    Computes the DOP error for all calibration points given the current guess
    of the bottom 3 rows of the calibration matrix.
    Residual = (S1^2 + S2^2 + S3^2 - S0^2) / (S0^2)
    This matches the metric Q in Equation (8) of the paper.
    
    Args:
        params_flat: (12,) flattened bottom 3 rows of C.
        D: (4, N) matrix of detector voltages.
        C0: (4,) fixed top row.
    
    Returns:
        residuals: (N,) array of DOP errors for each calibration SOP.
    """
    # Reconstruct the full 4x4 calibration matrix
    C_bottom = params_flat.reshape(3, 4)
    C = np.vstack([C0.reshape(1, 4), C_bottom])
    
    # Compute the Stokes vectors for all calibration points
    S = C @ D  # Shape: (4, N)
    S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]
    
    # Calculate residual: (S1^2 + S2^2 + S3^2 - S0^2) / (S0^2)
    # This forces DOP = 1
    residual = (S1**2 + S2**2 + S3**2 - S0**2) / (S0**2 + 1e-12)  # Add epsilon to avoid division by zero
    return residual

def calibrate_polarimeter(D_matrix: np.ndarray, constant_power: bool = True) -> tuple:
    """
    Performs the complete reference-free calibration of the polarimeter.
    
    Steps:
        1. Fit the top row (power coefficients) assuming constant power = 1.
        2. Generate a tetrahedral initial guess for the full matrix.
        3. Use non-linear least squares to refine the bottom 3 rows,
           enforcing DOP = 1 for all calibration points.
    
    Args:
        D_matrix: (4, N) matrix where each column is the averaged (4,) detector
                  readings for one calibration SOP. N must be >= 20.
        constant_power: If True, assumes S0 = 1 for all points. If False,
                        you must modify the function to accept a power vector.
    
    Returns:
        C_matrix: (4, 4) fully calibrated matrix.
        final_dop: (N,) array of DOPs computed with the final matrix (should be ~1).
    """
    if D_matrix.shape[1] < 12:
        print("Warning: Very few calibration points (<12). Recommend at least 20 for convergence.")
    
    # Step 1: Fit the top row
    if constant_power:
        C0 = fit_power_row(D_matrix, target_power=None)
    else:
        # Placeholder: In a future iteration, you can pass a power vector
        raise ValueError("Non-constant power requires measured power vector input.")
    
    # Step 2: Generate initial guess for the full matrix
    C_guess = generate_tetrahedral_guess(C0)
    initial_params = C_guess[1:].flatten()  # Flatten the bottom 3 rows
    
    # Step 3: Non-linear optimization to enforce DOP = 1
    result = least_squares(
        dop_residuals,
        initial_params,
        args=(D_matrix, C0),
        method='trf',  # Trust Region Reflective, robust for small-to-medium problems
        max_nfev=1000,
        verbose=0
    )
    
    # Reconstruct the final calibration matrix
    C_bottom_final = result.x.reshape(3, 4)
    C_matrix = np.vstack([C0.reshape(1, 4), C_bottom_final])
    
    # Compute final DOPs to verify calibration quality
    S = C_matrix @ D_matrix
    S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]
    final_dop = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-12)
    
    return C_matrix, final_dop

# Add this function to the existing calibration.py file

def generate_synthetic_calibration_data(num_sop: int = 30, noise_std: float = 1e-3, seed: int = 42) -> tuple:
    """
    Generates synthetic raw detector data (D_matrix) for validating the calibration routine.
    
    Assumptions:
        - True polarimeter follows the ideal tetrahedral projection (Equation 9 in the paper).
        - Input SOPs are perfectly polarized (DOP = 1) with constant power (S0 = 1).
        - Shot/Johnson noise is modeled as additive zero-mean Gaussian noise on the detector voltages.
    
    Args:
        num_sop: Number of random SOPs to generate (recommended >= 20).
        noise_std: Standard deviation of Gaussian noise added to each detector value.
        seed: Random seed for reproducibility.
    
    Returns:
        D_matrix: (4, num_sop) matrix of synthetic noisy detector readings.
        C_true: (4, 4) ground-truth calibration matrix used to generate the data.
        S_true: (4, num_sop) ground-truth Stokes vectors (for comparison).
    """
    np.random.seed(seed)
    
    # 1. Build the "True" ideal tetrahedral calibration matrix (eta = 1, top row = [1,0,0,0])
    # This corresponds to a perfect polarimeter where detector 0 measures total power directly.
    C_true = np.array([
        [1.0, 0.0, 0.0, 0.0],                     # S0 = D0
        [3.0/4.0, -1.0/4.0, -1.0/4.0, -1.0/4.0], # S1 = (3D0 - D1 - D2 - D3)/4
        [0.0, np.sqrt(2)/2, -np.sqrt(2)/4, -np.sqrt(2)/4], # S2 = (2√2 D1 - √2 D2 - √2 D3)/4
        [0.0, 0.0, np.sqrt(6)/4, -np.sqrt(6)/4]   # S3 = (√6 D2 - √6 D3)/4
    ])
    
    # 2. Generate N random, uniformly distributed SOPs on the Poincare sphere.
    # Standard method: sample 3D normal distribution, then normalize to radius 1.
    raw_points = np.random.normal(0, 1, (3, num_sop))
    norms = np.linalg.norm(raw_points, axis=0, keepdims=True)
    s1, s2, s3 = raw_points / norms  # Shape: (3, N)
    
    # Assemble the full Stokes vectors: S0 = 1 (constant power)
    S_true = np.vstack([np.ones((1, num_sop)), s1, s2, s3])  # Shape: (4, N)
    
    # 3. Compute the ideal (noise-free) detector readings.
    # Since S = C @ D, we have D = inv(C) @ S
    C_inv = np.linalg.inv(C_true)
    D_ideal = C_inv @ S_true  # Shape: (4, N)
    
    # 4. Add realistic noise to the detector readings.
    # Detectors have shot noise; we model it as Gaussian with std = noise_std.
    noise = np.random.normal(0, noise_std, D_ideal.shape)
    D_matrix = D_ideal + noise
    
    return D_matrix, C_true, S_true