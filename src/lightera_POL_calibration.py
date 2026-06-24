# src/calibration.py
import numpy as np
import itertools
from scipy.optimize import least_squares

def fit_power_row(D: np.ndarray, target_power: np.ndarray or float = None) -> np.ndarray:
    """
    Solves for the top row (C0) of the calibration matrix using linear least squares.
    
    Args:
        D: (4, N) matrix of averaged detector voltages for N calibration SOPs.
        target_power: (N,) array of measured optical powers, OR a scalar constant.
                      If None, assumes constant power = 1.0.
    
    Returns:
        C0: (4,) array containing the top row coefficients [C00, C01, C02, C03].
    """
    if target_power is None:
        target_power = np.ones(D.shape[1])
    elif np.isscalar(target_power):
        target_power = target_power * np.ones(D.shape[1])
    
    target_power = np.asarray(target_power).flatten()
    if target_power.shape[0] != D.shape[1]:
        raise ValueError("target_power must have same length as number of calibration points")
    
    C0, _, _, _ = np.linalg.lstsq(D.T, target_power, rcond=None)
    return C0

def generate_tetrahedral_guess(C0: np.ndarray) -> np.ndarray:
    """
    Generates a 4x4 initial guess for the full calibration matrix based on an ideal tetrahedron.
    Follows Equation (9) from the paper.
    """
    eta = np.mean(C0)  # Average scale factor from the top row
    
    row0 = C0.copy()
    row1 = np.array([3.0 * eta, -eta, -eta, -eta]) / 4.0
    row2 = np.array([0.0, 2.0 * eta * np.sqrt(2), -eta * np.sqrt(2), -eta * np.sqrt(2)]) / 4.0
    row3 = np.array([0.0, 0.0, eta * np.sqrt(6), -eta * np.sqrt(6)]) / 4.0
    
    C_guess = np.vstack([row0, row1, row2, row3])
    return C_guess

def dop_residuals(params_flat: np.ndarray, D: np.ndarray, C0: np.ndarray) -> np.ndarray:
    """
    Residual function for the non-linear least squares optimizer.
    Computes (S1^2 + S2^2 + S3^2 - S0^2) / (S0^2) for all calibration points.
    """
    C_bottom = params_flat.reshape(3, 4)
    C = np.vstack([C0.reshape(1, 4), C_bottom])
    S = C @ D
    S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]
    residual = (S1**2 + S2**2 + S3**2 - S0**2)**2 / (S0**2)*2
    return residual

def calibrate_polarimeter(D_matrix: np.ndarray, constant_power_value: float = 1.0,
                          reg_weight: float = 0.01) -> tuple:
    """
    Performs reference‑free calibration with row scaling and regularisation.
    - Row scaling improves numerical conditioning.
    - Regularisation prevents degeneracy (collapse) by penalising deviation from the tetrahedral guess.
    
    Args:
        D_matrix: (4, N) raw detector readings.
        constant_power_value: The optical power (e.g., 1.328 mW).
        reg_weight: Weight of the regularisation penalty (typical: 0.001 to 0.1).
    
    Returns:
        C_matrix: (4, 4) calibration matrix that works on RAW (unscaled) data.
        final_dop: DOP values for the calibration SOPs.
    """
    if D_matrix.shape[1] < 12:
        print("Warning: Very few calibration points (<12).")
    
    # ---------- 1. Row scaling ----------
    # Compute scaling factors: mean absolute value per row
    row_scales = np.mean(np.abs(D_matrix), axis=1, keepdims=True)
    row_scales = np.maximum(row_scales, 1e-12)  # avoid zero
    
    # Scale the data: D_scaled = D_raw / row_scales
    D_scaled = D_matrix / row_scales
    
    # ---------- 2. Fit top row on scaled data ----------
    C0_scaled = fit_power_row(D_scaled, target_power=constant_power_value)
    
    # ---------- 3. Generate tetrahedral guess (on scaled data) ----------
    C_guess_scaled = generate_tetrahedral_guess(C0_scaled)
    initial_params = C_guess_scaled[1:].flatten()
    C_guess_bottom = C_guess_scaled[1:, :]  # bottom rows for regularisation
    
    # ---------- 4. Regularised non‑linear optimisation ----------
    result = least_squares(
        dop_residuals_with_reg,  # NEW residual (see below)
        initial_params,
        args=(D_scaled, C0_scaled, C_guess_bottom, reg_weight),
        method='trf',
        max_nfev=2000,
        verbose=0
    )
    
    # ---------- 5. Build final scaled matrix ----------
    C_bottom_scaled = result.x.reshape(3, 4)
    C_scaled = np.vstack([C0_scaled.reshape(1, 4), C_bottom_scaled])
    
    # ---------- 6. Convert back to RAW (unscaled) calibration matrix ----------
    # D_scaled = diag(1/row_scales) @ D_raw
    # S = C_scaled @ D_scaled = C_scaled @ diag(1/row_scales) @ D_raw
    # So C_raw = C_scaled @ diag(1/row_scales)
    inv_scales = 1.0 / row_scales.flatten()
    S_inv = np.diag(inv_scales)
    C_matrix = C_scaled @ S_inv
    
    # ---------- 7. Verify DOP on raw data ----------
    S = C_matrix @ D_matrix
    S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]
    final_dop = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-12)
    
    return C_matrix, final_dop

def find_detector_order(D_matrix: np.ndarray, fixed_index: int = 0, constant_power_value: float = 1.0) -> tuple:
    """
    Determines the correct physical order of the 4 detector channels by brute-force
    permutation search. Uses the tetrahedral initial guess and compares the raw
    detector vector D with the computed Stokes vector S. The correct order minimizes
    the sum of Euclidean distances between normalized D and normalized S.

    Args:
        D_matrix: (4, N) matrix where row 0 is the known D0 channel.
        fixed_index: The row index corresponding to D0 (default: 0).
        constant_power_value: The constant optical power (e.g., 1.328 mW).

    Returns:
        best_permutation: Tuple of length 4, e.g., (0, 2, 1, 3).
        best_distance: The total distance for the best permutation.
        results: Dictionary with all permutations and their total distances.
    """
    num_channels = D_matrix.shape[0]
    if num_channels != 4:
        raise ValueError("This function only works for 4-channel polarimeters.")
    
    other_indices = [i for i in range(num_channels) if i != fixed_index]
    
    best_distance = np.inf
    best_perm = None
    results = {}
    
    for perm_tuple in itertools.permutations(other_indices):
        perm_order = (fixed_index,) + perm_tuple
        D_perm = D_matrix[perm_order, :]  # shape: (4, N)
        
        # Fit top row with the known constant power
        C0 = fit_power_row(D_perm, target_power=constant_power_value)
        C_guess = generate_tetrahedral_guess(C0)
        
        # Compute Stokes vectors using the initial guess
        S = C_guess @ D_perm  # shape: (4, N)
        
        # Normalize each column of D_perm and S to unit norm
        # (we treat each 4-element vector as a point in 4D space)
        D_norm = D_perm / np.linalg.norm(D_perm, axis=0, keepdims=True)
        S_norm = S / np.linalg.norm(S, axis=0, keepdims=True)
        
        # Compute Euclidean distance between corresponding columns
        diff = D_norm - S_norm
        distances = np.linalg.norm(diff, axis=0)  # shape: (N,)
        total_distance = np.sum(distances)
        
        results[perm_order] = total_distance
        
        if total_distance < best_distance:
            best_distance = total_distance
            best_perm = perm_order
    
    return best_perm, best_distance, results

def dop_residuals_with_reg(params_flat: np.ndarray, D: np.ndarray, C0: np.ndarray,
                        C_guess_bottom: np.ndarray, reg_weight: float) -> np.ndarray:
    """
    Residual: DOP error (paper's metric) + Frobenius penalty to stay close to initial guess.
    """
    C_bottom = params_flat.reshape(3, 4)
    C = np.vstack([C0.reshape(1, 4), C_bottom])
    S = C @ D
    S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]

    # Paper's DOP residual per point
    dop_resid = (S1**2 + S2**2 + S3**2 - S0**2) / (S0**2 + 1e-12)

    # Regularisation: Frobenius norm difference from initial guess
    reg_penalty = reg_weight * np.linalg.norm(C_bottom - C_guess_bottom, 'fro')

    # Append the penalty as a single extra residual
    residuals = np.append(dop_resid, reg_penalty)
    return residuals