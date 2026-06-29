"""Functions for reference‑free polarimeter calibration."""

import numpy as np

def fit_power_row(D: np.ndarray, S0: np.ndarray) -> np.ndarray:
    """
    Compute the top row of the calibration matrix C that maps detector
    readings to total optical power.

    Parameters
    ----------
    D : ndarray, shape (4, N)
        Detector readings for N random SOPs. Each column is D^(j).
    S0 : ndarray, shape (N,)
        Measured (or assumed) total power for each SOP.
        For constant‑power calibration, S0 is all ones.

    Returns
    -------
    c0 : ndarray, shape (4,)
        Coefficients C_{00}, C_{01}, C_{02}, C_{03} of the power row,
        minimising || D^T c0 - S0 ||^2.

    Notes
    -----
    This solves the linear least‑squares problem by forming the 4x4 system
    Z c0 = X, where Z_ki = sum_j D_k^j D_i^j and X_k = sum_j D_k^j S0^j.
    """
    D = np.atleast_2d(D)
    S0 = np.atleast_1d(S0)
    if D.shape[0] != 4:
        raise ValueError("D must have 4 rows (one per detector).")
    if D.shape[1] != len(S0):
        raise ValueError("Number of columns in D must match length of S0.")

    Z = D @ D.T          # 4x4
    X = D @ S0           # 4
    c0 = np.linalg.solve(Z, X)
    return c0

"""Functions for reference‑free polarimeter calibration."""

import numpy as np
from scipy.optimize import least_squares


def fit_power_row(D: np.ndarray, S0: np.ndarray) -> np.ndarray:
    """
    ... (previous docstring unchanged) ...
    """
    # (copy the existing implementation here)
    D = np.atleast_2d(D)
    S0 = np.atleast_1d(S0)
    if D.shape[0] != 4:
        raise ValueError("D must have 4 rows (one per detector).")
    if D.shape[1] != len(S0):
        raise ValueError("Number of columns in D must match length of S0.")
    Z = D @ D.T          # 4x4
    X = D @ S0           # 4
    c0 = np.linalg.solve(Z, X)
    return c0


def fit_polarization_rows(c0_fixed: np.ndarray, D: np.ndarray,
                          C_guess: np.ndarray = None) -> np.ndarray:
    """
    Fit the bottom three rows of the calibration matrix C by enforcing DOP=1.

    Parameters
    ----------
    c0_fixed : ndarray, shape (4,)
        The top row coefficients (from Stage 1).
    D : ndarray, shape (4, N)
        Detector readings for N SOPs.
    C_guess : ndarray, shape (4, 4), optional
        Initial guess for the full matrix. If None, a tetrahedral guess
        is built using the mean of c0_fixed.

    Returns
    -------
    C : ndarray, shape (4, 4)
        Full calibration matrix (without slow‑axis alignment).
        The top row is exactly c0_fixed.
    """
    if C_guess is None:
        eta = np.mean(c0_fixed)
        C_guess = np.array([
            [4 * c0_fixed[0], 4 * c0_fixed[1], 4 * c0_fixed[2], 4 * c0_fixed[3]],
            [3 * eta,          -eta,            -eta,            -eta],
            [0,                2 * eta * np.sqrt(2), -eta * np.sqrt(2), -eta * np.sqrt(2)],
            [0,                0,               eta * np.sqrt(6), -eta * np.sqrt(6)]
        ]) / 4.0

    # Variables to optimise: bottom 3 rows flattened (length 12)
    def residual(vars):
        # Build full matrix
        C = np.empty((4, 4))
        C[0, :] = c0_fixed
        C[1:, :] = vars.reshape(3, 4)
        # Compute Stokes vectors for all measurements
        S = C @ D           # shape (4, N)
        S0 = S[0, :]
        S_pol = S[1:, :]    # S1, S2, S3
        # DOP=1 residual: (S1^2+S2^2+S3^2 - S0^2) / S0^2
        return (np.sum(S_pol**2, axis=0) - S0**2) / S0**2

    # Initial guess for the bottom rows
    x0 = C_guess[1:, :].ravel()

    result = least_squares(residual, x0, method='trf')
    if not result.success:
        print("Warning: optimizer did not converge perfectly.")

    # Construct final matrix
    C_opt = np.empty((4, 4))
    C_opt[0, :] = c0_fixed
    C_opt[1:, :] = result.x.reshape(3, 4)
    return C_opt


def align_slow_axis(C: np.ndarray, detector_index: int = 1) -> np.ndarray:
    """
    Rotate the bottom three rows of C so that the Stokes vector
    corresponding to a pure signal on detector `detector_index`
    lies on the positive S1 axis.

    Parameters
    ----------
    C : ndarray, shape (4, 4)
        Calibration matrix (S = C D).
    detector_index : int
        Index (0‑based) of the detector aligned to the slow axis.
        By convention D1 → index 0.

    Returns
    -------
    C_rot : ndarray, shape (4, 4)
        Rotated matrix. S0 is unchanged.
    """
    # The column of C that corresponds to detector_index
    v = C[1:, detector_index]   # (S1, S2, S3) for that detector
    # Rotate v to (norm(v), 0, 0) – i.e. align with S1 axis
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        raise ValueError("Detector projection vector is zero; cannot align.")
    v_hat = v / v_norm
    target = np.array([1.0, 0.0, 0.0])

    # Rotation matrix that maps v_hat to target (Rodrigues formula)
    k = np.cross(v_hat, target)
    k_norm = np.linalg.norm(k)
    if k_norm < 1e-12:
        # v_hat already points to target (or exactly opposite)
        if np.dot(v_hat, target) > 0:
            R = np.eye(3)       # no rotation needed
        else:
            R = -np.eye(3)      # flip
    else:
        k = k / k_norm
        theta = np.arccos(np.clip(np.dot(v_hat, target), -1.0, 1.0))
        # Rodrigues formula
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    # Apply rotation to the bottom three rows
    C_rot = C.copy()
    C_rot[1:, :] = R @ C[1:, :]
    return C_rot


def compute_dop_std(C: np.ndarray, D: np.ndarray) -> float:
    """
    Compute the standard deviation of the degree of polarisation
    over all SOP measurements.

    Parameters
    ----------
    C : ndarray, shape (4, 4)
        Calibration matrix.
    D : ndarray, shape (4, N)
        Detector data.

    Returns
    -------
    sigma : float
        Standard deviation of DOP.
    """
    S = C @ D
    S0 = S[0, :]
    dop = np.sqrt(np.sum(S[1:, :]**2, axis=0)) / S0
    return float(np.std(dop))

def fit_polarization_rows_no_D0(c0_fixed: np.ndarray, D: np.ndarray,
                                C_guess: np.ndarray = None) -> np.ndarray:
    """
    Experimental: Fit the bottom three rows using only D1,D2,D3.
    The column for D0 is forced to zero. This is appropriate when
    D0 behaves as a pure power monitor with negligible polarisation
    sensitivity.

    Parameters
    ----------
    c0_fixed : ndarray, shape (4,)
        Top row from Stage 1.
    D : ndarray, shape (4, N)
        Detector matrix.
    C_guess : ndarray, shape (4,4), optional
        If provided, only the 3x3 block (rows 1..3, cols 1..3) is used.

    Returns
    -------
    C : ndarray, shape (4,4)
        Calibration matrix with C[i,0] = 0 for i=1,2,3.
    """
    D_pol = D[1:, :]   # D1, D2, D3 only

    if C_guess is None:
        eta = np.mean(c0_fixed)
        # ideal 3x3 tetrahedron
        C_guess_3x3 = np.array([
            [3*eta, -eta, -eta],
            [0, 2*eta*np.sqrt(2), -eta*np.sqrt(2)],
            [0, 0, eta*np.sqrt(6)]
        ]) / 4.0
    else:
        C_guess_3x3 = C_guess[1:, 1:]

    def residual(vars):
        M = vars.reshape(3, 3)
        S0 = c0_fixed @ D                # power uses all detectors
        S_pol = M @ D_pol                # polarization from D1-D3
        return (np.sum(S_pol**2, axis=0) - S0**2) / S0**2

    x0 = C_guess_3x3.ravel()
    from scipy.optimize import least_squares
    result = least_squares(residual, x0, method='trf')
    if not result.success:
        print("Warning: optimizer did not converge perfectly.")

    C = np.zeros((4, 4))
    C[0, :] = c0_fixed
    C[1:, 1:] = result.x.reshape(3, 3)
    return C