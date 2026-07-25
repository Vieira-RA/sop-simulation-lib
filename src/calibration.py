"""
calibration.py - Reference-free polarimeter calibration (Mikhailov et al., 2014)
Everything in one file: loader, dark correction, calibration, Stokes utils.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import least_squares


# ----------------------------------------------------------------------
# 1. OSCILLOSCOPE DATA LOADER
# ----------------------------------------------------------------------

def load_oscilloscope_trace(filepath):
    """
    Load a voltage trace from a Tektronix/Keysight style CSV.
    Skips 6 header lines, takes the last column (voltage).
    """
    return np.genfromtxt(filepath, delimiter=',', skip_header=6, usecols=-1)


def load_calibration_dataset(base_dir, num_sops=70, num_channels=4):
    """
    Load all SOP traces, subtract dark current, return mean voltage per channel.
    
    Expected files in base_dir:
        DARK_CURRENT000_Ch1.csv, DARK_CURRENT000_Ch2.csv, ...
        SOP001_Ch1.csv, SOP001_Ch2.csv, ..., SOP070_Ch4.csv
    
    Returns:
        detector_matrix: shape (num_sops, 4) - mean voltage per SOP, dark subtracted
        dark_vector: shape (4,) - mean dark voltage per channel
    """
    base_dir = Path(base_dir)
    
    # Load dark current
    dark_vector = np.zeros(num_channels)
    for ch in range(1, num_channels + 1):
        fname = base_dir / f"DARK_CURRENT000_Ch{ch}.csv"
        trace = load_oscilloscope_trace(fname)
        dark_vector[ch - 1] = np.mean(trace)
    
    # Load SOP data
    detector_matrix = np.zeros((num_sops, num_channels))
    for sop_idx in range(1, num_sops + 1):
        for ch in range(1, num_channels + 1):
            fname = base_dir / f"SOP{sop_idx:03d}_Ch{ch}.csv"
            trace = load_oscilloscope_trace(fname)
            mean_v = np.mean(trace)
            # Dark subtraction (negative values are OK for inverting TIA)
            detector_matrix[sop_idx - 1, ch - 1] = mean_v - dark_vector[ch - 1]
    
    return detector_matrix, dark_vector


# ----------------------------------------------------------------------
# 2. STOKES UTILITIES
# ----------------------------------------------------------------------

def dop(S):
    """Degree of Polarization from Stokes vector [S0, S1, S2, S3]."""
    S = np.asarray(S)
    if S.ndim == 1:
        return np.sqrt(S[1]**2 + S[2]**2 + S[3]**2) / (S[0] + 1e-12)
    else:
        s0 = S[:, 0]
        return np.sqrt(S[:, 1]**2 + S[:, 2]**2 + S[:, 3]**2) / (s0 + 1e-12)


def normalize_stokes(S):
    """Normalize polarization part of Stokes vector (preserve S0)."""
    S = np.asarray(S).copy()
    if S.ndim == 1:
        norm = np.sqrt(S[1]**2 + S[2]**2 + S[3]**2)
        if norm > 1e-12:
            S[1:] = S[1:] / norm
    else:
        norms = np.sqrt(S[:, 1]**2 + S[:, 2]**2 + S[:, 3]**2)
        S[:, 1:] = S[:, 1:] / (norms[:, None] + 1e-12)
    return S


# ----------------------------------------------------------------------
# 3. REFERENCE-FREE CALIBRATION
# ----------------------------------------------------------------------

def fit_power_row(detector_matrix, power_vector):
    """
    Fit top row of calibration matrix: power = detector_matrix @ c0.
    Uses least squares (robust to noise).
    """
    c0, _, _, _ = np.linalg.lstsq(detector_matrix, power_vector, rcond=None)
    return c0


# In calibration.py - replace these functions

def _dop_cost_function(c_flat, detector_matrix, c0, guess_flat=None, reg_lambda=1e-6):
    """
    Cost function with optional Tikhonov regularization.
    Minimizes (DOP^2 - 1)^2 / S0^4 + reg_lambda * ||C_lower - C_guess_lower||^2
    """
    C_lower = c_flat.reshape(3, 4)
    C_full = np.vstack([c0, C_lower])

    S = C_full @ detector_matrix.T
    S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]

    eps = 1e-12
    dop_error = (S1**2 + S2**2 + S3**2 - S0**2) / (S0**2 + eps)

    # Regularization term (keep lower rows close to the initial guess)
    if guess_flat is not None:
        reg_term = np.sqrt(reg_lambda) * (c_flat - guess_flat)
        return np.concatenate([dop_error, reg_term])
    else:
        return dop_error


def fit_polarization_rows(detector_matrix, c0, initial_guess=None, reg_lambda=1e-6):
    """
    Fit the lower 3 rows of the calibration matrix using the DOP=1 constraint
    with Tikhonov regularization to prevent collapse to a single point.

    Args:
        detector_matrix: np.ndarray of shape (N, 4).
        c0: np.ndarray of shape (4,), the calibrated top row.
        initial_guess: optional np.ndarray of shape (12,).
        reg_lambda: regularization strength. Increase if collapse occurs.
    """
    if initial_guess is None:
        eta = np.mean(np.abs(c0))
        guess_matrix = np.array([
            [4 * c0[0], 4 * c0[1], 4 * c0[2], 4 * c0[3]],
            [3 * eta, -eta, -eta, -eta],
            [0, 2 * eta * np.sqrt(2), -eta * np.sqrt(2), -eta * np.sqrt(2)],
            [0, 0, eta * np.sqrt(6), -eta * np.sqrt(6)]
        ]) / 4.0
        initial_guess = guess_matrix[1:, :].flatten()

    result = least_squares(
        _dop_cost_function,
        initial_guess,
        args=(detector_matrix, c0, initial_guess, reg_lambda),  # Pass guess and lambda
        method='lm',
        max_nfev=2000,
    )

    if not result.success:
        print(f"Warning: Calibration fit did not converge: {result.message}")

    C_lower_opt = result.x.reshape(3, 4)
    C_full = np.vstack([c0, C_lower_opt])
    return C_full

# In calibration.py - replace the calibrate_polarimeter function with this:

def calibrate_polarimeter(detector_matrix, power_vector, anchor_to_s1=True,
                          return_guess=False, reg_lambda=1e-6, variance_weight=0.0,
                          initial_guess=None):
    """
    Full reference-free calibration with optional custom initial guess.

    Args:
        detector_matrix: (N, 4) dark-corrected mean voltages.
        power_vector: (N,) measured optical power for each SOP.
        anchor_to_s1: if True, aligns detector 0 to Stokes S1 axis.
        return_guess: if True, returns (C, guess_matrix).
        reg_lambda: regularization strength.
        variance_weight: weight for variance maximization (0 = off).
        initial_guess: optional 12-element array for the lower 3 rows.
                       If None, uses the tetrahedral guess from the paper.

    Returns:
        C: (4, 4) calibration matrix.
        guess_matrix: (4, 4) initial guess matrix (only if return_guess=True).
    """
    # Step 1: Fit the power row (always done from scratch)
    c0 = fit_power_row(detector_matrix, power_vector)

    # Step 2: Build or use the initial guess for the lower 3 rows
    if initial_guess is None:
        # Default tetrahedral guess (Eq. 9 from the paper)
        eta = np.mean(np.abs(c0))
        guess_matrix = np.array([
            [4 * c0[0], 4 * c0[1], 4 * c0[2], 4 * c0[3]],
            [3 * eta, -eta, -eta, -eta],
            [0, 2 * eta * np.sqrt(2), -eta * np.sqrt(2), -eta * np.sqrt(2)],
            [0, 0, eta * np.sqrt(6), -eta * np.sqrt(6)]
        ]) / 4.0
        initial_guess = guess_matrix[1:4, :].flatten()
    else:
        # User provided a custom initial guess (e.g., rotated version)
        initial_guess = np.asarray(initial_guess).flatten()
        if len(initial_guess) != 12:
            raise ValueError("initial_guess must have 12 elements.")
        # Reconstruct the full guess matrix for return_guess
        guess_matrix = np.vstack([c0, initial_guess.reshape(3, 4)])

    # Step 3: Non-linear fit for the lower rows
    C = fit_polarization_rows(detector_matrix, c0, initial_guess, reg_lambda, variance_weight)

    # Step 4: Anchor to S1 (if requested)
    R_full = np.eye(4)
    if anchor_to_s1:
        vec = C[:, 0]
        pol_vec = vec[1:] / (np.linalg.norm(vec[1:]) + 1e-12)
        target = np.array([1.0, 0.0, 0.0])
        v = np.cross(pol_vec, target)
        s = np.linalg.norm(v)
        c = np.dot(pol_vec, target)
        if s > 1e-12:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        else:
            R = np.eye(3)
        R_full = np.eye(4)
        R_full[1:4, 1:4] = R
        C = R_full @ C

    # Step 5: Return results
    if return_guess:
        # Apply the same rotation to the guess matrix for a fair comparison
        guess_rotated = R_full @ guess_matrix
        return C, guess_rotated

    return C

def apply_calibration(C, detector_vector):
    """
    Apply calibration to a single 4-element detector vector.
    Returns Stokes vector [S0, S1, S2, S3].
    """
    return C @ detector_vector

# Add this to the end of your calibration.py file

def run_sanity_checks(raw_detector_matrix, dark_vector, corrected_detector_matrix, 
                      power_vector, C_matrix, num_points_to_show=5):
    """
    Run comprehensive sanity checks on the polarimeter calibration.

    Checks performed:
        1. Raw signal vs Dark (ensures the oscilloscope saw light)
        2. Dark correction polarity (confirms inverting TIA behavior)
        3. Power reconstruction accuracy (S0 vs measured power)
        4. DOP (should be ~1 with std < 0.01)
        5. Condition number of C (should be < 10 for a good tetrahedron)
        6. Poincaré sphere coverage (ensures SOPs were sufficiently varied)

    Args:
        raw_detector_matrix: (N, 4) raw mean voltages before dark correction.
        dark_vector: (4,) mean dark voltage per channel.
        corrected_detector_matrix: (N, 4) dark-subtracted values.
        power_vector: (N,) measured optical powers.
        C_matrix: (4, 4) calibrated matrix.
        num_points_to_show: int, how many SOP indices to show in sample output.

    Returns:
        dict: Dictionary of all check results.
    """
    print("\n" + "="*60)
    print("             POLARIMETER CALIBRATION SANITY CHECKS")
    print("="*60)
    
    results = {}
    N = corrected_detector_matrix.shape[0]
    
    # ------------------------------------------------------------------
    # CHECK 1: Raw signal vs Dark (Did the scope see the laser?)
    # ------------------------------------------------------------------
    print("\n[1] SIGNAL STRENGTH CHECK")
    print("-------------------------")
    # Average raw signal across all SOPs (per channel)
    mean_raw = np.mean(raw_detector_matrix, axis=0)
    # Difference between raw (with light) and dark
    signal_above_dark = mean_raw - dark_vector
    
    print(f"  Dark vector (V):        [{dark_vector[0]:.4f}, {dark_vector[1]:.4f}, {dark_vector[2]:.4f}, {dark_vector[3]:.4f}]")
    print(f"  Mean raw signal (V):    [{mean_raw[0]:.4f}, {mean_raw[1]:.4f}, {mean_raw[2]:.4f}, {mean_raw[3]:.4f}]")
    print(f"  Signal - Dark (V):      [{signal_above_dark[0]:.4f}, {signal_above_dark[1]:.4f}, {signal_above_dark[2]:.4f}, {signal_above_dark[3]:.4f}]")
    
    # FAIL if any channel has signal_above_dark < 0.001 V (too weak)
    if np.max(np.abs(signal_above_dark)) < 0.001:
        print("  ⚠️  WARNING: Signal is very weak (< 1 mV). Check laser power or scope settings.")
        results['signal_strength'] = 'WEAK'
    elif np.max(np.abs(signal_above_dark)) < 0.01:
        print("  ⚠️  NOTICE: Signal is moderate (10 mV). Acceptable but ensure SNR is > 10.")
        results['signal_strength'] = 'MODERATE'
    else:
        print("  ✅ Signal strength is good.")
        results['signal_strength'] = 'GOOD'
    
    # ------------------------------------------------------------------
    # CHECK 2: Dark correction polarity (Are corrected values negative?)
    # ------------------------------------------------------------------
    print("\n[2] POLARITY CHECK (Inverting TIA?)")
    print("-----------------------------------")
    # Sample a few SOPs to show polarity
    sample_indices = np.linspace(0, N-1, min(num_points_to_show, N), dtype=int)
    sample_corrected = corrected_detector_matrix[sample_indices, :]
    
    # Count how many corrected values are negative
    n_negative = np.sum(corrected_detector_matrix < 0)
    total_values = N * 4
    neg_frac = n_negative / total_values
    
    print(f"  Fraction of corrected values that are negative: {neg_frac*100:.1f}%")
    print(f"  Sample corrected values (SOP indices {sample_indices.tolist()}):")
    for i, idx in enumerate(sample_indices):
        print(f"    SOP{idx+1:03d}: [{sample_corrected[i,0]:.4f}, {sample_corrected[i,1]:.4f}, {sample_corrected[i,2]:.4f}, {sample_corrected[i,3]:.4f}]")
    
    if neg_frac > 0.8:
        print("  ✅ Most values are negative. This confirms an INVERTING TIA. Matrix C will handle the sign.")
        results['polarity'] = 'INVERTING'
    elif neg_frac < 0.2:
        print("  ℹ️  Most values are positive. You likely have a non-inverting TIA (or very low signal).")
        results['polarity'] = 'NON_INVERTING'
    else:
        print("  ℹ️  Mixed polarity. Could be weak signal, DC drift, or a mix of channels.")
        results['polarity'] = 'MIXED'
    
    # ------------------------------------------------------------------
    # CHECK 3: Power reconstruction accuracy
    # ------------------------------------------------------------------
    print("\n[3] POWER RECONSTRUCTION CHECK")
    print("-------------------------------")
    # C[0, :] is the power row
    S0_calc = corrected_detector_matrix @ C_matrix[0, :]
    
    # Relative error (ignore points where power is very low)
    eps = 1e-12
    rel_error = np.abs(S0_calc - power_vector) / (np.abs(power_vector) + eps)
    mean_rel_error = np.mean(rel_error)
    max_rel_error = np.max(rel_error)
    
    print(f"  Mean relative power error: {mean_rel_error*100:.3f}%")
    print(f"  Max relative power error:  {max_rel_error*100:.3f}%")
    
    # Check if S0 is positive (should be!)
    if np.any(S0_calc < 0):
        n_neg_power = np.sum(S0_calc < 0)
        print(f"  ❌ ERROR: {n_neg_power} / {N} SOPs have NEGATIVE calculated power!")
        print("     This indicates the top row of C has the wrong sign.")
        results['power_reconstruction'] = 'FAILED'
    elif mean_rel_error > 0.1:
        print(f"  ⚠️  WARNING: Mean power error > 10%. Check your power meter readings.")
        results['power_reconstruction'] = 'POOR'
    else:
        print(f"  ✅ Power reconstruction is accurate.")
        results['power_reconstruction'] = 'GOOD'
    
    # ------------------------------------------------------------------
    # CHECK 4: DOP (Degree of Polarization)
    # ------------------------------------------------------------------
    print("\n[4] DOP CHECK (Should be ~1.0)")
    print("------------------------------")
    S_all = corrected_detector_matrix @ C_matrix.T  # (N, 4)
    dop_values = dop(S_all)
    mean_dop = np.mean(dop_values)
    std_dop = np.std(dop_values)
    max_dop = np.max(dop_values)
    min_dop = np.min(dop_values)
    
    print(f"  Mean DOP:  {mean_dop:.6f}")
    print(f"  Std DOP:   {std_dop:.6f}")
    print(f"  Min DOP:   {min_dop:.6f}")
    print(f"  Max DOP:   {max_dop:.6f}")
    
    if std_dop > 0.02:
        print("  ❌ FAIL: Std DOP > 0.02. Calibration did not converge well.")
        print("     Try: increasing number of SOPs, checking input DOP, or ensuring SOPs cover the sphere.")
        results['dop'] = 'FAILED'
    elif std_dop > 0.01:
        print("  ⚠️  WARNING: Std DOP > 0.01. Acceptable but could be better (< 0.01).")
        results['dop'] = 'ACCEPTABLE'
    else:
        print("  ✅ Excellent DOP consistency (< 1% std).")
        results['dop'] = 'EXCELLENT'
    
    # ------------------------------------------------------------------
    # CHECK 5: Condition Number of C
    # ------------------------------------------------------------------
    print("\n[5] MATRIX CONDITION NUMBER")
    print("---------------------------")
    cond_num = np.linalg.cond(C_matrix)
    print(f"  Condition number of C: {cond_num:.3f}")
    
    if cond_num > 100:
        print("  ❌ FAIL: Condition number > 100. The tetrahedron is nearly degenerate.")
        print("     Check grating angles and beat length separation.")
        results['condition'] = 'FAILED'
    elif cond_num > 10:
        print("  ⚠️  WARNING: Condition number > 10. Suboptimal geometry. Noise amplification may be high.")
        results['condition'] = 'SUBOPTIMAL'
    else:
        print("  ✅ Well-conditioned matrix. Good tetrahedral geometry.")
        results['condition'] = 'GOOD'
    
    # ------------------------------------------------------------------
    # CHECK 6: Poincaré Sphere Coverage
    # ------------------------------------------------------------------
    print("\n[6] POINCARÉ SPHERE COVERAGE")
    print("----------------------------")
    # Compute normalized Stokes vectors (remove S0)
    S1, S2, S3 = S_all[:, 1], S_all[:, 2], S_all[:, 3]
    # Normalize to unit sphere
    norm = np.sqrt(S1**2 + S2**2 + S3**2) + 1e-12
    s1_norm = S1 / norm
    s2_norm = S2 / norm
    s3_norm = S3 / norm
    
    # Check spread: variance of each component
    var_s1 = np.var(s1_norm)
    var_s2 = np.var(s2_norm)
    var_s3 = np.var(s3_norm)
    total_var = var_s1 + var_s2 + var_s3
    
    print(f"  Variance of S1: {var_s1:.4f}")
    print(f"  Variance of S2: {var_s2:.4f}")
    print(f"  Variance of S3: {var_s3:.4f}")
    print(f"  Total variance: {total_var:.4f} (theoretical max for uniform sphere ~ 0.333)")
    
    if total_var < 0.05:
        print("  ❌ FAIL: Very low variance. SOPs are clustered.")
        print("     You need to scramble the polarization controller more broadly.")
        results['coverage'] = 'POOR'
    elif total_var < 0.15:
        print("  ⚠️  WARNING: Moderate variance. Coverage is limited.")
        results['coverage'] = 'MODERATE'
    else:
        print("  ✅ Good coverage of the Poincaré sphere.")
        results['coverage'] = 'GOOD'
    
    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("                      SUMMARY")
    print("="*60)
    pass_count = sum([1 for v in results.values() if v in ['GOOD', 'EXCELLENT', 'INVERTING', 'NON_INVERTING']])
    warn_count = sum([1 for v in results.values() if v in ['WEAK', 'MODERATE', 'ACCEPTABLE', 'SUBOPTIMAL', 'MIXED']])
    fail_count = sum([1 for v in results.values() if v in ['FAILED', 'POOR', 'WEAK']])
    
    print(f"  ✅ PASS:  {pass_count} checks")
    print(f"  ⚠️  WARN:  {warn_count} checks")
    print(f"  ❌ FAIL:  {fail_count} checks")
    
    if fail_count > 0:
        print("\n  ❌ CALIBRATION NOT RECOMMENDED. Fix the failed checks above.")
    elif warn_count > 0:
        print("\n  ⚠️  CALIBRATION ACCEPTABLE but with warnings. Review them.")
    else:
        print("\n  ✅ CALIBRATION SUCCESSFUL. Ready for use.")
    
    print("="*60 + "\n")
    
    return results

# Add this to calibration.py

def plot_poincare_coverage(detector_matrix, guess_matrix, C, save_path='poincare_comparison.png'):
    """
    Plot superimposed Poincaré sphere coverage of the initial guess vs final matrix.

    Args:
        detector_matrix: (N, 4) dark-corrected mean voltages.
        guess_matrix: (4, 4) initial tetrahedral guess matrix.
        C: (4, 4) final optimized calibration matrix.
        save_path: str, path to save the .png file.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not installed. Skipping Poincaré plot.")
        return

    # Compute Stokes vectors
    S_guess = detector_matrix @ guess_matrix.T  # (N, 4)
    S_final = detector_matrix @ C.T            # (N, 4)

    # Extract and normalize polarization components (S1, S2, S3)
    def normalize_pol(S):
        s1, s2, s3 = S[:, 1], S[:, 2], S[:, 3]
        norm = np.sqrt(s1**2 + s2**2 + s3**2) + 1e-12
        return s1/norm, s2/norm, s3/norm

    s1g, s2g, s3g = normalize_pol(S_guess)
    s1f, s2f, s3f = normalize_pol(S_final)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Initial guess (blue circles, transparent)
    ax.scatter(s1g, s2g, s3g, c='blue', marker='o', s=25, alpha=0.4, label='Initial Guess (Tetrahedral)')

    # Final optimized (red triangles, solid)
    ax.scatter(s1f, s2f, s3f, c='red', marker='^', s=30, alpha=0.8, label='Optimized Calibration')

    ax.set_xlabel('$S_1$')
    ax.set_ylabel('$S_2$')
    ax.set_zlabel('$S_3$')
    ax.set_title('Poincaré Sphere Coverage: Initial Guess vs Final Calibration')
    ax.legend()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])  # Ensure sphere looks spherical

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Poincaré plot saved to: {save_path}")


import itertools

def find_channel_order(detector_matrix, power_vector, reg_lambda=1e-3):
    """
    Brute-force search over all 24 channel permutations to find the physical order
    that matches the tetrahedral geometry.

    Args:
        detector_matrix: (N, 4) dark-corrected data with UNKNOWN column order.
        power_vector: (N,) measured optical powers.
        reg_lambda: regularization strength (use a moderate value like 1e-3).

    Returns:
        best_perm: tuple, e.g., (2, 0, 3, 1) meaning physical channel 2 is D0, etc.
        best_C: (4, 4) calibrated matrix for the best permutation.
        results: list of dicts for all permutations (for debugging).
    """
    best_score = np.inf
    best_perm = None
    best_C = None
    results = []

    print("Searching all 24 channel permutations...")
    for perm in itertools.permutations(range(4)):
        # Reorder columns according to this permutation
        det_perm = detector_matrix[:, perm]

        # Run calibration (without returning guess to save time)
        try:
            C = calibrate_polarimeter(
                det_perm, 
                power_vector, 
                anchor_to_s1=True, 
                return_guess=False, 
                reg_lambda=reg_lambda
            )
            
            # Compute metrics
            cond_num = np.linalg.cond(C)
            S_all = det_perm @ C.T
            dop_vals = dop(S_all)
            std_dop = np.std(dop_vals)
            mean_dop = np.mean(dop_vals)
            
            # Score: prioritize low condition number, then low std DOP
            # A perfect tetrahedron has cond ~ sqrt(3) ≈ 1.73.
            # We heavily penalize condition numbers > 100 (degenerate).
            score = cond_num * (1 + 10 * std_dop)
            
            results.append({
                'perm': perm,
                'cond': cond_num,
                'std_dop': std_dop,
                'mean_dop': mean_dop,
                'score': score
            })
            
            if score < best_score:
                best_score = score
                best_perm = perm
                best_C = C
                
        except Exception as e:
            print(f"  Perm {perm} failed: {e}")
            continue

    # Sort results by score for easy viewing
    results.sort(key=lambda x: x['score'])
    
    print("\nTop 5 best permutations:")
    print(" Rank | Permutation  | Cond Num  | Std DOP  | Score")
    print("------|--------------|-----------|----------|--------")
    for i, res in enumerate(results[:5]):
        print(f"  {i+1:2}   | {res['perm']}     | {res['cond']:.2e} | {res['std_dop']:.6f} | {res['score']:.2e}")

    return best_perm, best_C, results

# In calibration.py - Replace the cost function and fitting function

def _dop_cost_with_variance(c_flat, detector_matrix, c0, guess_flat, reg_lambda, variance_weight):
    """
    Cost function with:
    1. DOP=1 constraint (standard)
    2. Tikhonov regularization (keep near initial guess)
    3. SPREAD CONSTRAINT: Maximize variance of S1, S2, S3 across SOPs.
    """
    C_lower = c_flat.reshape(3, 4)
    C_full = np.vstack([c0, C_lower])

    # 1. Compute Stokes vectors: (4, N)
    S = C_full @ detector_matrix.T
    S0, S1, S2, S3 = S[0, :], S[1, :], S[2, :], S[3, :]

    eps = 1e-12
    # 2. DOP residuals: (N,) array
    dop_residual = (S1**2 + S2**2 + S3**2 - S0**2) / (S0**2 + eps)

    # 3. Regularization residual: (12,) array
    reg_residual = np.sqrt(reg_lambda) * (c_flat - guess_flat) if guess_flat is not None else np.array([])

    # 4. NEW: Spread constraint.
    #    We compute the variance of the three normalized polarization components.
    #    If variance is small, this residual becomes negative, forcing the optimizer
    #    to increase variance to drive the squared error down.
    norm = np.sqrt(S1**2 + S2**2 + S3**2) + eps
    s1_norm = S1 / norm
    s2_norm = S2 / norm
    s3_norm = S3 / norm
    
    total_variance = np.var(s1_norm) + np.var(s2_norm) + np.var(s3_norm)
    
    # The negative sign is critical: we want to maximize variance.
    # We apply sqrt to scale it similarly to the DOP residuals.
    variance_residual = - variance_weight * np.sqrt(total_variance + 1e-12)

    # Concatenate all residuals: DOP errors + Regularization + Variance penalty
    return np.concatenate([dop_residual, reg_residual, [variance_residual]])


def fit_polarization_rows(detector_matrix, c0, initial_guess=None, reg_lambda=1e-6, variance_weight=0.0):
    """
    Fit the lower 3 rows. 
    variance_weight: If > 0, actively pulls the SOPs apart on the sphere.
    """
    if initial_guess is None:
        eta = np.mean(np.abs(c0))
        guess_matrix = np.array([
            [4 * c0[0], 4 * c0[1], 4 * c0[2], 4 * c0[3]],
            [3 * eta, -eta, -eta, -eta],
            [0, 2 * eta * np.sqrt(2), -eta * np.sqrt(2), -eta * np.sqrt(2)],
            [0, 0, eta * np.sqrt(6), -eta * np.sqrt(6)]
        ]) / 4.0
        initial_guess = guess_matrix[1:, :].flatten()

    result = least_squares(
        _dop_cost_with_variance,
        initial_guess,
        args=(detector_matrix, c0, initial_guess, reg_lambda, variance_weight),
        method='lm',
        max_nfev=5000,  # Allow more iterations for the harder problem
        ftol=1e-12,
        xtol=1e-12
    )

    if not result.success:
        print(f"Warning: Calibration fit did not converge: {result.message}")

    C_lower_opt = result.x.reshape(3, 4)
    C_full = np.vstack([c0, C_lower_opt])
    return C_full


# ALSO UPDATE calibrate_polarimeter to accept variance_weight
def calibrate_polarimeter(detector_matrix, power_vector, anchor_to_s1=True, 
                          return_guess=False, reg_lambda=1e-6, variance_weight=0.0):
    """
    Args:
        variance_weight: Weight for the variance maximization term.
                         Start with 0.01, increase to 0.1, 1.0 to see effects.
    """
    c0 = fit_power_row(detector_matrix, power_vector)

    eta = np.mean(np.abs(c0))
    guess_matrix = np.array([
        [4 * c0[0], 4 * c0[1], 4 * c0[2], 4 * c0[3]],
        [3 * eta, -eta, -eta, -eta],
        [0, 2 * eta * np.sqrt(2), -eta * np.sqrt(2), -eta * np.sqrt(2)],
        [0, 0, eta * np.sqrt(6), -eta * np.sqrt(6)]
    ]) / 4.0
    initial_guess = guess_matrix[1:, :].flatten()

    # Pass variance_weight to the fit function
    C = fit_polarization_rows(detector_matrix, c0, initial_guess, reg_lambda, variance_weight)

    # Anchoring (rotation) remains the same
    R_full = np.eye(4)
    if anchor_to_s1:
        vec = C[:, 0]
        pol_vec = vec[1:] / (np.linalg.norm(vec[1:]) + 1e-12)
        target = np.array([1.0, 0.0, 0.0])
        v = np.cross(pol_vec, target)
        s = np.linalg.norm(v)
        c = np.dot(pol_vec, target)
        if s > 1e-12:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        else:
            R = np.eye(3)
        R_full = np.eye(4)
        R_full[1:, 1:] = R
        C = R_full @ C

    if return_guess:
        guess_matrix_rotated = R_full @ guess_matrix
        return C, guess_matrix_rotated

    return C

# Add these functions to your existing calibration.py

import numpy as np
from scipy.stats import special_ortho_group  # for random rotations

def random_rotation_3d():
    """Generate a random 3x3 rotation matrix (SO(3))."""
    # Use the Haar distribution (uniform over SO(3))
    return special_ortho_group.rvs(3)

def apply_rotation_to_guess(guess_matrix, R):
    """
    Apply a 3D rotation R to the lower 3 rows (S1,S2,S3) of the guess matrix.
    The top row (S0) remains unchanged.
    """
    C_rot = guess_matrix.copy()
    # R is 3x3, multiply with the 3x4 submatrix (rows 1..3)
    C_rot[1:4, :] = R @ C_rot[1:4, :]
    return C_rot


def multi_start_calibrate(detector_matrix, power_vector, anchor_to_s1=True,
                          reg_lambda=1e-6, n_restarts=20):
    """
    Multi‑start calibration: try n_restarts random rotations of the
    tetrahedral initial guess, run the non‑linear fit, and keep the best.

    Args:
        detector_matrix: (N, 4) dark‑corrected data.
        power_vector: (N,) measured powers.
        anchor_to_s1: whether to anchor final matrix to S1.
        reg_lambda: regularization strength.
        n_restarts: number of random rotations to try.

    Returns:
        best_C: (4,4) best calibration matrix.
        best_score: the score of the best.
        best_rotation: the 3x3 rotation used for the best.
        all_results: list of dicts for each run (for debugging).
    """
    # 1. Compute the initial tetrahedral guess (without any rotation)
    c0 = fit_power_row(detector_matrix, power_vector)
    eta = np.mean(np.abs(c0))
    guess_matrix = np.array([
        [4 * c0[0], 4 * c0[1], 4 * c0[2], 4 * c0[3]],
        [3 * eta, -eta, -eta, -eta],
        [0, 2 * eta * np.sqrt(2), -eta * np.sqrt(2), -eta * np.sqrt(2)],
        [0, 0, eta * np.sqrt(6), -eta * np.sqrt(6)]
    ]) / 4.0

    best_score = np.inf
    best_C = None
    best_rotation = None
    all_results = []

    for i in range(n_restarts):
        # Generate random 3D rotation
        R = random_rotation_3d()
        rotated_guess = apply_rotation_to_guess(guess_matrix, R)

        # Run the calibration with this rotated initial guess
        try:
            # Use the updated calibrate_polarimeter with initial_guess
            C = calibrate_polarimeter(
                detector_matrix,
                power_vector,
                anchor_to_s1=anchor_to_s1,
                return_guess=False,
                reg_lambda=reg_lambda,
                initial_guess=rotated_guess[1:4, :].flatten()  # lower 12 elements
            )
        except Exception as e:
            print(f"  Restart {i+1}: fit failed: {e}")
            continue

        # Evaluate metrics
        S_all = detector_matrix @ C.T
        dop_vals = dop(S_all)
        std_dop = np.std(dop_vals)
        mean_dop = np.mean(dop_vals)

        # Sphere variance (normalized S1,S2,S3)
        s1, s2, s3 = S_all[:, 1], S_all[:, 2], S_all[:, 3]
        norm = np.sqrt(s1**2 + s2**2 + s3**2) + 1e-12
        s1n, s2n, s3n = s1/norm, s2/norm, s3/norm
        var_total = np.var(s1n) + np.var(s2n) + np.var(s3n)

        cond_num = np.linalg.cond(C)

        # Score: we want low std_dop, high variance, low condition number.
        score = std_dop * (1 + 10 / (var_total + 1e-6)) * (1 + cond_num / 100)

        all_results.append({
            'rotation': R,
            'std_dop': std_dop,
            'mean_dop': mean_dop,
            'var_total': var_total,
            'cond_num': cond_num,
            'score': score,
            'C': C
        })

        if score < best_score:
            best_score = score
            best_C = C
            best_rotation = R

    print(f"Multi‑start: tried {len(all_results)} successful rotations. Best score = {best_score:.4f}")
    return best_C, best_score, best_rotation, all_results