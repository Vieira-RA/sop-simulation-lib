"""Outlier filtering and adaptive weighted‑mean estimation."""
import numpy as np

def iqr_filter(data, factor=1.0):
    """
    Return boolean mask of inliers using the inter‑quartile range.

    Points within [Q1 – factor*IQR, Q3 + factor*IQR] are kept.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (data >= lower) & (data <= upper)


def adaptive_weighted_mean(distances, inlier_mask, D_abs,
                           c_bisquare=2.5, iqr_factor=1.5):
    """
    Compute a robust, wavelength‑separation‑weighted mean distance.

    Parameters
    ----------
    distances : ndarray (n_pairs,)
        Raw distance estimates for all channel pairs.
    inlier_mask : ndarray, bool
        Mask of pairs that passed IQR filtering.
    D_abs : ndarray (n_pairs,)
        Absolute integrated dispersion |∫D dλ| in ps/km for each pair.
    c_bisquare : float
        Tuning constant for Tukey bisquare weights (2.5 gives moderate
        down‑weighting of outliers inside the inlier set).
    iqr_factor : float
        Factor multiplying IQR to define the scale for robustness weights.

    Returns
    -------
    weighted_mean : float
        The final distance estimate (km).
    full_weights : ndarray (n_pairs,)
        Weight assigned to each pair (zero for outliers).
    """
    Z = distances[inlier_mask]
    D = D_abs[inlier_mask]
    if len(Z) == 0:
        return np.nan, np.zeros_like(distances)

    # 1. Wavelength‑separation weight (variance ∝ 1/D² → weight ∝ D²)
    w_lambda = D ** 2
    w_lambda /= np.median(w_lambda)

    # 2. Robustness weight (Tukey bisquare)
    median_Z = np.median(Z)
    residuals = np.abs(Z - median_Z)
    q1 = np.percentile(Z, 25)
    q3 = np.percentile(Z, 75)
    iqr = q3 - q1
    scale = iqr_factor * iqr
    if scale < 1e-12:
        w_robust = np.ones_like(Z)
    else:
        t = residuals / (c_bisquare * scale)
        w_robust = np.where(t < 1.0, (1 - t**2)**2, 0.0)

    w_total = w_lambda * w_robust
    weighted_mean = np.sum(w_total * Z) / np.sum(w_total)

    full_weights = np.zeros_like(distances)
    full_weights[inlier_mask] = w_total
    return weighted_mean, full_weights