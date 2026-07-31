"""Stokes‑noise addition utilities."""
import numpy as np

def add_noise_to_stokes(stokes, snr_db, rng=None):
    """
    Add independent zero‑mean Gaussian noise to each Stokes component.

    Parameters
    ----------
    stokes : ndarray, shape (n_samples, 3)
        Clean Stokes vectors.
    snr_db : float
        Desired signal‑to‑noise ratio in dB:
        SNR = 10 * log10(mean(|S|^2) / variance_per_component).
    rng : numpy.random.RandomState, optional
        Reproducible random generator.

    Returns
    -------
    noisy : ndarray, shape (n_samples, 3)
    """
    if rng is None:
        rng = np.random
    signal_power = np.mean(np.sum(stokes**2, axis=1))
    snr_lin = 10**(snr_db / 10.0)
    noise_var = signal_power / snr_lin
    if noise_var <= 0:
        return stokes
    noise = np.sqrt(noise_var) * rng.randn(*stokes.shape)
    return stokes + noise