"""
src/correlation_models.py

Reusable functions for generating stationary random processes with
exponential or triangular autocorrelation, and for analysing their ACF.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.signal import correlate

def generate_ou_process(duration: float, Tc: float, dt: float, seed: int = None):
    """
    Generate an Ornstein-Uhlenbeck (OU) process with exponential autocorrelation.
    The continuous-time OU process satisfies: dX = - (1/Tc) X dt + sqrt(2/Tc) dW,
    resulting in R(τ) = exp(-|τ|/Tc) (variance = 1).

    Parameters:
        duration : total time [s]
        Tc       : correlation time [s]
        dt       : sampling interval (small for good approximation) [s]
        seed     : random seed (optional)

    Returns:
        t : 1D array of time points
        X : 1D array of process values (variance approx. 1)
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(duration / dt) + 1
    t = np.linspace(0, duration, n_steps)

    # Euler-Maruyama discretisation of the OU SDE
    X = np.zeros(n_steps)
    X[0] = np.random.normal(0, 1)  # stationary initial condition
    sigma = np.sqrt(2.0 / Tc)      # diffusion coefficient

    for i in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i-1] - (1.0 / Tc) * X[i-1] * dt + sigma * dW

    return t, X

def generate_zoh_process(duration: float, Tc: float, dt: float, seed: int = None):
    """
    Generate a zero-order hold (ZOH) process from independent samples.
    New value is drawn every Tc seconds and held constant between updates.
    Autocorrelation is triangular: R(τ) = 1 - |τ|/Tc for |τ| ≤ Tc, else 0.
    Variance = 1.

    Parameters:
        duration : total time [s]
        Tc       : hold time / correlation time [s]
        dt       : sampling interval (for output) [s]
        seed     : random seed (optional)

    Returns:
        t : 1D array of time points
        X : 1D array of process values (variance approx. 1)
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(duration / dt) + 1
    t = np.linspace(0, duration, n_steps)

    # Number of independent intervals
    n_blocks = int(np.ceil(duration / Tc))
    # Generate i.i.d. Gaussian samples (mean 0, variance 1)
    samples = np.random.normal(0, 1, size=n_blocks)

    # Build piecewise constant signal
    X = np.zeros(n_steps)
    for i, t_val in enumerate(t):
        block_idx = int(t_val / Tc)
        if block_idx < n_blocks:
            X[i] = samples[block_idx]
        else:
            X[i] = samples[-1]   # last block for any overshoot

    return t, X

def compute_acf(t, X, max_lag: float = None):
    """
    Compute the unbiased autocorrelation function of a uniformly sampled process.
    Uses scipy.signal.correlate with mode='full' and normalises by (N - |k|).

    Parameters:
        t         : time array (assumed uniform dt)
        X         : process values
        max_lag   : maximum lag in seconds (if None, uses full length)

    Returns:
        lags : 1D array of time lags (seconds)
        acf  : 1D array of autocorrelation values (R(0)=1)
    """
    dt = t[1] - t[0]
    N = len(X)
    # Unbiased correlation estimate
    corr = correlate(X, X, mode='full')
    # Keep only positive lags (including zero)
    corr = corr[N-1:] / (N - np.arange(N))
    # Normalise so that R(0) = 1
    corr = corr / corr[0]

    lags = np.arange(N) * dt
    if max_lag is not None:
        idx = lags <= max_lag
        lags, corr = lags[idx], corr[idx]
    return lags, corr

def area_under_acf(lags, acf):
    """
    Integrate the autocorrelation function over lag (two-sided).
    For a stationary process with one-sided ACF ρ(τ), the two-sided area
    is ∫_{-∞}^{∞} R(τ) dτ = 2 ∫_{0}^{∞} R(τ) dτ.
    This function assumes lags start at 0 and extend to positive values,
    and returns the two-sided area using Simpson's rule.

    Parameters:
        lags : 1D array of positive lags (including 0)
        acf  : 1D array of R(τ) values (R(0)=1)

    Returns:
        area : two-sided integral
    """
    # One-sided integral from 0 to max_lag
    one_sided = simpson(acf, x=lags)
    return 2.0 * one_sided