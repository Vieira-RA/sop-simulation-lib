"""
Functions for generating digital communication signals and
general signal processing routines (FFT, cross‑correlation, peak fitting).
"""

import numpy as np
from scipy.signal import upfirdn

def cross_correlation_2d_fft(sig1, sig2, normalize=True):
    """
    Combined cross‑correlation of two 2D vector time series using the FFT.

    The correlation is the sum of the per‑component cross‑correlations,
    equivalent to the dot‑product correlation at every lag.

    Parameters
    ----------
    sig1, sig2 : ndarray, shape (N, 2)
        Real 2D vector signals. Each row is one time sample.
    normalize : bool
        If True, normalise by the geometric mean of the signal energies.

    Returns
    -------
    lags : ndarray
        Lag indices (in samples) from -(len(sig2)-1) to (len(sig1)-1).
    corr : ndarray
        Correlation values for each lag.
    """
    a_x, a_y = sig1[:, 0], sig1[:, 1]
    b_x, b_y = sig2[:, 0], sig2[:, 1]

    N = len(a_x) + len(b_x) - 1
    A_x = np.fft.fft(a_x, n=N)
    A_y = np.fft.fft(a_y, n=N)
    B_x = np.fft.fft(b_x, n=N)
    B_y = np.fft.fft(b_y, n=N)

    corr_x = np.fft.ifft(A_x * np.conj(B_x)).real
    corr_y = np.fft.ifft(A_y * np.conj(B_y)).real
    corr = corr_x + corr_y

    if normalize:
        ener1 = np.sum(a_x**2 + a_y**2)
        ener2 = np.sum(b_x**2 + b_y**2)
        corr = corr / np.sqrt(ener1 * ener2)

    corr = np.fft.fftshift(corr)
    lags = np.arange(-len(b_x) + 1, len(a_x))
    return lags, corr


def parabolic_fit(y, peak_idx):
    """
    Subsample peak interpolation using a parabolic fit through three points.

    Parameters
    ----------
    y : ndarray, 1D
        Correlation function (e.g., absolute value of cross‑correlation).
    peak_idx : int
        Index of the maximum in `y`.

    Returns
    -------
    delta : float
        Fractional offset (in samples) from `peak_idx`.
    peak_val : float
        Interpolated peak value.
    """
    if peak_idx == 0 or peak_idx == len(y) - 1:
        return 0.0, y[peak_idx]

    y_left = y[peak_idx - 1]
    y_center = y[peak_idx]
    y_right = y[peak_idx + 1]

    denom = y_left - 2 * y_center + y_right
    if abs(denom) < 1e-15:
        return 0.0, y_center

    delta = (y_left - y_right) / (2 * denom)
    peak_val = y_center - (y_left - y_right) * delta / 4
    return delta, peak_val

def generate_dp_qpsk(
    n_symbols: int,
    baud_rate: float,
    samples_per_symbol: int = 4,
    rolloff: float = 0.1,
    span: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dual‑polarisation QPSK signal with root‑raised‑cosine pulse shaping.

    Parameters
    ----------
    n_symbols : int
        Number of QPSK symbols per polarisation.
    baud_rate : float
        Symbol rate in Hz (or GBd if time units are ns).
    samples_per_symbol : int, optional
        Upsampling factor (default 4).
    rolloff : float, optional
        Roll‑off factor of the RRC filter (0 < rolloff <= 1).
    span : int, optional
        Filter span in symbols (one‑sided length = span//2).
    seed : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    t : ndarray
        Time vector in seconds (same length as the shaped signal).
    sig_x : ndarray
        Complex baseband envelope for the X polarisation.
    sig_y : ndarray
        Complex baseband envelope for the Y polarisation.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate random bits and Gray‑map to QPSK symbols
    # QPSK constellations: (1+1j), (1-1j), (-1+1j), (-1-1j) / sqrt(2)
    bits_x = rng.integers(0, 2, size=(n_symbols, 2))
    bits_y = rng.integers(0, 2, size=(n_symbols, 2))

    gray_map = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    sym_x = gray_map[bits_x[:, 0] * 2 + bits_x[:, 1]]
    sym_y = gray_map[bits_y[:, 0] * 2 + bits_y[:, 1]]

    # 2. Upsample (insert zeros) – shape (n_symbols * samples_per_symbol,)
    sym_x_up = np.zeros(n_symbols * samples_per_symbol, dtype=complex)
    sym_x_up[::samples_per_symbol] = sym_x
    sym_y_up = np.zeros(n_symbols * samples_per_symbol, dtype=complex)
    sym_y_up[::samples_per_symbol] = sym_y

    # 3. Design RRC filter
    t_filt, rrc = _rrc_filter(baud_rate, samples_per_symbol, rolloff, span)

    # 4. Convolve (filter) using upfirdn (upsample inside already done, so use 'up' mode
    #    but we already upsampled; just use np.convolve with mode='same' to keep length)
    sig_x = np.convolve(sym_x_up, rrc, mode="same")
    sig_y = np.convolve(sym_y_up, rrc, mode="same")

    # 5. Time vector
    dt = 1 / (baud_rate * samples_per_symbol)
    t = np.arange(len(sig_x)) * dt

    return t, sig_x, sig_y


def _rrc_filter(
    baud_rate: float,
    samples_per_symbol: int,
    rolloff: float,
    span: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a root‑raised‑cosine filter.

    Returns (time vector, filter taps) with peak at t=0.
    """
    Ts = 1 / baud_rate
    dt = Ts / samples_per_symbol
    N = span * samples_per_symbol
    t = np.arange(-N // 2, N // 2 + 1) * dt

    # handle t=0 case safely
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(np.pi * t / Ts * (1 - rolloff)) + 4 * rolloff * t / Ts * np.cos(np.pi * t / Ts * (1 + rolloff))
        den = np.pi * t / Ts * (1 - (4 * rolloff * t / Ts) ** 2)
        h = num / den
    h[t == 0] = 1.0 + rolloff * (4 / np.pi - 1)

    # handle the other singular point
    idx = np.abs(t) == Ts / (4 * rolloff)
    if np.any(idx):
        h[idx] = rolloff / np.sqrt(2) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
        )

    h /= np.sqrt(np.sum(h**2))  # energy normalisation
    return t, h