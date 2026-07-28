"""
Utilities for chromatic-dispersion-induced propagation delays.

These functions compute the relative arrival time of WDM channels due
to fibre chromatic dispersion and apply arbitrary fractional delays to
time-domain signals using the Fourier shift theorem.

All delays are referenced to the earliest arriving channel, so the
minimum delay is always zero.
"""

from __future__ import annotations

import numpy as np

C = 299792458.0  # m/s


# ---------------------------------------------------------------------
# Wavelength / frequency conversion
# ---------------------------------------------------------------------

def frequency_to_wavelength(freq_hz: np.ndarray) -> np.ndarray:
    """
    Convert optical frequency to wavelength.

    Parameters
    ----------
    freq_hz : ndarray
        Optical frequency [Hz].

    Returns
    -------
    wavelength_nm : ndarray
        Wavelength [nm].
    """
    return C / freq_hz * 1e9


def wavelength_to_frequency(wavelength_nm: np.ndarray) -> np.ndarray:
    """
    Convert wavelength to optical frequency.

    Parameters
    ----------
    wavelength_nm : ndarray
        Wavelength [nm].

    Returns
    -------
    freq_hz : ndarray
        Optical frequency [Hz].
    """
    return C / (wavelength_nm * 1e-9)


# ---------------------------------------------------------------------
# Dispersion integral
# ---------------------------------------------------------------------

def integrated_dispersion(
    lambda1_nm: float,
    lambda2_nm: float,
    D0: float = 20.2,
    S0: float = 0.06,
    lambda0_nm: float = 1550.0,
) -> float:
    """
    Integral of the chromatic-dispersion curve.

    Computes

        ∫ D(λ) dλ

    assuming

        D(λ) = D0 + S0 (λ − λ0)

    Parameters
    ----------
    lambda1_nm
        Initial wavelength [nm].

    lambda2_nm
        Final wavelength [nm].

    D0
        Dispersion at λ0 [ps/(nm km)].

    S0
        Dispersion slope [ps/(nm² km)].

    lambda0_nm
        Zero-slope reference wavelength [nm].

    Returns
    -------
    integral : float

        Value of

            D0(λ2−λ1)
            + S0/2[(λ2−λ0)^2 − (λ1−λ0)^2]

        Units:

            ps/km
    """

    return (
        D0 * (lambda2_nm - lambda1_nm)
        + 0.5
        * S0
        * (
            (lambda2_nm - lambda0_nm) ** 2
            - (lambda1_nm - lambda0_nm) ** 2
        )
    )


# ---------------------------------------------------------------------
# Relative delays
# ---------------------------------------------------------------------

def relative_channel_delays(
    wavelengths_nm: np.ndarray,
    distance_km: float,
    reference: str = "earliest",
    D0: float = 20.2,
    S0: float = 0.06,
    lambda0_nm: float = 1550.0,
) -> np.ndarray:
    """
    Compute chromatic-dispersion delay for every channel.

    Parameters
    ----------
    wavelengths_nm
        Channel wavelengths [nm].

    distance_km
        Distance between perturbation and receiver.

    reference

        "earliest"
            Earliest arriving channel has zero delay.

        "centre"
            Centre channel has zero delay.

    Returns
    -------
    delays : ndarray

        Relative delays [seconds].
    """

    wavelengths_nm = np.asarray(wavelengths_nm)

    if reference == "centre":
        lambda_ref = wavelengths_nm[len(wavelengths_nm) // 2]
    elif reference == "earliest":
        lambda_ref = np.min(wavelengths_nm)
    else:
        raise ValueError("reference must be 'earliest' or 'centre'")

    integral = np.array([
        integrated_dispersion(
            lambda_ref,
            lam,
            D0=D0,
            S0=S0,
            lambda0_nm=lambda0_nm,
        )
        for lam in wavelengths_nm
    ])

    #
    # Units:
    #
    # integral       -> ps/km
    #
    # × distance     -> ps
    #
    delay_ps = integral * distance_km

    delay_s = delay_ps * 1e-12

    #
    # Earliest channel always has zero delay.
    #
    delay_s -= np.min(delay_s)

    return delay_s


# ---------------------------------------------------------------------
# FFT fractional delay
# ---------------------------------------------------------------------

def fractional_delay_fft(
    signal: np.ndarray,
    dt: float,
    delay: float,
) -> np.ndarray:
    """
    Apply an arbitrary time delay using the Fourier shift theorem.

    Parameters
    ----------
    signal

        Complex or real signal.

    dt

        Sampling interval [s].

    delay

        Positive delay [s].

    Returns
    -------
    delayed_signal
    """

    n = len(signal)

    freq = np.fft.fftfreq(n, d=dt)

    spectrum = np.fft.fft(signal)

    phase = np.exp(1j * 2 * np.pi * freq * delay)

    delayed = np.fft.ifft(spectrum * phase)

    #
    # Preserve purely real signals.
    #
    if np.isrealobj(signal):
        delayed = delayed.real

    return delayed


# ---------------------------------------------------------------------
# Jones matrices
# ---------------------------------------------------------------------

def delay_jones_sequence(
    U: np.ndarray,
    dt: float,
    delay: float,
) -> np.ndarray:
    """
    Apply the same fractional delay to every Jones-matrix element.

    Parameters
    ----------
    U

        Shape

            (Ntime,2,2)

    dt

        Sampling interval [s].

    delay

        Delay [s].

    Returns
    -------
    U_delayed

        Delayed Jones matrices.
    """

    U = np.asarray(U)

    if U.ndim != 3:
        raise ValueError("Expected array of shape (Ntime,2,2).")

    if U.shape[1:] != (2, 2):
        raise ValueError("Expected Jones matrices of shape (2,2).")

    out = np.empty_like(U)

    for i in range(2):
        for j in range(2):

            out[:, i, j] = fractional_delay_fft(
                U[:, i, j],
                dt,
                delay,
            )

    return out