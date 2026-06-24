"""
Ornstein‑Uhlenbeck birefringence model for PMD simulations.

This module implements a continuous random birefringence profile:
  - β₀(z) is a 3‑D Ornstein‑Uhlenbeck process with correlation length L_F
    and a magnitude set by the PMD coefficient D_pmd.
  - β′(z) = ∂β/∂ω is a white noise process with variance per component
    γ₀² / (3·Δz), consistent with the Shtaif–Mecozzi continuous model.
"""

import numpy as np


def generate_ou_birefringence(
    L: float,
    L_F: float,
    D_pmd: float,
    lambda0: float = 1550e-9,
    dz: float = 0.1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate β₀(z) as an OU process and β′(z) as white noise on a fine grid.

    Parameters
    ----------
    L : float
        Fibre length [m].
    L_F : float
        Birefringence correlation length [m].
    D_pmd : float
        PMD coefficient [s/√m], e.g. 3.16e-15 for 0.1 ps/√km.
    lambda0 : float
        Central wavelength [m].
    dz : float
        Spatial step [m]. Must be much smaller than the beat length to
        ensure Ω = |β₀|·dz ≪ 1, so that the full 3‑D contribution of
        β′Δz is retained.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    z : (N,) ndarray
        Position grid [m].
    beta0 : (3, N) ndarray
        Static birefringence vector at each z [rad/m].
    beta_prime : (3, N) ndarray
        Frequency derivative of birefringence [rad/(m·rad/s)].
    """
    rng = np.random.default_rng(seed)
    c = 299792458.0
    omega0 = 2.0 * np.pi * c / lambda0

    # ---- OU process parameters for β₀ ----
    # Desired rms magnitude of β₀: we use the same value as the wave‑plate
    # model for a realistic beat length, though it does not affect the PMD
    # statistics as long as it is constant (only rotates the PMD vector).
    beta_rms = (omega0 / np.sqrt(L_F)) * D_pmd

    # For a 3‑D OU process dβ = -β/τ dz + σ_OU dW
    tau = L_F          # correlation time (in metres)
    # Steady‑state variance per component: σ_OU² * τ / 2
    # Total mean square |β|² = 3 * (σ_OU² τ / 2) = beta_rms²
    sigma_OU = np.sqrt(2.0 * beta_rms**2 / (3.0 * tau))

    # ---- Spatial grid ----
    n_points = int(np.round(L / dz)) + 1
    z = np.linspace(0.0, L, n_points)

    # ---- Generate β₀ via Euler–Maruyama ----
    beta0 = np.zeros((3, n_points), dtype=float)
    # initial condition from steady‑state distribution
    beta0[:, 0] = rng.normal(scale=beta_rms / np.sqrt(3.0), size=3)

    for k in range(1, n_points):
        dz_actual = z[k] - z[k - 1]
        drift = -beta0[:, k - 1] / tau * dz_actual
        diffusion = sigma_OU * np.sqrt(dz_actual) * rng.normal(size=3)
        beta0[:, k] = beta0[:, k - 1] + drift + diffusion

    # ---- White noise for β′ ----
    gamma0_sq = D_pmd**2
    # Variance per component: γ₀² / (3·dz)
    var_beta_prime = gamma0_sq / (3.0 * dz)
    sigma_beta_prime = np.sqrt(var_beta_prime)

    beta_prime = rng.normal(scale=sigma_beta_prime, size=(3, n_points))

    return z, beta0, beta_prime