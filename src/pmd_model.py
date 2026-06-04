"""Generate random birefringence profiles using the wave‑plate model."""

import numpy as np


def generate_birefringence_profile(
    L: float,
    L_F: float,
    D_pmd: float,
    lambda0: float = 1550e-9,
    dz: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a piecewise‑constant random birefringence vector β₀(z) at λ₀.

    Implements the random wave‑plate model:
      - Fibre divided into segments of length L_F.
      - In each segment β is a constant vector with random direction
        uniformly distributed on the Poincaré sphere.
      - Magnitude is determined by the PMD coefficient D_pmd.

    Parameters
    ----------
    L : float
        Total fibre length [m].
    L_F : float
        Birefringence correlation length (segment length) [m].
    D_pmd : float
        PMD coefficient [s/√m], e.g. 3.16e-15 for 0.1 ps/√km.
    lambda0 : float
        Central wavelength [m], default 1550 nm.
    dz : float, optional
        Spatial step for the output grid. If None, uses L_F/10.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    z : (N,) ndarray
        Position grid [m].
    beta0 : (3, N) ndarray
        Birefringence vector [rad/m] at each z.
    """
    rng = np.random.default_rng(seed)
    c = 299792458.0                     # m/s
    omega0 = 2 * np.pi * c / lambda0    # rad/s

    # Magnitude from PMD relation: D_pmd² = (2 L_F / ω₀²) β_mag²
    beta_mag = (omega0 / np.sqrt(2 * L_F)) * D_pmd

    # Build spatial grid
    if dz is None:
        dz = L_F / 10.0
    n_points = int(np.ceil(L / dz)) + 1
    z = np.linspace(0, L, n_points)

    # Number of complete segments; last one may be shorter
    n_segments = int(np.ceil(L / L_F))
    beta0 = np.zeros((3, n_points), dtype=float)

    for seg in range(n_segments):
        # Random unit vector on the 3‑sphere
        n_hat = rng.normal(size=3)
        n_hat /= np.linalg.norm(n_hat)

        # Indices belonging to this segment
        seg_start = seg * L_F
        seg_end = min((seg + 1) * L_F, L)
        mask = (z >= seg_start) & (z <= seg_end + 1e-12)  # include endpoint

        beta0[:, mask] = beta_mag * n_hat[:, np.newaxis]

    return z, beta0