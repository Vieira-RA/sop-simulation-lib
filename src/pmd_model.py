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

    # Magnitude from PMD relation:
    #   ⟨τ²⟩/L = (2 L_int / ω₀²) ⟨β²⟩,
    # where L_int is the integral correlation length.
    # For the piecewise‑constant wave‑plate model the autocorrelation is triangular,
    # giving L_int = L_F / 2.  Substituting L_int yields:
    #   D_pmd² = (L_F / ω₀²) β_mag²   →   β_mag = (ω₀ / √L_F) D_pmd.
    L_int = L_F / 2.0
    beta_mag = (omega0 / np.sqrt(2.0 * L_int)) * D_pmd   # same as (omega0 / sqrt(L_F)) * D_pmd

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

def scale_birefringence_to_wavelength(
    beta0: np.ndarray,
    lambda0: float,
    lambda_target: float,
) -> np.ndarray:
    """
    Scale birefringence profile to a new wavelength using first‑order PMD.

    β(λ_target, z) = β₀(z) * (λ₀ / λ_target)

    Parameters
    ----------
    beta0 : (3, N) ndarray
        Birefringence profile at the reference wavelength λ₀ [rad/m].
    lambda0 : float
        Reference wavelength [m].
    lambda_target : float
        Target wavelength [m].

    Returns
    -------
    beta : (3, N) ndarray
        Birefringence at λ_target.
    """
    scale = lambda0 / lambda_target   # because ω/ω₀ = λ₀/λ
    return beta0 * scale


def generate_multiwavelength_birefringence(
    L: float,
    L_F: float,
    D_pmd: float,
    lambda0: float,
    wavelengths: list[float],
    dz: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    """
    Generate β₀(z) and scaled profiles for multiple wavelengths.

    Parameters
    ----------
    L, L_F, D_pmd, lambda0, dz, seed : see generate_birefringence_profile
    wavelengths : list of float
        Target wavelengths [m].

    Returns
    -------
    z : (N,) ndarray
        Position grid [m].
    beta0 : (3, N) ndarray
        Reference birefringence at λ₀.
    profiles : dict
        {lambda: (3, N) ndarray} scaled birefringence for each target wavelength.
    """
    z, beta0 = generate_birefringence_profile(
        L=L, L_F=L_F, D_pmd=D_pmd, lambda0=lambda0, dz=dz, seed=seed
    )

    profiles = {}
    for wl in wavelengths:
        profiles[wl] = scale_birefringence_to_wavelength(
            beta0, lambda0, wl
        )

    return z, beta0, profiles

# Add this function inside src/pmd_model.py (after the existing functions)

def extract_pmd_vector(
    U1: np.ndarray,
    U2: np.ndarray,
    delta_omega: float,
) -> tuple[np.ndarray, float]:
    """
    Compute the PMD vector from two Jones matrices at nearby frequencies.

    Uses the central difference approximation:
        dU/dω ≈ (U2 - U1) / Δω
    and the relation  Ω = (dU/dω) U† = -i/2 (τ·σ).

    Parameters
    ----------
    U1, U2 : (2,2) complex ndarray
        Jones matrices at ω - Δω/2 and ω + Δω/2.
    delta_omega : float
        Frequency difference ω2 - ω1 [rad/s].

    Returns
    -------
    tau : (3,) ndarray
        PMD vector [s].
    dgd : float
        Differential group delay = |τ| [s].
    """
    from fiber_propagation import PAULI  # (3,2,2) Pauli matrices

    dU_dw = (U2 - U1) / delta_omega
    Omega = dU_dw @ U1.conj().T

    tau = np.array(
        [1j * np.trace(Omega @ PAULI[k]) for k in range(3)],
        dtype=complex,
    ).real
    
    dgd = np.linalg.norm(tau)
    return tau, dgd