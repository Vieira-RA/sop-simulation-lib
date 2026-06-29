"""Generate random birefringence profiles using the wave‑plate model."""

import numpy as np
from fiber_propagation import propagate_unitary

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

# ===== Add to src/pmd_model.py =====

def generate_pmd_waveplates(
    L: float,
    L_F: float,
    D_pmd: float,
    lambda0: float,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate piecewise‑constant birefringence and its frequency derivative
    for the random wave‑plate model.

    For each segment of length L_F:
        - β₀ is a constant vector with random direction and magnitude
          derived from the PMD coefficient (as in the existing function).
        - β′ is an independent vector with zero‑mean Gaussian components,
          scaled so that ∫ β′ dz over the link yields the correct
          mean‑square DGD.

    Parameters
    ----------
    L : float
        Total fibre length [m].
    L_F : float
        Birefringence correlation length (segment length) [m].
    D_pmd : float
        PMD coefficient [s/√m], e.g. 3.16e-15 for 0.1 ps/√km.
    lambda0 : float
        Central wavelength [m].
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    z : (N,) ndarray
        Position grid (fine enough to resolve segments).
    beta0 : (3, N) ndarray
        Static birefringence vector at each z [rad/m].
    beta_prime : (3, N) ndarray
        Frequency derivative of birefringence at each z [rad/(m·rad/s)].
    L_seg : float
        Actual segment length used (may be slightly adjusted).
    """
    rng = np.random.default_rng(seed)
    c = 299792458.0
    omega0 = 2 * np.pi * c / lambda0

    # ---- Determine number of segments and actual L_seg ----
    n_segments = int(np.ceil(L / L_F))
    L_seg = L / n_segments   # all segments equal length

    # ---- PMD coefficient squared ----
    gamma0_sq = D_pmd**2   # = <tau^2> / L

    # ---- Scaling for β₀ magnitude (same as before) ----
    # We want the contribution from β₀ to the DGD to be negligible
    # (it only rotates the PMD vector). However, we need a non‑zero
    # β₀ to rotate the polarization. We use the same magnitude as in
    # the existing generate_birefringence_profile, but note that the
    # actual value does not affect the final PMD statistics because
    # it is frequency‑independent. We set it to the value that gives
    # a reasonable rotation per segment, e.g., using the PMD coefficient.
    # Actually, we can set β₀ magnitude arbitrarily, but to stay
    # consistent with the physical fibre, we use the relation:
    #   β_mag = (omega0 / sqrt(L_F)) * D_pmd   (from the wave‑plate model)
    # This is the same as in generate_birefringence_profile.
    beta_mag = (omega0 / np.sqrt(L_F)) * D_pmd

    # ---- Scaling for β′ (frequency derivative) ----
    # In the white‑noise limit, β′ satisfies:
    #   <β′_i(z) β′_j(z')> = (1/3) gamma0^2 δ_ij δ(z-z')
    # For a discrete segment of length L_seg, the variance per component
    # is gamma0^2 / (3 * L_seg).
    #var_beta_prime = gamma0_sq / (3 * L_seg)

    #######     This is cheap AI explanation, not trustworth       ###########
    # However, for segments much longer than the beat length (large Ω),
    # the segment PMD vector is dominated by the projection of β'Δz
    # onto the local birefringence axis. This yields only 1/3 of the        #This is cheap AI explanation, not trustworth
    # total variance of β'Δz. To obtain the desired mean‑square DGD         #I don't know true reason
    # we therefore multiply the theoretical white‑noise variance by 3.
    #######     This is cheap AI explanation, not trustworth       ###########

    var_beta_prime = 3 * gamma0_sq / (3 * L_seg)
    sigma_beta_prime = np.sqrt(var_beta_prime)

    # ---- Build the profiles on a fine z grid ----
    # We'll use a step dz = L_seg / 10 to get smooth representation.
    dz = L_seg / 1
    n_points = int(np.ceil(L / dz)) + 1
    z = np.linspace(0, L, n_points)

    beta0 = np.zeros((3, n_points))
    beta_prime = np.zeros((3, n_points))

    for seg in range(n_segments):
        # Random unit vector for β₀ direction
        n_hat = rng.normal(size=3)
        n_hat /= np.linalg.norm(n_hat)
        beta0_seg = beta_mag * n_hat

        # Random vector for β′ (each component independent Gaussian)
        beta_prime_seg = rng.normal(scale=sigma_beta_prime, size=3)

        # Indices belonging to this segment
        seg_start = seg * L_seg
        seg_end = (seg + 1) * L_seg
        mask = (z >= seg_start) & (z < seg_end)
        # Handle the last point exactly at L
        if seg == n_segments - 1:
            mask |= (z == L)

        beta0[:, mask] = beta0_seg[:, np.newaxis]
        beta_prime[:, mask] = beta_prime_seg[:, np.newaxis]

    return z, beta0, beta_prime, L_seg


def beta_at_frequency(
    beta0: np.ndarray,
    beta_prime: np.ndarray,
    delta_omega: float,
) -> np.ndarray:
    """
    Compute the total birefringence vector at a frequency offset.

    β(ω₀+δω, z) = β₀(z) + δω * β′(z)

    Parameters
    ----------
    beta0, beta_prime : (3, N) ndarray
        Static and derivative birefringence profiles.
    delta_omega : float
        Frequency offset from ω₀ [rad/s].

    Returns
    -------
    beta : (3, N) ndarray
        Total birefringence at that frequency.
    """
    return beta0 + delta_omega * beta_prime


def jones_at_frequency(
    z: np.ndarray,
    beta0: np.ndarray,
    beta_prime: np.ndarray,
    delta_omega: float,
    U0: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the Jones matrix at a given frequency offset using the
    existing propagate_unitary function.

    Parameters
    ----------
    z : (N,) ndarray
        Position grid.
    beta0, beta_prime : (3, N) ndarray
        Birefringence profiles.
    delta_omega : float
        Frequency offset [rad/s].
    U0 : (2,2) complex ndarray, optional
        Initial Jones matrix (default: identity).

    Returns
    -------
    U : (2,2) complex ndarray
        Jones matrix at the end of the fibre.
    """
    beta = beta_at_frequency(beta0, beta_prime, delta_omega)
    return propagate_unitary(z, beta, U0)