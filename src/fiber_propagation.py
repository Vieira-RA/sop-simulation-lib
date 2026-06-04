"""Fibre propagation models for Jones and Stokes calculus."""

import numpy as np

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = np.stack([sigma_x, sigma_y, sigma_z], axis=0)  # shape (3,2,2)


def propagate_unitary(
    z: np.ndarray,
    beta: np.ndarray,
    U0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve dU/dz = -i/2 [β(z)·σ] U using piecewise‑constant birefringence.

    Parameters
    ----------
    z : (N,) array_like
        Position grid [m]. Must be strictly increasing.
    beta : (3, N) array_like
        Birefringence vector β(z) at each point [rad/m].
        beta[:, k] corresponds to position z[k].
    U0 : (2,2) array_like, optional
        Jones matrix at z[0]. Defaults to identity.

    Returns
    -------
    U : (2,2) complex ndarray
        Jones matrix at the end of the fibre z[-1].
    """
    z = np.asarray(z, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if z.ndim != 1 or z.shape[0] < 2:
        raise ValueError("z must be 1D with at least 2 points")
    if beta.shape != (3, z.shape[0]):
        raise ValueError(
            f"beta must have shape (3, {z.shape[0]}), got {beta.shape}"
        )

    if U0 is None:
        U = np.eye(2, dtype=complex)
    else:
        U = np.asarray(U0, dtype=complex)

    # Iterate over segments
    for k in range(len(z) - 1):
        dz = z[k + 1] - z[k]
        beta_k = beta[:, k]                     # β at the start of the segment
        theta = np.linalg.norm(beta_k) * dz

        if theta == 0:
            U_step = np.eye(2, dtype=complex)
        else:
            n = beta_k / np.linalg.norm(beta_k)  # rotation axis (unit)
            # n·σ = n_x σ_x + n_y σ_y + n_z σ_z
            n_dot_sigma = np.tensordot(n, PAULI, axes=1)  # (2,2)
            U_step = (
                np.cos(theta / 2) * np.eye(2, dtype=complex)
                - 1j * np.sin(theta / 2) * n_dot_sigma
            )

        # Left‑multiply because dU = (-i/2 β·σ) U ⇒ U(z+dz) = exp(...) U(z)
        U = U_step @ U

    return U