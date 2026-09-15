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


def _tree_product(M):
    """Product M[N-1] @ ... @ M[0] by binary-tree reduction (batched matmuls)."""
    M = np.asarray(M, dtype=complex)
    while M.shape[0] > 1:
        n = M.shape[0]
        if n % 2:
            last = M[-1:]
            M = M[:-1]
            M = np.concatenate([M[1::2] @ M[0::2], last], axis=0)
        else:
            M = M[1::2] @ M[0::2]
    return M[0]


def propagate_unitary_vectorized(
    z: np.ndarray,
    beta0: np.ndarray,
    delta: np.ndarray | float | None = None,
    U0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Vectorized (exact) propagation for magnitude modulation of birefringence.

    Solves  dU/dz = -i/2 [beta(z,t).sigma] U  with

        beta(z, t) = beta0(z) * (1 + delta(z, t))

    using the closed form of each segment's SU(2) matrix and a binary-tree
    reduction of the matrix product.  Mathematically equivalent to
    propagate_unitary (differences are floating-point rounding only), but
    ~100-200x faster because the segment loop is vectorized.

    Parameters
    ----------
    z : (N,) array_like
        Position grid [m]. Must be strictly increasing.
    beta0 : (3, N) array_like
        Static birefringence vector beta0(z) [rad/m].
    delta : (N,), (N, nt) array_like or scalar, optional
        Relative modulation of the birefringence magnitude, sampled on the
        z grid.  Defaults to 0 (pure static fibre).
    U0 : (2,2) array_like, optional
        Jones matrix at z[0]. Defaults to identity.

    Returns
    -------
    U : (2,2) or (nt, 2, 2) complex ndarray
        Jones matrix at the fibre end for each snapshot.  Shape (2,2) when
        delta is a scalar or 1D, else (nt, 2, 2).
    """
    z = np.asarray(z, dtype=float)
    beta0 = np.asarray(beta0, dtype=float)

    if z.ndim != 1 or z.shape[0] < 2:
        raise ValueError("z must be 1D with at least 2 points")
    if beta0.shape != (3, z.shape[0]):
        raise ValueError(
            f"beta0 must have shape (3, {z.shape[0]}), got {beta0.shape}"
        )

    # ---- time-independent parts (one per segment; segment start = z[:-1]) ----
    beta_seg = beta0[:, :-1]                    # (3, M), M = N-1
    dz = np.diff(z)                             # (M,)
    mag = np.linalg.norm(beta_seg, axis=0)      # (M,)
    theta0 = mag * dz                           # (M,)

    n_hat = np.zeros_like(beta_seg)
    safe = mag > 0
    n_hat[:, safe] = beta_seg[:, safe] / mag[safe][None, :]

    # S[k] = n_hat[k] . sigma   (M, 2, 2)  -- time-independent
    S = np.einsum("jk,jab->kab", n_hat, PAULI)

    # ---- modulation ----
    if delta is None:
        delta = 0.0
    delta_in = np.asarray(delta, dtype=float)
    single = delta_in.ndim <= 1
    if delta_in.ndim == 0:
        delta = np.full((z.shape[0], 1), float(delta_in))
    elif delta_in.ndim == 1:
        delta = delta_in[:, None]               # (N, 1)
    else:
        delta = delta_in                        # (N, nt)
    if delta.shape[0] != z.shape[0]:
        raise ValueError(
            f"delta first axis must match z ({z.shape[0]}), got {delta.shape[0]}"
        )
    delta = delta[:-1]                          # (M, nt) align with segments

    if U0 is None:
        U0 = np.eye(2, dtype=complex)
    else:
        U0 = np.asarray(U0, dtype=complex)

    nt = delta.shape[1]
    theta = theta0[:, None] * (1.0 + delta)     # (M, nt)
    c = np.cos(theta / 2.0)                     # (M, nt)
    s = np.sin(theta / 2.0)                     # (M, nt)
    I2 = np.eye(2, dtype=complex)

    U = np.empty((nt, 2, 2), dtype=complex)
    for k in range(nt):
        M = c[:, k][:, None, None] * I2[None] - 1j * s[:, k][:, None, None] * S
        U[k] = _tree_product(M) @ U0

    return U[0] if single else U