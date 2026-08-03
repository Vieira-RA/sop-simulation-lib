"""Quaternion representation of SU(2) Jones matrices and sign regularization."""

import numpy as np

def jones_to_quaternion(U: np.ndarray) -> np.ndarray:
    """
    Convert an SU(2) Jones matrix to a unit quaternion.

    Parameters
    ----------
    U : (2,2) complex ndarray
        Unitary matrix with det = 1 (after common‑phase removal).

    Returns
    -------
    q : (4,) float ndarray
        Quaternion (q0, q1, q2, q3) satisfying U = q0*I - i*(q·σ).
    """
    α = U[0, 0]
    β = U[0, 1]
    q0 = np.real(α)
    q1 = -np.imag(α)
    q2 = -np.imag(β)
    q3 = -np.real(β)
    q = np.array([q0, q1, q2, q3])
    return q / np.linalg.norm(q)   # ensure unit norm despite numerical errors


def quaternion_to_jones(q: np.ndarray) -> np.ndarray:
    """
    Reconstruct an SU(2) Jones matrix from a unit quaternion.

    Parameters
    ----------
    q : (4,) float ndarray

    Returns
    -------
    U : (2,2) complex ndarray
    """
    q0, q1, q2, q3 = q
    U = np.array(
        [
            [q0 - 1j * q1, -q3 - 1j * q2],
            [q3 - 1j * q2, q0 + 1j * q1],
        ]
    )
    return U


def regularize_signs(Q: np.ndarray) -> np.ndarray:
    """
    Remove sign ambiguity from a sequence of quaternions.

    For each consecutive pair, if the dot product is negative,
    flip the sign of the later quaternion so that the sequence
    follows the shortest path on the 3‑sphere.

    Parameters
    ----------
    Q : (N, 4) float ndarray
        Sequence of quaternions (may have arbitrary sign flips).

    Returns
    -------
    Q_reg : (N, 4) float ndarray
        Sequence with consistent signs.
    """
    Q_reg = np.copy(Q)
    for i in range(1, len(Q_reg)):
        if np.dot(Q_reg[i - 1], Q_reg[i]) < 0:
            Q_reg[i] = -Q_reg[i]
    return Q_reg


def quaternion_to_rotation_vector(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert a unit quaternion to the rotation vector φ.

    U = exp(-i/2 φ·σ)  ↔  q0 = cos(φ/2),  q_vec = (φ/φ) sin(φ/2).

    Parameters
    ----------
    q : (4,) float ndarray
        (q0, q1, q2, q3)
    eps : float
        Tolerance for small‑angle approximation.

    Returns
    -------
    phi : (3,) float ndarray
        Rotation vector (radians).
    """
    q0 = np.clip(q[0], -1.0, 1.0)
    q_vec = q[1:]
    norm_q_vec = np.linalg.norm(q_vec)

    if 1.0 - q0**2 < eps:
        # small rotation angle: φ ≈ 2 q_vec
        return 2.0 * q_vec
    else:
        phi_mag = 2.0 * np.arccos(q0)
        return phi_mag * q_vec / norm_q_vec