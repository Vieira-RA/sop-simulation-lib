"""
Rotation matrices and Stokes‑vector geometry.
"""

import numpy as np

# Pauli matrices for Jones → rotation conversion
_sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
_sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = np.stack([_sigma_x, _sigma_y, _sigma_z], axis=0)


def jones_to_rotation_matrix(U):
    """
    Convert a 2×2 unitary Jones matrix U to its 3×3 rotation matrix R
    acting on real Stokes vectors: s_out = R @ s_in.

    Parameters
    ----------
    U : ndarray, shape (2, 2), complex
        Unitary Jones matrix.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Real orthogonal rotation matrix.
    """
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = 0.5 * np.real(
                np.trace(PAULI[i] @ U @ PAULI[j] @ U.conj().T)
            )
    return R


def rotate_centroid_to_north_pole(S):
    """
    Rotate a set of Stokes vectors so that their mean (centroid) points
    to the North Pole (0,0,1).

    Parameters
    ----------
    S : ndarray, shape (N, 3)
        Sequence of Stokes vectors.

    Returns
    -------
    S_rot : ndarray, shape (N, 3)
        Rotated Stokes vectors.
    R : ndarray, shape (3, 3)
        Rotation matrix used.
    """
    centroid = np.mean(S, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-15:
        return S, np.eye(3)

    u = centroid / centroid_norm
    north_pole = np.array([0.0, 0.0, 1.0])

    if np.allclose(u, north_pole):
        return S, np.eye(3)
    if np.allclose(u, -north_pole):
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        return S @ R.T, R

    v = np.cross(u, north_pole)
    s = np.linalg.norm(v)
    c = np.dot(u, north_pole)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
    S_rot = S @ R.T
    return S_rot, R