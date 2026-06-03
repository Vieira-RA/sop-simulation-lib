"""
Quaternion / Cayley-Klein propagation of the Jones matrix along a fibre.

Equation (3) of Mecozzi 2025 without the polarisation-averaged phase term:

    dU/dz = -i/2 (β(z,t)·σ) U

U(z) is a 2×2 unitary matrix, stored as a unit quaternion (q0,q1,q2,q3).

Steps are performed analytically using the exponential map,
avoiding trigonometric functions during integration.
"""

import numpy as np
from typing import Tuple, Callable

# ----------------------------------------------------------------------
#  Quaternion ↔ Jones matrix conversions
# ----------------------------------------------------------------------

def quaternion_to_jones(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion to a 2×2 Jones matrix.

    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion (q0, q1, q2, q3) with q0²+q1²+q2²+q3² = 1.

    Returns
    -------
    U : ndarray, shape (2,2), complex
        Jones matrix: U = q0 I - i (q1 σ1 + q2 σ2 + q3 σ3).
    """
    q0, q1, q2, q3 = q
    U = np.array([[q0 - 1j*q3, -q2 - 1j*q1],
                  [q2 - 1j*q1,  q0 + 1j*q3]], dtype=complex)
    return U


def jones_to_quaternion(U: np.ndarray) -> np.ndarray:
    """
    Convert a Jones matrix (unitary, det=1) to a unit quaternion.

    The formula extracts the real components according to Appendix A of Mecozzi 2025.

    Parameters
    ----------
    U : ndarray, shape (2,2), complex
        Unitary Jones matrix.

    Returns
    -------
    q : ndarray, shape (4,)
        Quaternion (q0, q1, q2, q3) real.
    """
    # U = [[q0-i q3, -q2-i q1],
    #      [q2-i q1,  q0+i q3]]
    q0 = 0.5 * np.real(U[0,0] + U[1,1])
    q1 = 0.5j * (U[1,0] + U[0,1])
    q2 = 0.5  * (U[1,0] - U[0,1])
    q3 = 0.5j * (U[0,0] - U[1,1])
    # All components must be real (up to rounding)
    q = np.array([q0, np.real(q1), np.real(q2), np.real(q3)])
    return q


# ----------------------------------------------------------------------
#  Rotation vector ↔ quaternion
# ----------------------------------------------------------------------

def rotation_vector_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Compute the rotation vector φ from a unit quaternion.

    Equation (A12): φ = 2 arccos(q0) * (q1,q2,q3) / sqrt(q1²+q2²+q3²).

    If q0 ≈ 1 (no rotation), φ = (0,0,0).

    Parameters
    ----------
    q : array_like, shape (4,)
        Unit quaternion.

    Returns
    -------
    phi : ndarray, shape (3,)
        Rotation vector.
    """
    q0 = q[0]
    vec = q[1:4]
    norm_vec = np.linalg.norm(vec)
    if norm_vec < 1e-15:
        return np.zeros(3)
    angle = 2.0 * np.arccos(np.clip(q0, -1.0, 1.0))
    return angle * vec / norm_vec


def quaternion_from_rotation_vector(phi: np.ndarray) -> np.ndarray:
    """
    Build a unit quaternion representing the rotation exp(-i/2 φ·σ).

    The resulting quaternion is q = (cos(|φ|/2), sin(|φ|/2) * φ/|φ|).

    Parameters
    ----------
    phi : array_like, shape (3,)
        Rotation vector.

    Returns
    -------
    q : ndarray, shape (4,)
        Unit quaternion (q0,q1,q2,q3).
    """
    phi = np.asarray(phi)
    angle = np.linalg.norm(phi)
    if angle < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = phi / angle
    half_angle = 0.5 * angle
    q0 = np.cos(half_angle)
    q_vec = np.sin(half_angle) * axis
    return np.array([q0, q_vec[0], q_vec[1], q_vec[2]])


# ----------------------------------------------------------------------
#  Quaternion composition (Hamilton product)
# ----------------------------------------------------------------------

def compose_quaternions(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Hamilton product r = p * q.

    This corresponds to multiplication of the associated Jones matrices:
        U_r = U_p · U_q.

    Parameters
    ----------
    p, q : ndarray, shape (4,)
        Input quaternions.

    Returns
    -------
    r : ndarray, shape (4,)
        Product quaternion.
    """
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q

    r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1
    r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0
    return np.array([r0, r1, r2, r3])


# ----------------------------------------------------------------------
#  Propagation step in quaternion form
# ----------------------------------------------------------------------

def propagation_step(q: np.ndarray, beta: np.ndarray, dz: float) -> np.ndarray:
    """
    Advance the quaternion by one spatial step Δz.

    Using the local birefringence vector β (rad/unit length),
    the incremental rotation is U_inc = exp(-i/2 (β·σ) Δz).
    In quaternion form:
        q_inc = quaternion_from_rotation_vector(β * Δz)
        q_new = compose_quaternions(q_inc, q)

    Parameters
    ----------
    q : ndarray, shape (4,)
        Current unit quaternion.
    beta : array_like, shape (3,)
        Local birefringence vector (β1,β2,β3) at this z.
    dz : float
        Step length (same units as 1/|β|).

    Returns
    -------
    q_new : ndarray, shape (4,)
        Updated unit quaternion.
    """
    phi_step = np.asarray(beta) * dz
    q_inc = quaternion_from_rotation_vector(phi_step)
    return compose_quaternions(q_inc, q)


# ----------------------------------------------------------------------
#  Propagation over the whole fibre (constant β for a given time)
# ----------------------------------------------------------------------

def propagate_fibre(L: float, dz: float,
                    beta_func: Callable[[float], np.ndarray]) -> np.ndarray:
    """
    Propagate the identity Jones matrix from z=0 to z=L for one time sample.

    Parameters
    ----------
    L : float
        Total fibre length (same units as dz).
    dz : float
        Integration step.
    beta_func : callable
        Function beta_func(z) returning the 3‑element β vector at position z.

    Returns
    -------
    q_end : ndarray, shape (4,)
        Unit quaternion at the fibre output.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    z = 0.0
    while z < L:
        step = min(dz, L - z)
        beta = beta_func(z)
        q = propagation_step(q, beta, step)
        z += step
    return q


# ----------------------------------------------------------------------
#  Time‑series with sign regularisation (Appendix A)
# ----------------------------------------------------------------------

def propagate_time_series(times: np.ndarray, L: float, dz: float,
                          beta_func_zt: Callable[[float, float], np.ndarray]
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate the Jones matrix for each time sample, applying sign‑regularisation.

    For each t in `times`, the quaternion is obtained by spatial propagation
    with β(z,t). Then it is sign‑corrected against the previously regularised
    quaternion using the dot product: if dot < 0, the new quaternion is flipped.

    Parameters
    ----------
    times : array_like, 1‑D
        Time samples (arbitrary units, consistent with beta_func_zt).
    L, dz : float
        Fibre length and step.
    beta_func_zt : callable
        beta = beta_func_zt(z, t) returns the birefringence vector at (z,t).

    Returns
    -------
    q_series : ndarray, shape (len(times), 4)
        Regularised quaternions.
    phi_series : ndarray, shape (len(times), 3)
        Corresponding rotation vectors (computed from the regularised quaternions).
    """
    n = len(times)
    q_series = np.empty((n, 4))
    phi_series = np.empty((n, 3))

    q_prev = None
    for i, t in enumerate(times):
        # Create z-dependent beta for this time
        def beta_z(z):
            return beta_func_zt(z, t)
        q_new = propagate_fibre(L, dz, beta_z)

        # Sign regularisation (skip first sample)
        if i > 0:
            s = np.dot(q_prev, q_new)
            if s < 0:
                q_new = -q_new
        q_series[i] = q_new
        phi_series[i] = rotation_vector_from_quaternion(q_new)
        q_prev = q_new

    return q_series, phi_series


# ----------------------------------------------------------------------
#  Stokes transformation helper
# ----------------------------------------------------------------------

def apply_rotation_to_stokes(phi: np.ndarray, s_in: np.ndarray) -> np.ndarray:
    """
    Rotate a 3‑element Stokes vector using the rotation vector φ.

    The Jones matrix U = exp(-i/2 φ·σ) corresponds to a 3D rotation
    of the Stokes vector by angle |φ| around axis φ/|φ|.

    Parameters
    ----------
    phi : array_like, shape (3,)
        Rotation vector.
    s_in : array_like, shape (3,)
        Input normalised Stokes vector (s1,s2,s3).

    Returns
    -------
    s_out : ndarray, shape (3,)
        Rotated Stokes vector.
    """
    phi = np.asarray(phi)
    angle = np.linalg.norm(phi)
    if angle < 1e-15:
        return np.asarray(s_in).copy()
    axis = phi / angle
    # Rodrigues rotation formula
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    s_out = (cos_a * np.asarray(s_in)
             + sin_a * np.cross(axis, s_in)
             + (1 - cos_a) * np.dot(axis, s_in) * axis)
    return s_out

def propagate_time_series_unwrapped(
    times: np.ndarray,
    L: float,
    dz: float,
    beta_func_zt: Callable[[float, float], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as propagate_time_series, but returns an unwrapped continuous rotation
    vector by integrating incremental rotations between time samples.

    Parameters
    ----------
    times : array_like
    L, dz : float
    beta_func_zt : callable(z, t) -> (3,) array

    Returns
    -------
    q_series : ndarray (n,4)   sign‑regularised quaternions
    phi_unwrapped : ndarray (n,3)   continuous rotation vector (no 2π jumps)
    """
    n = len(times)
    q_series = np.empty((n, 4))
    phi_unwrapped = np.empty((n, 3))
    phi_current = np.zeros(3)  # starting reference (identity rotation)

    q_prev = None
    for i, t in enumerate(times):
        # propagate through fibre for this time instant
        def beta_z(z):
            return beta_func_zt(z, t)
        q_new = propagate_fibre(L, dz, beta_z)

        # sign regularisation
        if i > 0:
            if np.dot(q_prev, q_new) < 0:
                q_new = -q_new

        q_series[i] = q_new

        # compute incremental rotation from previous sample
        if i == 0:
            # first sample: total rotation is just the quaternion's vector
            phi_current = rotation_vector_from_quaternion(q_new)
        else:
            # delta_q = q_new * conj(q_prev)
            q_prev_conj = np.array([q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]])
            delta_q = compose_quaternions(q_new, q_prev_conj)
            # ensure delta_q is on the "near" hemisphere (already true if time step small)
            if delta_q[0] < 0:
                delta_q = -delta_q
            # extract small rotation vector
            dphi = rotation_vector_from_quaternion(delta_q)
            phi_current = phi_current + dphi

        phi_unwrapped[i] = phi_current
        q_prev = q_new

    return q_series, phi_unwrapped