"""Plotting utilities for polarization optics."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_poincare_sphere(stokes_vectors, ax=None, title="Poincaré sphere",
                         color='C0', alpha=0.7, s=30, show_sphere=True):
    """
    Plot 3D points on the Poincaré sphere.

    Parameters
    ----------
    stokes_vectors : ndarray, shape (3, N) or (4, N)
        If 4‑row, only rows 1‑3 (S1,S2,S3) are used and normalized to
        ensure they lie on the unit sphere.
    ax : matplotlib Axes3D or None
        If None, a new figure is created.
    title : str
        Title for the axes.
    color : any valid matplotlib color
        Color of the points.
    alpha : float
        Transparency of the points.
    s : float
        Marker size.
    show_sphere : bool
        If True, draw a wireframe sphere of radius 1.

    Returns
    -------
    ax : Axes3D
        The axes object with the plot.
    """
    # Extract S1,S2,S3 and normalize if needed
    if stokes_vectors.shape[0] == 4:
        S = stokes_vectors[1:, :]
    else:
        S = stokes_vectors

    # Normalize each vector to unit length (assumes non‑zero)
    norms = np.sqrt(np.sum(S**2, axis=0))
    S_norm = S / norms

    # Create figure if no axes given
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Draw the sphere
    if show_sphere:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, rstride=5, cstride=5,
                          color='gray', alpha=0.2, linewidth=0.5)

    # Plot points
    ax.scatter(S_norm[0], S_norm[1], S_norm[2],
               c=color, alpha=alpha, s=s)

    # Labels and limits
    ax.set_xlabel('S₁')
    ax.set_ylabel('S₂')
    ax.set_zlabel('S₃')
    ax.set_title(title)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

    return ax