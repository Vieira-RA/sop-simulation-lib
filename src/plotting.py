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

    """Publication‑quality plotting helpers for sweep analysis."""

"""Publication‑quality plotting helpers for sweep analysis."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, ListedColormap

def plot_std_heatmap(std_data, SNRS, BANDWIDTHS, title, filename,
                     vmax=100, gamma=0.4):
    """
    Heatmap of standard deviation with >vmax shown in red.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(SNRS, BANDWIDTHS)

    cmap = plt.cm.plasma.copy()
    cmap.set_over('red')
    cmap.set_bad('white')

    norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
    data_clipped = np.clip(std_data, 0, vmax)

    c = ax.pcolormesh(X, Y, data_clipped, shading='auto',
                      cmap=cmap, norm=norm)

    red_mask = (std_data > vmax) & (~np.isnan(std_data))
    if red_mask.any():
        red_data = np.ma.masked_where(~red_mask, np.ones_like(std_data))
        ax.pcolormesh(X, Y, red_data, shading='auto',
                      cmap=ListedColormap(['red']), vmin=0, vmax=1)

    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bandwidth (Hz)')
    ax.set_title(title)
    cbar = fig.colorbar(c, ax=ax, extend='max',
                        label='Standard deviation (km)')
    cbar.set_ticks(np.linspace(0, vmax, 6))
    cbar.set_ticklabels([f'{t:.0f}' for t in np.linspace(0, vmax, 6)])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


from matplotlib.colors import TwoSlopeNorm   # add this import at the top

def plot_bias_heatmap(bias_data, SNRS, BANDWIDTHS, title, filename,
                      vmin=-50, vmax=50, cmap='coolwarm'):
    """
    Heatmap of bias (estimated – true distance) with zero exactly at the
    white centre of a diverging colormap.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(SNRS, BANDWIDTHS)

    # --- Use a norm that centers the colormap at zero ---
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('white')

    c = ax.pcolormesh(X, Y, bias_data, shading='auto',
                      cmap=cmap_obj, norm=norm)

    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bandwidth (Hz)')
    ax.set_title(title)
    cbar = fig.colorbar(c, ax=ax, extend='both', label='Bias (km)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_success_rate_heatmap(success, SNRS, BANDWIDTHS, title, filename):
    """
    Heatmap of success rate (fraction of realisations within ±10% of truth).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(SNRS, BANDWIDTHS)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad('white')
    c = ax.pcolormesh(X, Y, success, shading='auto',
                      cmap=cmap, vmin=0, vmax=1)
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bandwidth (Hz)')
    ax.set_title(title)
    cbar = fig.colorbar(c, ax=ax, label='Success rate')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()