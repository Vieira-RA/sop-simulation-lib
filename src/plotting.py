import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, ListedColormap, TwoSlopeNorm

# ----- global style settings applied when this module is imported -----
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

def plot_std_heatmap(std_data, SNRS, BANDWIDTHS, title, filename,
                     vmax=100, gamma=0.4):
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
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Standard deviation (km)', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_bias_heatmap(bias_data, SNRS, BANDWIDTHS, title, filename,
                      vmin=-50, vmax=50, cmap='coolwarm'):
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(SNRS, BANDWIDTHS)

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
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Bias (km)', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_success_rate_heatmap(success, SNRS, BANDWIDTHS, title, filename):
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
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Success rate', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()