# -*- coding: utf-8 -*-
"""
OpInf Visualization Library (Robust & Publication-Ready)
------------------------------------------------------
Features:
- MPI Corporate Design integration
- Safe PGF Export (avoids 'text.usetex' crashes in Spyder)
- Hybrid rasterization for efficient 2D/3D plotting
"""

__all__ = [
    "plot_inputs",
    "plot_3D",
    "plot_3D_flat",
    "plot_entries",
    "plot_PDE_data",
    "plot_compare_PDE_data",
    "plot_1D_comparison",
    "plot_PDE_dynamics_2D",
    "plot_PDE_dynamics_3D",
    "plot_POD_modes",
    "plot_reduced_trajectories"
]

import os
import re
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import MaxNLocator
import opinf.parameters

# Load parameters
Params = opinf.parameters.Params()

###############################################################################
# 1. CONFIGURATION & STYLING
###############################################################################

# MPI Colors Definition
MPI_COLORS = {
    'mpi_blue': (51/255, 165/255, 195/255),
    'mpi_red': (120/255, 0/255, 75/255),
    'mpi_green': (0/255, 118/255, 117/255),
    'mpi_grey': (135/255, 135/255, 141/255),
    'mpi_beige': (236/255, 233/255, 212/255),
    'mpi_charcoal': (60/255, 60/255, 65/255),
    'mpi_dark_blue': (20/255, 65/255, 85/255)
}

# Custom diverging colormap (Blue -> Beige -> Red)
MPI_CMAP = LinearSegmentedColormap.from_list(
    "mpi_diverging",
    [MPI_COLORS['mpi_blue'], MPI_COLORS['mpi_beige'], MPI_COLORS['mpi_red']]
)

# CRITICAL: Safe configuration for robust plotting
# We set text.usetex = False to prevent zlib/cmr9 crashes in IDEs like Spyder.
# PGF output remains functional, and LaTeX will render the text properly upon compilation.
PUBLICATION_RC = {
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",  # Uses serif in the plot; LaTeX handles PGF font styling
    "text.usetex": False,
    "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc} \usepackage{amsmath} \usepackage{siunitx}",
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
}

###############################################################################
# 2. HELPER FUNCTIONS
###############################################################################

def _handle_time_units(t):
    """Automatically scales time values and returns appropriate units (s, min, or h)."""
    t_max = np.max(t)
    if t_max > 12000:
        return t / 3600.0, r"time $t\,/\,\mathrm{h}$"
    elif t_max > 200:
        return t / 60.0, r"time $t\,/\,\mathrm{min}$"
    return t, r"time $t\,/\,\mathrm{s}$"

def _find_nearest_index(array, value):
    """Returns the index of the closest element in an array to a given value."""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def _fix_pgf_mathdefault(filepath):
    """Removes \mathdefault macros from generated PGF files for LaTeX compatibility."""
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if r'\mathdefault' in content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.replace(r'\mathdefault', ''))

def _save_and_fix(fig, name, folder='./results/figures/', formats=['pgf']):
    """
    Saves the figure in the specified formats.

    Note on PGF format: For heatmaps (imshow/pcolormesh), Matplotlib automatically
    generates supplementary -imgX.png files in the target directory. These are
    required by the PGF file to render properly in LaTeX. Do not delete them.
    """
    os.makedirs(folder, exist_ok=True)

    # Clean the filename to ensure cross-platform compatibility
    clean_name = re.sub(r'[^\w\-]', '_', name.replace(" ", "_"))
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')

    # Option A: PGF (Optimal for LaTeX; generates supplemental PNGs for rasterized elements)
    if 'pgf' in formats:
        pgf_path = os.path.join(folder, f"{clean_name}.pgf")
        try:
            fig.savefig(pgf_path, bbox_inches='tight')
            # Uncomment the line below if _fix_pgf_mathdefault is required in your pipeline
            # _fix_pgf_mathdefault(pgf_path)
        except Exception as e:
            print(f"Warning: PGF save failed: {e}")

    # Option B: PDF (Self-contained vector graphic, standard for LaTeX \includegraphics)
    if 'pdf' in formats:
        pdf_path = os.path.join(folder, f"{clean_name}.pdf")
        fig.savefig(pdf_path, bbox_inches='tight')

    # Option C: PNG (Quick preview, non-vectorized)
    if 'png' in formats:
        png_path = os.path.join(folder, f"{clean_name}_preview.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')

def _add_split_labels(ax, t, split_ratio, orientation='vertical', line_color='white'):
    """
    Overlays 'train' and 'test' domain labels directly onto the axes.
    """
    if split_ratio >= 1.0 or split_ratio <= 0.0:
        return

    split_idx = min(int(len(t) * split_ratio), len(t) - 1)

    # Ensure compatibility with Pandas series if provided
    if hasattr(t, 'iloc'):
        split_val = t.iloc[split_idx]
        t_min, t_max = t.min(), t.max()
    else:
        split_val = t[split_idx]
        t_min, t_max = t.min(), t.max()

    # Define common text properties
    text_props = dict(
        fontweight='bold',
        color=MPI_COLORS['mpi_charcoal'],
        fontsize=10,
    )

    if orientation == 'vertical':
        # --- 1D CASE ---
        ax.axvspan(split_val, t_max, color=MPI_COLORS['mpi_grey'], alpha=0.15, zorder=0, lw=0)
        ax.axvline(split_val, color=line_color, linestyle=':', alpha=0.8, linewidth=1.5)

        trans = ax.get_xaxis_transform()

        # Calculate offset to position text close to the divider line
        x_range = t_max - t_min
        offset = x_range * 0.02

        # Train: Left of the divider
        ax.text(split_val - offset, 0.9, 'train', transform=trans,
                ha='right', va='baseline', **text_props)

        # Test: Right of the divider
        ax.text(split_val + offset, 0.9, 'test', transform=trans,
                ha='left', va='baseline', **text_props)

    else:
        # --- 2D CASE (Horizontal) ---
        ax.axhline(split_val, color=line_color, linestyle='--', linewidth=1.5, alpha=0.8)

        x_min, x_max = ax.get_xlim()

        # Calculate vertical offset
        t_range = t_max - t_min
        offset = t_range * 0.025

        # Test: Later in time (positioned above the line)
        ax.text(x_max * 0.98, split_val + offset, 'test',
                ha='right', va='bottom', **text_props)

        # Train: Earlier in time (positioned below the line)
        ax.text(x_max * 0.98, split_val - offset, 'train',
                ha='right', va='top', **text_props)

###############################################################################
# 3. PLOTTING FUNCTIONS
###############################################################################

# We use @mpl.rc_context to apply configurations strictly locally.
# This prevents unintended global state modifications and crashes.

@mpl.rc_context(PUBLICATION_RC)
def plot_inputs(t, entries_load, entries_Tcool):
    t_plot, t_lbl = _handle_time_units(t)
    fig, ax1 = plt.subplots(figsize=(8, 4))

    c1 = MPI_COLORS['mpi_blue']
    ax1.set_xlabel(t_lbl)
    ax1.set_ylabel("Input Load", color=c1)
    ax1.plot(t_plot, entries_load, color=c1, label="Input Load")
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    c2 = MPI_COLORS['mpi_red']
    ax2.set_ylabel("Cooling Temp / K", color=c2)
    ax2.plot(t_plot, entries_Tcool, color=c2, linestyle='--', label="T_cool")
    ax2.tick_params(axis="y", labelcolor=c2)

    fig.tight_layout()
    # _save_and_fix(fig, "inputs_over_time")
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_3D(z, t, F, name='', function_name='Function f'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(z, t)

    # Time-based colormapping
    norm_t = (t - t.min()) / (t.max() - t.min())
    color_array = np.array([MPI_CMAP(val) for val in norm_t])
    color_array = np.repeat(color_array[:, np.newaxis, :], x.shape[1], axis=1)

    ax.plot_surface(x, y, F.T, facecolors=color_array, linewidth=0, shade=False, rasterized=True)

    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_ylabel(r"time $t\,/\,\mathrm{s}$")
    ax.set_zlabel(function_name)
    ax.view_init(elev=25, azim=-120)

    plt.tight_layout()
    # _save_and_fix(fig, f"3D_{name}")
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_3D_flat(z, t, F, name='', function_name='Function f'):
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(z, t, F.T, cmap=MPI_CMAP, shading='auto', rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(function_name)

    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_ylabel(r"time $t\,/\,\mathrm{s}$")
    fig.tight_layout()
    # _save_and_fix(fig, f"2D_Flat_{name}")
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_entries(time_values, entries_train, entries_test):
    fig, ax = plt.subplots(figsize=(8, 5))

    for j, val in enumerate(entries_train):
        lbl = "Train" if j == 0 else None
        ax.plot(time_values, val, color=MPI_COLORS['mpi_grey'], lw=2, label=lbl, alpha=0.6)

    n_test = len(entries_test)
    colors = [MPI_CMAP(i) for i in np.linspace(0, 1, n_test)]
    for j, val in enumerate(entries_test):
        ax.plot(time_values, val, color=colors[j], lw=2, label=f"Test {j+1}")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='best')
    ax.grid(True, linestyle=":")

    fig.tight_layout()
    # _save_and_fix(fig, "entries_plot")
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_PDE_data(Z, z, t, function_name='Function f'):
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_t = np.linspace(t.min(), t.max(), 6)
    sample_idxs = [_find_nearest_index(t, time) for time in sample_t]
    colors = [MPI_CMAP(i) for i in np.linspace(0.1, 1.0, len(sample_idxs))]

    for i, idx in enumerate(sample_idxs):
        ax.plot(z, Z[:, idx], color=colors[i], lw=2, label=f"t={t[idx]:.0f}s")

    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_ylabel(function_name)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, linestyle=":")
    ax.set_xlim(z.min(), z.max())

    fig.tight_layout()
    # _save_and_fix(fig, f"Snapshot_{function_name}")
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_compare_PDE_data(Z_true, Z_pred, z, t, title, function_name='Function f'):
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_t = np.linspace(t.min(), t.max(), 5)
    sample_idxs = [_find_nearest_index(t, time) for time in sample_t]
    colors = [MPI_CMAP(i) for i in np.linspace(0.1, 1.0, len(sample_idxs))]

    # Dummy legend entry for ground truth styling
    ax.plot([], [], color=MPI_COLORS['mpi_grey'], lw=2, label='True')

    for i, idx in enumerate(sample_idxs):
        ax.plot(z, Z_true[:, idx], color=MPI_COLORS['mpi_grey'], lw=2, alpha=0.5)
        ax.plot(z, Z_pred[:, idx], color=colors[i], ls='--', lw=2, label=f"Pred t={t[idx]:.0f}")

    ax.set_title(title)
    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_ylabel(function_name)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, linestyle=":")

    fig.tight_layout()
    # _save_and_fix(fig, f"Compare_{title}")
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_1D_comparison(t, y_true, y_pred, title, ylabel, train_ratio=1.0, draw_split=True):
    """
    Plots a 1D time-series comparison between ground truth and prediction.
    Train/Test boundaries are safely bounded within the plot area.
    """
    t_plot, t_label = _handle_time_units(t)
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot trajectories
    ax.plot(t_plot, y_true.flatten(), color=MPI_COLORS['mpi_blue'], label="true data", lw=2)
    # Predicted data is dashed to distinguish overlaps
    ax.plot(t_plot, y_pred.flatten(), color=MPI_COLORS['mpi_red'], label="OpInf pred.", lw=2, linestyle="--")

    # Styling
    ax.set_xlim(t_plot.min(), t_plot.max())
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabel)
    # Title intentionally removed per user request, but variable kept for file naming

    # Subtle grid
    ax.grid(True, linestyle=":", alpha=0.5)

    # MaxNLocator prevents y-axis clutter
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Legend dynamically finds the optimal location
    ax.legend(frameon=True, framealpha=0.9, loc='best')

    if draw_split:
        _add_split_labels(ax, t_plot, train_ratio, orientation='vertical')

    fig.tight_layout()

    # --- Standardize filename ---
    clean_title = str(title[0]) if isinstance(title, list) and title else str(title)
    # Truncate string at " vs " to avoid excessively long filenames
    clean_title = clean_title.split(' vs')[0].split('_vs_')[0]
    clean_title = clean_title.strip().replace(' ', '_')

    _save_and_fix(fig, f"1D_{clean_title}")
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_PDE_dynamics_2D(z, t, F, F_pred, filename_suffix="",
                         train_ratio=1.0, draw_split=True, tiny=False,
                         val_label=r'$T \,/\, \mathrm{K}$'):
    """
    Generates an optimized 2D heatmap comparing true dynamics, predictions, and absolute error.
    Colorbar titles are slightly offset to the right to avoid overlapping with ticks.
    """
    # 1. Setup mode-specific aesthetics (standard vs. presentation/tiny scale)
    if tiny:
        fs_title = 22
        fs_label = 22
        fs_ticks = 22
        cbar_bins = 3
        y_locator_bins = 3
        pad_title = 24
    else:
        fs_title = 12
        fs_label = 12
        fs_ticks = 12
        cbar_bins = 5
        y_locator_bins = 4
        pad_title = 12

    # 2. Data preparation
    t_plot = t / 3600.0
    t_label = r'time $t \,/\, \mathrm{h}$'
    z_label = r'reactor length $z \,/\, \mathrm{m}$'

    if F.shape[0] != len(t):
        F, F_pred = F.T, F_pred.T

    vmin = min(F.min(), F_pred.min())
    vmax = max(F.max(), F_pred.max())
    abs_error = np.abs(F - F_pred)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    pcm_opts = dict(cmap=MPI_CMAP, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)

    # 3. Plotting
    # (a) Ground Truth
    ax0 = axes[0]
    ax0.pcolormesh(z, t_plot, F, **pcm_opts)
    ax0.set_title(r'\textbf{(a)} ground truth', fontsize=fs_title, pad=pad_title)
    ax0.set_ylabel(t_label, fontsize=fs_label)
    ax0.set_xlabel(z_label, fontsize=fs_label)
    ax0.tick_params(axis='both', labelsize=fs_ticks)
    if tiny:
        ax0.yaxis.set_major_locator(MaxNLocator(nbins=y_locator_bins))

    # (b) Prediction
    ax1 = axes[1]
    pcm1 = ax1.pcolormesh(z, t_plot, F_pred, **pcm_opts)
    ax1.set_title(r'\textbf{(b)} prediction', fontsize=fs_title, pad=pad_title)
    ax1.set_xlabel(z_label, fontsize=fs_label)
    ax1.tick_params(axis='x', labelsize=fs_ticks)
    ax1.tick_params(left=False)

    # Prediction Colorbar
    cbar1 = fig.colorbar(pcm1, ax=ax1, orientation='vertical', shrink=0.9, pad=0.02)
    cbar1.locator = MaxNLocator(nbins=cbar_bins)
    cbar1.update_ticks()
    cbar1.ax.tick_params(labelsize=fs_ticks)
    # Shift title slightly to the right to clear ticks
    cbar1.ax.set_title(val_label, fontsize=fs_label, pad=10, x=2)

    # (c) Error
    ax2 = axes[2]
    pcm2 = ax2.pcolormesh(z, t_plot, abs_error, cmap='Greys', shading='auto', rasterized=True)

    ax2.set_title(r'\textbf{(c)} abs. error', fontsize=fs_title, pad=pad_title)
    ax2.set_xlabel(z_label, fontsize=fs_label)
    ax2.tick_params(axis='x', labelsize=fs_ticks)
    ax2.tick_params(left=False)

    # Error Colorbar
    cbar2 = fig.colorbar(pcm2, ax=ax2, orientation='vertical', shrink=0.9, pad=0.02)
    cbar2.locator = MaxNLocator(nbins=cbar_bins)
    cbar2.update_ticks()
    cbar2.ax.tick_params(labelsize=fs_ticks)
    cbar2.ax.set_title(val_label, fontsize=fs_label, pad=10, x=2)

    # 4. Domain Split Line
    if draw_split and 0.0 < train_ratio < 1.0:
        for i, ax in enumerate(axes):
            _add_split_labels(ax, t_plot, train_ratio, orientation='horizontal', line_color='black')

    # 5. Filename sanitization
    if isinstance(filename_suffix, list):
        filename_suffix = str(filename_suffix[0]) if filename_suffix else ""

    clean_suffix = str(filename_suffix)
    clean_suffix = clean_suffix.replace('_mathrm', '').replace('mathrm', '')
    clean_suffix = clean_suffix.split('_true_data')[0].split(' - ')[0].split('-')[0].strip('_')

    base_name = f"2D_Dynamics_{clean_suffix}"
    _save_and_fix(fig, base_name)
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_PDE_dynamics_3D(z, t, F, F_pred, title_list, function_name='f', draw_split=True):
    """
    Simplified, robust 3D surface plot comparing true vs. predicted dynamics.
    """
    t_plot = t / 3600.0
    fig = plt.figure(figsize=(10, 3))
    axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]

    x, y = np.meshgrid(z, t_plot)
    data_list = [F, F_pred, F - F_pred]

    for i, ax in enumerate(axes):
        ax.plot_surface(x, y, data_list[i].T, cmap=MPI_CMAP, rasterized=True, linewidth=0)
        ax.set_title(title_list[i])
        ax.set_xlabel("z")
        ax.set_ylabel("t")

    fig.tight_layout()
    # _save_and_fix(fig, f"3D_Dyn_{title_list[0]}")
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_POD_modes(z, basis, min_idx, max_idx, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [MPI_CMAP(i) for i in np.linspace(0.1, 1, max_idx - min_idx)]

    for i, j in enumerate(range(min_idx, max_idx)):
        ax.plot(z, basis[:, j], color=colors[i], lw=2, label=f"Mode {j+1}")

    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, linestyle=":")

    fig.tight_layout()
    # _save_and_fix(fig, f"POD_{title}")
    plt.show()

@mpl.rc_context(PUBLICATION_RC)
def plot_reduced_trajectories(t, reduced_data, min_idx, max_idx, title="Reduced"):
    """
    Plots the reduced-order state trajectories over time.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min_idx, min(max_idx, reduced_data.shape[0])):
        ax.plot(t, reduced_data[i], label=f"Mode {i+1}")

    ax.legend()
    ax.set_title(title)

    fig.tight_layout()
    # _save_and_fix(fig, f"Red_Traj_{title}")
    plt.show()
