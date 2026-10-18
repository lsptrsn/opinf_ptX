# -*- coding: utf-8 -*-
"""
OpInf Visualization Library (Dissertation Version)
------------------------------------------------------
Features:
- MPI Corporate Design integration
- Strict PGF/LaTeX configuration for publication-ready outputs
- Advanced GridSpec layouts for flawless alignment
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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

# ==========================================
# 1. CORPORATE DESIGN & STYLE SETUP
# ==========================================

MPI_COLORS = {
    'mpi_blue':     (51/255, 165/255, 195/255),
    'mpi_red':      (120/255, 0/255, 75/255),
    'mpi_green':    (0/255, 118/255, 117/255),
    'mpi_grey':     (135/255, 135/255, 141/255),
    'mpi_charcoal': (60/255, 60/255, 65/255),
    'mpi_beige':    (236/255, 233/255, 212/255),
    'black':        (0, 0, 0),
    'white':        (1, 1, 1)
}

_colors = [MPI_COLORS['mpi_blue'], MPI_COLORS['mpi_beige'], MPI_COLORS['mpi_red']]
MPI_CMAP = LinearSegmentedColormap.from_list("mpi_diverging", _colors, N=256)

USE_PGF = True
DOC_WIDTH_INCH = 5.8

def setup_style():
    """Applies the publication style settings globally."""
    if USE_PGF:
        mpl.use("pgf")
        mpl.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc} \usepackage{amsmath} \usepackage{lmodern} \usepackage{sfmath} \usepackage{siunitx}",
        })

    mpl.rcParams.update({
        'font.family': 'serif', # Konsistent auf Serif umgestellt
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'grid.linewidth': 0.5,
        'lines.markersize': 3.5,
        'figure.figsize': (DOC_WIDTH_INCH, DOC_WIDTH_INCH * 1.4),
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15
    })

setup_style()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def _handle_time_units(t):
    """Automatically scales time values and returns appropriate LaTeX units."""
    t_max = np.max(t)
    if t_max > 12000:
        return t / 3600.0, r"time $t\,/\,\mathrm{h}$"
    elif t_max > 200:
        return t / 60.0, r"time $t\,/\,\mathrm{min}$"
    return t, r"time $t\,/\,\mathrm{s}$"

def _find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def _save_and_fix(fig, name, folder='./results/figures/', formats=['pdf', 'pgf']):
    os.makedirs(folder, exist_ok=True)
    clean_name = re.sub(r'[^\w\-]', '_', name.replace(" ", "_"))
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    for fmt in formats:
        try:
            filepath = os.path.join(folder, f"{clean_name}.{fmt}")
            fig.savefig(filepath, dpi=300)
        except Exception as e:
            print(f"Warning: Failed to save {fmt} format. Error: {e}")

def _add_split_labels(ax, t, split_ratio, orientation='vertical', line_color='black'):
    if split_ratio >= 1.0 or split_ratio <= 0.0:
        return
    split_idx = min(int(len(t) * split_ratio), len(t) - 1)
    split_val = t.iloc[split_idx] if hasattr(t, 'iloc') else t[split_idx]
    t_min, t_max = t.min(), t.max()
    text_props = dict(fontweight='normal', color=MPI_COLORS['mpi_charcoal'], fontsize=9)

    if orientation == 'vertical':
        ax.axvline(split_val, color=line_color, linestyle=':', alpha=0.8, linewidth=1.5)
        trans = ax.get_xaxis_transform()
        offset = (t_max - t_min) * 0.02
        ax.text(split_val - offset, 0.9, 'train', transform=trans, ha='right', va='baseline', **text_props)
        ax.text(split_val + offset, 0.9, 'test', transform=trans, ha='left', va='baseline', **text_props)
    else:
        ax.axhline(split_val, color=line_color, linestyle='--', linewidth=1.5, alpha=0.8)
        x_min, x_max = ax.get_xlim()
        offset = (t_max - t_min) * 0.025
        ax.text(x_max * 0.98, split_val + offset, 'test', ha='right', va='bottom', **text_props)
        ax.text(x_max * 0.98, split_val - offset, 'train', ha='right', va='top', **text_props)

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_1D_comparison(t, y_true, y_pred,
                       title="outlet CO$_2$ flow rate (reference vs. OpInf prediction)",
                       ylabel=r"flow rate / Ln$\cdot$min$^{-1}$",
                       train_ratio=1.0, draw_split=True,
                       label_true='reference', label_pred='OpInf'):
    """1D time-series comparison utilizing the dissertation style."""
    t_plot, t_label = _handle_time_units(t)
    fig, ax = plt.subplots(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.35))

    ax.plot(t_plot, y_true.flatten(), color=MPI_COLORS['mpi_grey'], label=label_true, lw=1.5, alpha=0.9)
    ax.plot(t_plot, y_pred.flatten(), color=MPI_COLORS['mpi_red'], label=label_pred, lw=1.2, linestyle="--")

    ax.set_xlim(t_plot.min(), t_plot.max())
    ax.set_xlabel(t_label)
    ax.set_ylabel(ylabel)
    ax.set_title(f"(a) {title}", loc='left', pad=8)

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.legend(frameon=True, framealpha=1.0, facecolor='white', edgecolor='black',
              loc='upper left', bbox_to_anchor=(0.08, 0.95))

    if draw_split:
        _add_split_labels(ax, t_plot, train_ratio, orientation='vertical')

    _save_and_fix(fig, "04_1D_model_prediction_exp1")
    if not USE_PGF: plt.show()


def plot_PDE_dynamics_2D(z, t, F, F_pred, filename_suffix="", train_ratio=1.0, draw_split=True, val_label=r'$T \,/\, \mathrm{K}$', tiny=False):
    """2D heatmap comparison with space-saving titles and optimized width ratios."""
    t_plot, t_label = _handle_time_units(t)
    if F.shape[0] != len(t):
        F, F_pred = F.T, F_pred.T

    vmin = min(F.min(), F_pred.min())
    vmax = max(F.max(), F_pred.max())
    abs_error = np.abs(F - F_pred)

    # --- Nomenclature routing ---
    if "Temperature" in filename_suffix:
        cbar_symbol = r'$T \,/\, \mathrm{K}$'
        title_ref = "reference"
        title_pred = "OpInf" if tiny else "OpInf prediction"
        save_name = "05_2D_Dynamics_Temperature_syn_traj1" if tiny else f"04_2D_Dynamics_Temperature_exp1"
    else:
        # Fallback to conversion metrics
        cbar_symbol = r'$X \,/\, -$'
        title_ref = "reference"
        title_pred = "OpInf" if tiny else "OpInf prediction"
        save_name = "05_2D_Dynamics_Conv_syn_traj1" if tiny else f"04_2D_Dynamics_Conv_{filename_suffix}_exp1"

    fig_width = DOC_WIDTH_INCH
    fig_height = DOC_WIDTH_INCH * 0.4
    title_pad = 10
    z_label = r'reactor length $z \,/\, \mathrm{m}$'

    # --- Tiny mode configuration ---
    if tiny:
        # Scale typography for minimal plot dimensions
        rc_updates = {
            'font.size': 14,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
        }
        x_ticks_ax0 = [0.0, 1.0]
        x_ticks_all = [0.0, 1.0, 2.0]
    else:
        rc_updates = {}
        x_ticks_ax0 = [0.0, 0.5, 1.0, 1.5]
        x_ticks_all = [0.0, 0.5, 1.0, 1.5, 2.0]

    # Standard width ratios for the gridspec
    wr = [1, 0.08, 1, 0.04, 0.05, 0.35, 1, 0.04, 0.05]

    # Apply local font scaling
    with plt.rc_context(rc_updates):
        fig = plt.figure(figsize=(fig_width, fig_height))
        # Remove horizontal whitespace to maximize plot width
        gs = fig.add_gridspec(1, 9, width_ratios=wr, wspace=0.0)
        pcm_opts = dict(cmap=MPI_CMAP, vmin=vmin, vmax=vmax, shading='auto', rasterized=True)

        # --- (b) Reference ---
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.pcolormesh(z, t_plot, F, **pcm_opts)
        ax0.set_title(f'(b) {title_ref}', loc='left', pad=title_pad)
        ax0.set_ylabel(t_label)
        # Hide x-axis label in tiny mode to prevent clutter
        ax0.set_xlabel(z_label if not tiny else "")
        ax0.yaxis.set_major_locator(MaxNLocator(nbins=4))
        # Exclude rightmost tick to prevent overlap with the adjacent plot
        ax0.set_xticks(x_ticks_ax0)
        ax0.set_xlim(0, 2.0)

        # --- (c) Prediction ---
        ax1 = fig.add_subplot(gs[0, 2], sharey=ax0)
        pcm1 = ax1.pcolormesh(z, t_plot, F_pred, **pcm_opts)
        ax1.set_title(f'(c) {title_pred}', loc='left', pad=title_pad)
        ax1.set_xlabel(z_label)
        ax1.set_xticks(x_ticks_all)
        ax1.set_xlim(0, 2.0)
        plt.setp(ax1.get_yticklabels(), visible=False)

        cax1 = fig.add_subplot(gs[0, 4])
        cbar1 = fig.colorbar(pcm1, cax=cax1, orientation='vertical')
        cbar1.locator = MaxNLocator(nbins=3)
        cbar1.update_ticks()
        cax1.text(0.5, 1.03, cbar_symbol, transform=cax1.transAxes, ha='center', va='bottom')

        # --- (d) Absolute Error ---
        ax2 = fig.add_subplot(gs[0, 6], sharey=ax0)
        pcm2 = ax2.pcolormesh(z, t_plot, abs_error, cmap='Greys', shading='auto', rasterized=True)
        ax2.set_title(r'(d) abs. error', loc='left', pad=title_pad)
        ax2.set_xlabel(z_label if not tiny else "")
        ax2.set_xticks(x_ticks_all)
        ax2.set_xlim(0, 2.0)
        plt.setp(ax2.get_yticklabels(), visible=False)

        cax2 = fig.add_subplot(gs[0, 8])
        cbar2 = fig.colorbar(pcm2, cax=cax2, orientation='vertical')
        cbar2.locator = MaxNLocator(nbins=3)
        cbar2.update_ticks()

        delta_label = cbar_symbol.replace('$', r'$\Delta ', 1)
        cax2.text(0.5, 1.03, delta_label, transform=cax2.transAxes, ha='center', va='bottom')

        if draw_split and 0.0 < train_ratio < 1.0:
            for ax in [ax0, ax1, ax2]:
                _add_split_labels(ax, t_plot, train_ratio, orientation='horizontal', line_color='black')

        # Enforce zero-padding
        fig.tight_layout(pad=0.0, w_pad=0.0)

        _save_and_fix(fig, save_name)
        if not USE_PGF: plt.show()

# ==========================================
# 4. REMAINING PLOTS (Adapted to Style)
# ==========================================

def plot_inputs(t, entries_load, entries_Tcool):
    t_plot, t_lbl = _handle_time_units(t)
    fig, ax1 = plt.subplots(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.4))

    ax1.set_xlabel(t_lbl)
    ax1.set_ylabel("Input Load", color=MPI_COLORS['mpi_blue'])
    ax1.plot(t_plot, entries_load, color=MPI_COLORS['mpi_blue'], label="Input Load", lw=1.5)
    ax1.tick_params(axis="y", labelcolor=MPI_COLORS['mpi_blue'])
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Cooling Temp / K", color=MPI_COLORS['mpi_red'])
    ax2.plot(t_plot, entries_Tcool, color=MPI_COLORS['mpi_red'], linestyle='--', label="T_cool", lw=1.2)
    ax2.tick_params(axis="y", labelcolor=MPI_COLORS['mpi_red'])

    _save_and_fix(fig, "inputs_over_time")
    if not USE_PGF: plt.show()


def plot_3D(z, t, F, name='', function_name='Function f'):
    fig = plt.figure(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.8))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(z, t)

    norm_t = (t - t.min()) / (t.max() - t.min())
    color_array = np.array([MPI_CMAP(val) for val in norm_t])
    color_array = np.repeat(color_array[:, np.newaxis, :], x.shape[1], axis=1)

    ax.plot_surface(x, y, F.T, facecolors=color_array, linewidth=0, shade=False, rasterized=True)
    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_ylabel(r"time $t\,/\,\mathrm{s}$")
    ax.set_zlabel(function_name)
    ax.view_init(elev=25, azim=-120)

    _save_and_fix(fig, f"3D_{name}")
    if not USE_PGF: plt.show()


def plot_3D_flat(z, t, F, name='', function_name='Function f'):
    fig, ax = plt.subplots(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.5))
    pcm = ax.pcolormesh(z, t, F.T, cmap=MPI_CMAP, shading='auto', rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(function_name)

    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_ylabel(r"time $t\,/\,\mathrm{s}$")

    _save_and_fix(fig, f"2D_Flat_{name}")
    if not USE_PGF: plt.show()


def plot_entries(time_values, entries_train, entries_test):
    fig, ax = plt.subplots(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.4))

    for j, val in enumerate(entries_train):
        lbl = "Train" if j == 0 else None
        ax.plot(time_values, val, color=MPI_COLORS['mpi_grey'], lw=1.5, label=lbl, alpha=0.6)

    colors = [MPI_CMAP(i) for i in np.linspace(0, 1, len(entries_test))]
    for j, val in enumerate(entries_test):
        ax.plot(time_values, val, color=colors[j], lw=1.2, label=f"Test {j+1}")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='best')
    ax.grid(True, linestyle=":")

    _save_and_fix(fig, "entries_plot")
    if not USE_PGF: plt.show()


def plot_PDE_data(Z, z, t, function_name='Function f'):
    fig, ax = plt.subplots(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.4))
    sample_t = np.linspace(t.min(), t.max(), 6)
    sample_idxs = [_find_nearest_index(t, time) for time in sample_t]
    colors = [MPI_CMAP(i) for i in np.linspace(0.1, 1.0, len(sample_idxs))]

    for i, idx in enumerate(sample_idxs):
        ax.plot(z, Z[:, idx], color=colors[i], lw=1.5, label=f"t={t[idx]:.0f}s")

    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_ylabel(function_name)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, linestyle=":")
    ax.set_xlim(z.min(), z.max())

    _save_and_fix(fig, f"Snapshot_{function_name}")
    if not USE_PGF: plt.show()


def plot_compare_PDE_data(Z_true, Z_pred, z, t, title, function_name='Function f'):
    fig, ax = plt.subplots(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.4))
    sample_t = np.linspace(t.min(), t.max(), 5)
    sample_idxs = [_find_nearest_index(t, time) for time in sample_t]
    colors = [MPI_CMAP(i) for i in np.linspace(0.1, 1.0, len(sample_idxs))]

    ax.plot([], [], color=MPI_COLORS['mpi_grey'], lw=1.5, label='True')

    for i, idx in enumerate(sample_idxs):
        ax.plot(z, Z_true[:, idx], color=MPI_COLORS['mpi_grey'], lw=1.5, alpha=0.5)
        ax.plot(z, Z_pred[:, idx], color=colors[i], ls='--', lw=1.2, label=f"Pred t={t[idx]:.0f}")

    ax.set_title(title, loc='left', pad=10)
    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_ylabel(function_name)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, linestyle=":")

    _save_and_fix(fig, f"Compare_{title}")
    if not USE_PGF: plt.show()


def plot_PDE_dynamics_3D(z, t, F, F_pred, title_list, function_name='f', draw_split=True):
    t_plot, _ = _handle_time_units(t)
    fig = plt.figure(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.35))
    axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]

    x, y = np.meshgrid(z, t_plot)
    data_list = [F, F_pred, F - F_pred]

    for i, ax in enumerate(axes):
        ax.plot_surface(x, y, data_list[i].T, cmap=MPI_CMAP, rasterized=True, linewidth=0)
        ax.set_title(title_list[i], loc='left', pad=10)
        ax.set_xlabel("z")
        ax.set_ylabel("t")

    _save_and_fix(fig, f"3D_Dyn_{title_list[0]}")
    if not USE_PGF: plt.show()


def plot_POD_modes(z, basis, min_idx, max_idx, title):
    fig, ax = plt.subplots(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.4))
    colors = [MPI_CMAP(i) for i in np.linspace(0.1, 1, max_idx - min_idx)]

    for i, j in enumerate(range(min_idx, max_idx)):
        ax.plot(z, basis[:, j], color=colors[i], lw=1.2, label=f"Mode {j+1}")

    ax.set_xlabel(r"reactor length $z\,/\,\mathrm{m}$")
    ax.set_title(title, loc='left', pad=10)
    ax.legend(loc='best')
    ax.grid(True, linestyle=":")

    _save_and_fix(fig, f"POD_{title}")
    if not USE_PGF: plt.show()


def plot_reduced_trajectories(t, reduced_data, min_idx, max_idx, title="Reduced"):
    t_plot, t_label = _handle_time_units(t)
    fig, ax = plt.subplots(figsize=(DOC_WIDTH_INCH, DOC_WIDTH_INCH * 0.4))

    for i in range(min_idx, min(max_idx, reduced_data.shape[0])):
        ax.plot(t_plot, reduced_data[i], lw=1.2, label=f"Mode {i+1}")

    ax.legend(loc='best')
    ax.set_title(title, loc='left', pad=10)
    ax.set_xlabel(t_label)
    ax.grid(True, linestyle=":", alpha=0.5)

    _save_and_fix(fig, f"Red_Traj_{title}")
    if not USE_PGF: plt.show()
