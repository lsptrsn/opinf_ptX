import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from typing import Optional

# --- Konfiguration ---

MPI_COLORS = {
    'mpi_blue': (51/255, 165/255, 195/255),
    'mpi_red': (120/255, 0/255, 75/255),
    'mpi_green': (0/255, 118/255, 117/255),
    'mpi_grey': (135/255, 135/255, 141/255),
    'mpi_beige': (236/255, 233/255, 212/255),
    'mpi_charcoal': (60/255, 60/255, 65/255),
}

PUBLICATION_RC = {
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": False,
    "pgf.preamble": r"\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc} \usepackage{amsmath} \usepackage{siunitx}",
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "lines.linewidth": 2,
    "lines.markersize": 4
}

# Hilfsfunktion für einheitliches Styling
def _style_axis(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=10)

    ax.grid(True, linestyle=":", alpha=0.6, color=MPI_COLORS['mpi_grey'], zorder=0)

    # Weniger Ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# --- Plotting Functions ---

@mpl.rc_context(PUBLICATION_RC)
def plot_control_trajectory(
    time_arr: np.ndarray,
    U_opt: np.ndarray,
    u_min: float,
    u_max: float,
    save_path: Optional[Path] = None,
):
    """Plot optimal control trajectory."""
    # Zeitumrechnung
    t_plot = time_arr / 3600.0
    t_control = t_plot[:-1]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # Control Signal
    ax.plot(t_control, U_opt, '-o', color=MPI_COLORS['mpi_blue'],
            label='optimal control', zorder=3)

    # Bounds
    ax.axhline(u_min, color=MPI_COLORS['mpi_charcoal'], linestyle='--', alpha=0.6,
               label=r'$u_{\mathrm{min}}$')
    ax.axhline(u_max, color=MPI_COLORS['mpi_charcoal'], linestyle='--', alpha=0.6,
               label=r'$u_{\mathrm{max}}$')

    _style_axis(ax, r'time $t \,/\, \mathrm{h}$', r'coolant temperature $T_c \,/\, \mathrm{K}$',
                r'Optimal Coolant Temperature')

    ax.legend(loc='best', frameon=True, framealpha=0.9)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.show()


@mpl.rc_context(PUBLICATION_RC)
def plot_temperature_profile(
    time_arr: np.ndarray,
    T_hot_time: np.ndarray,
    T_hot_max: float,
    slack: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
):
    """Plot maximum temperature profile over time."""
    # Zeitumrechnung
    t_plot = time_arr / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # Temperature Curve
    ax.plot(t_plot, T_hot_time, '-o', color=MPI_COLORS['mpi_red'],
            label='max. temperature', zorder=3)

    # Constraint
    ax.axhline(T_hot_max, color=MPI_COLORS['mpi_charcoal'], linestyle='--', linewidth=1.5,
               label=r'constraint')

    # Highlight violations
    violations = T_hot_time > T_hot_max + 0.1
    if np.any(violations):
        ax.scatter(t_plot[violations], T_hot_time[violations],
                   color='black', s=40, zorder=5, marker='x',
                   label='violation')

    _style_axis(ax, r'time $t \,/\, \mathrm{h}$', r'max. temperature $T_{\mathrm{max}} \,/\, \mathrm{K}$',
                r'Hotspot Temperature Profile')

    ax.legend(loc='best', frameon=True, framealpha=0.9)

    # Slack info box
    if slack is not None:
        max_slack = np.max(slack)
        if max_slack > 1e-6:
            ax.text(0.02, 0.95, f'max slack: {max_slack:.2e}',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=MPI_COLORS['mpi_beige'], alpha=0.8))

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.show()


@mpl.rc_context(PUBLICATION_RC)
def plot_conversion_profile(
    time_arr: np.ndarray,
    conv_outlet: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Plot outlet conversion over time."""
    # Zeitumrechnung
    t_plot = time_arr / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    ax.plot(t_plot, conv_outlet, '-o', color=MPI_COLORS['mpi_green'],
            label='outlet conversion', zorder=3)

    mean_conv = np.mean(conv_outlet)
    ax.axhline(mean_conv, color=MPI_COLORS['mpi_green'], linestyle='--', linewidth=1.5,
               alpha=0.6, label=f'mean = {mean_conv:.3f}')

    _style_axis(ax, r'time $t \,/\, \mathrm{h}$', r'conversion $X \,/\, -$',
                r'Outlet Conversion')

    ax.legend(loc='best', frameon=True, framealpha=0.9)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.show()


@mpl.rc_context(PUBLICATION_RC)
def plot_disturbance_profile(
    time_arr: np.ndarray,
    load_profile: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Plot disturbance (load) profile."""
    # Zeitumrechnung
    t_plot = time_arr / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    ax.plot(t_plot, load_profile, '-o', color=MPI_COLORS['mpi_charcoal'],
            label='load profile', zorder=3)

    _style_axis(ax, r'time $t \,/\, \mathrm{h}$', r'inlet load $V_{\mathrm{in}} \,/\, (\mathrm{Nl}\,\mathrm{min}^{-1})$',
                r'Disturbance Profile')

    # Legend ist hier oft redundant, aber der Konsistenz halber:
    # ax.legend(loc='best')

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.show()


@mpl.rc_context(PUBLICATION_RC)
def create_summary_plot(
    time_arr: np.ndarray,
    U_opt: np.ndarray,
    T_hot_time: np.ndarray,
    conv_outlet: np.ndarray,
    load_profile: np.ndarray,
    T_hot_max: float,
    u_min: float,
    u_max: float,
    save_path: Optional[Path] = None,
    # NEUE ARGUMENTE
    T_hot_linear: Optional[np.ndarray] = None,
    conv_linear: Optional[np.ndarray] = None
):
    """
    Create comprehensive 4-panel summary plot.
    """
    FONT_SIZE = 12
    t_plot = time_arr / 3600.0
    t_control = t_plot[:-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True, sharex=True)
    fig.get_layout_engine().set(h_pad=0.15, w_pad=0.1)

    # 1. Control (Blue)
    ax0 = axes[0, 0]
    ax0.plot(t_control, U_opt, '-o', color=MPI_COLORS['mpi_blue'], markersize=3, label='control input')
    ax0.set_ylim(bottom=510)
    _style_axis(ax0, '', r'$T_c \,/\, \mathrm{K}$', r'Optimal control')

    # 2. Temperature (Red)
    ax1 = axes[0, 1]

    # Linear Prediction (Model) - Dashed
    if T_hot_linear is not None:
        ax1.plot(t_plot, T_hot_linear, '--', color='black', linewidth=1.5, alpha=0.6, label='OpInf (Model)')
        ax1.fill_between(t_plot, T_hot_time, T_hot_linear, color='gray', alpha=0.15) # Mismatch Shading

    # Reality (CNN) - Solid
    ax1.plot(t_plot, T_hot_time, '-o', color=MPI_COLORS['mpi_red'], markersize=3, label='CNN (Reality)')
    ax1.set_ylim(bottom=650)
    ax1.axhline(T_hot_max, color=MPI_COLORS['mpi_charcoal'], linestyle=':', linewidth=1.5, label='Constraint')
    ax1.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
    _style_axis(ax1, '', r'$T_{\mathrm{max}} \,/\, \mathrm{K}$', r'Hotspot temperature')

    # 3. Conversion (Green)
    ax2 = axes[1, 0]

    # Linear Prediction (Model) - Dashed
    if conv_linear is not None:
        ax2.plot(t_plot, conv_linear, '--', color='black', linewidth=1.5, alpha=0.6, label='OpInf (Model)')

    # Reality (CNN) - Solid
    ax2.plot(t_plot, conv_outlet, '-o', color=MPI_COLORS['mpi_green'], markersize=3, label='CNN (Reality)')
    ax2.set_ylim(bottom=0.7)

    mean_conv = np.mean(conv_outlet)
    ax2.axhline(mean_conv, color=MPI_COLORS['mpi_green'], linestyle=':', alpha=0.6, label=f'Mean: {mean_conv:.3f}')
    ax2.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
    _style_axis(ax2, r'time $t \,/\, \mathrm{h}$', r'$X \,/\, -$', r'Conversion')

    # 4. Load (Charcoal)
    ax3 = axes[1, 1]
    ax3.plot(t_plot, load_profile, '-o', color=MPI_COLORS['mpi_charcoal'], markersize=3, label='load profile')
    _style_axis(ax3, r'time $t \,/\, \mathrm{h}$', r'Load', r'Disturbance')

    for ax in axes.flatten():
        ax.title.set_fontsize(FONT_SIZE)
        ax.xaxis.label.set_size(FONT_SIZE)
        ax.yaxis.label.set_size(FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved summary plot: {save_path}")
    plt.show()
