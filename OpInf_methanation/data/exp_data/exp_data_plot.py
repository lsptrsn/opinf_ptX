# -*- coding: utf-8 -*-
"""
Experimental Data Preprocessing & Visualization for OpInf
Description:
    Handles raw experimental data including unit conversion, resampling,
    and physics-constrained cleaning. Generates publication-quality
    plots using the PGF backend for seamless LaTeX integration.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

# --- 1. LATEX / PGF BACKEND CONFIGURATION ---
# This configuration ensures that the plots match the document's font and style.
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\usepackage{siunitx}",
    ]),
    'font.size': 12,           # Main font size
    'axes.labelsize': 12,      # Axis labels
    'legend.fontsize': 10,     # Legend font size
    'xtick.labelsize': 12,     # X-axis tick labels
    'ytick.labelsize': 12,     # Y-axis tick labels
    'axes.linewidth': 1.0,     # Slightly thicker axes for visibility
    'lines.linewidth': 1.8,
})

# --- 2. CORPORATE DESIGN SETUP ---
MPI_COLORS = {
    'mpi_blue': (51/255, 165/255, 195/255),
    'mpi_red': (120/255, 0/255, 75/255),
    'mpi_green': (0/255, 118/255, 117/255),
    'mpi_grey': (135/255, 135/255, 141/255),
    'mpi_beige': (236/255, 233/255, 212/255),
    'mpi_charcoal': (60/255, 60/255, 65/255),
}

# --- CONFIGURATION CONSTANTS ---
SETTLING_TIME_S = 180.0

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def process_experiment_data(file_path):
    """
    Loads experimental data from parquet, resets index, renames columns,
    and performs unit conversions (Celsius to Kelvin, minutes to seconds).
    """
    df = pd.read_parquet(file_path)
    df.reset_index(drop=True, inplace=True)

    # Mapping raw experimental keys (German) to internal variables (English)
    input_map = {
        'Laufzeit in min': 'runtime_min',
        'F_H2_in_Soll': 'F_H2_setpoint',
        'F_H2_in': 'F_H2_in',
        'F_CO2_in_Soll': 'F_CO2_setpoint',
        'F_CO2_in': 'F_CO2_in',
        'F_Ar_in': 'F_Ar_in',
        'Ftot_in': 'Ftot_in'
    }

    cols_to_fetch = [c for c in input_map.keys() if c in df.columns]
    extra_cols = ['T_gas_einlass_Soll', 'T_oil_Mantel_Soll',
                  'T_gas_einlass_1', 'T_oil_Mantel']

    df_inputs = df[cols_to_fetch + extra_cols].copy()
    df_inputs.rename(columns=input_map, inplace=True)

    # Time Conversion: min -> s
    if 'runtime_min' in df_inputs.columns:
        t_min = df_inputs['runtime_min'] - df_inputs['runtime_min'].iloc[0]
        df_inputs['runtime_s'] = t_min * 60.0
        df_inputs.drop(columns=['runtime_min'], inplace=True)

    # Temperature Conversions: °C -> K
    df_inputs['T_gas_in_setpoint'] = df_inputs['T_gas_einlass_Soll'] + 273.15
    df_inputs['T_gas_in'] = df_inputs['T_gas_einlass_1'] + 273.15
    df_inputs['T_cool_setpoint'] = df_inputs['T_oil_Mantel_Soll'] + 273.15
    df_inputs['T_cool'] = df_inputs['T_oil_Mantel'] + 273.15
    df_inputs.drop(columns=extra_cols, inplace=True)

    # Extract Spatial Temperature Profiles (z-positions)
    z_cols = [c for c in df.columns if c.startswith('z_in:')]
    temp_data = {}
    if z_cols:
        z_positions = []
        for col in z_cols:
            try:
                pos = float(col.split(':')[1])
                z_positions.append((col, pos))
            except ValueError:
                continue
        z_positions.sort(key=lambda x: x[1])
        z_start = z_positions[0][1]

        for col_name, pos in z_positions:
            new_pos = pos - z_start
            new_col_name = f"z: {new_pos:.4f}"
            temp_data[new_col_name] = df[col_name] + 273.15
        df_temps = pd.DataFrame(temp_data, index=df.index)
    else:
        df_temps = pd.DataFrame(temp_data, index=df.index)

    # Conversion/Flow Rate Output Data
    conv_map = {
        'Umsatz_CO2_MS': 'X_CO2_out_MS',
        'Umsatz_CO2_FTC': 'X_CO2_out_FTC',
        'F_CO2_MS_out': 'F_CO2_out'
    }
    available_conv_cols = [c for c in conv_map.keys() if c in df.columns]
    df_conv = df[available_conv_cols].copy()
    df_conv.rename(columns=conv_map, inplace=True)

    # Normalize conversion percentages to [0, 1]
    for col in ['X_CO2_out_MS', 'X_CO2_out_FTC']:
        if col in df_conv.columns:
            df_conv[col] = df_conv[col] / 100.0

    return df_inputs, df_temps, df_conv


def handle_nans(df):
    """Aggressively handles missing values via linear interpolation and fill methods."""
    if df.isnull().values.any():
        df = df.interpolate(method='linear', limit_direction='both', axis=0)
        df = df.ffill().bfill()
        if df.isnull().values.any():
            df = df.fillna(0)
    return df


def ensure_min_ylim_span(ax, min_span=0.2):
    """Adjusts the y-axis limits to ensure a minimum display span."""
    ymin, ymax = ax.get_ylim()
    current_span = ymax - ymin
    if current_span < min_span:
        center = (ymax + ymin) / 2
        ax.set_ylim(center - min_span / 2, center + min_span / 2)


def fix_pgf_mathdefault(filepath):
    """
    Removes the '\mathdefault' command from PGF files which can cause
    rendering issues in some LaTeX environments.
    """
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if r'\mathdefault' in content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.replace(r'\mathdefault', ''))

# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def _add_split_labels(ax, t, split_ratio, orientation='vertical', line_color='white', col_idx=0):
    """
    Adds visual labels for Train/Test split markers directly onto the plot.
    """
    if split_ratio >= 1.0 or split_ratio <= 0.0:
        return

    split_idx = min(int(len(t) * split_ratio), len(t) - 1)
    split_val = t.iloc[split_idx] if hasattr(t, 'iloc') else t[split_idx]
    t_max, t_min = t.max(), t.min()

    text_props = dict(color=MPI_COLORS['mpi_charcoal'])

    if orientation == 'vertical':
        # Highlight test region with a subtle background
        ax.axvspan(split_val, t_max, color=MPI_COLORS['mpi_grey'], alpha=0.15, zorder=0, lw=0)

        # Vertical separator line
        ax.axvline(split_val, color=line_color, linestyle=':', alpha=0.8, linewidth=1.5)

        trans = ax.get_xaxis_transform()

        # Calculate positions for the labels
        train_pos = t_min + (split_val - t_min) * 0.91
        test_pos = ((split_val + t_max) / 2) * 1.03

        # Add labels only for specific columns to avoid clutter
        if col_idx == 1:
            ax.text(train_pos, 0.92, 'train', transform=trans, ha='right', va='top', **text_props)
            ax.text(test_pos, 0.92, 'test', transform=trans, ha='center', va='top', **text_props)


def plot_combined_experiments_row_based(df1, name1, df2, name2, split_ratio=0.7):
    """
    Creates a 2x4 layout plot comparing two experiments.
    Rows: (a) Experiment 1, (b) Experiment 2.
    Columns: H2 flow, CO2 flow, gas temperature, cooling temperature.
    """
    fig, axs = plt.subplots(2, 4, figsize=(10, 5.0), sharex=False)

    cols_config = [
        ('F_H2_in', 'F_H2_setpoint', r'$F_{\mathrm{H}_2, \mathrm{in}} \,/\, \mathrm{Ln \, min}^{-1}$'),
        ('F_CO2_in', 'F_CO2_setpoint', r'$F_{\mathrm{CO}_2, \mathrm{in}} \,/\, \mathrm{Ln \, min}^{-1}$'),
        ('T_gas_in', 'T_gas_in_setpoint', r'$T_{\mathrm{gas, in}} \,/\, \mathrm{K}$'),
        ('T_cool', 'T_cool_setpoint', r'$T_{\mathrm{cool}} \,/\, \mathrm{K}$')
    ]

    t1 = df1['runtime_s'] / 3600.0
    t2 = df2['runtime_s'] / 3600.0

    data_setup = [
        (0, t1, df1, MPI_COLORS['mpi_blue'], name1, r'\textbf{(a)}'),
        (1, t2, df2, MPI_COLORS['mpi_green'], name2, r'\textbf{(b)}')
    ]

    for row_idx, t_data, df_data, color, exp_name, label_letter in data_setup:
        for col_idx, (act, setp, ylabel) in enumerate(cols_config):
            ax = axs[row_idx, col_idx]

            # Plot actual sensor data and setpoints
            ax.plot(t_data, df_data[act], color=color, linestyle='-', lw=1.2, label='sensor data')
            ax.plot(t_data, df_data[setp], color='black', linestyle='--', alpha=0.8, lw=1.2, label='setpoint')

            # Axis formatting
            ax.set_ylabel(ylabel)
            if row_idx == 1:
                ax.set_xlabel(r"time $t\,/\,\mathrm{h}$")
            ax.set_xlim(left=0, right=t_data.max())

            # Annotate Train/Test split
            _add_split_labels(ax, t_data, split_ratio, orientation='vertical', line_color='grey', col_idx=col_idx)

            # Refined aesthetics
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)

            if 'F_' in act:
                ensure_min_ylim_span(ax, min_span=0.2)

            # Legends and Subplot Titles
            if col_idx == 0:
                ax.legend(frameon=True, framealpha=0.9, loc='upper right', handlelength=1.0)
                # Label letters (a)/(b) and experiment descriptions
                ax.text(-0.35, 1.05, label_letter, transform=ax.transAxes, fontweight='bold', va='bottom')
                ax.text(0.0, 1.05, f"{exp_name}", transform=ax.transAxes, va='bottom')

    # Global layout adjustments
    plt.subplots_adjust(
        left=0.08, right=0.98, bottom=0.12, top=0.88,
        wspace=0.45, hspace=0.30
    )

    # Export paths
    save_dir = "./results/figures/"
    os.makedirs(save_dir, exist_ok=True)
    pgf_path = os.path.join(save_dir, "Combined_RowBased_Labeled.pgf")
    png_path = os.path.join(save_dir, "Combined_RowBased_Labeled.png")

    print(f"Saving PGF to {pgf_path}...")
    plt.savefig(pgf_path)
    fix_pgf_mathdefault(pgf_path)

    print(f"Saving Preview PNG to {png_path}...")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    file_ramps = "dyn_T_ramps_exp.parquet"
    file_steps = "dyn_var_T_var_F_exp.parquet"

    print("--- Starting Processing Pipeline ---")

    # 1. Load and Clean Dataset 1 (Ramps)
    print(f"Processing: {file_ramps}")
    inputs_ramps, temps_ramps, conv_ramps = process_experiment_data(file_ramps)
    inputs_ramps = handle_nans(inputs_ramps)

    # 2. Load and Clean Dataset 2 (Steps)
    print(f"Processing: {file_steps}")
    inputs_steps, temps_steps, conv_steps = process_experiment_data(file_steps)
    inputs_steps = handle_nans(inputs_steps)

    # 3. Visualization
    print("Generating comparison plot...")
    plot_combined_experiments_row_based(
        inputs_ramps, "temperature ramps",
        inputs_steps, "load steps and temperature ramps"
    )

    print("Pipeline complete. Output saved to ./results/figures/")
