#!/usr/bin/env python3
"""
Input Trajectory Visualization for OpInf Methanation Experiment
Description:
    Processes multiple experimental trajectories, merges them into training
    and test sets, and generates a publication-quality 1x4 combined figure.
    The style follows MPI corporate identity (Blue/Red) with high-quality
    LaTeX/PGF rendering.
"""

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# --- 1. LATEX / PGF BACKEND CONFIGURATION ---
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
    'font.size': 12,           # Base font size
    'axes.labelsize': 12,      # Axis labels
    'legend.fontsize': 10,     # Legend font size
    'xtick.labelsize': 12,     # Tick labels
    'ytick.labelsize': 12,
    'axes.linewidth': 1.0,     # Thicker lines for visibility in print
    'lines.linewidth': 1.8,
    'axes.titlesize': 12,
})

# --- 2. CONFIGURATION ---

# Data Selection
exclude = {5, 6, 9, 11}
TEST_CASES_INDICES = [0, 1, 8]
CASES_TO_PROCESS = [x for x in range(0, 14) if x not in exclude]

OUTPUT_SUFFIX = "_training"
INPUT_SUFFIX_BASE = "opinf"
SECONDS_TO_REMOVE = 600.0  # Initial transient to be removed

VARS_TO_STACK = ["time", "load", "cooling_temperature"]
VARS_SINGLE_INSTANCE = ["z"]

# MPI Corporate Colors
MPI_COLORS = {
    'mpi_blue': (51/255, 165/255, 195/255),
    'mpi_red': (120/255, 0/255, 75/255),
    'mpi_charcoal': (60/255, 60/255, 65/255), # Neutral dark grey
}

# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_single_instance_files():
    """Copies static variables (e.g., spatial grid z) from a reference case."""
    if not CASES_TO_PROCESS: return
    ref_case = CASES_TO_PROCESS[0]
    for var_name in VARS_SINGLE_INSTANCE:
        input_file = f"{var_name}_{INPUT_SUFFIX_BASE}_{ref_case}.npy"
        output_file = f"{var_name}{OUTPUT_SUFFIX}.npy"
        if os.path.exists(input_file):
            data = np.load(input_file)
            np.save(output_file, data)

def precompute_start_indices():
    """Finds the index where the time exceeds SECONDS_TO_REMOVE for each case."""
    indices = {}
    for case_num in CASES_TO_PROCESS:
        try:
            time_data = np.load(f"time_{INPUT_SUFFIX_BASE}_{case_num}.npy").squeeze()
            indices[case_num] = np.argmax(time_data >= SECONDS_TO_REMOVE)
        except FileNotFoundError:
            indices[case_num] = None
    return indices

def process_stackable_files(case_start_indices):
    """Stacks time-series data from all cases, removing initial transients."""
    length_report = {var: [] for var in VARS_TO_STACK}
    for var_name in VARS_TO_STACK:
        stack_list = []
        lengths = []
        for case_num in CASES_TO_PROCESS:
            start = case_start_indices.get(case_num)
            length = 0
            if start is not None:
                try:
                    data = np.load(f"{var_name}_{INPUT_SUFFIX_BASE}_{case_num}.npy").squeeze()
                    if data.ndim == 1:
                        chunk = data[start:]
                        length = chunk.shape[0]
                    elif data.ndim == 2:
                        chunk = data[:, start:]
                        length = chunk.shape[1]
                    else:
                        chunk = data
                        length = 1
                    stack_list.append(chunk)
                except FileNotFoundError:
                    pass
            lengths.append(length)
        length_report[var_name] = lengths
        if stack_list:
            combined = np.hstack(stack_list)
            if var_name == "time":
                combined -= combined[0] # Relative time for training
            np.save(f"{var_name}{OUTPUT_SUFFIX}.npy", combined)
    return length_report

# =============================================================================
# PLOTTING HELPERS
# =============================================================================



def _fix_pgf_mathdefault(filepath):
    """Removes mathdefault strings in PGF files for better LaTeX rendering."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if r'\mathdefault' in content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content.replace(r'\mathdefault', ''))

def get_global_limits(load_data, temp_data):
    """Computes fixed axis limits for consistent visualization across plots."""
    l_min, l_max = -1, 22
    t_min, t_max = np.min(temp_data), np.max(temp_data)
    t_pad = (t_max - t_min) * 0.05
    return (l_min, l_max), (t_min - t_pad, t_max + t_pad)

def draw_on_axis(ax_left, t, load, temp, title_str, limits_load, limits_temp, is_training_bundle=False):
    """
    Plots Load and Temperature on a dual-axis system.
    Style: Blue (Load) on left axis, Red (Temp) on right axis. Text remains black.
    """
    ax_right = ax_left.twinx()

    c_load = MPI_COLORS['mpi_blue']
    c_temp = MPI_COLORS['mpi_red']
    c_text = "black"

    # Plotting Logic
    if is_training_bundle:
        # Multiple trajectories in one plot
        for i in range(len(t)):
            t_seg_h = t[i] / 3600.0
            ax_left.plot(t_seg_h, load[i], color=c_load, alpha=0.9, lw=1)
            ax_right.plot(t_seg_h, temp[i], color=c_temp, alpha=0.9, lw=1, linestyle='--')
    else:
        # Single trajectory
        t_plot = t / 3600.0
        ax_left.plot(t_plot, load, color=c_load, lw=2.0)
        ax_right.plot(t_plot, temp, color=c_temp, lw=2.0, linestyle='--')

    # Axis Limits & Ticks
    ax_left.set_ylim(limits_load)
    ax_right.set_ylim(limits_temp)
    ax_left.set_yticks([0, 5, 10, 15, 20])

    # Left Axis Styling (Load -> Blue)
    ax_left.spines['left'].set_color(c_load)
    ax_left.spines['left'].set_linewidth(1.3)
    ax_left.tick_params(axis='y', colors=c_load, labelcolor=c_text, width=1.3)

    # Right Axis Styling (Temp -> Red)
    ax_right.spines['right'].set_color(c_temp)
    ax_right.spines['right'].set_linewidth(1.3)
    ax_right.spines['left'].set_visible(False)
    ax_right.tick_params(axis='y', colors=c_temp, labelcolor=c_text, width=1.3)

    # X-Axis Styling (Neutral)
    ax_left.set_xlabel(r"time $t\,/\,\mathrm{h}$", color=c_text)
    ax_left.tick_params(axis='x', colors=c_text, labelcolor=c_text)
    ax_left.spines['bottom'].set_color(c_text)
    ax_left.spines['top'].set_color(c_text)

    # Grid & Title
    ax_left.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
    ax_left.set_title(title_str, pad=12, color=c_text, fontweight='normal', fontsize=12)

    return ax_right

# =============================================================================
# MAIN PLOTTING FUNCTION
# =============================================================================

def generate_plots_combined(t_full, l_full, c_full, length_report):
    """Generates the final 1x4 layout combining training and test trajectories."""
    print("\n--- Generating Combined 1x4 Plot (Publication Style) ---")

    lim_load, lim_temp = get_global_limits(l_full, c_full)
    lengths = length_report['load']
    current_idx = 0

    training_data = {'t': [], 'l': [], 'c': []}
    test_data_list = []

    # Segment data into Training and Test lists
    for i, length in enumerate(lengths):
        if length == 0: continue
        end = current_idx + length
        ts = t_full[current_idx:end]
        ts_rel = ts - ts[0]
        ls = l_full[current_idx:end]
        cs = c_full[current_idx:end]

        if i in TEST_CASES_INDICES:
            test_data_list.append((ts_rel, ls, cs, CASES_TO_PROCESS[i]))
        else:
            training_data['t'].append(ts_rel)
            training_data['l'].append(ls)
            training_data['c'].append(cs)
        current_idx = end

    # Create Subplots
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharey=False)
    right_axes = []

    # 1. Training Overview (Left-most)
    ax_r_train = draw_on_axis(axes[0], training_data['t'], training_data['l'], training_data['c'],
                              "all training trajectories", lim_load, lim_temp, is_training_bundle=True)
    axes[0].set_xlim(left=0)
    right_axes.append(ax_r_train)

    # 2. Individual Test Cases
    for i, (ts, ls, cs, case_id) in enumerate(test_data_list):
        if i >= 3: break
        title = f"test trajectory {i+1}"
        ax_r = draw_on_axis(axes[i+1], ts, ls, cs, title, lim_load, lim_temp)
        axes[i+1].set_xlim(left=0)
        right_axes.append(ax_r)

    # Label Handling (Color coded to match axis colors)
    axes[0].set_ylabel(r'$F_{\mathrm{in}}\,/\, \mathrm{Ln \, min}^{-1}$', color=MPI_COLORS['mpi_blue'])
    right_axes[3].set_ylabel(r'$T_{\mathrm{cool}} \,/\, \mathrm{K}$', color=MPI_COLORS['mpi_red'])

    # Clean internal ticks (Labels only on outer edges)
    for ax in axes[1:4]:
         plt.setp(ax.get_yticklabels(), visible=False)
    for ax_r in right_axes[0:3]:
        plt.setp(ax_r.get_yticklabels(), visible=False)

    # Legend Construction
    h_load = Line2D([0], [0], color=MPI_COLORS['mpi_blue'], lw=2, label='input load')
    h_temp = Line2D([0], [0], color=MPI_COLORS['mpi_red'], lw=2, linestyle='--', label='cooling temp.')
    axes[3].legend(handles=[h_load, h_temp], loc='lower left', frameon=True, framealpha=0.9)

    # Final Layout Tuning
    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.16, top=0.88, wspace=0.12)

    # Save to Disk
    save_dir = "./results/figures/combined/"
    os.makedirs(save_dir, exist_ok=True)
    filename = "Input_Traj_Final"
    pgf_path = os.path.join(save_dir, f"{filename}.pgf")

    plt.savefig(pgf_path, bbox_inches='tight')
    if '_fix_pgf_mathdefault' in globals():
        _fix_pgf_mathdefault(pgf_path)

    plt.savefig(os.path.join(save_dir, f"{filename}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=150, bbox_inches='tight')
    print(f"  -> Saved {filename} to {save_dir}")
    plt.show()

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    process_single_instance_files()
    indices = precompute_start_indices()
    report = process_stackable_files(indices)

    try:
        t_data = np.load(f"time{OUTPUT_SUFFIX}.npy")
        l_data = np.load(f"load{OUTPUT_SUFFIX}.npy")
        c_data = np.load(f"cooling_temperature{OUTPUT_SUFFIX}.npy")

        generate_plots_combined(t_data, l_data, c_data, report)

    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback; traceback.print_exc()

    print("\n--- Preprocessing and Plotting Complete ---")
