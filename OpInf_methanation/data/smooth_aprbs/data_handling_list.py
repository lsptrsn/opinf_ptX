#!/usr/bin/env python3
"""
Data preprocessing script.
Functionality:
1. Filters cases based on 'exclude' list.
2. Cuts initial transients (SECONDS_TO_REMOVE).
3. Stacks data horizontally for training.
4. PLOTTING: Generates individual 3D surface plots for EACH case and EACH matrix variable.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# --- Configuration ---
exclude = {5, 6, 9, 11}
CASES_TO_PROCESS = [x for x in range(0, 14) if x not in exclude]

OUTPUT_SUFFIX = "_training"
INPUT_SUFFIX_BASE = "opinf"
SECONDS_TO_REMOVE = 600.0

VARS_SINGLE_INSTANCE = ["z"]

VARS_TO_STACK = [
    "time",
    "center_temperature",
    "cooling_temperature",
    "load",
    "conversion"
]

# Variables usually not plotted as 3D surfaces (e.g. 1D vectors)
VARS_TO_SKIP_PLOTTING = ["time"]

# --- End of Configuration ---

print(f"!!! CONFIGURATION CHECK !!!")
print(f"Processing Cases: {CASES_TO_PROCESS}")
print("---------------------------------------------------\n")


def process_single_instance_files():
    """
    Loads static files (z coordinate).
    Uses the first available case as reference.
    """
    if not CASES_TO_PROCESS:
        return

    ref_case = CASES_TO_PROCESS[0]
    print(f"--- Processing Static Files (Ref: Case {ref_case}) ---")

    for var_name in VARS_SINGLE_INSTANCE:
        input_file = f"{var_name}_{INPUT_SUFFIX_BASE}_{ref_case}.npy"
        output_file = f"{var_name}{OUTPUT_SUFFIX}.npy"

        try:
            data = np.load(input_file)
            np.save(output_file, data)
            print(f"  Saved '{output_file}'")
        except Exception as e:
            print(f"  Error loading {input_file}: {e}")
    print("---------------------------------------------------\n")


def precompute_start_indices():
    """
    Calculates the array index where time >= SECONDS_TO_REMOVE for each case.
    """
    print(f"--- Calculating Start Indices ---")
    case_start_indices = {}

    for case_num in CASES_TO_PROCESS:
        time_file = f"time_{INPUT_SUFFIX_BASE}_{case_num}.npy"
        try:
            time_data = np.load(time_file).squeeze()
            start_index = np.argmax(time_data >= SECONDS_TO_REMOVE)

            # Safety check if threshold is never reached
            if start_index == 0 and time_data[0] < SECONDS_TO_REMOVE:
                 if not np.any(time_data >= SECONDS_TO_REMOVE):
                    start_index = 0 # Fallback

            case_start_indices[case_num] = start_index
        except Exception:
            case_start_indices[case_num] = 0

    return case_start_indices


def process_stackable_files(case_start_indices):
    """
    Stacks files horizontally to create the big training set.
    """
    print(f"--- Stacking Data ---")
    length_report = {var_name: [] for var_name in VARS_TO_STACK}

    for var_name in VARS_TO_STACK:
        data_list = []

        for case_num in CASES_TO_PROCESS:
            start_idx = case_start_indices.get(case_num, 0)
            input_file = f"{var_name}_{INPUT_SUFFIX_BASE}_{case_num}.npy"

            try:
                data = np.load(input_file).squeeze()
                import pdb; pdb.set_trace()

                # Slice data based on dimensions
                if data.ndim == 1:
                    cut_data = data[start_idx:]
                    length_report[var_name].append(cut_data.shape[0])
                elif data.ndim == 2:
                    cut_data = data[:, start_idx:]
                    length_report[var_name].append(cut_data.shape[1])
                else:
                    cut_data = data
                    length_report[var_name].append(1)

                data_list.append(cut_data)
            except:
                length_report[var_name].append(0)

        if data_list:
            combined = np.hstack(data_list)
            # Normalize time if it is the time variable
            if var_name == "time":
                combined = combined - combined[0]

            np.save(f"{var_name}{OUTPUT_SUFFIX}.npy", combined)
            print(f"  Saved combined {var_name}")

    return length_report


def plot_individual_3d_cases(case_start_indices):
    """
    Generates a 3D plot for EACH case in CASES_TO_PROCESS.
    Only plots 2D matrices (Space x Time).
    """
    print("\n--- Generating Individual 3D Plots per Case ---")

    # Load spatial coordinate z
    try:
        z = np.load(f"z{OUTPUT_SUFFIX}.npy")
    except FileNotFoundError:
        print("  Error: z-coordinate file not found. Skipping 3D plots.")
        return

    # Iterate through every allowed case
    for case_num in CASES_TO_PROCESS:
        start_idx = case_start_indices.get(case_num, 0)

        # Load time for this specific case
        time_file = f"time_{INPUT_SUFFIX_BASE}_{case_num}.npy"
        if not os.path.exists(time_file):
            continue

        t_raw = np.load(time_file).squeeze()
        # Cut time vector according to threshold
        t_vec = t_raw[start_idx:]

        # Shift time to start at 0 for better visualization (optional)
        t_vec = t_vec - t_vec[0]

        # Meshgrid (Time x Space)
        # Note: Meshgrid must match data dimensions (Rows=Z, Cols=Time)
        T_mesh, Z_mesh = np.meshgrid(t_vec, z)

        print(f"\nProcessing Case {case_num}...")

        # Check every variable for this case
        for var_name in VARS_TO_STACK:
            if var_name in VARS_TO_SKIP_PLOTTING:
                continue

            file_name = f"{var_name}_{INPUT_SUFFIX_BASE}_{case_num}.npy"
            if not os.path.exists(file_name):
                continue

            data_raw = np.load(file_name).squeeze()

            # We only want 3D plots for Matrices (2D arrays)
            if data_raw.ndim != 2:
                continue

            # Cut data (Spatial x Time)
            data_cut = data_raw[:, start_idx:]

            # Dimension Check
            if data_cut.shape != T_mesh.shape:
                print(f"  Skipping {var_name}: Shape mismatch {data_cut.shape} vs mesh {T_mesh.shape}")
                continue

            # Plotting
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            surf = ax.plot_surface(T_mesh, Z_mesh, data_cut, cmap='coolwarm',
                                   edgecolor='none', alpha=0.9)

            ax.set_title(f"Case {case_num}: {var_name}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Position z [m]")
            ax.set_zlabel("Value")

            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            plt.tight_layout()
            plt.show() # Shows plot immediately in Spyder

if __name__ == "__main__":
    process_single_instance_files()
    start_indices = precompute_start_indices()
    process_stackable_files(start_indices)

    # Trigger the per-case 3D plotting
    plot_individual_3d_cases(start_indices)

    print("\nDone.")
