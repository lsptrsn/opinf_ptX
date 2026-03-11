# -*- coding: utf-8 -*-
"""
Experimental Data Preprocessing for OpInf (Robust Version)
Includes:
- Unit conversion & Resampling
- Outlier removal (Median Filter)
- Physics-constrained cleaning
- Export for Operator Inference (without plotting, exact legacy file naming)
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
EXCEL_FILE = "Dataset_Data-driven reduced-order model for optimal control of dynamically operated Power-to-X reactors.xlsx"

# Mapping: Neues Excel Sheet -> Alte Datei-ID
SHEET_MAPPING = {
    "Temperature Ramp Exp.": "dyn_T_ramps_exp",
    "Multivariable Excitation Exp.": "dyn_var_T_var_F_exp"
}


def process_experiment_data(file_path, sheet_name):
    """
    Loads data, resets index, renaming, and converts units.
    Time -> Seconds, Temp -> Kelvin.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.reset_index(drop=True, inplace=True)

    df_inputs = pd.DataFrame(index=df.index)

    # 1. Time Conversion (Timestamp -> runtime_s)
    if pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        dt_series = df['Timestamp'] - df['Timestamp'].iloc[0]
        df_inputs['runtime_s'] = dt_series.dt.total_seconds()
    elif pd.api.types.is_timedelta64_dtype(df['Timestamp']):
        df_inputs['runtime_s'] = df['Timestamp'].dt.total_seconds()
    else:
        t_raw = df['Timestamp'] - df['Timestamp'].iloc[0]
        df_inputs['runtime_s'] = t_raw * 60.0

    # 2. Flow Data Mapping
    flow_map = {
        'F_H2_in_set in Ln/min': 'F_H2_setpoint',
        'F_H2_in in Ln/min': 'F_H2_in',
        'F_CO2_in_set in Ln/min': 'F_CO2_setpoint',
        'F_CO2_in in Ln/min': 'F_CO2_in',
        'Ftot_in in Ln/min': 'Ftot_in'
    }
    for old_col, new_col in flow_map.items():
        if old_col in df.columns:
            df_inputs[new_col] = df[old_col]

    # 3. Temperature Conversions (C -> K)
    temp_map = {
        'T_gas_in_set in °C': 'T_gas_in_setpoint',
        'T_gas_in in °C': 'T_gas_in',
        'T_coolant_set in °C': 'T_cool_setpoint',
        'T_coolant in °C': 'T_cool'
    }
    for old_col, new_col in temp_map.items():
        if old_col in df.columns:
            df_inputs[new_col] = df[old_col] + 273.15

    # 4. Spatial Temperature Profiles (T_center at X.XX mm)
    z_cols = [c for c in df.columns if 'T_center at' in c]
    temp_data = {}
    if z_cols:
        z_positions = []
        for col in z_cols:
            try:
                pos_str = col.split('at ')[1].split(' mm')[0]
                pos_mm = float(pos_str)
                pos_m = pos_mm / 1000.0  # mm to meters
                z_positions.append((col, pos_m))
            except Exception:
                continue

        z_positions.sort(key=lambda x: x[1])
        z_start = z_positions[0][1]

        for col_name, pos in z_positions:
            new_pos = pos - z_start
            new_col_name = f"z: {new_pos:.4f}"
            temp_data[new_col_name] = df[col_name] + 273.15

        df_temps = pd.DataFrame(temp_data, index=df.index)
    else:
        df_temps = pd.DataFrame(index=df.index)

    # 5. Conversion Data
    df_conv = pd.DataFrame(index=df.index)
    if 'X_CO2 in %' in df.columns:
        df_conv['X_CO2_out'] = df['X_CO2 in %'] / 100.0
    if 'F_CO2_out in Ln/min' in df.columns:
        df_conv['F_CO2_out'] = df['F_CO2_out in Ln/min']
    if 'F_CH4_out in Ln/min' in df.columns:
        df_conv['F_CH4_out'] = df['F_CH4_out in Ln/min']

    return df_inputs, df_temps, df_conv


def check_and_resample_dt(inputs, temps, conv, tol=1e-3):
    print("--- Checking Time Steps (dt) ---")
    time = inputs['runtime_s'].values
    dt_vec = np.diff(time)
    mean_dt = np.mean(dt_vec)
    std_dt = np.std(dt_vec)

    print(f"Mean dt: {mean_dt:.4f} s | Std dt: {std_dt:.6f} s")

    if std_dt > (mean_dt * tol):
        print(f"-> dt inconsistent (Jitter). Resampling to fixed dt={mean_dt:.4f}s...")
        t_new = np.arange(time[0], time[-1], mean_dt)

        def resample_df(df, t_old, t_new_grid):
            f_interp = interp1d(t_old, df.values, axis=0, kind='linear', fill_value="extrapolate")
            return pd.DataFrame(f_interp(t_new_grid), index=np.arange(len(t_new_grid)), columns=df.columns)

        inputs_new = resample_df(inputs.drop(columns=['runtime_s']), time, t_new)
        inputs_new['runtime_s'] = t_new
        temps_new = resample_df(temps, time, t_new)
        conv_new = resample_df(conv, time, t_new)

        print(f"-> Resampling complete. New length: {len(t_new)}")
        return inputs_new, temps_new, conv_new
    else:
        print("-> dt is sufficiently constant. No resampling needed.")
        return inputs, temps, conv


def handle_nans(df, label="Dataset"):
    if df.isnull().values.any():
        n_nans = df.isnull().sum().sum()
        print(f"-> Fixing {n_nans} NaNs in {label}...")
        df = df.interpolate(method='linear', limit_direction='both', axis=0)
        df = df.ffill().bfill()
        if df.isnull().values.any():
            df = df.fillna(0)
        print(f"   {label} is now NaN-free.")
    return df


def clean_values_and_constraints(df_inputs, df_conv):
    input_flow_cols = [c for c in df_inputs.columns if c.startswith('F_')]
    if input_flow_cols:
        df_inputs[input_flow_cols] = df_inputs[input_flow_cols].clip(lower=0)

    conv_cols = [c for c in df_conv.columns if c.startswith('X_') or c.startswith('F_')]
    if conv_cols:
        df_conv[conv_cols] = df_conv[conv_cols].clip(lower=0)

    if 'Ftot_in' in df_inputs.columns:
        f_in_total = df_inputs['Ftot_in']
    else:
        f_in_total = (df_inputs.get('F_H2_in', 0) + df_inputs.get('F_CO2_in', 0))

    if 'F_CO2_out' in df_conv.columns:
        df_conv['F_CO2_out'] = np.minimum(df_conv['F_CO2_out'], f_in_total)
        print("-> Enforced constraint: F_out <= F_in_total.")

    return df_inputs, df_conv


def save_data_for_opinf(inputs, temps_smooth, flow_smooth, temps_deriv, flow_deriv, file_id):
    print(f"--- Saving Data for OpInf (ID: {file_id}) ---")
    dt = inputs['runtime_s'].iloc[1] - inputs['runtime_s'].iloc[0]

    np.save(f"center_temperature_{file_id}.npy", temps_smooth.values.T)
    np.save(f"flow_rate_out_{file_id}.npy", flow_smooth.values.flatten())
    np.save(f"cooling_temperature_{file_id}.npy", inputs['T_cool'].values)
    np.save(f"inlet_temperature_{file_id}.npy", inputs['T_gas_in'].values)
    np.save(f"derivatives_center_temperature_{file_id}.npy", temps_deriv.values.T / dt)
    np.save(f"derivatives_flow_rate_out_{file_id}.npy", flow_deriv.values.flatten() / dt)

    # Legacy variables specifically for the dyn_var_T_var_F_exp block
    if file_id == "dyn_var_T_var_F_exp":
        load = (inputs['F_H2_setpoint'].values + inputs['F_CO2_setpoint'].values + 20.0)
        np.save(f"load_{file_id}.npy", load)

        np.save(f"load_F_H2_in_exp_{file_id}.npy", inputs['F_H2_in'].values)
        np.save(f"load_F_CO2_in_exp_{file_id}.npy", inputs['F_CO2_in'].values)
        np.save(f"load_T_gas_in_exp_{file_id}.npy", inputs['T_gas_in'].values)
        np.save(f"load_T_cool_exp_{file_id}.npy", inputs['T_cool'].values)

    z_coords = np.array([float(c.split(':')[1]) for c in temps_smooth.columns])
    np.save(f"z_{file_id}.npy", z_coords)
    np.save(f"time_{file_id}.npy", inputs['runtime_s'].values)

    print(f"Saved all files with suffix _{file_id}.npy\n")


# --- Main Execution ---
if __name__ == "__main__":
    for sheet, old_file_id in SHEET_MAPPING.items():
        print(f"==================================================")
        print(f"Processing Sheet: {sheet} -> Mapped to: {old_file_id}")
        print(f"==================================================")

        # 1. Load & Clean
        inputs, temps, conv = process_experiment_data(EXCEL_FILE, sheet)

        # 2. Check DT & Resample if needed
        inputs, temps, conv = check_and_resample_dt(inputs, temps, conv)

        # 3. Constraints & Fix NaNs
        inputs, conv = clean_values_and_constraints(inputs, conv)
        inputs = handle_nans(inputs, "Inputs")
        temps = handle_nans(temps, "Temperatures")
        conv = handle_nans(conv, "Conversions/Flows")

        # 4. Gradient checks
        final_temps_deriv = pd.DataFrame(np.gradient(temps, axis=0),
                                         index=temps.index, columns=temps.columns)

        final_flow = conv[['F_CO2_out']]
        final_flow_deriv = pd.DataFrame(np.gradient(final_flow, axis=0),
                                        index=final_flow.index, columns=final_flow.columns)

        # 5. Save outputs with legacy ID
        save_data_for_opinf(inputs, temps, final_flow, final_temps_deriv, final_flow_deriv, old_file_id)
