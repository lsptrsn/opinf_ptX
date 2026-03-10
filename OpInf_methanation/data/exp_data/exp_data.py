# -*- coding: utf-8 -*-
"""
Experimental Data Preprocessing for OpInf (Robust Version)
Includes:
- Unit conversion & Resampling
- Outlier removal (Median Filter)
- Automatic Savitzky-Golay parameter tuning
- Physics-constrained cleaning
- Export for Operator Inference
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURATION ---
filename = "dyn_var_T_var_F_exp.parquet"
# filename = "dyn_T_ramps_exp.parquet"
SETTLING_TIME_S = 180.0  # 3 min * 60


def process_experiment_data(file_path):
    """
    Loads data, resets index, renaming, and converts units.
    Time -> Seconds, Temp -> Kelvin.
    """
    df = pd.read_parquet(file_path)
    df.reset_index(drop=True, inplace=True)

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
        # Normalize start to 0 and convert to seconds
        t_min = df_inputs['runtime_min'] - df_inputs['runtime_min'].iloc[0]
        df_inputs['runtime_s'] = t_min * 60.0
        df_inputs.drop(columns=['runtime_min'], inplace=True)

    # Temperature Conversions (C -> K)
    df_inputs['T_gas_in_setpoint'] = df_inputs['T_gas_einlass_Soll'] + 273.15
    df_inputs['T_gas_in'] = df_inputs['T_gas_einlass_1'] + 273.15
    df_inputs['T_cool_setpoint'] = df_inputs['T_oil_Mantel_Soll'] + 273.15
    df_inputs['T_cool'] = df_inputs['T_oil_Mantel'] + 273.15
    df_inputs.drop(columns=extra_cols, inplace=True)

    # Spatial Temperature Profiles
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

    # Conversion Data
    conv_map = {
        'Umsatz_CO2_MS': 'X_CO2_out_MS',
        'Umsatz_CO2_FTC': 'X_CO2_out_FTC',
        'F_CO2_MS_out': 'F_CO2_out'
    }
    available_conv_cols = [c for c in conv_map.keys() if c in df.columns]
    df_conv = df[available_conv_cols].copy()
    df_conv.rename(columns=conv_map, inplace=True)

    if 'X_CO2_out_MS' in df_conv.columns:
        df_conv['X_CO2_out_MS'] = df_conv['X_CO2_out_MS'] / 100
    if 'X_CO2_out_FTC' in df_conv.columns:
        df_conv['X_CO2_out_FTC'] = df_conv['X_CO2_out_FTC'] / 100

    return df_inputs, df_temps, df_conv


def check_and_resample_dt(inputs, temps, conv, tol=1e-3):
    """
    Checks if time steps are constant. If variance is high, resamples data
    to a fixed grid (Constant dt).
    """
    print("\n--- Checking Time Steps (dt) ---")
    time = inputs['runtime_s'].values
    dt_vec = np.diff(time)
    mean_dt = np.mean(dt_vec)
    std_dt = np.std(dt_vec)

    print(f"Mean dt: {mean_dt:.4f} s | Std dt: {std_dt:.6f} s")

    # If standard deviation of dt is significant (>0.1% of mean), resample
    if std_dt > (mean_dt * tol):
        print(f"-> dt inconsistent (Jitter). Resampling to fixed dt={mean_dt:.4f}s...")

        t_new = np.arange(time[0], time[-1], mean_dt)

        def resample_df(df, t_old, t_new_grid):
            f_interp = interp1d(t_old, df.values, axis=0, kind='linear',
                                fill_value="extrapolate")
            return pd.DataFrame(f_interp(t_new_grid),
                                index=np.arange(len(t_new_grid)),
                                columns=df.columns)

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
    """Aggressively handles NaNs: Interpolate -> FFill -> BFill -> Fill 0."""
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
    """
    1. Clips negative values to 0.
    2. Enforces physical constraint: F_out <= F_in_total.
    """
    # 1. Negative Values
    input_flow_cols = [c for c in df_inputs.columns if c.startswith('F_')]
    if input_flow_cols:
        df_inputs[input_flow_cols] = df_inputs[input_flow_cols].clip(lower=0)

    conv_cols = [c for c in df_conv.columns if c.startswith('X_') or c.startswith('F_')]
    if conv_cols:
        df_conv[conv_cols] = df_conv[conv_cols].clip(lower=0)

    # 2. Mass Balance Constraint: F_out <= F_in_tot
    if 'Ftot_in' in df_inputs.columns:
        f_in_total = df_inputs['Ftot_in']
    else:
        f_in_total = (df_inputs.get('F_H2_in', 0) +
                      df_inputs.get('F_CO2_in', 0) +
                      df_inputs.get('F_Ar_in', 0))

    if 'F_CO2_out' in df_conv.columns:
        df_conv['F_CO2_out'] = np.minimum(df_conv['F_CO2_out'], f_in_total)
        print("-> Enforced constraint: F_out <= F_in_total.")

    print("-> Negative values in Flows and Conversions set to 0.")
    return df_inputs, df_conv

def analyze_noise(df, actual_col, setpoint_col, settling_time_s, label):
    """Calculates noise stats for stable plateaus."""
    df['block_id'] = (df[setpoint_col].diff() != 0).cumsum()
    stats_list = []
    valid_indices = []
    grouped = df.groupby('block_id')

    for _, group in grouped:
        t_start = group['runtime_s'].iloc[0]
        t_end = group['runtime_s'].iloc[-1]
        if (t_end - t_start) > settling_time_s:
            stable = group[group['runtime_s'] >= (t_start + settling_time_s)]
            if not stable.empty:
                mean_val = stable[actual_col].mean()
                std_val = stable[actual_col].std()
                noise_pct = ((std_val / mean_val) * 100
                             if abs(mean_val) > 1e-6 else 0)
                stats_list.append({'Std_Dev': std_val, 'Noise_Pct': noise_pct})
                valid_indices.extend(stable.index.tolist())

    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        avg_noise = stats_df['Noise_Pct'].mean()
        avg_std = stats_df['Std_Dev'].mean()
        print(f"[{label}] Avg Noise: {avg_noise:.3f}% | Avg Std: {avg_std:.4f}")
    else:
        avg_noise, avg_std = 0, 0
    return df.loc[valid_indices], avg_std, avg_noise


def train_continuous_baseline(inputs, temps, temps_deriv, flow, flow_deriv):
    """Tests CONTINUOUS time dynamics with EXTENDED STATE."""
    print("\n--- Training Continuous Baseline (Extended State: T + Flow) ---")

    dt = inputs['runtime_s'].iloc[1] - inputs['runtime_s'].iloc[0]

    # 1. Build State Vector X = [T, F_out]
    X_state = np.hstack([temps.values, flow.values])

    # 2. Build Control Vector U
    control_cols = ['T_gas_in', 'T_cool', 'F_H2_in', 'F_CO2_in']
    X_control = inputs[control_cols].values
    X_train = np.hstack([X_state, X_control])

    # 3. Build Target Vector dX/dt (Scale by 1/dt to get physical units)
    Y_temps = temps_deriv.values / dt
    Y_flow = flow_deriv.values / dt
    Y_train = np.hstack([Y_temps, Y_flow])

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, Y_train)
    r2_lin = lin_reg.score(X_train, Y_train)
    print(f"Linear Model (Ax + Bu -> dx/dt) R²: {r2_lin:.5f}")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=10, max_depth=10, n_jobs=-1,
                               random_state=42)
    rf.fit(X_train, Y_train)
    r2_rf = rf.score(X_train, Y_train)
    print(f"Random Forest (Non-linear)       R²: {r2_rf:.5f}")

    if r2_lin > 0.8:
        print(">> CONCLUSION: Linearity looks good.")
    elif r2_rf > r2_lin + 0.2:
        print(">> CONCLUSION: Non-linear effects dominant.")
    else:
        print(">> CONCLUSION: Derivative prediction is hard (Noise/Disturbances).")


def plot_phase_space(smoothed, deriv, dt, z_indices=[10, 50, 100]):
    """Plots dT/dt vs T."""
    fig, ax = plt.subplots(figsize=(8, 6))
    deriv_scaled = deriv / dt
    for idx in z_indices:
        if idx < len(smoothed.columns):
            col_name = smoothed.columns[idx]
            ax.plot(smoothed[col_name], deriv_scaled[col_name], lw=0.8,
                    alpha=0.7, label=f'Pos {col_name}')
    ax.set_xlabel('T [K]')
    ax.set_ylabel('dT/dt [K/s]')
    ax.set_title('Phase Space (State vs Derivative)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_results(inputs, temps, conv, filename, title_prefix=""):
    t = inputs['runtime_s']
    clean_title = filename.replace('.parquet', '')

    # --- PLOT 1: Conversion & Outlet Flow ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    if 'X_CO2_out_MS' in conv.columns:
        ax1.plot(t, conv['X_CO2_out_MS'], label='X_CO2 (MS)', color='tab:blue')
    if 'X_CO2_out_FTC' in conv.columns:
        ax1.plot(t, conv['X_CO2_out_FTC'], label='X_CO2 (FTC)',
                 color='tab:cyan', linestyle='--')
    ax1.set_xlabel('Runtime [s]')
    ax1.set_ylabel('Conversion [-]')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if 'F_CO2_out' in conv.columns:
        ax2.plot(t, conv['F_CO2_out'], label='F_CO2_out', color='tab:red')
    ax2.set_ylabel('Outlet Flow [Nl/min]', color='tab:red')

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right')
    plt.title(f'{title_prefix}Conversion and Flow - {clean_title}')
    plt.tight_layout()

    # --- PLOT 2: Inputs & Noise ---
    fig2, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    vars_to_analyze = [
        ('F_H2_in', 'F_H2_setpoint', axs[0, 0], 'H2 Flow'),
        ('F_CO2_in', 'F_CO2_setpoint', axs[0, 1], 'CO2 Flow'),
        ('T_gas_in', 'T_gas_in_setpoint', axs[1, 0], 'Gas Inlet T'),
        ('T_cool', 'T_cool_setpoint', axs[1, 1], 'Coolant T')
    ]
    for act_col, set_col, ax, title in vars_to_analyze:
        ax.plot(t, inputs[set_col], 'k--', alpha=0.6, label='Setpoint')
        ax.plot(t, inputs[act_col], color='tab:blue', alpha=0.6, label='Actual')

        stable_data, avg_std, avg_pct = analyze_noise(
            inputs, act_col, set_col, SETTLING_TIME_S, title)
        if not stable_data.empty:
            ax.scatter(stable_data['runtime_s'], stable_data[act_col],
                       color='tab:green', s=1, alpha=0.5)
            textstr = f'Noise: {avg_pct:.3f}%\n$\sigma$: {avg_std:.3f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.03, 0.93, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axs[1, 0].set_xlabel('Runtime [s]')
    axs[1, 1].set_xlabel('Runtime [s]')
    plt.suptitle(f'{title_prefix}Inputs & Noise Analysis - {clean_title}')
    plt.tight_layout()

    # --- PLOT 3: 3D Surface ---
    z_cols = [c for c in temps.columns if c.startswith('z:')]
    if z_cols:
        z_vals = [float(c.split(':')[1]) for c in z_cols]
        T_mat = temps[z_cols].values
        X_grid, Y_grid = np.meshgrid(z_vals, t)

        fig4 = plt.figure(figsize=(12, 8))
        ax4 = fig4.add_subplot(111, projection='3d')
        skip = max(1, len(t) // 150) # Downsample for plotting speed
        ax4.plot_surface(X_grid[::skip], Y_grid[::skip], T_mat[::skip],
                         cmap='inferno', edgecolor='none', alpha=0.9)
        ax4.set_xlabel('z [m]')
        ax4.set_ylabel('Time [s]')
        ax4.set_zlabel('T [K]')
        ax4.view_init(elev=30, azim=-120)
        plt.title(f'{title_prefix}3D Temperature - {clean_title}')
        plt.tight_layout()

    plt.show()


def save_data_for_opinf(inputs, temps_smooth, flow_smooth,
                        temps_deriv, flow_deriv, filename):
    print("\n--- Saving Data for OpInf ---")
    file_id = filename.replace('.parquet', '')
    dt = inputs['runtime_s'].iloc[1] - inputs['runtime_s'].iloc[0]

    np.save(f"center_temperature_{file_id}.npy", temps_smooth.values.T)
    np.save(f"flow_rate_out_{file_id}.npy", flow_smooth.values.flatten())
    np.save(f"cooling_temperature_{file_id}.npy", inputs['T_cool'].values)
    np.save(f"inlet_temperature_{file_id}.npy", inputs['T_gas_in'].values)
    np.save(f"derivatives_center_temperature_{file_id}.npy", temps_deriv.values.T / dt)
    np.save(f"derivatives_flow_rate_out_{file_id}.npy", flow_deriv.values.flatten() / dt)

    print(f"Saved derivatives (scaled by 1/dt = {1/dt:.2f} Hz)")

    load = (inputs['F_H2_setpoint'].values + inputs['F_CO2_setpoint'].values +
            20.0)

    if file_id == "dyn_var_T_var_F_exp":
        np.save(f"load_{file_id}.npy", load)
        np.save(f"load_F_H2_in_exp_{file_id}.npy", inputs['F_H2_in'].values)
        np.save(f"load_F_CO2_in_exp_{file_id}.npy", inputs['F_Ar_in'].values)
        np.save(f"load_F_Ar_in_exp_{file_id}.npy", inputs['F_CO2_in'].values)
        np.save(f"load_T_gas_in_exp_{file_id}.npy", inputs['T_gas_in'].values)
        np.save(f"load_T_cool_exp_{file_id}.npy", inputs['T_cool'].values)

    z_coords = np.array([float(c.split(':')[1]) for c in temps_smooth.columns])
    np.save(f"z_{file_id}.npy", z_coords)

    np.save(f"time_{file_id}.npy", inputs['runtime_s'].values)
    print(f"Saved all files with suffix _{file_id}.npy")


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Processing: {filename}")

    # 1. Load & Clean
    inputs, temps, conv = process_experiment_data(filename)

    # 2. Check DT & Resample if needed
    inputs, temps, conv = check_and_resample_dt(inputs, temps, conv)

    # 3. Constraints & Fix NaNs
    inputs, conv = clean_values_and_constraints(inputs, conv)
    inputs = handle_nans(inputs, "Inputs")
    temps = handle_nans(temps, "Temperatures")
    conv = handle_nans(conv, "Conversions/Flows")

    # 4. gradient checks
    final_temps = temps
    # Simple gradient (d/dn)
    final_temps_deriv = pd.DataFrame(np.gradient(final_temps, axis=0),
                                     index=temps.index, columns=temps.columns)
    flow_df = conv[['F_CO2_out']]
    final_flow = flow_df
    final_flow_deriv = pd.DataFrame(np.gradient(final_flow, axis=0),
                                    index=flow_df.index, columns=flow_df.columns)
    title_prefix = "[Raw] "

    # 5. Baseline Check (Linearity R2)
    # train_continuous_baseline(inputs, final_temps, final_temps_deriv,
    #                           final_flow, final_flow_deriv)

    # 6. Visuals & Save
    conv_plotting = conv.copy()
    conv_plotting['F_CO2_out'] = final_flow['F_CO2_out']
    dt = inputs['runtime_s'].iloc[1] - inputs['runtime_s'].iloc[0]

    indices_to_plot = [0, len(final_temps.columns)//2, len(final_temps.columns)-1]
    plot_phase_space(final_temps, final_temps_deriv, dt,
                     z_indices=indices_to_plot)

    plot_results(inputs, final_temps, conv_plotting, filename, title_prefix)
    save_data_for_opinf(inputs, final_temps, final_flow, final_temps_deriv, final_flow_deriv, filename)
