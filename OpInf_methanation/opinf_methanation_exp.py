"""
OpInf Methanation Experiment - Main Script
"""

# --- Standard libraries ---
import sys
import time
import warnings

# --- Third-party libraries ---
import numpy as np
import matplotlib as mpl

# --- Local libraries ---
import opinf

# Suppress warnings
warnings.filterwarnings("ignore")


# Matplotlib PGF settings for publication-quality plots
mpl.rcParams['text.usetex'] = False
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
    ])
})


# =============================================================================
# SETUP
# =============================================================================
device = opinf.utils.get_device()


# =============================================================================
# HELPER FUNCTIONS (GLOBAL)
# =============================================================================

def simulate_and_reconstruct(
    y0_reduced, t, t_span, entries, model_ops, recon_data, Params, seed=None
):
    """
    Integrate ROM and reconstruct full-state solution.
    Handles Flow (F) and Temperature (T) using Linear/NL-POD reconstruction.
    """
    # 1. Scale initial condition
    y0 = y0_reduced.flatten()
    if Params.reduced_scaling:
        y0_list = []
        current_idx = 0

        def scale_and_append(rank, key):
            nonlocal current_idx
            if rank > 0:
                _, y_scaled = opinf.utils.scaled_states(
                    y0_reduced[current_idx: current_idx + rank],
                    scaling_fac=recon_data[key]
                )
                y0_list.append(y_scaled)
                current_idx += rank

        scale_and_append(recon_data['r_F'], 'scaling_fac_F')
        scale_and_append(recon_data['r_T'], 'scaling_fac_T')
        scale_and_append(recon_data['r_w1'], 'scaling_fac_w1')
        scale_and_append(recon_data['r_w2'], 'scaling_fac_w2')
        y0 = np.concatenate([arr for arr in y0_list if arr.size > 0])

    # 2. Integrate ROM
    try:
        if Params.lag_max == 0:
            sol_reduced = opinf.models.integrate(
                t_span, y0, t, entries,
                model_ops['A'], model_ops['B'], model_ops['C'],
                model_ops['H'], model_ops['E'], model_ops['F']
            )
        else:
            dt_recon = recon_data.get('dt', t[1] - t[0])
            n_lags = len(model_ops['E']) if model_ops['E'] else 0
            delays = [(l + 1) * dt_recon for l in range(n_lags)]
            sol_reduced = opinf.models.integrate_dde(
                t_span, y0, t, entries,
                model_ops['A'], model_ops['B'], model_ops['C'],
                model_ops['H'], model_ops['E'], model_ops['F'], delays
            )
    except Exception as e:
        print(f"  Integration failed: {e}")
        return np.zeros((y0.shape[0], len(t))), np.zeros((y0.shape[0], len(t)))

    # Ensure 2D shape
    if sol_reduced.ndim == 1:
        if sol_reduced.shape[0] == t.shape[0]:
            sol_reduced = sol_reduced.reshape(1, -1)
        else:
            sol_reduced = sol_reduced.reshape(-1, 1)

    # Padding if necessary
    if sol_reduced.shape[-1] != t.shape[0]:
        if sol_reduced.shape[-1] < t.shape[0]:
            padding = np.tile(
                sol_reduced[:, -1:],
                (1, t.shape[0] - sol_reduced.shape[-1])
            )
            sol_reduced = np.hstack((sol_reduced, padding))
        else:
            sol_reduced = sol_reduced[:, :t.shape[0]]

    # 3. Unscale reduced solution
    sol_reduced_unscaled = sol_reduced.copy()
    if Params.reduced_scaling:
        current_idx = 0

        def unscale_block(rank, key):
            nonlocal current_idx
            if rank > 0:
                sol_reduced_unscaled[current_idx: current_idx + rank, :] = \
                    opinf.utils.unscaled_states(
                        recon_data[key],
                        sol_reduced[current_idx: current_idx + rank, :]
                    )
                current_idx += rank

        unscale_block(recon_data['r_F'], 'scaling_fac_F')
        unscale_block(recon_data['r_T'], 'scaling_fac_T')
        unscale_block(recon_data['r_w1'], 'scaling_fac_w1')
        unscale_block(recon_data['r_w2'], 'scaling_fac_w2')

    # 4. Reconstruct full state
    V_active = recon_data['V_reduced'][:, :sol_reduced_unscaled.shape[0]]
    sol = V_active @ sol_reduced_unscaled

    # 5. Unscale physical state
    n_F = recon_data['state_scaling_params']['n_F']
    n_T = recon_data['state_scaling_params']['n_T']

    if sol.ndim == 1:
        sol = sol.reshape(-1, 1)

    sol = sol[:n_F + n_T]
    F_sol = sol[:n_F]
    T_sol = sol[n_F: n_F + n_T]

    if Params.state_scaling == 'min-max':
        F_from = recon_data['state_scaling_params']['F_scaled_from']
        T_from = recon_data['state_scaling_params']['T_scaled_from']
        F_sol = opinf.pre.apply_minmax(F_sol, F_from, inverse=True)
        T_sol = opinf.pre.apply_minmax(T_sol, T_from, inverse=True)
    elif Params.state_scaling == 'simple':
        F_sol = F_sol * recon_data['state_scaling_params']['max_F']
        T_sol = T_sol * recon_data['state_scaling_params']['max_T']

    sol = np.vstack((F_sol, T_sol))

    # Add shift back
    ref_states_used = recon_data['reference_states'][:n_F + n_T]
    sol = opinf.pre.unshift(sol, ref_states_used)

    # 6. Optional NL-POD correction
    V_reduced_nonlin = recon_data.get('V_reduced_nonlin', None)
    Xi = recon_data.get('Xi', None)

    if Params.basis in ['NL-POD', 'AM']:
        res = opinf.post.reconstruct_solution(
            sol=sol, sol_reduced=sol_reduced_unscaled,
            X_train_true=recon_data['F_train'],
            T_train_true=recon_data['T_train'],
            V_reduced=recon_data['V_reduced'],
            V_reduced_nonlin=V_reduced_nonlin, Xi=Xi, seed=seed
        )
        sol = res[0] if isinstance(res, tuple) else res

    return sol, sol_reduced_unscaled


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(Params, seed=None):
    """
    Main function of the OpInf methanization experiment.
    """
    if seed is None:
        seed = 0
    opinf.utils.set_deterministic(seed)

    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    if Params.output:
        print(f"Loading data from: {Params.data_dir} "
              f"with suffix: {Params.file_suffix}")

    try:
        z_all = np.load(f"{Params.data_dir}/z{Params.file_suffix}.npy")
        t_full = np.load(f"{Params.data_dir}/time{Params.file_suffix}.npy")
        try:
            F_all_full = np.load(
                f"{Params.data_dir}/flow_rate_out{Params.file_suffix}.npy"
            )
        except Exception:
            F_all_full = np.load(
                f"{Params.data_dir}/conversion{Params.file_suffix}.npy"
            )
        T_all_full = np.load(
            f"{Params.data_dir}/center_temperature{Params.file_suffix}.npy"
        )
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Data file not found: {e.filename}.")
        sys.exit(1)

    # Construct input matrix
    if 'B' in Params.model_structure:
        input_names = ['load', 'cooling_temperature', 'inlet_temperature']
        input_list = []
        for name in input_names:
            try:
                arr = np.load(
                    f"{Params.data_dir}/{name}{Params.file_suffix}.npy"
                ).squeeze()

                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[1] == 1 and t_full.size > 1:
                    arr = np.repeat(arr, t_full.size, axis=1)

                input_list.append(arr)
            except FileNotFoundError:
                if Params.output:
                    print(f"Warning: Input {name} not found. Using zeros.")
                input_list.append(np.ones((1, t_full.size)))

        entries_full = np.vstack(input_list)
    else:
        entries_full = np.ones((3, t_full.size))

    # Initialize derivatives
    Fdot_full = np.zeros_like(F_all_full)
    Tdot_full = np.zeros_like(T_all_full)

    # =========================================================================
    # 2. PRE-PROCESSING (SMOOTHING ONLY)
    # =========================================================================
    if F_all_full.ndim == 1:
        F_all_full = F_all_full.reshape(1, -1)
    if T_all_full.ndim == 1:
        T_all_full = T_all_full.reshape(1, -1)

    F_raw = F_all_full.copy()
    T_raw = T_all_full.copy()

    if Params.use_filter:
        F_all_full = opinf.pre.apply_smoothing(
            F_all_full, Params.filter_window, Params.filter_poly
        )
        T_all_full = opinf.pre.apply_smoothing(
            T_all_full, Params.filter_window, Params.filter_poly
        )
        Params.true_derivatives = False

    F_all, T_all = F_all_full, T_all_full
    Fdot, Tdot = np.zeros_like(F_all), np.zeros_like(T_all)

    # =========================================================================
    # 2. TRUNCATION & HIDDEN STATES
    # =========================================================================
    t, F_all, T_all, entries, Fdot, Tdot = (
        t_full, F_all_full, T_all_full, entries_full, Fdot_full, Tdot_full
    )

    if F_all.ndim == 1:
        F_all = F_all.reshape(1, -1)
    if Fdot.ndim == 1:
        Fdot = Fdot.reshape(1, -1)
    if F_raw.ndim == 1:
        F_raw = F_raw.reshape(1, -1)
    if T_raw.ndim == 1:
        T_raw = T_raw.reshape(1, -1)

    # --- A. Time Truncation ---
    if Params.time_to_leave_out > 0:
        if Params.split == 'condition':
            if Params.output:
                print(f"Condition Split: Truncating first "
                      f"{Params.time_to_leave_out} steps of EACH trajectory.")

            indices_reset = np.where(np.diff(t) < 0)[0] + 1
            traj_starts = np.concatenate(([0], indices_reset))
            traj_ends = np.concatenate((indices_reset, [len(t)]))
            mask = np.ones(len(t), dtype=bool)

            for start, end in zip(traj_starts, traj_ends):
                cut_until = start + Params.time_to_leave_out
                if cut_until < end:
                    mask[start:cut_until] = False
                else:
                    mask[start:end] = False

            t = t[mask]
            F_all, T_all = F_all[:, mask], T_all[:, mask]
            entries, Fdot, Tdot = (
                entries[:, mask], Fdot[:, mask], Tdot[:, mask]
            )
            F_raw, T_raw = F_raw[:, mask], T_raw[:, mask]
        else:
            if Params.output:
                print(f"Truncating first {Params.time_to_leave_out} "
                      f"time steps (globally).")

            t = t[Params.time_to_leave_out:]
            F_all = F_all[:, Params.time_to_leave_out:]
            T_all = T_all[:, Params.time_to_leave_out:]
            entries = entries[:, Params.time_to_leave_out:]
            Fdot = Fdot[:, Params.time_to_leave_out:]
            Tdot = Tdot[:, Params.time_to_leave_out:]
            F_raw = F_raw[:, Params.time_to_leave_out:]
            T_raw = T_raw[:, Params.time_to_leave_out:]

    # --- B. Matrix Completion (Reconstructing Spatial Flow F) ---
    if Params.hidden_states or F_all.shape[0] == 1:
        if Params.split == 'condition':
            print("\nCRITICAL ERROR: Matrix Completion not supported "
                  "for 'condition' split.")
            sys.exit(1)

        if Params.output:
            print("Performing Matrix Completion for Spatial Flow Profile...")

        Params.hidden_states = True
        load_input = entries[0, :]
        F_in_calculated = np.maximum((load_input - 20.0) / 5.0, 0)

        F_all_z_end = F_all[-1, :]
        try:
            F_all = opinf.utils.matrix_completion_F(
                F_in_calculated, F_all_z_end, T_all
            )
        except NameError:
            F_all = opinf.utils.matrix_completion(
                F_in_calculated, F_all_z_end, T_all
            )

        F_raw_z_end = F_raw[-1, :]
        try:
            F_raw = opinf.utils.matrix_completion_F(
                F_in_calculated, F_raw_z_end, T_raw
            )
        except NameError:
            F_raw = opinf.utils.matrix_completion(
                F_in_calculated, F_raw_z_end, T_raw
            )

    # =========================================================================
    # 3. SAMPLING
    # =========================================================================
    (t_sampled, z_F, F_all_sampled, Fdot_sampled, entries_sampled) = \
        opinf.pre.sampled_data(t, z_all, F_all, Fdot, entries)

    (_, z_T, T_all_sampled, Tdot_sampled, _) = \
        opinf.pre.sampled_data(t, z_all, T_all, Tdot, entries)

    t, entries = t_sampled, entries_sampled
    F_all, T_all = F_all_sampled, T_all_sampled
    Fdot, Tdot = Fdot_sampled, Tdot_sampled

    (_, _, F_raw_sampled, _, _) = opinf.pre.sampled_data(
        t, z_all, F_raw, Fdot, entries
    )
    (_, _, T_raw_sampled, _, _) = opinf.pre.sampled_data(
        t, z_all, T_raw, Tdot, entries
    )

    F_raw, T_raw = F_raw_sampled, T_raw_sampled
    z_all = z_T

    # Define main states (remove boundary z=0)
    F = F_all[1:, :]
    T = T_all[1:, :]
    z = z_all[1:]
    Params.batch_size = F.shape[1]

    # =========================================================================
    # 4. LIFTING
    # =========================================================================
    T_REF, Ea_R = 555.0, 10000.0
    w1_all = 1.0 / T_REF - 1.0 / np.maximum(T_all, 1e-6)
    w2_all = np.exp(Ea_R * w1_all)

    w1 = w1_all[1:, :] if Params.apply_lifting_1 else np.zeros_like(T)
    w2 = w2_all[1:, :] if Params.apply_lifting_2 else np.zeros_like(T)

    # =========================================================================
    # 5. TRAIN/TEST SPLIT
    # =========================================================================
    train_indices_sorted, test_indices_sorted = None, None
    evaluation_results_list = []

    datasets = [
        F, F_all, F_raw, Fdot,
        T, T_all, T_raw, Tdot,
        entries, t, w1, w2
    ]

    if Params.split == 'time':
        if Params.output:
            print(f"Splitting data by time (Train Ratio: {Params.train_ratio})")
        split_data = [
            opinf.pre.train_test_split_time(d, Params.train_ratio)
            for d in datasets
        ]

    elif Params.split == 'condition':
        if Params.output:
            print(f"Splitting data by condition "
                  f"(Train Ratio: {Params.train_ratio})")

        split_points = np.where(np.diff(t) < 0)[0] + 1
        traj_starts = np.concatenate(([0], split_points))
        traj_ends = np.concatenate((split_points, [len(t)]))
        number_condition = len(traj_starts)

        if Params.output:
            print(f"  -> Found {number_condition} trajectories.")

        num_training = int(np.round(Params.train_ratio * number_condition))
        if Params.output:
            print(f"  -> Using {num_training} for Training.")

        rng = np.random.default_rng(0)
        all_indices = np.arange(number_condition)
        rng.shuffle(all_indices)

        train_indices_sorted = np.sort(all_indices[:num_training])
        test_indices_sorted = np.sort(all_indices[num_training:])

        if Params.output:
            print(f"  -> Train Indices: {train_indices_sorted}")

        split_data = []
        for d in datasets:
            d_train, d_test = opinf.pre.train_test_split_conditions(
                d, traj_starts, traj_ends,
                train_indices_sorted, test_indices_sorted
            )
            split_data.append((d_train, d_test))

        for i in range(number_condition):
            status = 'Train' if i in train_indices_sorted else 'Test'
            evaluation_results_list.append({
                'Case': i,
                'Status': status,
                'rel_froerr': np.nan,
                'start_idx': traj_starts[i],
                'end_idx': traj_ends[i]
            })
    else:
        raise ValueError(f"Error: Invalid Params.split value: {Params.split}")

    (
        (F_train, F_test), (F_all_train, F_all_test),
        (F_raw_train, F_raw_test), (Fdot_train, Fdot_test),
        (T_train, T_test), (T_all_train, T_all_test),
        (T_raw_train, T_raw_test), (Tdot_train, Tdot_test),
        (entries_train, entries_test), (t_train, t_test),
        (w1_train, w1_test), (w2_train, w2_test)
    ) = split_data

    # =========================================================================
    # 6. PRE-PROCESSING (SCALING, NOISE, SHIFTING)
    # =========================================================================
    # 1. Input Scaling
    if Params.input_scaling:
        input_scaling_factors = np.max(
            np.abs(entries_train), axis=1, keepdims=True
        )
        input_scaling_factors[input_scaling_factors == 0] = 1.0

        entries_train /= input_scaling_factors
        entries_test /= input_scaling_factors
        if Params.split == 'condition':
            entries /= input_scaling_factors

    # 2. Noise Injection
    if Params.artificial_noise:
        if Params.output:
            print(f"\nAdding artificial noise to training states. "
                  f"Level: {Params.noise_level:.4f}")
        F_train = opinf.utils.add_noise(
            F_train, noise_level=Params.noise_level, seed=0
        )
        T_train = opinf.utils.add_noise(
            T_train, noise_level=Params.noise_level, seed=0
        )

    # 3. Shifting
    shift_by_F = np.zeros((F.shape[0], 1))
    shift_by_T = np.zeros((T.shape[0], 1))

    if F_train.size > 0 and Params.state_shifting != 'none':
        if Params.state_shifting == 'mean':
            shift_by_F = np.mean(F_train, axis=1, keepdims=True)
            shift_by_T = np.mean(T_train, axis=1, keepdims=True)
        elif Params.state_shifting == 'steady-state':
            shift_by_F = F_train[:, 0:1]
            shift_by_T = T_train[:, 0:1]
        elif Params.state_shifting == 'max':
            shift_by_F = np.max(F_train, axis=1, keepdims=True)
            shift_by_T = np.max(T_train, axis=1, keepdims=True)

    F_train_shifted, F_train_ss = opinf.pre.shift(F_train, shift_by=shift_by_F)
    T_train_shifted, T_train_ss = opinf.pre.shift(T_train, shift_by=shift_by_T)
    w1_train_shifted, w1_train_ss = opinf.pre.shift(w1_train, shift_by=None)
    w2_train_shifted, w2_train_ss = opinf.pre.shift(w2_train, shift_by=None)

    reference_states = np.concatenate(
        (F_train_ss, T_train_ss, w1_train_ss, w2_train_ss)
    )

    F_test_shifted, _ = opinf.pre.shift(F_test, shift_by=shift_by_F)
    T_test_shifted, _ = opinf.pre.shift(T_test, shift_by=shift_by_T)
    w1_test_shifted, _ = opinf.pre.shift(w1_test, shift_by=None)
    w2_test_shifted, _ = opinf.pre.shift(w2_test, shift_by=None)

    # 4. Scaling
    max_F, max_T = 1.0, 1.0
    F_scaled_from, F_scaled_to = None, None
    T_scaled_from, T_scaled_to = None, None
    w1_scaled_from, w1_scaled_to = None, None
    w2_scaled_from, w2_scaled_to = None, None

    if Params.state_scaling == 'min-max':
        # Apply min-max (without clipping) on train
        F_train_shifted, _, F_scaled_from = opinf.pre.scale(
            F_train_shifted, (0, 1)
        )
        T_train_shifted, _, T_scaled_from = opinf.pre.scale(
            T_train_shifted, (0, 1)
        )
        w1_train_shifted, _, w1_scaled_from = opinf.pre.scale(
            w1_train_shifted, (0, 1)
        )
        w2_train_shifted, _, w2_scaled_from = opinf.pre.scale(
            w2_train_shifted, (0, 1)
        )

        w1_train_shifted = np.nan_to_num(w1_train_shifted)
        w2_train_shifted = np.nan_to_num(w2_train_shifted)

        # Apply to test using global helper
        F_test_shifted = opinf.pre.apply_minmax(F_test_shifted, F_scaled_from)
        T_test_shifted = opinf.pre.apply_minmax(T_test_shifted, T_scaled_from)
        w1_test_shifted = opinf.pre.apply_minmax(
            w1_test_shifted, w1_scaled_from
        )
        w2_test_shifted = opinf.pre.apply_minmax(
            w2_test_shifted, w2_scaled_from
        )

        w1_test_shifted = np.nan_to_num(w1_test_shifted)
        w2_test_shifted = np.nan_to_num(w2_test_shifted)

    elif Params.state_scaling == 'simple':
        max_F = (np.max(np.abs(F_train_shifted))
                 if F_train_shifted.size > 0 else 1.0)
        max_T = (np.max(np.abs(T_train_shifted))
                 if T_train_shifted.size > 0 else 1.0)

        if max_F == 0:
            max_F = 1.0
        if max_T == 0:
            max_T = 1.0

        F_train_shifted /= max_F
        T_train_shifted /= max_T
        F_test_shifted /= max_F
        T_test_shifted /= max_T

    Q_train_shifted = np.concatenate(
        (F_train_shifted, T_train_shifted, w1_train_shifted, w2_train_shifted)
    )
    Q_test_shifted = np.concatenate(
        (F_test_shifted, T_test_shifted, w1_test_shifted, w2_test_shifted)
    )

    # =========================================================================
    # 7. POD BASIS
    # =========================================================================
    if Params.output:
        print("\n--- Computing POD Basis (SVD) ---")

    if F_train_shifted.shape[0] == 1:
        V_F_full = np.ones((1, 1))
        S_F = np.array([np.linalg.norm(F_train_shifted)])
    else:
        [V_F_full, S_F] = opinf.basis.pod(
            states=F_train_shifted, r="full", mode='dense'
        )

    [V_T_full, S_T] = opinf.basis.pod(
        states=T_train_shifted, r="full", mode='dense'
    )
    [V_w1_full, S_w1] = opinf.basis.pod(
        states=w1_train_shifted, r="full", mode='dense'
    )
    [V_w2_full, S_w2] = opinf.basis.pod(
        states=w2_train_shifted, r="full", mode='dense'
    )

    r_F = opinf.basis.cumulative_energy(
        S_F, Params.energy_threshold_single, name_tag='Flow F'
    )
    if r_F == 0 and F_train_shifted.shape[0] > 0:
        r_F = 1
    Params.r_X, Params.r_F = r_F, r_F

    r_T = opinf.basis.cumulative_energy(
        S_T, Params.energy_threshold_single, name_tag='Temperature T'
    )
    Params.r_T = r_T

    if np.all(w1_train_shifted == 0):
        r_w1 = 0
    else:
        r_w1 = opinf.basis.cumulative_energy(
            S_w1, Params.energy_threshold_single, name_tag='Lifting w1'
        )
    Params.r_w1 = r_w1

    if np.all(w2_train_shifted == 0):
        r_w2 = 0
    else:
        r_w2 = opinf.basis.cumulative_energy(
            S_w2, Params.energy_threshold_single, name_tag='Lifting w2'
        )
    Params.r_w2 = r_w2

    Params.ROM_order = r_F + r_T + r_w1 + r_w2
    if Params.output:
        print(f"\nTotal ROM Order: {Params.ROM_order} "
              f"(F={r_F}, T={r_T}, w1={r_w1}, w2={r_w2})")

    Q_reduced, V_reduced, V_reduced_nonlin, Xi = \
        opinf.basis.get_basis_and_reduced_data(
            V_F_full, V_T_full, V_w1_full, V_w2_full,
            Q_train_shifted, reference_states
        )

    V_F, V_T = V_F_full[:, :r_F], V_T_full[:, :r_T]

    if Params.output:
        print(f"V_reduced shape: {V_reduced.shape}")
        if r_F > 0 and V_F.shape[0] > 1:
            opinf.utils.plot_POD_modes(z, V_F, 0, r_F, 'Flow F modes')
        if r_T > 0:
            opinf.utils.plot_POD_modes(z, V_T, 0, r_T, 'Temperature T modes')

    # =========================================================================
    # 8. DERIVATIVE ASSIGNMENT
    # =========================================================================
    if Params.output:
        print("\n--- Estimating Time Derivatives (Numerical) ---")

    n_snapshots = Q_train_shifted.shape[1]
    dt = t_train[1] - t_train[0]

    if Params.output:
        print(f"Method: {Params.ddt_method}, dt: {dt:.4f} s")

    Fdot_train = opinf.utils.estimate_derivatives(
        F_train_shifted, dt, entries_train, method=Params.ddt_method
    )
    Tdot_train = opinf.utils.estimate_derivatives(
        T_train_shifted, dt, entries_train, method=Params.ddt_method
    )

    if Params.r_w1 > 0:
        w1dot_train = opinf.utils.estimate_derivatives(
            w1_train_shifted, dt, entries_train, method=Params.ddt_method
        )
    else:
        w1dot_train = np.zeros_like(w1_train_shifted)

    if Params.r_w2 > 0:
        w2dot_train = opinf.utils.estimate_derivatives(
            w2_train_shifted, dt, entries_train, method=Params.ddt_method
        )
    else:
        w2dot_train = np.zeros_like(w2_train_shifted)

    Qdot_train = np.vstack(
        (Fdot_train, Tdot_train, w1dot_train, w2dot_train)
    )

    if Qdot_train.shape[0] != V_reduced.shape[0]:
        if Params.output:
            print(f"CRITICAL ERROR: Shape Mismatch! "
                  f"V={V_reduced.shape[0]}, Qdot={Qdot_train.shape[0]}")

        diff = V_reduced.shape[0] - Qdot_train.shape[0]
        if diff > 0:
            Qdot_train = np.vstack(
                (Qdot_train, np.zeros((diff, n_snapshots)))
            )
        elif diff < 0:
            Qdot_train = Qdot_train[:V_reduced.shape[0], :]

    Qdot_reduced = V_reduced.T @ Qdot_train
    if Params.ddt_postprocessing:
        Qdot_reduced = opinf.pre.remove_spikes(Qdot_reduced)

    # =========================================================================
    # 9. REDUCED SCALING
    # =========================================================================
    idx_F_end = r_F
    idx_T_end = idx_F_end + r_T
    idx_w1_end = idx_T_end + r_w1
    idx_w2_end = idx_w1_end + r_w2

    scaling_fac_F = 1.0
    scaling_fac_T = 1.0
    scaling_fac_w1 = 1.0
    scaling_fac_w2 = 1.0

    if Params.reduced_scaling:
        def scale_block(q, dq, key):
            if q.shape[0] > 0:
                sf, q_s = opinf.utils.scaled_states(q)
                return q_s, dq / sf, sf
            return q, dq, 1.0

        qF, dqF, scaling_fac_F = scale_block(
            Q_reduced[0:idx_F_end], Qdot_reduced[0:idx_F_end], 'F'
        )
        qT, dqT, scaling_fac_T = scale_block(
            Q_reduced[idx_F_end:idx_T_end], Qdot_reduced[idx_F_end:idx_T_end], 'T'
        )
        qw1, dqw1, scaling_fac_w1 = scale_block(
            Q_reduced[idx_T_end:idx_w1_end],
            Qdot_reduced[idx_T_end:idx_w1_end], 'w1'
        )
        qw2, dqw2, scaling_fac_w2 = scale_block(
            Q_reduced[idx_w1_end:idx_w2_end],
            Qdot_reduced[idx_w1_end:idx_w2_end], 'w2'
        )

        Q_list, Qdot_list = [], []
        if r_F > 0:
            Q_list.append(qF)
            Qdot_list.append(dqF)
        if r_T > 0:
            Q_list.append(qT)
            Qdot_list.append(dqT)
        if r_w1 > 0:
            Q_list.append(qw1)
            Qdot_list.append(dqw1)
        if r_w2 > 0:
            Q_list.append(qw2)
            Qdot_list.append(dqw2)

        Q_reduced = np.concatenate(Q_list)
        Qdot_reduced = np.concatenate(Qdot_list)

    if Params.output:
        if r_F > 0:
            opinf.utils.plot_reduced_trajectories(
                t_train, Q_reduced, 0, r_F, 'Reduced: F'
            )
        if r_T > 0:
            opinf.utils.plot_reduced_trajectories(
                t_train, Q_reduced, r_F, r_F + r_T, 'Reduced: T'
            )

    # =========================================================================
    # 10. MODEL TRAINING
    # =========================================================================
    Params.input_dim = entries_train.shape[0]
    rom = opinf.models.create_rom(non_markov=False, seed=seed)

    if Params.use_PINN:
        if Params.output:
            print("Training via PINN-OpInf...")
        model, history = opinf.training.train_pinn_opinf(
            Q_reduced, t_train, entries_train, rom, r_F
        )
        A_OpInf, B_OpInf, C_OpInf, H_OpInf = opinf.training.learned_model(model)
        dynamics_pred = A_OpInf @ Q_reduced
        if 'B' in Params.model_structure:
            dynamics_pred += B_OpInf @ entries_train
        error_reduced_states = Qdot_reduced - dynamics_pred
    else:
        if Params.output:
            print("Training via Standard OpInf (Gradient Descent)...")
        model, loss_track, error_reduced_states = opinf.training.train_model(
            Q_reduced, Qdot_reduced, t_train, entries_train, rom, seed=seed
        )
        A_OpInf, B_OpInf, C_OpInf, H_OpInf = opinf.training.learned_model(model)

    # Non-Markovian
    E_list, F_list = [], []
    if Params.lag_max > 0:
        if Params.output:
            print(f"\nInferring Non-Markovian Terms "
                  f"(Max Lag: {Params.lag_max})")

        resid = error_reduced_states.copy()
        orig_struct = Params.model_structure
        Params.model_structure = orig_struct.replace("H", "").replace("C", "")
        norm_prev = np.linalg.norm(resid, 'fro')

        for lag in range(1, Params.lag_max + 1):
            if resid.shape[1] <= lag:
                break

            x_l = Q_reduced[:, :-lag]
            u_l = entries_train[:, :-lag]
            res_t = resid[:, lag:]
            t_v = t_train[lag:]

            rom_nm = opinf.models.create_rom(non_markov=True)
            mod_l, _, _ = opinf.training.train_model(x_l, res_t, t_v, u_l, rom_nm)
            E_l, F_l, _, _ = opinf.training.learned_model(mod_l)

            E_list.append(E_l)
            F_list.append(F_l)
            resid[:, lag:] -= (E_l @ x_l + F_l @ u_l)

            n_curr = np.linalg.norm(resid, 'fro')
            if (norm_prev - n_curr) / norm_prev < 1e-4 and lag > 1:
                break
            norm_prev = n_curr

        Params.model_structure = orig_struct

    # =========================================================================
    # 11. SIMULATE & EVALUATE (LINEAR BASELINE & DATA COLLECTION)
    # =========================================================================
    model_ops = {
        "A": A_OpInf, "B": B_OpInf, "C": C_OpInf,
        "H": H_OpInf, "E": E_list, "F": F_list
    }

    state_s_params = {
        'n_F': F.shape[0], 'n_T': T.shape[0],
        'max_F': max_F, 'max_T': max_T,
        'F_scaled_from': F_scaled_from, 'F_scaled_to': F_scaled_to,
        'T_scaled_from': T_scaled_from, 'T_scaled_to': T_scaled_to
    }

    recon_data = {
        "r_F": r_F, "r_T": r_T, "r_w1": r_w1, "r_w2": r_w2,
        "scaling_fac_F": scaling_fac_F, "scaling_fac_T": scaling_fac_T,
        "scaling_fac_w1": scaling_fac_w1, "scaling_fac_w2": scaling_fac_w2,
        "V_reduced": V_reduced, "V_reduced_nonlin": V_reduced_nonlin, "Xi": Xi,
        "F_train": F_train, "T_train": T_train,
        "reference_states": reference_states,
        "state_scaling_params": state_s_params, "dt": dt,
        "cnn_pack": None  # No CNN yet, establishing baseline
    }

    results_storage = []  # Collects data for CNN and Global Plot
    rel_froerr_test = np.nan

    # --- BRANCH A: CONDITION SPLIT ---
    if Params.split == 'condition':
        if Params.output:
            print("\n" + "=" * 60 + "\nLINEAR EVALUATION "
                  "(and Data Collection)\n" + "=" * 60)

        rel_froerr_linear_all = []

        for result_entry in evaluation_results_list:
            status = result_entry['Status']
            FORCE_SIM = Params.use_CNN

            if status == 'Train' and not Params.evaluate_train_cases and not FORCE_SIM:
                continue

            orig_output = Params.output
            if status == 'Train':
                Params.output = False

            try:
                s, e = result_entry['start_idx'], result_entry['end_idx']
                t_s, u_s = t[s:e] - t[s], entries[:, s:e]
                F_inn, T_inn = F[:, s:e], T[:, s:e]
                F_raw_c, T_raw_c = F_raw[:, s:e], T_raw[:, s:e]

                # Start Vector (Shift/Scale)
                f0, _ = opinf.pre.shift(F_inn[:, 0:1], shift_by=shift_by_F)
                t0, _ = opinf.pre.shift(T_inn[:, 0:1], shift_by=shift_by_T)
                w10, _ = opinf.pre.shift(w1[:, s:s + 1], None)
                w20, _ = opinf.pre.shift(w2[:, s:s + 1], None)

                if Params.state_scaling == 'min-max':
                    f0 = opinf.pre.apply_minmax(f0, F_scaled_from)
                    t0 = opinf.pre.apply_minmax(t0, T_scaled_from)
                    w10 = opinf.pre.apply_minmax(w10, w1_scaled_from)
                    w20 = opinf.pre.apply_minmax(w20, w2_scaled_from)
                elif Params.state_scaling == 'simple':
                    f0 /= max_F
                    t0 /= max_T

                y0 = opinf.utils.reduced_state(
                    np.vstack((f0, t0, w10, w20)), V_reduced
                )

                # Simulate (linear only)
                sol_lin, sol_red = simulate_and_reconstruct(
                    y0, t_s, (0, t_s[-1]), u_s,
                    model_ops, recon_data, Params, seed
                )
                if sol_lin.ndim == 1:
                    sol_lin = sol_lin.reshape(-1, 1)
                    sol_red = sol_red.reshape(-1, 1)

                # Store for CNN
                results_storage.append({
                    'Status': status, 'sol_reduced': sol_red,
                    'F_true': F_inn, 'T_true': T_inn,
                    't': t[s:e], 'z': z_all,
                    'F_raw': F_raw_c, 'T_raw_c': T_raw_c,
                    'err_linear': np.nan  # Placeholder
                })

                # Calculate linear error
                err_lin = opinf.post.run_postprocessing(
                    sol_lin, Params, F_inn, T_inn,
                    z_all, t_s, r_F, r_T, F_raw_c, T_raw_c, draw_split=False
                )

                results_storage[-1]['err_linear'] = err_lin
                if status == 'Test':
                    rel_froerr_linear_all.append(err_lin)

                if Params.output:
                    print(f"Case {result_entry['Case']} ({status}): "
                          f"Linear Error = {err_lin:.4e}")

            except Exception as e:
                print(f"Error {result_entry['Case']}: {e}")
            finally:
                Params.output = orig_output

        if rel_froerr_linear_all:
            rel_froerr_test = np.mean(rel_froerr_linear_all)

    # --- BRANCH B: TIME SPLIT ---
    else:
        if Params.output:
            print("\n" + "=" * 60 + "\nTIME SPLIT SIMULATION\n" + "=" * 60)

        y0_gl = opinf.utils.reduced_state(
            Q_train_shifted[:, :len(E_list) + 1], V_reduced
        )

        # Simulate full range
        sol_lin, sol_red = simulate_and_reconstruct(
            y0_gl, t, (np.min(t), np.max(t)), entries,
            model_ops, recon_data, Params, seed
        )
        if sol_lin.ndim == 1:
            sol_lin = sol_lin.reshape(-1, 1)
            sol_red = sol_red.reshape(-1, 1)

        split_idx = t_train.shape[0]

        # Store train segment
        results_storage.append({
            'Status': 'Train', 'sol_reduced': sol_red[:, :split_idx],
            'F_true': F_train, 'T_true': T_train, 't': t[:split_idx],
            'z': z_all, 'F_raw': F_raw[:, :split_idx],
            'T_raw_c': T_raw[:, :split_idx],
            'err_linear': 0.0
        })

        # Store test segment
        if split_idx < sol_lin.shape[1]:
            if Params.output:
                print("TEST SET RESULTS (Linear):")

            rel_froerr_test = opinf.post.run_postprocessing(
                sol_lin[:, split_idx:], Params,
                F_test, T_test, z_all, t_test, r_F, r_T,
                F_raw[:, split_idx:], T_raw[:, split_idx:],
                draw_split=False
            )

            results_storage.append({
                'Status': 'Test', 'sol_reduced': sol_red[:, split_idx:],
                'F_true': F_test, 'T_true': T_test, 't': t[split_idx:],
                'z': z_all, 'F_raw': F_raw[:, split_idx:],
                'T_raw_c': T_raw[:, split_idx:],
                'err_linear': rel_froerr_test
            })

        # Global Linear Plot (Train + Test)
        if Params.output:
            print("\nGLOBAL LINEAR RESULT:")
            opinf.post.run_postprocessing(
                sol_lin, Params, F, T, z_all, t, r_F, r_T,
                F_raw, T_raw, draw_split=True
            )

    # =========================================================================
    # 12. CNN ERROR CORRECTION (Post-Processing)
    # =========================================================================
    if Params.use_CNN and len(results_storage) > 0:
        if Params.output:
            print("\n" + "=" * 60 + "\nCNN CORRECTION PHASE\n" + "=" * 60)

        # 1. Collect training data
        train_sol_red = [
            r['sol_reduced'] for r in results_storage if r['Status'] == 'Train'
        ]
        train_F_true = [
            r['F_true'] for r in results_storage if r['Status'] == 'Train'
        ]
        train_T_true = [
            r['T_true'] for r in results_storage if r['Status'] == 'Train'
        ]

        if len(train_sol_red) == 0:
            print("Warning: No training simulations found. Cannot train CNN.")
        else:
            # 2. Train Global CNN
            cnn_pack = opinf.post.train_global_cnn(
                train_sol_red, train_F_true, train_T_true, V_reduced, seed
            )

            # 3. Apply to test cases
            rel_froerr_cnn_all = []

            for res in results_storage:
                if res['Status'] == 'Train':
                    continue

                # Inference
                sol_cnn_phys = opinf.post.apply_cnn_inference(
                    res['sol_reduced'], cnn_pack
                )

                # Eval
                if Params.output:
                    print(f"Evaluating CNN on "
                          f"{res.get('Case', 'Test Segment')}...")

                err_cnn = opinf.post.run_postprocessing(
                    sol_cnn_phys, Params,
                    res['F_true'], res['T_true'],
                    res['z'], res['t'],
                    r_F, r_T,
                    res['F_raw'], res['T_raw_c'],
                    draw_split=False
                )

                rel_froerr_cnn_all.append(err_cnn)
                if Params.output:
                    print(f"-> Linear: {res['err_linear']:.4e} | "
                          f"CNN: {err_cnn:.4e}")

            if rel_froerr_cnn_all:
                mean_cnn = np.mean(rel_froerr_cnn_all)
                print(f"\nFinal Test Error (Linear): {rel_froerr_test:.4e}")
                print(f"Final Test Error (CNN):    {mean_cnn:.4e}")
                rel_froerr_test = mean_cnn

            # 4. Global visualization
            if Params.output:
                print("\n" + "-" * 40 + "\nGLOBAL VISUALIZATION "
                      "(Train + Test combined)\n" + "-" * 40)

                # Sort segments by time
                sorted_res = sorted(results_storage, key=lambda x: x['t'][0])

                full_sol_list = []
                full_F_list = []
                full_T_list = []
                full_F_raw_list = []
                full_T_raw_list = []
                full_t_list = []

                for res in sorted_res:
                    sol_seg = opinf.post.apply_cnn_inference(
                        res['sol_reduced'], cnn_pack
                    )

                    full_sol_list.append(sol_seg)
                    full_F_list.append(res['F_true'])
                    full_T_list.append(res['T_true'])
                    full_F_raw_list.append(res['F_raw'])
                    full_T_raw_list.append(res['T_raw_c'])
                    full_t_list.append(res['t'])

                # Stitch arrays together
                sol_global = np.hstack(full_sol_list)
                F_true_global = np.hstack(full_F_list)
                T_true_global = np.hstack(full_T_list)
                F_raw_global = np.hstack(full_F_raw_list)
                T_raw_global = np.hstack(full_T_raw_list)
                t_global = np.concatenate(full_t_list)

                # Plot global
                opinf.post.run_postprocessing(
                    sol_global, Params,
                    F_true_global, T_true_global,
                    z_all, t_global,
                    r_F, r_T,
                    F_raw_global, T_raw_global,
                    draw_split=True
                )

    # =========================================================================
    # SAVE
    # =========================================================================
    if Params.save_results:
        if Params.output:
            print("\nSaving results...")

        y0_s = locals().get('y0_gl', None)
        if y0_s is None:
            if 'Q_reduced' in locals() and Q_reduced.shape[1] > 0:
                y0_s = Q_reduced[:, 0]
            else:
                y0_s = np.zeros(Params.ROM_order)

        if y0_s.shape[0] != Params.ROM_order:
            y0_fix = np.zeros(Params.ROM_order)
            m = min(y0_s.shape[0], Params.ROM_order)
            y0_fix[:m] = y0_s[:m]
            y0_s = y0_fix

        model_dict = {
            "A_OpInf": A_OpInf, "B_OpInf": B_OpInf,
            "C_OpInf": C_OpInf, "H_OpInf": H_OpInf,
            "E_list": E_list, "F_list": F_list,
            "inputs": entries, "time": t, "initial_values": y0_s,
            "basis": V_reduced, "scaling_fac_F": scaling_fac_F,
            "scaling_fac_T": scaling_fac_T,
            "scaling_fac_w1": scaling_fac_w1,
            "scaling_fac_w2": scaling_fac_w2,
            "r_F": r_F, "r_T": r_T, "r_w1": r_w1, "r_w2": r_w2,
            "reference_states_shifting": reference_states,
            "state_scaling_params": state_s_params
        }

        save_dir = opinf.utils.save_results(
            model=model_dict, Params=Params, save_dir=None, decoder=None,
            input_scaler_F=None, input_scaler_T=None,
            target_scaler_F=None, target_scaler_T=None
        )

        if Params.output:
            print(f"Saved to: {save_dir}")

    return rel_froerr_test


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print(f"Used device: {device}")

    params = opinf.parameters.Params()
    params.output = True
    params.save_results = False

    seed = 0
    tic = time.time()
    err_test = main(params, seed=seed)

    print("runtime in seconds:", time.time() - tic)
