"""Identify a reduced-order model from a reactor record and evaluate it.

The settings of the final models are in configs.py; run.py applies them and
calls main(). Module constants below are the switches a configuration can set.
"""
import os
import sys
import time
import traceback
import warnings

import numpy as np
import matplotlib as mpl

import opinf

warnings.filterwarnings("ignore")

# The graphics module switches matplotlib to pgf at import; plain text here.
mpl.rcParams['text.usetex'] = False
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([r"\usepackage[utf8x]{inputenc}",
                               r"\usepackage[T1]{fontenc}"]),
})

# --- switches -----------------------------------------------------------------
# On integration failure: "zeros" scores the run as 100 % error (used in the
# hyperparameter search), "raise" propagates the exception.
INTEGRATION_FAILURE = "zeros"

# Basis names that carry a polynomial manifold correction.
MANIFOLD_BASES = ('NL-POD', 'AM', 'MPOD', 'MAM')

# Measurement repair (Sections 3.1 and 5.2): runs of identical samples of at
# least this length are interpolated along z, and the listed temperature
# nodes are replaced throughout by their neighbours. None / () disable.
REPAIR_FLATLINES_MIN_SAMPLES = None
BAD_TEMPERATURE_NODES = ()

# Memory terms (evaluated in the ablation only): fit to the derivative
# residual, delay spacing in samples, and "stable" or "free" operators.
LAG_FIT_ON_DERIVATIVE = True
LAG_STEP = 1
LAG_OPERATOR_STRUCTURE = "stable"

# PINN surrogate when Params.use_PINN: "operators" uses its derivative as the
# target, "smoother" only replaces the measured latent states.
PINN_MODE = "operators"

# Test error from a simulation started at the beginning of the record
# ("record") or at the measured state at the start of the test segment
# ("segment", as used inside an optimizer). Both are printed.
TEST_FORECAST_FROM = "record"

# Directory and tag for the saved test-segment forecasts; None saves nothing.
SAVE_PREDICTIONS_TO = None
PREDICTION_TAG = None

device = opinf.utils.get_device()


def repair_flatlines(T, z, min_samples, dt=None, verbose=False):
    """Replace frozen sensor stretches by linear interpolation along z.

    A run of at least ``min_samples`` identical samples is treated as missing
    and filled from the nodes that are valid at the same instant.
    """
    T = np.array(T, dtype=float, copy=True)
    z = np.asarray(z, dtype=float).reshape(-1)
    invalid = np.zeros(T.shape, dtype=bool)
    for i in range(T.shape[0]):
        change = np.flatnonzero(np.diff(T[i]) != 0) + 1
        edges = np.concatenate(([0], change, [T.shape[1]]))
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a >= min_samples:
                invalid[i, a:b] = True

    for k in np.flatnonzero(invalid.any(axis=0)):
        bad = invalid[:, k]
        if bad.all():
            continue
        T[bad, k] = np.interp(z[bad], z[~bad], T[~bad, k])

    if verbose:
        rows = np.flatnonzero(invalid.any(axis=1))
        scale = (dt / 60.0) if dt else None
        for i in rows:
            n = int(invalid[i].sum())
            span = f"{n * scale:.1f} min" if scale else f"{n} samples"
            print(f"  frozen sensor repaired: node {i} (z {z[i]:.4f} m), "
                  f"{span}")
    return T


def replace_nodes(T, nodes):
    """Replace temperature nodes by interpolation between their neighbours."""
    T = np.array(T, dtype=float)
    nodes_first = T.shape[0] < T.shape[1]
    Tn = T if nodes_first else T.T
    bad = sorted(set(int(i) for i in nodes))
    good = [i for i in range(Tn.shape[0]) if i not in bad]
    for i in bad:
        lo = max((g for g in good if g < i), default=None)
        hi = min((g for g in good if g > i), default=None)
        if lo is None or hi is None:
            Tn[i] = Tn[hi if lo is None else lo]
        else:
            w = (i - lo) / (hi - lo)
            Tn[i] = (1 - w) * Tn[lo] + w * Tn[hi]
    return T


def simulate_and_reconstruct(y0_reduced, t, t_span, entries, model_ops,
                             recon_data, Params):
    """Integrate the reduced model and reconstruct the full state."""
    # 1. Initial condition in the scaled reduced coordinates.
    y0 = np.asarray(y0_reduced, dtype=float)
    if Params.reduced_scaling:
        y0_list = []
        current_idx = 0

        def scale_and_append(rank, key):
            nonlocal current_idx
            if rank > 0:
                _, y_scaled = opinf.utils.scaled_states(
                    y0_reduced[current_idx: current_idx + rank],
                    scaling_fac=recon_data[key])
                y0_list.append(y_scaled)
                current_idx += rank

        scale_and_append(recon_data['r_F'], 'scaling_fac_F')
        scale_and_append(recon_data['r_T'], 'scaling_fac_T')
        scale_and_append(recon_data['r_w1'], 'scaling_fac_w1')
        scale_and_append(recon_data['r_w2'], 'scaling_fac_w2')
        y0 = np.concatenate([arr for arr in y0_list if arr.size > 0])

    # 2. Integrate.
    try:
        if Params.lag_max == 0:
            sol_reduced = opinf.models.integrate(
                t_span, y0, t, entries,
                model_ops['A'], model_ops['B'], model_ops['C'],
                model_ops['H'], model_ops['E'], model_ops['F'])
        else:
            dt_recon = recon_data.get('dt', t[1] - t[0])
            n_lags = len(model_ops['E']) if model_ops['E'] else 0
            delays = [(lag + 1) * LAG_STEP * dt_recon for lag in range(n_lags)]
            sol_reduced = opinf.models.integrate_dde(
                t_span, y0, t, entries,
                model_ops['A'], model_ops['B'], model_ops['C'],
                model_ops['H'], model_ops['E'], model_ops['F'], delays)
    except Exception as exc:
        if INTEGRATION_FAILURE == "raise":
            raise
        n_full = (recon_data['state_scaling_params']['n_F']
                  + recon_data['state_scaling_params']['n_T'])
        print(f"  Integration failed: {type(exc).__name__}: {exc}")
        if Params.output:
            traceback.print_exc()
        print("  Returning a zero trajectory (scored as 100 % error).")
        return (np.zeros((n_full, len(t))),
                np.zeros((y0.shape[0], len(t))))

    if sol_reduced.ndim == 1:
        if sol_reduced.shape[0] == t.shape[0]:
            sol_reduced = sol_reduced.reshape(1, -1)
        else:
            sol_reduced = sol_reduced.reshape(-1, 1)

    if sol_reduced.shape[-1] != t.shape[0]:
        if sol_reduced.shape[-1] < t.shape[0]:
            padding = np.tile(sol_reduced[:, -1:],
                              (1, t.shape[0] - sol_reduced.shape[-1]))
            sol_reduced = np.hstack((sol_reduced, padding))
        else:
            sol_reduced = sol_reduced[:, :t.shape[0]]

    # 3. Undo the reduced scaling.
    sol_reduced_unscaled = sol_reduced.copy()
    if Params.reduced_scaling:
        current_idx = 0

        def unscale_block(rank, key):
            nonlocal current_idx
            if rank > 0:
                sol_reduced_unscaled[current_idx: current_idx + rank, :] = \
                    opinf.utils.unscaled_states(
                        recon_data[key],
                        sol_reduced[current_idx: current_idx + rank, :])
                current_idx += rank

        unscale_block(recon_data['r_F'], 'scaling_fac_F')
        unscale_block(recon_data['r_T'], 'scaling_fac_T')
        unscale_block(recon_data['r_w1'], 'scaling_fac_w1')
        unscale_block(recon_data['r_w2'], 'scaling_fac_w2')

    # 4. Linear reconstruction in the shifted and scaled space, plus the
    # manifold correction, which lives in the same space.
    V_active = recon_data['V_reduced'][:, :sol_reduced_unscaled.shape[0]]
    sol = V_active @ sol_reduced_unscaled
    if Params.basis in MANIFOLD_BASES:
        res = opinf.post.reconstruct_solution(
            sol=sol, sol_reduced=sol_reduced_unscaled,
            V_reduced_nonlin=recon_data.get('V_reduced_nonlin', None),
            Xi=recon_data.get('Xi', None))
        sol = res[0] if isinstance(res, tuple) else res

    # 5. Back to physical units.
    n_F = recon_data['state_scaling_params']['n_F']
    n_T = recon_data['state_scaling_params']['n_T']
    if sol.ndim == 1:
        sol = sol.reshape(-1, 1)
    sol = sol[:n_F + n_T]
    F_sol = sol[:n_F]
    T_sol = sol[n_F: n_F + n_T]
    if Params.state_scaling == 'simple':
        F_sol = F_sol * recon_data['state_scaling_params']['max_F']
        T_sol = T_sol * recon_data['state_scaling_params']['max_T']
    sol = np.vstack((F_sol, T_sol))
    sol = opinf.pre.unshift(sol, recon_data['reference_states'][:n_F + n_T])

    return sol, sol_reduced_unscaled


def main(Params, seed=None):
    """Identify a reduced model from the record and evaluate it."""
    if seed is None:
        seed = 0
    opinf.utils.set_deterministic(seed)
    if Params.state_scaling not in ('simple', 'none'):
        raise ValueError("state_scaling must be 'simple' or 'none'")

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
                f"{Params.data_dir}/flow_rate_out{Params.file_suffix}.npy")
        except FileNotFoundError:
            F_all_full = np.load(
                f"{Params.data_dir}/conversion{Params.file_suffix}.npy")
        T_all_full = np.load(
            f"{Params.data_dir}/center_temperature{Params.file_suffix}.npy")
    except FileNotFoundError as exc:
        print(f"CRITICAL ERROR: data file not found: {exc.filename}.")
        sys.exit(1)

    if BAD_TEMPERATURE_NODES:
        T_all_full = replace_nodes(T_all_full, BAD_TEMPERATURE_NODES)
        if Params.output:
            print(f"  temperature nodes {sorted(BAD_TEMPERATURE_NODES)} "
                  f"replaced by interpolation between their neighbours")

    # Inputs; the order fixes the columns of B, and entries[0] is the load.
    if 'B' in Params.model_structure:
        input_names = list(getattr(Params, 'input_names',
                                   ['load', 'cooling_temperature',
                                    'inlet_temperature']))
        input_list = []
        for name in input_names:
            path = f"{Params.data_dir}/{name}{Params.file_suffix}.npy"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"input channel '{name}' not found at {path}")
            arr = np.load(path).squeeze()
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] == 1 and t_full.size > 1:
                arr = np.repeat(arr, t_full.size, axis=1)
            input_list.append(arr)
        entries_full = np.vstack(input_list)
    else:
        entries_full = np.ones((3, t_full.size))

    # =========================================================================
    # 2. REPAIR, TRUNCATION, SPATIAL COMPLETION
    # =========================================================================
    if F_all_full.ndim == 1:
        F_all_full = F_all_full.reshape(1, -1)
    if T_all_full.ndim == 1:
        T_all_full = T_all_full.reshape(1, -1)

    if REPAIR_FLATLINES_MIN_SAMPLES:
        T_all_full = repair_flatlines(
            T_all_full, z_all, REPAIR_FLATLINES_MIN_SAMPLES,
            dt=float(np.median(np.diff(t_full))), verbose=Params.output)

    # Errors are measured against these copies.
    F_raw = F_all_full.copy()
    T_raw = T_all_full.copy()

    t = t_full
    F_all, T_all = F_all_full, T_all_full
    entries = entries_full
    Fdot = np.zeros_like(F_all)
    Tdot = np.zeros_like(T_all)

    # --- A. truncation of the start-up of each record or trajectory ---
    if Params.time_to_leave_out > 0:
        if Params.split == 'condition':
            indices_reset = np.where(np.diff(t) < 0)[0] + 1
            traj_starts = np.concatenate(([0], indices_reset))
            traj_ends = np.concatenate((indices_reset, [len(t)]))
            mask = np.ones(len(t), dtype=bool)
            for start, end in zip(traj_starts, traj_ends):
                mask[start:min(start + Params.time_to_leave_out, end)] = False
        else:
            mask = np.arange(len(t)) >= Params.time_to_leave_out
        t = t[mask]
        F_all, T_all = F_all[:, mask], T_all[:, mask]
        entries = entries[:, mask]
        Fdot, Tdot = Fdot[:, mask], Tdot[:, mask]
        F_raw, T_raw = F_raw[:, mask], T_raw[:, mask]

    # --- B. completion of the unobserved flow profile (Section 3.2.3) ---
    if Params.hidden_states or F_all.shape[0] == 1:
        if Params.split == 'condition':
            print("CRITICAL ERROR: matrix completion is not supported for "
                  "the condition split.")
            sys.exit(1)
        if Params.output:
            print("Reconstructing the spatial flow profile...")
        Params.hidden_states = True

        # Stoichiometric feed (CO2 + 4 H2): F_CO2,in = load / 5.
        STOICH_H2_CO2 = 4.0
        F_in_calculated = np.maximum(
            entries[0, :] / (1.0 + STOICH_H2_CO2), 0.0)
        completion = getattr(opinf.utils, 'matrix_completion_F',
                             opinf.utils.matrix_completion)
        F_all = completion(F_in_calculated, F_all[-1, :], T_all)
        F_raw = completion(F_in_calculated, F_raw[-1, :], T_raw)

    # =========================================================================
    # 3. SUBSAMPLING
    # =========================================================================
    (t_sampled, _z_F, F_all_sampled, Fdot_sampled, entries_sampled) = \
        opinf.pre.sampled_data(t, z_all, F_all, Fdot, entries)
    (_, z_T, T_all_sampled, Tdot_sampled, _) = \
        opinf.pre.sampled_data(t, z_all, T_all, Tdot, entries)
    (_, _, F_raw_sampled, _, _) = \
        opinf.pre.sampled_data(t, z_all, F_raw, Fdot, entries)
    (_, _, T_raw_sampled, _, _) = \
        opinf.pre.sampled_data(t, z_all, T_raw, Tdot, entries)

    t, entries = t_sampled, entries_sampled
    F_all, T_all = F_all_sampled, T_all_sampled
    Fdot, Tdot = Fdot_sampled, Tdot_sampled
    F_raw, T_raw = F_raw_sampled, T_raw_sampled
    z_all = z_T

    # Holdout for the hyperparameter search: the last part of the record is
    # hidden, the search is scored on the validation window before it.
    holdout = float(getattr(Params, 'holdout_ratio', 0.0))
    if holdout > 0.0:
        n_t = t.shape[0]
        if Params.split == 'condition':
            starts = np.concatenate(([0], np.where(np.diff(t) < 0)[0] + 1))
            n_keep = max(1, int(round((1.0 - holdout) * len(starts))))
            keep = int(starts[n_keep]) if n_keep < len(starts) else n_t
        else:
            keep = max(2, int(round((1.0 - holdout) * n_t)))
        if Params.output:
            print(f"Holding out the last {n_t - keep} of {n_t} samples.")
        t = t[:keep]
        F_all, T_all = F_all[:, :keep], T_all[:, :keep]
        F_raw, T_raw = F_raw[:, :keep], T_raw[:, :keep]
        entries = entries[:, :keep]
        Fdot, Tdot = Fdot[:, :keep], Tdot[:, :keep]

    # States without the inlet node, which is a boundary condition.
    F = F_all[1:, :]
    T = T_all[1:, :]
    z = z_all[1:]

    # Lifted variables (not used in the paper): kept as empty blocks.
    T_REF, Ea_R = 555.0, 10000.0
    w1_all = 1.0 / T_REF - 1.0 / np.maximum(T_all, 1e-6)
    w2_all = np.exp(Ea_R * w1_all)
    w1 = w1_all[1:, :] if Params.apply_lifting_1 else np.zeros_like(T)
    w2 = w2_all[1:, :] if Params.apply_lifting_2 else np.zeros_like(T)

    # =========================================================================
    # 4. TRAIN / TEST SPLIT
    # =========================================================================
    evaluation_results_list = []
    datasets = [F, T, entries, t, w1, w2]

    if Params.split == 'time':
        if Params.output:
            print(f"Splitting by time (train ratio {Params.train_ratio})")
        split_data = [opinf.pre.train_test_split_time(d, Params.train_ratio)
                      for d in datasets]
    elif Params.split == 'condition':
        split_points = np.where(np.diff(t) < 0)[0] + 1
        traj_starts = np.concatenate(([0], split_points))
        traj_ends = np.concatenate((split_points, [len(t)]))
        number_condition = len(traj_starts)
        num_training = int(np.round(Params.train_ratio * number_condition))

        rng = np.random.default_rng(0)
        all_indices = np.arange(number_condition)
        rng.shuffle(all_indices)
        train_indices_sorted = np.sort(all_indices[:num_training])
        test_indices_sorted = np.sort(all_indices[num_training:])
        if Params.output:
            print(f"Splitting by condition: {number_condition} trajectories, "
                  f"training on {train_indices_sorted}")

        split_data = [
            opinf.pre.train_test_split_conditions(
                d, traj_starts, traj_ends,
                train_indices_sorted, test_indices_sorted)
            for d in datasets]
        for i in range(number_condition):
            evaluation_results_list.append({
                'Case': i,
                'Status': 'Train' if i in train_indices_sorted else 'Test',
                'start_idx': traj_starts[i], 'end_idx': traj_ends[i]})
    else:
        raise ValueError(f"invalid Params.split: {Params.split}")

    ((F_train, F_test), (T_train, T_test),
     (entries_train, entries_test), (t_train, t_test),
     (w1_train, w1_test), (w2_train, w2_test)) = split_data

    # =========================================================================
    # 5. SCALING AND SHIFTING (Section 3.1)
    # =========================================================================
    if Params.input_scaling:
        input_scaling_factors = np.max(np.abs(entries_train), axis=1,
                                       keepdims=True)
        input_scaling_factors[input_scaling_factors == 0] = 1.0
        entries = entries / input_scaling_factors
        entries_train = entries_train / input_scaling_factors
        entries_test = entries_test / input_scaling_factors

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
        (F_train_ss, T_train_ss, w1_train_ss, w2_train_ss))

    max_F, max_T = 1.0, 1.0
    if Params.state_scaling == 'simple':
        max_F = (np.max(np.abs(F_train_shifted))
                 if F_train_shifted.size > 0 else 1.0)
        max_T = (np.max(np.abs(T_train_shifted))
                 if T_train_shifted.size > 0 else 1.0)
        max_F = max_F if max_F != 0 else 1.0
        max_T = max_T if max_T != 0 else 1.0
        F_train_shifted = F_train_shifted / max_F
        T_train_shifted = T_train_shifted / max_T

    Q_train_shifted = np.concatenate(
        (F_train_shifted, T_train_shifted, w1_train_shifted, w2_train_shifted))

    # =========================================================================
    # 6. POD BASIS (Section 3.2)
    # =========================================================================
    if F_train_shifted.shape[0] == 1:
        V_F_full = np.ones((1, 1))
        S_F = np.array([np.linalg.norm(F_train_shifted)])
    else:
        V_F_full, S_F = opinf.basis.pod(states=F_train_shifted, r="full",
                                        mode='dense')
    V_T_full, S_T = opinf.basis.pod(states=T_train_shifted, r="full",
                                    mode='dense')
    V_w1_full, S_w1 = opinf.basis.pod(states=w1_train_shifted, r="full",
                                      mode='dense')
    V_w2_full, S_w2 = opinf.basis.pod(states=w2_train_shifted, r="full",
                                      mode='dense')

    r_F = opinf.basis.cumulative_energy(
        S_F, Params.energy_threshold_single, name_tag='Flow F')
    if r_F == 0 and F_train_shifted.shape[0] > 0:
        r_F = 1
    r_T = opinf.basis.cumulative_energy(
        S_T, Params.energy_threshold_single, name_tag='Temperature T')
    r_w1 = 0 if np.all(w1_train_shifted == 0) else \
        opinf.basis.cumulative_energy(S_w1, Params.energy_threshold_single,
                                      name_tag='Lifting w1')
    r_w2 = 0 if np.all(w2_train_shifted == 0) else \
        opinf.basis.cumulative_energy(S_w2, Params.energy_threshold_single,
                                      name_tag='Lifting w2')
    Params.r_F = Params.r_X = r_F
    Params.r_T, Params.r_w1, Params.r_w2 = r_T, r_w1, r_w2
    Params.ROM_order = r_F + r_T + r_w1 + r_w2
    if Params.output:
        print(f"ROM order: {Params.ROM_order} (F = {r_F}, T = {r_T})")

    Q_reduced, V_reduced, V_reduced_nonlin, Xi = \
        opinf.basis.get_basis_and_reduced_data(
            V_F_full, V_T_full, V_w1_full, V_w2_full,
            Q_train_shifted, reference_states)

    if Params.output:
        V_F, V_T = V_F_full[:, :r_F], V_T_full[:, :r_T]
        if r_F > 0 and V_F.shape[0] > 1:
            opinf.utils.plot_POD_modes(z, V_F, 0, r_F, 'Flow F modes')
        if r_T > 0:
            opinf.utils.plot_POD_modes(z, V_T, 0, r_T, 'Temperature T modes')

    # =========================================================================
    # 7. TIME DERIVATIVES (derivative term and PINN only)
    # =========================================================================
    n_snapshots = Q_train_shifted.shape[1]
    dt = t_train[1] - t_train[0]
    Fdot_train = opinf.utils.estimate_derivatives(
        F_train_shifted, dt, entries_train, method=Params.ddt_method)
    Tdot_train = opinf.utils.estimate_derivatives(
        T_train_shifted, dt, entries_train, method=Params.ddt_method)
    w1dot_train = (opinf.utils.estimate_derivatives(
        w1_train_shifted, dt, entries_train, method=Params.ddt_method)
        if r_w1 > 0 else np.zeros_like(w1_train_shifted))
    w2dot_train = (opinf.utils.estimate_derivatives(
        w2_train_shifted, dt, entries_train, method=Params.ddt_method)
        if r_w2 > 0 else np.zeros_like(w2_train_shifted))
    Qdot_train = np.vstack((Fdot_train, Tdot_train, w1dot_train, w2dot_train))

    diff = V_reduced.shape[0] - Qdot_train.shape[0]
    if diff > 0:
        Qdot_train = np.vstack((Qdot_train, np.zeros((diff, n_snapshots))))
    elif diff < 0:
        Qdot_train = Qdot_train[:V_reduced.shape[0], :]
    Qdot_reduced = V_reduced.T @ Qdot_train

    # =========================================================================
    # 8. REDUCED SCALING
    # =========================================================================
    idx_F_end = r_F
    idx_T_end = idx_F_end + r_T
    idx_w1_end = idx_T_end + r_w1
    idx_w2_end = idx_w1_end + r_w2
    scaling_fac_F = scaling_fac_T = scaling_fac_w1 = scaling_fac_w2 = 1.0

    if Params.reduced_scaling:
        def scale_block(q, dq):
            if q.shape[0] > 0:
                sf, q_s = opinf.utils.scaled_states(q)
                return q_s, dq / sf, sf
            return q, dq, 1.0

        qF, dqF, scaling_fac_F = scale_block(
            Q_reduced[0:idx_F_end], Qdot_reduced[0:idx_F_end])
        qT, dqT, scaling_fac_T = scale_block(
            Q_reduced[idx_F_end:idx_T_end], Qdot_reduced[idx_F_end:idx_T_end])
        qw1, dqw1, scaling_fac_w1 = scale_block(
            Q_reduced[idx_T_end:idx_w1_end],
            Qdot_reduced[idx_T_end:idx_w1_end])
        qw2, dqw2, scaling_fac_w2 = scale_block(
            Q_reduced[idx_w1_end:idx_w2_end],
            Qdot_reduced[idx_w1_end:idx_w2_end])

        Q_list, Qdot_list = [], []
        for rank, q_blk, dq_blk in ((r_F, qF, dqF), (r_T, qT, dqT),
                                    (r_w1, qw1, dqw1), (r_w2, qw2, dqw2)):
            if rank > 0:
                Q_list.append(q_blk)
                Qdot_list.append(dq_blk)
        Q_reduced = np.concatenate(Q_list)
        Qdot_reduced = np.concatenate(Qdot_list)

    if Params.output:
        if r_F > 0:
            opinf.utils.plot_reduced_trajectories(
                t_train, Q_reduced, 0, r_F, 'Reduced: F')
        if r_T > 0:
            opinf.utils.plot_reduced_trajectories(
                t_train, Q_reduced, r_F, r_F + r_T, 'Reduced: T')

    # =========================================================================
    # 9. TRAINING (Section 3.2.4)
    # =========================================================================
    Params.input_dim = entries_train.shape[0]
    rom = opinf.models.create_rom(non_markov=False, seed=seed)
    # The rate bound on H needs the size of the states it acts on.
    opinf.models._nonparametric.set_state_scale(rom, Q_reduced)
    # Field blocks for the equal field weighting in the loss.
    opinf.training._model_inference.FIELD_SIZES = (r_F, r_T)

    if Params.use_PINN and PINN_MODE == "smoother":
        if Params.output:
            print("Training on the states of the PINN surrogate...")
        stages = (Params.PINN_stage_config_A, Params.PINN_stage_config_B)
        Params.PINN_stage_config_A, Params.PINN_stage_config_B = 1.0, 0.0
        try:
            _, history, _ = opinf.training.train_pinn_opinf(
                Q_reduced, t_train, entries_train, rom, r_F, seed=seed)
        finally:
            Params.PINN_stage_config_A, Params.PINN_stage_config_B = stages
        model, loss_track, error_reduced_states = opinf.training.train_model(
            history["surrogate_states"], history["surrogate_ddt"],
            t_train, entries_train, rom, seed=seed)
    elif Params.use_PINN:
        if Params.output:
            print("Training via PINN-OpInf...")
        model, history, error_reduced_states = opinf.training.train_pinn_opinf(
            Q_reduced, t_train, entries_train, rom, r_F, seed=seed)
    else:
        if Params.output:
            print("Training via gradient-based OpInf...")
        model, loss_track, error_reduced_states = opinf.training.train_model(
            Q_reduced, Qdot_reduced, t_train, entries_train, rom, seed=seed)

    # Read now: the memory-term fits below overwrite both.
    Params.last_ddt_residual = float(getattr(Params, 'ddt_residual', np.nan))
    Params.last_rollout = dict(getattr(Params, 'rollout', {}) or {})

    A_OpInf, B_OpInf, C_OpInf, H_OpInf = opinf.training.learned_model(model)

    # --- memory terms (Section 3.2.3), fitted lag by lag to the residual ---
    E_list, F_list = [], []
    if Params.lag_max > 0:
        if Params.output:
            print(f"Inferring memory terms (max lag {Params.lag_max})")
        resid = error_reduced_states.copy()
        orig_struct = Params.model_structure
        orig_loss = Params.ROM_loss
        orig_stability = Params.stability
        Params.model_structure = orig_struct.replace("H", "").replace("C", "")
        if LAG_FIT_ON_DERIVATIVE:
            Params.ROM_loss = 'ddt'
        if LAG_OPERATOR_STRUCTURE == "free":
            Params.stability = 'none'
        norm_prev = np.linalg.norm(resid, 'fro')
        try:
            for lag in range(1, Params.lag_max + 1):
                k = lag * LAG_STEP
                if resid.shape[1] <= k:
                    break
                x_l = Q_reduced[:, :-k]
                u_l = entries_train[:, :-k]
                rom_nm = opinf.models.create_rom(non_markov=True, seed=0)
                mod_l, _, _ = opinf.training.train_model(
                    x_l, resid[:, k:], t_train[k:], u_l, rom_nm, seed=0,
                    scale_reference="init")
                E_l, F_l, _, _ = opinf.training.learned_model(mod_l)
                E_list.append(E_l)
                F_list.append(F_l)
                resid[:, k:] -= (E_l @ x_l + F_l @ u_l)
                n_curr = np.linalg.norm(resid, 'fro')
                if (norm_prev - n_curr) / norm_prev < 1e-4 and lag > 1:
                    break
                norm_prev = n_curr
        finally:
            Params.model_structure = orig_struct
            Params.ROM_loss = orig_loss
            Params.stability = orig_stability

    # =========================================================================
    # 10. SIMULATE AND EVALUATE
    # =========================================================================
    model_ops = {"A": A_OpInf, "B": B_OpInf, "C": C_OpInf,
                 "H": H_OpInf, "E": E_list, "F": F_list}
    state_s_params = {
        'n_F': F.shape[0], 'n_T': T.shape[0], 'max_F': max_F, 'max_T': max_T,
        'F_scaled_from': None, 'F_scaled_to': None,
        'T_scaled_from': None, 'T_scaled_to': None,
    }
    recon_data = {
        "r_F": r_F, "r_T": r_T, "r_w1": r_w1, "r_w2": r_w2,
        "scaling_fac_F": scaling_fac_F, "scaling_fac_T": scaling_fac_T,
        "scaling_fac_w1": scaling_fac_w1, "scaling_fac_w2": scaling_fac_w2,
        "V_reduced": V_reduced, "V_reduced_nonlin": V_reduced_nonlin, "Xi": Xi,
        "F_train": F_train, "T_train": T_train,
        "reference_states": reference_states,
        "state_scaling_params": state_s_params, "dt": dt,
    }

    def initial_state(F0, T0, w10, w20, w1_shift=None, w2_shift=None):
        """Reduced initial state from measured fields at one instant."""
        f0, _ = opinf.pre.shift(F0, shift_by=shift_by_F)
        t0, _ = opinf.pre.shift(T0, shift_by=shift_by_T)
        w10, _ = opinf.pre.shift(w10, shift_by=w1_shift)
        w20, _ = opinf.pre.shift(w20, shift_by=w2_shift)
        if Params.state_scaling == 'simple':
            f0, t0 = f0 / max_F, t0 / max_T
        return opinf.utils.reduced_state(np.vstack((f0, t0, w10, w20)),
                                         V_reduced)

    results_storage = []
    rel_froerr_test = np.nan
    y0_gl = None
    cnn_pack = None

    # --- condition split: every held-out trajectory from its own start ---
    if Params.split == 'condition':
        rel_froerr_linear_all = []
        evaluate_train_cases = getattr(Params, 'evaluate_train_cases', False)
        for entry in evaluation_results_list:
            status = entry['Status']
            if (status == 'Train' and not evaluate_train_cases
                    and not Params.use_CNN):
                continue
            orig_output = Params.output
            if status == 'Train':
                Params.output = False
            try:
                start, end = entry['start_idx'], entry['end_idx']
                t_s = t[start:end] - t[start]
                u_s = entries[:, start:end]
                F_inn, T_inn = F[:, start:end], T[:, start:end]
                F_raw_c, T_raw_c = F_raw[:, start:end], T_raw[:, start:end]
                y0 = initial_state(F_inn[:, 0:1], T_inn[:, 0:1],
                                   w1[:, start:start + 1],
                                   w2[:, start:start + 1])
                sol_lin, sol_red = simulate_and_reconstruct(
                    y0, t_s, (0, t_s[-1]), u_s, model_ops, recon_data, Params)
                if sol_lin.ndim == 1:
                    sol_lin = sol_lin.reshape(-1, 1)
                    sol_red = sol_red.reshape(-1, 1)
                err_lin = opinf.post.run_postprocessing(
                    sol_lin, Params, F_inn, T_inn, z_all, t_s, r_F, r_T,
                    F_raw_c, T_raw_c, draw_split=False)
                results_storage.append({
                    'Case': entry['Case'], 'Status': status,
                    'sol_reduced': sol_red, 'F_true': F_inn, 'T_true': T_inn,
                    't': t[start:end], 'z': z_all,
                    'F_raw': F_raw_c, 'T_raw_c': T_raw_c,
                    'err_linear': err_lin})
                if status == 'Test':
                    rel_froerr_linear_all.append(err_lin)
                if Params.output:
                    print(f"Case {entry['Case']} ({status}): "
                          f"linear error {err_lin:.4e}")
            except Exception as exc:
                print(f"Case {entry['Case']} failed: "
                      f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            finally:
                Params.output = orig_output
        if rel_froerr_linear_all:
            rel_froerr_test = float(np.mean(rel_froerr_linear_all))

    # --- time split: test segment after the training record ---
    else:
        y0_gl = opinf.utils.reduced_state(
            Q_train_shifted[:, :len(E_list) + 1], V_reduced)
        sol_lin, sol_red = simulate_and_reconstruct(
            y0_gl, t, (np.min(t), np.max(t)), entries,
            model_ops, recon_data, Params)
        if sol_lin.ndim == 1:
            sol_lin = sol_lin.reshape(-1, 1)
            sol_red = sol_red.reshape(-1, 1)
        split_idx = t_train.shape[0]

        results_storage.append({
            'Status': 'Train', 'sol_reduced': sol_red[:, :split_idx],
            'F_true': F_train, 'T_true': T_train, 't': t[:split_idx],
            'z': z_all, 'F_raw': F_raw[:, :split_idx],
            'T_raw_c': T_raw[:, :split_idx], 'err_linear': 0.0})

        if split_idx < sol_lin.shape[1]:
            if Params.output:
                print("TEST SEGMENT, SIMULATED FROM THE START OF THE RECORD:")
            err_record = opinf.post.run_postprocessing(
                sol_lin[:, split_idx:], Params, F_test, T_test, z_all, t_test,
                r_F, r_T, F_raw[:, split_idx:], T_raw[:, split_idx:],
                draw_split=False)

            y0_seg = initial_state(F_test[:, 0:1], T_test[:, 0:1],
                                   w1_test[:, 0:1], w2_test[:, 0:1],
                                   w1_train_ss, w2_train_ss)
            t_seg = t[split_idx:]
            sol_seg, sol_red_seg = simulate_and_reconstruct(
                y0_seg, t_seg, (t_seg[0], t_seg[-1]), entries[:, split_idx:],
                model_ops, recon_data, Params)
            if sol_seg.ndim == 1:
                sol_seg = sol_seg.reshape(-1, 1)
                sol_red_seg = sol_red_seg.reshape(-1, 1)
            if Params.output:
                print("TEST SEGMENT, FORECAST FROM ITS OWN INITIAL STATE:")
            err_segment = opinf.post.run_postprocessing(
                sol_seg, Params, F_test, T_test, z_all, t_test,
                r_F, r_T, F_raw[:, split_idx:], T_raw[:, split_idx:],
                draw_split=False, plotting=False)

            Params.last_test_error_record = float(err_record)
            Params.last_test_error_segment = float(err_segment)
            from_segment = (TEST_FORECAST_FROM == "segment")
            rel_froerr_test = err_segment if from_segment else err_record

            if SAVE_PREDICTIONS_TO:
                os.makedirs(SAVE_PREDICTIONS_TO, exist_ok=True)
                rollout = getattr(Params, "last_rollout", {}) or {}
                np.savez_compressed(
                    os.path.join(SAVE_PREDICTIONS_TO,
                                 f"{PREDICTION_TAG or 'run'}_seed{seed}.npz"),
                    sol=np.asarray(sol_seg, dtype=np.float32),
                    F_raw=np.asarray(F_raw[:, split_idx:], dtype=np.float32),
                    T_raw=np.asarray(T_raw[:, split_idx:], dtype=np.float32),
                    z_all=np.asarray(z_all, dtype=float),
                    r_F=int(r_F), r_T=int(r_T),
                    F_true=np.asarray(F_test, dtype=np.float32),
                    T_true=np.asarray(T_test, dtype=np.float32),
                    t=np.asarray(t_test),
                    rollout_200=float(rollout.get(200, np.nan)),
                    error=float(err_segment), seed=int(seed))

            results_storage.append({
                'Status': 'Test',
                'sol_reduced': (sol_red_seg if from_segment
                                else sol_red[:, split_idx:]),
                'F_true': F_test, 'T_true': T_test, 't': t[split_idx:],
                'z': z_all, 'F_raw': F_raw[:, split_idx:],
                'T_raw_c': T_raw[:, split_idx:],
                'err_linear': rel_froerr_test})

        if Params.output:
            print("\nWHOLE RECORD:")
        Params.last_global_error = opinf.post.run_postprocessing(
            sol_lin, Params, F, T, z_all, t, r_F, r_T,
            F_raw, T_raw, draw_split=True, plotting=Params.output)

    # =========================================================================
    # 11. NEURAL DECODER (Section 3.2.2)
    # =========================================================================
    if Params.use_CNN and results_storage:
        train = [r for r in results_storage if r['Status'] == 'Train']
        if not train:
            print("Warning: no training simulations, cannot fit the decoder.")
        else:
            cnn_pack = opinf.post.train_global_cnn(
                [r['sol_reduced'] for r in train],
                [r['F_true'] for r in train],
                [r['T_true'] for r in train], seed)

            rel_froerr_cnn_all = []
            for res in results_storage:
                if res['Status'] == 'Train':
                    continue
                sol_cnn_phys = opinf.post.apply_cnn_inference(
                    res['sol_reduced'], cnn_pack)
                err_cnn = opinf.post.run_postprocessing(
                    sol_cnn_phys, Params, res['F_true'], res['T_true'],
                    res['z'], res['t'], r_F, r_T,
                    res['F_raw'], res['T_raw_c'], draw_split=False)
                rel_froerr_cnn_all.append(err_cnn)

                # The decoder forecast next to the linear one in the saved file.
                if SAVE_PREDICTIONS_TO and Params.split == 'time':
                    path = os.path.join(
                        SAVE_PREDICTIONS_TO,
                        f"{PREDICTION_TAG or 'run'}_seed{seed}.npz")
                    if os.path.exists(path):
                        with np.load(path) as f:
                            saved = {k: f[k] for k in f.files}
                        dec = np.asarray(sol_cnn_phys, dtype=np.float32)
                        if dec.shape == saved["sol"].shape:
                            saved["sol_decoder"] = dec
                            np.savez_compressed(path, **saved)
                        else:
                            print(f"  decoder output {dec.shape} does not "
                                  f"match the saved forecast "
                                  f"{saved['sol'].shape}; not saved")
                if Params.output:
                    print(f"-> linear {res['err_linear']:.4e} | "
                          f"decoder {err_cnn:.4e}")

            if rel_froerr_cnn_all:
                mean_cnn = float(np.mean(rel_froerr_cnn_all))
                if Params.output:
                    print(f"\nTest error, linear:  {rel_froerr_test:.4e}")
                    print(f"Test error, decoder: {mean_cnn:.4e}")
                rel_froerr_test = mean_cnn

            if Params.output:
                ordered = sorted(results_storage, key=lambda r: r['t'][0])
                opinf.post.run_postprocessing(
                    np.hstack([opinf.post.apply_cnn_inference(
                        r['sol_reduced'], cnn_pack) for r in ordered]),
                    Params,
                    np.hstack([r['F_true'] for r in ordered]),
                    np.hstack([r['T_true'] for r in ordered]),
                    z_all, np.concatenate([r['t'] for r in ordered]),
                    r_F, r_T,
                    np.hstack([r['F_raw'] for r in ordered]),
                    np.hstack([r['T_raw_c'] for r in ordered]),
                    draw_split=True)

    # =========================================================================
    # 12. SAVE THE MODEL
    # =========================================================================
    if Params.save_results:
        if y0_gl is not None:
            y0_s = y0_gl
        elif Q_reduced.shape[1] > 0:
            y0_s = Q_reduced[:, 0]
        else:
            y0_s = np.zeros(Params.ROM_order)
        y0_s = np.asarray(y0_s).reshape(-1)
        if y0_s.shape[0] != Params.ROM_order:
            y0_fix = np.zeros(Params.ROM_order)
            n = min(y0_s.shape[0], Params.ROM_order)
            y0_fix[:n] = y0_s[:n]
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
            "state_scaling_params": state_s_params,
            # Column order of B and the factors the inputs were divided by.
            "input_names": np.asarray(list(Params.input_names)),
            "input_scaling_factors": (
                np.asarray(input_scaling_factors, dtype=float).ravel()
                if Params.input_scaling
                else np.ones(len(Params.input_names))),
        }
        save_dir = opinf.utils.save_results(
            model=model_dict, Params=Params, save_dir=None,
            cnn_pack=cnn_pack, seed=seed,
            metrics={
                "test_combined": float(rel_froerr_test),
                "global_combined": float(
                    getattr(Params, "last_global_error", np.nan)),
                "markovian_ddt_residual": float(
                    getattr(Params, "last_ddt_residual", np.nan)),
                "use_CNN": bool(Params.use_CNN),
                "use_PINN": bool(Params.use_PINN),
                "markovian_rollout": {
                    str(h): float(v) for h, v
                    in sorted(getattr(Params, "last_rollout", {}).items())},
            })
        if Params.output:
            print(f"Saved to: {save_dir}")

    return rel_froerr_test


if __name__ == "__main__":
    print(f"Used device: {device}")
    params = opinf.parameters.Params()
    params.output = True
    params.save_results = False
    tic = time.time()
    err_test = main(params, seed=1)
    print("runtime in seconds:", time.time() - tic)
