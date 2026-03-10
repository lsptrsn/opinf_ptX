#!/usr/bin/env python3
"""
Optimal Control for Reduced-Order Chemical Reactor Model (OpInf-ROM).
Description:
    Solves an open-loop optimal control problem (OCP) using CasADi and Ipopt.
    The goal is to maximize outlet conversion while adhering to spatial
    temperature constraints (hotspot prevention).
    Supports Multi-Fidelity optimization using an iterative back-off
    strategy to bridge the gap between linear ROM predictions and CNN reality.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import casadi as ca
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PhysicalConstraints, OptimizationConfig, get_ipopt_options
from utils import (
    get_device,
    load_results,
    create_cnn_casadi_function,
    reduced_to_full_casadi,
    reduced_to_full_numpy,
    forward_sim_reduced,
    check_dynamics_residuals,
    setup_results_dir,
    create_summary_plot
)
from models.conv_decoder import ConvDecoder

# =============================================================================
# GLOBAL SETTINGS & REPRODUCIBILITY
# =============================================================================

MODEL_ID = 'model_202601080938'
CONTROL_SCENARIO = 'control_scenario_0'
DECODER_CLASS = ConvDecoder

HOTSPOT_SAMPLE_RATE = 1
DATA_SAMPLE_RATE = 10
SKIP_TRANSIENT_S = 600  # Slicing to skip initial startup behavior

USE_MULTI_FIDELITY = True
MAX_LOOPS = 20

def set_random_seeds(seed: int = 0):
    """Set random seeds for system-wide reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# CASADI MODEL BUILDING
# =============================================================================

def build_casadi_integrator(A, H, B, C, n_red: int, dt: float) -> ca.Function:
    """
    Builds a CasADi integrator for reduced-order dynamics using CVODES.
    Handles the quadratic term H[x ⊗ x] via symbolic Kronecker expansion.
    """
    x_sym = ca.SX.sym('x', n_red)
    p_sym = ca.SX.sym('p', 3)  # Parameters: [load, coolant_temp, inlet_temp]

    # Efficient symbolic Kronecker product
    kron_list = [x_sym[i] * x_sym[j] for i in range(n_red) for j in range(n_red)]
    kron_xx = ca.vertcat(*kron_list)

    A_c = ca.DM(A)
    B_c = ca.DM(B)
    C_c = ca.DM(np.asarray(C).reshape((n_red, 1)))
    H_c = ca.DM(H.reshape((n_red, n_red * n_red)))

    # ODE Expression: dx/dt = Ax + H(x⊗x) + C + Bu
    f_expr = A_c @ x_sym + H_c @ kron_xx + C_c + B_c @ p_sym

    intg_opts = {'abstol': 1e-4, 'reltol': 1e-4, 'max_num_steps': 20000}
    integrator = ca.integrator(
        'integrator', 'cvodes',
        {'x': x_sym, 'p': p_sym, 'ode': f_expr}, 0, dt, intg_opts
    )
    return integrator


def setup_optimization_problem(
    n_red: int, Nt: int, N: int, y0: np.ndarray, s_reduce_np: np.ndarray,
    u_min_scaled: float, u_max_scaled: float,
    ramp_up_scaled: float, ramp_down_scaled: float, dt: float
):
    """
    Sets up the CasADi Opti stack with decision variables and
    asymmetric control rate constraints.
    """
    opti = ca.Opti()

    # Decision variables
    Xs = opti.variable(n_red, Nt) # Reduced states
    U = opti.variable(1, N)       # Control inputs (Coolant Temp)
    S = opti.variable(1, Nt)      # Slack variables for soft constraints

    # Initial condition
    Xs0 = (y0 / s_reduce_np).flatten()
    opti.subject_to(Xs[:, 0] == ca.DM(Xs0))

    # Control and Slack bounds
    opti.subject_to(opti.bounded(u_min_scaled, U, u_max_scaled))
    opti.subject_to(S >= 0)

    # Vectorized Asymmetric Control Rate Constraints
    delta_U = U[:, 1:] - U[:, :-1]
    opti.subject_to(delta_U <= ramp_up_scaled)    # Limit heating rate
    opti.subject_to(delta_U >= -ramp_down_scaled) # Limit cooling rate

    return opti, Xs, U, S


def add_dynamics_constraints(opti, Xs, U, integrator, time_arr, load_interp, inlet_interp, N):
    """Adds equality constraints for the ROM dynamics across the horizon."""
    for k in range(N):
        p_k = ca.vertcat(float(load_interp(time_arr[k])), U[:, k], float(inlet_interp(time_arr[k])))
        xf_scaled = integrator(x0=Xs[:, k], p=p_k)['xf']
        opti.subject_to(xf_scaled == Xs[:, k + 1])


def add_temperature_constraints(opti, Xs, S, reduced_to_full_func, T_hot_limit, Nt, n_F_full, sample_rate=1):
    """Adds spatial hotspot constraints. T_hot_limit can be scalar or a symbolic vector."""
    check_indices = range(0, Nt, sample_rate)
    for k in check_indices:
        full_k = reduced_to_full_func(Xs[:, k])
        T_state = full_k[n_F_full:] # Extract temperature part of state
        T_hot_k = ca.mmax(T_state)

        # Handle scalar limit vs. trajectory-based parameter vector
        limit_k = T_hot_limit if isinstance(T_hot_limit, (float, int)) else T_hot_limit[0, k]
        opti.subject_to(T_hot_k <= limit_k + S[0, k])


def build_objective(Xs, U, S, reduced_to_full_func, n_F_full, N, dt, alpha_u, slack_penalty, u_start_val=None):
    """Builds the composite objective: Maximize conversion + Smoothness + Slack + Soft-Start."""
    cost = 0
    T_total = N * dt

    # 1. Maximize Average Outlet Conversion
    cumulative_conversion = 0
    for k in range(N):
        full_next = reduced_to_full_func(Xs[:, k + 1])
        conv_spatial_mean = ca.sum1(full_next[:n_F_full, 0]) / n_F_full
        cumulative_conversion += conv_spatial_mean * dt
    cost += -(cumulative_conversion / T_total)

    # 2. Control Smoothness (Regularization)
    if alpha_u > 0:
        cost += (alpha_u / N) * ca.sumsqr(U[:, 1:] - U[:, :-1])

    # 3. Slack Penalty (Soft Constraints)
    if slack_penalty > 0:
        cost += (slack_penalty / N) * ca.sumsqr(S)

    # 4. Soft Start Penalty (Match existing initial state)
    if u_start_val is not None:
        cost += 1e5 * ca.sumsqr(U[:, 0] - u_start_val)

    return cost

# =============================================================================
# MAIN OPTIMIZATION ROUTINE
# =============================================================================



def main():
    set_random_seeds(0)
    device = get_device()

    constraints = PhysicalConstraints()
    opt_config = OptimizationConfig(
        alpha_u=100.0, slack_penalty=1e5,
        hotspot_sample_rate=HOTSPOT_SAMPLE_RATE,
        time_sampling_stride=DATA_SAMPLE_RATE
    )

    DATA_DIR = Path("data") / MODEL_ID
    CONTROL_DIR = Path("data") / CONTROL_SCENARIO
    RESULTS_DIR = setup_results_dir("results")

    # IPOPT Options
    ipopt_opts = get_ipopt_options(hessian_approximation='limited-memory', linear_solver='ma57', tolerance=1e-4)
    ipopt_opts['ipopt.mu_strategy'] = 'adaptive'
    ipopt_opts['ipopt.max_wall_time'] = 86400.0

    # Load Model Results
    results = load_results(DATA_DIR, decoder_class=DECODER_CLASS)
    A, B, C, H = results["A"], results["B"], results["C"], results["H"]
    basis, ref_states_shifting = results["basis"], results["reference_states_shifting"]

    sF, sT = float(results["scaling_fac_F"]), float(results["scaling_fac_T"])
    max_F, max_T = float(results["max_F"]), float(results["max_T"])
    scale_load, scale_Tcool = float(results["input_scaling_factors"][0]), float(results["input_scaling_factors"][1])

    n_red, n_full = int(A.shape[0]), int(basis.shape[0])
    n_F_full = n_full // 2 # Assumes [Flow, Temp] stacking

    # Scenario Loading & Slicing
    scenario_id = CONTROL_SCENARIO.split('_')[-1]
    time_raw = np.load(CONTROL_DIR / f"time_control_{scenario_id}.npy")
    load_raw = np.load(CONTROL_DIR / f"load_control_{scenario_id}.npy").flatten()
    Tcool_raw = np.load(CONTROL_DIR / f"cooling_temperature_control_{scenario_id}.npy").flatten()

    start_idx = np.where(time_raw >= SKIP_TRANSIENT_S)[0][0] if SKIP_TRANSIENT_S > 0 else 0
    time_raw, load_raw, Tcool_raw = time_raw[start_idx:] - time_raw[start_idx], load_raw[start_idx:], Tcool_raw[start_idx:]

    # Downsampling for Optimization
    stride = opt_config.time_sampling_stride
    time_opt = time_raw[::stride]
    dt_opt = float(time_opt[1] - time_opt[0])
    Nt, N = len(time_opt), len(time_opt) - 1

    # Interpolators for dynamics
    load_interp = interp1d(time_raw, load_raw / scale_load, kind='linear', fill_value="extrapolate")
    inlet_interp = interp1d(time_raw, np.zeros_like(load_raw), kind='linear', fill_value="extrapolate")

    # Constraint Scaling
    u_min_scaled, u_max_scaled = constraints.u_min / scale_Tcool, constraints.u_max / scale_Tcool
    ramp_up_scaled = (constraints.ramp_up_K_per_s / scale_Tcool) * dt_opt
    ramp_down_scaled = (constraints.ramp_down_K_per_s / scale_Tcool) * dt_opt

    # --- Initial Condition y0 ---
    F_phys_0 = np.load(CONTROL_DIR / f"conversion_opinf_control_{scenario_id}.npy")[1:, 0, start_idx].reshape(-1, 1)
    T_phys_0 = np.load(CONTROL_DIR / f"center_temperature_opinf_control_{scenario_id}.npy")[1:, 0, start_idx].reshape(-1, 1)

    f0, t0 = (F_phys_0 - ref_states_shifting.flatten()[:n_F_full].reshape(-1,1))/max_F, (T_phys_0 - ref_states_shifting.flatten()[n_F_full:2*n_F_full].reshape(-1,1))/max_T
    y0 = (basis.T @ np.vstack((f0, t0, np.zeros_like(f0), np.zeros_like(f0)))).flatten()
    y0 /= np.concatenate([np.full(results["r_F"], sF), np.full(results["r_T"], sT)])

    # --- Setup Multi-Fidelity Iteration ---
    print("\n" + "="*70 + "\n8. STARTING MULTI-FIDELITY OPTIMIZATION\n" + "="*70)

    integrator = build_casadi_integrator(A, H, B, C, n_red, dt_opt)

    # Define Wrappers
    def lin_wrapper(x): return reduced_to_full_casadi(x, results["basis_type"], results["r_F"], results["r_T"], sF, sT, basis, ref_states_shifting, max_F, max_T, results.get("V_reduced_nonlin"), results.get("Xi"), has_decoder=False, n_full=n_full, n_F_full=n_F_full)
    def cnn_check(x): return reduced_to_full_numpy(x, results["basis_type"], results["r_F"], results["r_T"], sF, sT, basis, ref_states_shifting, max_F, max_T, results.get("V_reduced_nonlin"), results.get("Xi"), has_decoder=True, decoder=results.get("decoder").to(device).eval(), scalers=(results.get("input_scaler"), results.get("target_scaler_F"), results.get("target_scaler_T")), n_F_full=n_F_full, device=device)

    # Build Graph once
    opti, Xs, U, S = setup_optimization_problem(n_red, Nt, N, y0, np.ones(n_red), u_min_scaled, u_max_scaled, ramp_up_scaled, ramp_down_scaled, dt_opt)
    p_T_limit = opti.parameter(1, Nt)

    add_dynamics_constraints(opti, Xs, U, integrator, time_opt, load_interp, inlet_interp, N)
    add_temperature_constraints(opti, Xs, S, lin_wrapper, p_T_limit, Nt, n_F_full, sample_rate=HOTSPOT_SAMPLE_RATE)
    opti.minimize(build_objective(Xs, U, S, lin_wrapper, n_F_full, N, dt_opt, opt_config.alpha_u, opt_config.slack_penalty, u_start_val=Tcool_raw[0]/scale_Tcool))
    opti.solver('ipopt', ipopt_opts)

    # Iteration Loop
    T_abs_limit = constraints.T_hot_max
    safety_margin = np.full((1, Nt), 2.0)
    current_limit_profile = T_abs_limit - safety_margin

    best_sol = {'conversion': -1.0, 'violation': 999.0, 'Xs': None, 'U': None, 'S': None, 'cnn_traj': None}
    polishing_active = False

    curr_Xs, curr_U, curr_S = None, None, None

    for loop in range(MAX_LOOPS):
        print(f"\n[LOOP {loop+1}] Limit Range: {np.min(current_limit_profile):.1f} - {np.max(current_limit_profile):.1f} K")

        opti.set_value(p_T_limit, current_limit_profile)
        if curr_Xs is not None:
            opti.set_initial(Xs, curr_Xs); opti.set_initial(U, curr_U); opti.set_initial(S, curr_S)

        try:
            sol = opti.solve()
            curr_Xs, curr_U, curr_S = sol.value(Xs), sol.value(U), sol.value(S)
        except:
            sol = opti.debug
            curr_Xs, curr_U, curr_S = sol.value(Xs), sol.value(U), sol.value(S)

        # Analysis
        sol_cnn = cnn_check(curr_Xs)
        T_hot_cnn = np.max(sol_cnn[n_F_full:2*n_F_full, :], axis=0).reshape(1, -1)
        T_hot_lin = np.max(lin_wrapper(curr_Xs), axis=0).reshape(1, -1) if isinstance(lin_wrapper(curr_Xs), np.ndarray) else np.max(ca.evalf(lin_wrapper(curr_Xs)), axis=0).reshape(1, -1)

        mismatch = T_hot_cnn - T_hot_lin
        violation = np.max(T_hot_cnn - T_abs_limit)
        conv = np.mean(sol_cnn[:n_F_full, -1])

        print(f"  Violation: {violation:.2f} K | Conversion: {conv:.4f}")

        # Update Best
        if violation < best_sol['violation'] - 0.2 or (violation <= 1.0 and conv > best_sol['conversion']):
            best_sol.update({'conversion': conv, 'violation': violation, 'Xs': curr_Xs, 'U': curr_U, 'S': curr_S, 'cnn_traj': sol_cnn})

        # State Machine & Back-off logic
        if not polishing_active and violation <= 0.5:
            print("  >> STABLE. Entering Polishing Phase."); polishing_active = True

        # Adaptive Profile Update
        smoothed_mismatch = gaussian_filter1d(mismatch, sigma=5)
        step = np.zeros_like(smoothed_mismatch)

        # Tighten where CNN is hotter than OpInf
        step[smoothed_mismatch > 0] = 1.0 if not (polishing_active and violation < 0.2) else 0.0
        # Relax only during polishing and if safe
        if polishing_active and violation < 0.8:
            step[smoothed_mismatch < 0] = np.where(smoothed_mismatch < -2.0, 0.2, 0.05)

        safety_margin += step * smoothed_mismatch
        current_limit_profile = np.clip(T_abs_limit - safety_margin, T_abs_limit-60, T_abs_limit+25)

        if polishing_active and (violation > 2.0 or loop > MAX_LOOPS - 5): break

    # --- Finalize ---
    print(f"\nOptimization Finalized. Best Conversion: {best_sol['conversion']:.4f}")

    # Save & Plot
    np.save(RESULTS_DIR / 'u_opt.npy', best_sol['U'] * scale_Tcool)
    np.save(RESULTS_DIR / 'sol_full_cnn.npy', best_sol['cnn_traj'])
    create_summary_plot(time_opt, best_sol['U']*scale_Tcool, np.max(best_sol['cnn_traj'][n_F_full:2*n_F_full, :], axis=0), best_sol['cnn_traj'][:n_F_full, -1], load_raw[::stride], T_abs_limit, constraints.u_min, constraints.u_max, save_path=RESULTS_DIR/'summary.png')

if __name__ == "__main__":
    main()
