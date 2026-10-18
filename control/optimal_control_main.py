"""
Optimal Control for Reduced-Order Chemical Reactor Model.

This script solves an open-loop optimal control problem to maximize
outlet conversion while respecting temperature constraints.

Variants are selected through environment variables, so that several can be
run side by side and compared from their summary files:

    OCP_MODEL     model folder under data/            (default below)
    OCP_STRATEGY  multi_fidelity | decoder | linear_margin | modifier
    OCP_ENVELOPE  margin of the state envelope as a fraction of the training
                  range of each reduced coordinate, or "off"
    OCP_CONV_CAP  upper bound on the outlet conversion, or "off"
    OCP_MARGIN_K  back-off of the temperature limit in kelvin for
                  linear_margin, modifier and slp (slp: the decoder is held
                  to the limit minus this back-off)
    OCP_DAMPING   relaxation of the modifier update (modifier only)
    OCP_AGREE_Q   quantile of the training disagreement between decoder and
                  linear reconstruction that bounds it in the optimum (slp),
                  or "off"
    OCP_TR0       initial trust radius as a fraction of the training range of
                  each reduced coordinate (slp)
    OCP_MAX_ITER  maximum number of linearizations (slp)
    OCP_ENSEMBLE  further model folders, comma-separated (ensemble)
    OCP_TRAIN_TRAJ  "all" when the model was trained on all ten trajectories
    OCP_TAG       name of the result subfolder (results/<tag>)
"""

import sys
import random
import os
import json
import time
from pathlib import Path
import numpy as np
import casadi as ca
import torch
from scipy.interpolate import interp1d
import dataclasses

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PhysicalConstraints, OptimizationConfig, get_ipopt_options

# The pipeline that trained the model supplies the decoder and its
# preprocessing, so that the control problem applies exactly the network
# that was trained. The device is fixed before the package is imported,
# because the package reads it at import time; for single latent vectors
# the CPU is faster than a GPU anyway.
os.environ.setdefault("OPINF_DEVICE", "cpu")
PIPELINE_DIR = Path(os.environ.get(
    "OPINF_PIPELINE_DIR",
    Path(__file__).resolve().parent.parent / "identification")).expanduser()
# Appended, not inserted: the control code has its own utils package, which
# must not be shadowed by anything in the pipeline folder.
sys.path.append(str(PIPELINE_DIR))

from utils import (
    get_device,
    load_results,
    create_cnn_casadi_function,
    reduced_to_full_casadi,
    reduced_to_full_numpy,
    forward_sim_reduced,
    check_dynamics_residuals,
    create_summary_plot,
    setup_results_dir,
)
from opinf.post._decoder import ConvDecoder


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

def _env_float(name, default):
    """A float from the environment; "off", "none" or empty give default."""
    value = os.environ.get(name)
    if value is None or value.strip().lower() in ("", "off", "none"):
        return default
    return float(value)


file_name = os.environ.get("OCP_MODEL", 'model_202610040028')
control_name = 'control_scenario_0'
decoder_class = ConvDecoder
hotspot_sample_rate = 1
data_sample_rate = 100
skip_first_seconds = 600 # Slicing to skip transient start

# How the temperature limit is enforced.
#   "multi_fidelity" : linear reconstruction in the constraints, the decoder
#                      as the check, limits tightened locally in a loop.
#   "decoder"        : the decoder itself in constraints and objective, in
#                      one pass.
#   "linear_margin"  : linear reconstruction in one pass, the limit lowered
#                      by OCP_MARGIN_K.
#   "modifier"       : linear reconstruction in the solver, corrected node by
#                      node and time point by time point by the difference
#                      between decoder and linear reconstruction along the
#                      previous solution; repeated until the decoder meets
#                      the limit (zeroth-order modifier adaptation). The
#                      decoder then decides where the hot zone is, which the
#                      back-off on the maximum cannot do, and the solver
#                      keeps the smooth linear constraints.
#   "slp"            : successive linearization of the decoder with a trust
#                      region. At every iterate the decoder's temperatures and
#                      outlet conversion are linearized in the reduced state
#                      (central differences), the problem is solved with these
#                      linear constraints and objective within a trust region,
#                      and the step is accepted and the radius adapted by the
#                      ratio of actual to predicted improvement. Optionally the
#                      disagreement between decoder and linear reconstruction
#                      is held within the range it has on the training data,
#                      which keeps the solution where the two independently
#                      built reconstructions agree.
#   "ensemble"       : linear reconstruction, lowered limit, several models at
#                      once; constraints for every model, mean conversion over
#                      the models.
STRATEGY = os.environ.get("OCP_STRATEGY", "multi_fidelity")
if STRATEGY not in ("multi_fidelity", "decoder", "linear_margin",
                    "modifier", "slp", "ensemble"):
    raise ValueError(f"unknown OCP_STRATEGY {STRATEGY!r}")
USE_MULTI_FIDELITY = (STRATEGY == "multi_fidelity")
MAX_LOOPS = 20

# Coverage of the reduced state. Every reduced coordinate is kept within the
# range it covered on the training trajectories, widened by this fraction of
# that range. A reduced model is identified on the states it was trained on;
# a new combination of feed and coolant can drive it into states it never
# saw even when each input stays within its training range, and there it
# extrapolates (on the synthetic model up to a hotspot of 1292 K and an
# outlet conversion of 2.7 at a coolant temperature inside the training
# range). None switches the envelope off.
ENVELOPE_MARGIN = _env_float("OCP_ENVELOPE", None)

# Upper bound on the outlet conversion in the optimisation model. The
# objective rewards conversion, and where the model extrapolates it promises
# conversions above one, which the optimiser then pursues. None switches the
# bound off.
CONVERSION_CAP = _env_float("OCP_CONV_CAP", None)

# Back-off of the temperature limit for STRATEGY = "linear_margin", in K.
MARGIN_K = _env_float("OCP_MARGIN_K", 20.0)

# Modifier adaptation: relaxation of the offset update (1 = take the new
# offsets as they are), maximum number of solves, and the decoder violation
# below which a solution counts as feasible.
MODIFIER_DAMPING = _env_float("OCP_DAMPING", 0.7)
MODIFIER_MAX_ITER = 15
MODIFIER_TOL_K = 0.5

# Successive linearization (slp).
SLP_MAX_ITER = int(_env_float("OCP_MAX_ITER", 25))
# "off" switches the agreement bound off. _env_float maps "off" to its
# default, which here is 0.99, so the switch is read explicitly.
AGREE_QUANTILE = (None if os.environ.get("OCP_AGREE_Q", "").strip().lower()
                  in ("off", "none") else _env_float("OCP_AGREE_Q", 0.99))
AGREE_STRIDE = 4          # every 4th node in the agreement constraint
TR_INIT = _env_float("OCP_TR0", 0.10)
# Merit of an iterate: mean outlet conversion minus SLP_RHO per kelvin of
# violation. It has to outweigh the gain in conversion per kelvin of
# hotspot (about 0.004 here), or the iterates settle slightly outside the
# limit; on a test problem a weight five times the sensitivity converged
# to the exact constrained optimum in ten iterations, a weight of a tenth
# of it not at all.
SLP_RHO = _env_float("OCP_RHO", 0.05)
# Start of the linearization: "linear" solves the problem with the linear
# reconstruction and a limit lowered by SLP_START_MARGIN_K first, which gives
# a point consistent with the coarse dynamics and close to the optimum;
# "cold" starts from the forward simulation at minimum coolant temperature.
SLP_START = os.environ.get("OCP_SLP_START", "linear")
SLP_START_MARGIN_K = _env_float("OCP_SLP_START_MARGIN", 20.0)

# Ensemble: further model folders under data/.
ENSEMBLE_MODELS = [m for m in os.environ.get("OCP_ENSEMBLE", "").split(",")
                   if m.strip()]

RUN_TAG = os.environ.get("OCP_TAG", "")

# Trajectories of the synthetic record the model was trained on (the
# condition split of the pipeline; seven of ten, the others held out). The
# envelope is taken from these only.
TRAIN_DATA_DIR = PIPELINE_DIR / "data" / "smooth_aprbs"
TRAIN_TRAJECTORIES = (tuple(range(10))
                      if os.environ.get("OCP_TRAIN_TRAJ", "") == "all"
                      else (2, 3, 4, 5, 6, 7, 9))
TRAIN_STRIDE = 10


def set_random_seeds(seed: int = 0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_casadi_integrator(A, H, B, C, n_red: int, dt: float) -> ca.Function:
    """Build CasADi integrator for reduced dynamics (CVODES)."""
    # Symbolic variables
    x_sym = ca.SX.sym('x', n_red)
    p_sym = ca.SX.sym('p', np.asarray(B).shape[1])  # inputs in B's order

    # Kronecker product: kron(x, x)
    kron_list = [x_sym[i] * x_sym[j] for i in range(n_red) for j in range(n_red)]
    kron_xx = ca.vertcat(*kron_list)

    # Convert to CasADi
    A_c = ca.DM(A)
    B_c = ca.DM(B)
    C_c = ca.DM(np.asarray(C).reshape((n_red, 1)))
    H_c = ca.DM(H.reshape((n_red, n_red * n_red)))

    # Dynamics expression
    f_expr = A_c @ x_sym + H_c @ kron_xx + C_c + B_c @ p_sym

    # Create integrator
    intg_opts = {'abstol': 1e-4, 'reltol': 1e-4, 'max_num_steps': 20000}
    integrator = ca.integrator(
        'integrator', 'cvodes',
        {'x': x_sym, 'p': p_sym, 'ode': f_expr}, 0, dt, intg_opts
    )

    return integrator


def setup_optimization_problem(
    n_red: int, Nt: int, N: int, y0: np.ndarray, s_reduce_np: np.ndarray,
    u_min_scaled: float, u_max_scaled: float,
    ramp_up_scaled: float,
    ramp_down_scaled: float,
    dt: float
):
    """Setup CasADi optimization problem with vectorized asymmetric constraints."""
    opti = ca.Opti()

    # === Decision variables ===
    Xs = opti.variable(n_red, Nt)
    U = opti.variable(1, N)
    S = opti.variable(1, Nt)

    # === Initial condition ===
    Xs0 = (y0 / s_reduce_np).flatten()
    opti.subject_to(Xs[:, 0] == ca.DM(Xs0))

    # === Control bounds ===
    opti.subject_to(opti.bounded(u_min_scaled, U, u_max_scaled))

    # === Slack bounds ===
    opti.subject_to(S >= 0)

    # === Control rate constraints (Vectorized) ===
    delta_U = U[:, 1:] - U[:, :-1]

    # 1. Limit Heating (Upper bound for changes)
    opti.subject_to(delta_U <= ramp_up_scaled)

    # 2. Limit Cooling (Lower bound for changes)
    opti.subject_to(delta_U >= -ramp_down_scaled)

    return opti, Xs, U, S


def add_dynamics_constraints(
    opti: ca.Opti, Xs: ca.MX, U: ca.MX, integrator: ca.Function,
    time_arr: np.ndarray, load_interp, input_vector, N: int
):
    """Add the dynamics as equality constraints, one integrator step each."""
    print("Adding dynamics constraints...")
    for k in range(N):
        load_k = float(load_interp(time_arr[k]))
        p_k = input_vector(load_k, U[0, k])
        xf_scaled = integrator(x0=Xs[:, k], p=p_k)['xf']
        opti.subject_to(xf_scaled == Xs[:, k + 1])
    print(f"  Added {N} dynamics constraints")


def add_state_envelope(opti: ca.Opti, Xs: ca.MX, lower: np.ndarray,
                       upper: np.ndarray, Nt: int):
    """Keep every reduced coordinate within its envelope after the start.

    The initial state is fixed by the measurement and is left out, so that a
    start slightly outside the envelope cannot make the problem infeasible.
    """
    lo = ca.repmat(ca.DM(np.asarray(lower, dtype=float).reshape(-1, 1)),
                   1, Nt - 1)
    hi = ca.repmat(ca.DM(np.asarray(upper, dtype=float).reshape(-1, 1)),
                   1, Nt - 1)
    opti.subject_to(opti.bounded(lo, Xs[:, 1:], hi))
    print(f"  Added the state envelope on {Xs.shape[0]} reduced coordinates "
          f"at {Nt - 1} time points")


def add_conversion_cap(opti: ca.Opti, Xs: ca.MX, reduced_to_full_func,
                       Nt: int, n_F_full: int, cap: float):
    """Bound the outlet conversion of the optimisation model from above."""
    for k in range(1, Nt):
        full_k = reduced_to_full_func(Xs[:, k])
        if full_k.shape[1] != 1:
            full_k = ca.reshape(full_k, (-1, 1))
        opti.subject_to(full_k[n_F_full - 1, 0] <= cap)
    print(f"  Added the outlet conversion cap {cap:.3f} at {Nt - 1} time "
          f"points")


def add_temperature_constraints(
    opti: ca.Opti, Xs: ca.MX, S: ca.MX, reduced_to_full_func,
    T_hot_max_input, Nt: int, n_F_full: int, sample_rate: int = 1,
    hotspot_nodes: slice = slice(None),
):
    """Keep every temperature node below the limit, softened by the slack.

    One inequality per node instead of one on the maximum. With the linear
    reconstruction each node temperature is linear in the reduced state, so
    the constraints are smooth and exact; the maximum over the nodes
    (ca.mmax) is neither, its gradient following only the node that happens
    to be hottest, which is what IPOPT converges poorly on. ``hotspot_nodes``
    restricts the nodes, for instance to the part of the reactor where the
    hotspot can occur, if the number of constraints becomes a burden.

    Scalar limits and a (1, Nt) parameter of limits per time point are both
    accepted.
    """
    sr = sample_rate[0] if isinstance(sample_rate, (list, tuple)) else sample_rate
    check_indices = range(0, Nt, sr)
    print(f"Adding temperature constraints (every {sr}. time point, "
          f"node-wise)...")

    for k in check_indices:
        full_k = reduced_to_full_func(Xs[:, k])
        T_nodes = full_k[n_F_full:][hotspot_nodes]
        if isinstance(T_hot_max_input, (float, int)):
            limit_k = T_hot_max_input
        else:
            limit_k = T_hot_max_input[0, k]
        opti.subject_to(T_nodes <= limit_k + S[0, k])

    print(f"  Added node-wise constraints at {len(check_indices)} time points")


def add_temperature_constraints_modified(
    opti: ca.Opti, Xs: ca.MX, S: ca.MX, reduced_to_full_func,
    T_limit: float, p_offset: ca.MX, Nt: int, n_F_full: int,
):
    """Node-wise temperature limit with an additive offset per node and time.

    The offset is a parameter, so the problem is built once and the offsets
    are updated between solves. With the linear reconstruction each node is
    linear in the reduced state, and the offset carries the decoder's
    correction to it.
    """
    for k in range(Nt):
        full_k = reduced_to_full_func(Xs[:, k])
        T_nodes = full_k[n_F_full:2 * n_F_full]
        opti.subject_to(T_nodes + p_offset[:, k] <= T_limit + S[0, k])
    print(f"  Added node-wise constraints with offsets at {Nt} time points")


def build_objective(
    Xs: ca.MX, U: ca.MX, S: ca.MX, reduced_to_full_func,
    n_F_full: int, N: int, dt: float,
    alpha_u: float, slack_penalty: float,
    u_start_val: float = None
) -> ca.MX:
    """Objective: mean outlet conversion, control smoothness, slack.

    The conversion term is the time average of the outlet conversion, the
    last node of the conversion block. The spatial mean over the whole
    profile, used before, rewards conversion near the inlet, which is what
    drives the hotspot, and it is not the quantity the result is reported
    in.

    Parameters
    ----------
    Xs : (n_red, N + 1) MX
        Scaled reduced states.
    U : (1, N) MX
        Scaled control, one value per interval.
    S : (1, N + 1) MX
        Slack of the temperature constraint.
    reduced_to_full_func : callable
        Maps one scaled reduced state to the physical state [F; T].
    n_F_full : int
        Number of nodes of the conversion block.
    N : int
        Number of control intervals.
    dt : float
        Interval length in seconds.
    alpha_u : float
        Weight of the squared control increments.
    slack_penalty : float
        Weight of the squared slack.
    u_start_val : float, optional
        Scaled control value the first interval is pulled towards.
    """
    print("Building objective function...")
    cost = 0
    T_total = N * dt

    # === 1. Mean outlet conversion, maximised ===
    cumulative_conversion = 0
    for k in range(N):
        full_next = reduced_to_full_func(Xs[:, k + 1])
        if full_next.shape[1] != 1:
            full_next = ca.reshape(full_next, (-1, 1))
        cumulative_conversion += full_next[n_F_full - 1, 0] * dt
    cost += -(cumulative_conversion / T_total)

    # === 2. Control smoothness ===
    if alpha_u > 0:
        diff_u = U[:, 1:] - U[:, :-1]
        cost += (alpha_u / N) * ca.sumsqr(diff_u)

    # === 3. Slack of the temperature constraint ===
    if slack_penalty > 0:
        cost += (slack_penalty / N) * ca.sumsqr(S)

    # === 4. Soft start: first control value near the current operation ===
    if u_start_val is not None:
        start_penalty_weight = 1e5
        cost += start_penalty_weight * ca.sumsqr(U[:, 0] - u_start_val)

    return cost


def main():
    """Main optimization routine."""
    t_start = time.time()

    # =========================================================================
    # 1. CONFIGURATION
    # =========================================================================
    print("\n" + "="*70)
    print("OPTIMAL CONTROL FOR REDUCED-ORDER REACTOR MODEL")
    print("="*70 + "\n")
    print(f"Model {file_name}, strategy {STRATEGY}, state envelope "
          f"{ENVELOPE_MARGIN}, conversion cap {CONVERSION_CAP}, margin "
          f"{MARGIN_K if STRATEGY in ('linear_margin', 'modifier', 'slp') else '-'}"
          f" K, tag "
          f"{RUN_TAG or '-'}")

    SEED = 0
    set_random_seeds(SEED)
    device = get_device()
    print(f"Using device: {device}\n")

    constraints = PhysicalConstraints()
    opt_config = OptimizationConfig(
        alpha_u=100.0,
        slack_penalty=1e5,
        max_time_steps=None,
        enable_caching=True,
        cache_size=1000,
        hotspot_sample_rate=hotspot_sample_rate ,
        time_sampling_stride=data_sample_rate
    )
    constraints.validate()

    DATA_DIR = Path("data") / file_name
    CONTROL_DIR = Path("data") / control_name
    results_root = Path("results") / RUN_TAG if RUN_TAG else Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR = setup_results_dir(str(results_root))

    ipopt_opts = get_ipopt_options(
        hessian_approximation='limited-memory',
        linear_solver='ma57',
        max_iter=1000,
        tolerance=1e-4,
        print_level=1,
    )
    ipopt_opts['ipopt.mu_strategy'] = 'adaptive'
    ipopt_opts['ipopt.max_wall_time'] = 86400.0 # 24h (Overnight)

    # =========================================================================
    # 2. LOAD MODEL DATA & PARAMETERS
    # =========================================================================
    results = load_results(DATA_DIR, decoder_class=decoder_class)

    # Operators
    A, B, C, H = results["A"], results["B"], results["C"], results["H"]
    basis = results["basis"]
    reference_states_shifting = results["reference_states_shifting"]

    # Scaling Factors
    sF = float(np.asarray(results["scaling_fac_F"]).item())
    sT = float(np.asarray(results["scaling_fac_T"]).item())
    max_F = float(results["max_F"])
    max_T = float(results["max_T"])

    # Inputs, in the order of the columns of B, and their scaling.
    input_names = results["input_names"]
    input_factor = dict(zip(input_names, results["input_scaling_factors"]))
    if np.asarray(B).shape[1] != len(input_names):
        raise ValueError(f"B has {np.asarray(B).shape[1]} columns, the model "
                         f"lists {len(input_names)} inputs: {input_names}")
    CONTROL_INPUT = "cooling_temperature"
    load_name = next((n for n in input_names if n.startswith("load")), None)
    if CONTROL_INPUT not in input_names or load_name is None:
        raise KeyError(f"the model's inputs {input_names} lack the coolant "
                       f"temperature or the load")
    scale_load = float(input_factor[load_name])
    scale_Tcool = float(input_factor[CONTROL_INPUT])

    # Inputs the scenario does not prescribe, such as an inlet temperature
    # that was constant in training, are held at their training mean. Their
    # column of B then contributes the same constant as in training; the zero
    # used before was an input the model never saw.
    held_inputs = {
        name: float(np.mean(results["inputs"][i]))
        for i, name in enumerate(input_names)
        if name not in (CONTROL_INPUT, load_name)}
    for name, value in held_inputs.items():
        print(f"  input '{name}' held at its training mean "
              f"{value * input_factor[name]:.4g} (scaled {value:.4f})")

    def input_matrix(load_scaled, cool_scaled):
        """Inputs for a numpy simulation, one row per input, in B's order."""
        load_scaled = np.atleast_1d(np.asarray(load_scaled, dtype=float))
        cool_scaled = np.broadcast_to(np.asarray(cool_scaled, dtype=float),
                                      load_scaled.shape)
        rows = [load_scaled if n == load_name
                else cool_scaled if n == CONTROL_INPUT
                else np.full(load_scaled.shape, held_inputs[n])
                for n in input_names]
        return np.vstack(rows)

    def input_vector(load_k, u_k):
        """Inputs for one CasADi step, in B's order."""
        return ca.vertcat(*[load_k if n == load_name
                            else u_k if n == CONTROL_INPUT
                            else held_inputs[n]
                            for n in input_names])

    # The controller may only use coolant temperatures the model was
    # identified on. Outside that range the reduced model extrapolates: on
    # the synthetic model a static coolant temperature of 673 K yields a
    # hotspot of 4970 K and a conversion of 693 %, and the linear
    # reconstruction and the decoder disagree by hundreds of kelvin.
    i_cool = input_names.index(CONTROL_INPUT)
    cool_train = np.asarray(results["inputs"][i_cool], dtype=float) * scale_Tcool
    cool_lo, cool_hi = float(np.min(cool_train)), float(np.max(cool_train))
    print(f"  coolant temperature in training: {cool_lo:.1f} to "
          f"{cool_hi:.1f} K; the control bounds are clipped to this range")
    constraints = dataclasses.replace(
        constraints,
        u_min=max(constraints.u_min, cool_lo),
        u_max=min(constraints.u_max, cool_hi))
    print(f"  control bounds: {constraints.u_min:.1f} to "
          f"{constraints.u_max:.1f} K")

    # Dimensions
    r_F, r_T = results["r_F"], results["r_T"]
    n_red = int(A.shape[0])
    n_full = int(basis.shape[0])
    n_F_full = n_full // 4

    # NL-POD & CNN
    basis_type = results.get("basis_type")
    V_reduced_nonlin = results.get("V_reduced_nonlin")
    Xi = results.get("Xi")

    decoder = results.get("decoder")
    has_decoder = (decoder is not None)
    if has_decoder: decoder = decoder.to(device).eval()
    if STRATEGY in ("multi_fidelity", "decoder", "modifier", "slp") \
            and not has_decoder:
        raise RuntimeError(f"strategy {STRATEGY} needs a decoder, and the "
                           f"model has none")

    scalers = (results.get("input_scaler"),
               results.get("target_scaler_F"),
               results.get("target_scaler_T"))

    # =========================================================================
    # 3. SCENARIO & TIME SAMPLING
    # =========================================================================
    scenario_id = control_name.split('_')[-1]
    print(f"Loading external control scenario: {control_name}...")

    # Load Raw (Fine Grid)
    load_profile_raw = np.load(CONTROL_DIR / f"load_control_{scenario_id}.npy").flatten()
    Tcool_profile_raw = np.load(CONTROL_DIR / f"cooling_temperature_control_{scenario_id}.npy").flatten()
    time_arr_raw = np.load(CONTROL_DIR / f"time_control_{scenario_id}.npy")

    # --- Slicing (Skip Start) ---
    start_idx = 0
    if skip_first_seconds > 0:
        # Find index where time >= skip_first_seconds
        indices = np.where(time_arr_raw >= skip_first_seconds)[0]
        if len(indices) > 0:
            start_idx = indices[0]
            print(f"  Skipping first {skip_first_seconds}s (Starting at index {start_idx})")

            # Slice arrays
            time_arr_raw = time_arr_raw[start_idx:]
            # Shift time to start at 0
            time_arr_raw = time_arr_raw - time_arr_raw[0]

            load_profile_raw = load_profile_raw[start_idx:]
            Tcool_profile_raw = Tcool_profile_raw[start_idx:]
        else:
            print(f"  Warning: skip_first_seconds={skip_first_seconds} exceeds data length.")

    # Global Truncation
    if opt_config.max_time_steps is not None:
        time_arr_raw = time_arr_raw[:opt_config.max_time_steps]
        load_profile_raw = load_profile_raw[:opt_config.max_time_steps]
        Tcool_profile_raw =  Tcool_profile_raw[:opt_config.max_time_steps]

    # Sampling & Scaling for Optimization
    load_profile_fine = load_profile_raw / scale_load
    Tcool_profile_fine = Tcool_profile_raw / scale_Tcool

    stride = opt_config.time_sampling_stride
    time_arr = time_arr_raw[::stride]
    load_profile = load_profile_fine[::stride]

    # Calculate Dt for Optimization
    dt = float(time_arr[1] - time_arr[0])

    # Interpolators (Fine Grid)
    load_interp = interp1d(time_arr_raw, load_profile_fine, kind='linear', fill_value="extrapolate")

    # Constraints Scaling
    u_min_scaled = constraints.u_min / scale_Tcool
    u_max_scaled = constraints.u_max / scale_Tcool
    ramp_up_step_scaled = (constraints.ramp_up_K_per_s / scale_Tcool) * dt
    ramp_down_step_scaled = (constraints.ramp_down_K_per_s / scale_Tcool) * dt

    # Setup Optimization Dimensions
    Nt = len(time_arr)
    N = Nt - 1

    print(f"\nProblem dimensions:")
    print(f"  Reduced dim: {n_red}, Full dim: {n_full}")
    print(f"  Time Stride: {stride} (Raw steps: {len(time_arr_raw)} -> Opt steps: {len(time_arr)})")
    print(f"  Optimization dt: {dt:.3f} s")
    print(f"  Reference Temp (Mean): {np.mean(reference_states_shifting[n_F_full:]):.2f} K")

    # =========================================================================
    # 4. INITIAL CONDITION
    # =========================================================================
    print("\nCalculating initial condition y0...")
    fname_F = f"conversion_opinf_control_{scenario_id}.npy"
    fname_T = f"center_temperature_opinf_control_{scenario_id}.npy"
    path_F = CONTROL_DIR / fname_F
    path_T = CONTROL_DIR / fname_T

    if not path_F.exists() or not path_T.exists():
        raise FileNotFoundError(f"Initial data missing: {path_F} or {path_T}")

    # Load Data
    F_data_full = np.load(path_F)
    T_data_full = np.load(path_T)

    # --- Robust Slice Helper ---
    def get_slice_at_k(data_array, time_idx):
        if data_array.ndim == 3 and data_array.shape[1] == 1:
            return data_array[1:, 0, time_idx].reshape(-1, 1)
        elif data_array.ndim == 3 and data_array.shape[2] == 1:
             return data_array[1:, time_idx, 0].reshape(-1, 1)
        elif data_array.ndim == 2:
            return data_array[1:, time_idx].reshape(-1, 1)
        else:
            return data_array.flatten()[1:389].reshape(-1, 1)

    # Extract initial physical state
    F_phys_0 = get_slice_at_k(F_data_full, start_idx)
    T_phys_0 = get_slice_at_k(T_data_full, start_idx)

    n_spatial = F_phys_0.shape[0]
    print(f"  Spatial points (phys, no BC): {n_spatial}")

    # Handle Reference States
    ref_vec = reference_states_shifting.flatten()
    if ref_vec.shape[0] != 4 * n_spatial:
         raise ValueError(f"Reference state size {ref_vec.shape[0]} does not match 4x spatial {4*n_spatial}")

    ref_F  = ref_vec[0 : n_spatial].reshape(-1, 1)
    ref_T  = ref_vec[n_spatial : 2*n_spatial].reshape(-1, 1)

    # Latent Scaling
    s_red_local = np.ones((n_red, 1))
    s_red_local[:r_F] = sF
    s_red_local[r_F:] = sT

    def latent_of(F_phys, T_phys):
        """Scaled reduced coordinates of physical profiles, one per column."""
        f = (F_phys - ref_F) / max_F
        t = (T_phys - ref_T) / max_T
        zeros = np.zeros_like(f)
        x_full_normalized = np.vstack((f, t, zeros, zeros))
        return (basis.T @ x_full_normalized) / s_red_local

    y0 = latent_of(F_phys_0, T_phys_0).flatten()

    print(f"  Loaded T0 mean: {np.mean(T_phys_0):.2f} K")
    print(f"  Calculated y0 norm: {np.linalg.norm(y0):.4f}")

    n_F_full = n_spatial
    print(f"  Updated n_F_full to: {n_F_full} (Spatial grid size)")

    # =========================================================================
    # 4b. STATE ENVELOPE OF THE TRAINING TRAJECTORIES
    # =========================================================================
    env_lo = env_hi = None
    Y_tr = y_rng = None
    if ENVELOPE_MARGIN is not None or STRATEGY == "slp":
        print("\nReduced states of the training trajectories...")
        F_tr = np.asarray(np.load(TRAIN_DATA_DIR / "conversion_training.npy"),
                          dtype=float)
        T_tr = np.asarray(np.load(TRAIN_DATA_DIR /
                                  "center_temperature_training.npy"),
                          dtype=float)
        t_tr = np.asarray(np.load(TRAIN_DATA_DIR / "time_training.npy"),
                          dtype=float).ravel()
        # Trajectories are separated by the resets of the time vector.
        starts = np.concatenate(([0], np.where(np.diff(t_tr) < 0)[0] + 1,
                                 [t_tr.size]))
        in_train = np.zeros(t_tr.size, dtype=bool)
        for j in TRAIN_TRAJECTORIES:
            in_train[starts[j]:starts[j + 1]] = True
        cols = np.where(in_train)[0][::TRAIN_STRIDE]
        Y_tr = latent_of(F_tr[1:, cols], T_tr[1:, cols])
        y_min, y_max = Y_tr.min(axis=1), Y_tr.max(axis=1)
        y_rng = np.maximum(y_max - y_min, 1e-6)
        print(f"  {len(starts) - 1} trajectories, {len(TRAIN_TRAJECTORIES)} "
              f"for training, {cols.size} snapshots")
        if ENVELOPE_MARGIN is not None:
            pad = ENVELOPE_MARGIN * (y_max - y_min)
            env_lo, env_hi = y_min - pad, y_max + pad
            inside0 = bool(np.all((y0 >= env_lo) & (y0 <= env_hi)))
            print(f"  state envelope: margin {ENVELOPE_MARGIN:.0%} of each "
                  f"range; initial state inside: {inside0}")

    # =========================================================================
    # 5. WRAPPERS & INTEGRATOR
    # =========================================================================
    print("\n" + "-"*40 + "\n5. BUILDING INTEGRATOR & WRAPPERS\n" + "-"*40)

    print("Building CasADi integrator...")
    integrator = build_casadi_integrator(A, H, B, C, n_red, dt)

    cnn_casadi_func = None
    if has_decoder:
        print("Creating CasADi CNN callback...")
        cnn_output_dim = 2 * n_F_full

        cnn_casadi_func = create_cnn_casadi_function(
            decoder, scalers, r_F, r_T, n_F_full, cnn_output_dim, device,
            opt_config.enable_caching, opt_config.cache_size
        )
        print("  ✓ CNN callback ready")

    # --- Wrapper for Optimizer (CasADi): decoder if loaded ---
    def reduced_to_full_casadi_wrapper(x_red_scaled):
        return reduced_to_full_casadi(
            x_red_scaled, basis_type, r_F, r_T, sF, sT, basis,
            reference_states_shifting, state_max_F=max_F, state_max_T=max_T,
            V_reduced_nonlin=V_reduced_nonlin, Xi=Xi, poly_degree=3,
            has_decoder=has_decoder, cnn_func=cnn_casadi_func,
            n_full=n_full, n_F_full=n_F_full
        )

    # --- Wrapper for Plotting/Checks (Numpy): decoder if loaded ---
    def reduced_to_full_numpy_wrapper(x_red_scaled):
        return reduced_to_full_numpy(
            x_red_scaled, basis_type, r_F, r_T, sF, sT, basis,
            reference_states_shifting, state_max_F=max_F, state_max_T=max_T,
            V_reduced_nonlin=V_reduced_nonlin, Xi=Xi, poly_degree=3,
            has_decoder=has_decoder, decoder=decoder, scalers=scalers,
            n_F_full=n_F_full, device=device
        )

    # --- Linear reconstruction (CasADi), for the solver ---
    def reduced_to_full_linear_opt_wrapper(x_red_scaled):
        return reduced_to_full_casadi(
            x_red_scaled, basis_type, r_F, r_T, sF, sT, basis,
            reference_states_shifting, state_max_F=max_F, state_max_T=max_T,
            V_reduced_nonlin=V_reduced_nonlin, Xi=Xi, poly_degree=3,
            has_decoder=False, cnn_func=None, n_full=n_full, n_F_full=n_F_full
        )

    # --- Linear reconstruction (Numpy), for analysis ---
    def reduced_to_full_linear_check_wrapper(x_red_scaled):
        return reduced_to_full_numpy(
            x_red_scaled, basis_type, r_F, r_T, sF, sT, basis,
            reference_states_shifting, state_max_F=max_F, state_max_T=max_T,
            V_reduced_nonlin=V_reduced_nonlin, Xi=Xi, poly_degree=3,
            has_decoder=False, decoder=None, scalers=None, n_F_full=n_F_full,
            device=device
        )

    # --- Decoder (Numpy), for the check ---
    def reduced_to_full_cnn_check_wrapper(x_red_scaled):
        return reduced_to_full_numpy(
            x_red_scaled, basis_type, r_F, r_T, sF, sT, basis,
            reference_states_shifting, state_max_F=max_F, state_max_T=max_T,
            V_reduced_nonlin=V_reduced_nonlin, Xi=Xi, poly_degree=3,
            has_decoder=True, decoder=decoder, scalers=scalers,
            n_F_full=n_F_full, device=device
        )

    # --- Helper for Fine-Grid Simulation ---
    def forward_sim_wrapper_fine(U_val_const):
        U_profile = input_matrix(load_interp(time_arr_raw), U_val_const)
        return forward_sim_reduced(y0, time_arr_raw, U_profile, A, H, B, C, n_red)

    # =========================================================================
    # 5b. STATIC COOLANT STRATEGY DIAGNOSTICS
    # =========================================================================
    print("\n" + "-"*40)
    print("5b. STATIC COOLANT STRATEGY DIAGNOSTICS")
    print("-"*40)

    def evaluate_static_coolant_strategy(label: str, u_phys_K: float):
        """
        Evaluate a constant coolant temperature strategy with the learned ROM
        and the available full-state reconstruction.

        Note:
            This evaluates the ROM + reconstruction/decoder.
            It is not the independent mechanistic plant validation unless
            reduced_to_full_numpy_wrapper represents that reference.
        """
        u_scaled = u_phys_K / scale_Tcool

        # Simulate reduced dynamics on fine grid
        X_red = forward_sim_wrapper_fine(u_scaled)

        # Reconstruct full state
        full = reduced_to_full_numpy_wrapper(X_red)
        full = np.asarray(full)

        # Split conversion / temperature blocks
        X_block = full[:n_F_full, :]
        T_block = full[n_F_full:2*n_F_full, :]

        # Metrics
        T_hot = np.max(T_block, axis=0)
        X_out = X_block[-1, :]

        # Additional metric because some scripts use spatial mean conversion
        X_spatial_mean = np.mean(X_block, axis=0)

        stats = {
            "label": label,
            "u_phys_K": float(u_phys_K),
            "u_phys_C": float(u_phys_K - 273.15),
            "max_T_hot_K": float(np.max(T_hot)),
            "T_violation_K": float(np.max(T_hot) - constraints.T_hot_max),
            "mean_X_out": float(np.mean(X_out)),
            "mean_X_spatial": float(np.mean(X_spatial_mean)),
            "min_X_out": float(np.min(X_out)),
            "max_X_out": float(np.max(X_out)),
        }
        if env_lo is not None:
            # How far the static strategy leaves the state envelope.
            outside = np.maximum(env_lo[:, None] - X_red, 0) + \
                np.maximum(X_red - env_hi[:, None], 0)
            stats["share_outside_envelope"] = float(
                np.mean(np.any(outside > 0, axis=0)))

        print(f"\n[{label}]")
        print(f"  T_cool:              {stats['u_phys_K']:.2f} K ({stats['u_phys_C']:.1f} °C)")
        print(f"  Max hotspot:         {stats['max_T_hot_K']:.2f} K")
        print(f"  Violation vs Tmax:   {stats['T_violation_K']:.2f} K")
        print(f"  Mean X_out:          {stats['mean_X_out']:.4f} ({100*stats['mean_X_out']:.2f} %)")
        print(f"  Mean X_spatial:      {stats['mean_X_spatial']:.4f} ({100*stats['mean_X_spatial']:.2f} %)")
        print(f"  X_out range:         {stats['min_X_out']:.4f} ... {stats['max_X_out']:.4f}")
        if "share_outside_envelope" in stats:
            print(f"  outside envelope:    {stats['share_outside_envelope']:.1%} of the time")

        return stats

    # Candidate static coolant strategies
    u_min_phys = constraints.u_min
    u_max_phys = constraints.u_max
    u_mid_phys = 0.5 * (constraints.u_min + constraints.u_max)
    u_mean_loaded_phys = float(np.mean(Tcool_profile_raw))

    static_stats = []
    static_stats.append(evaluate_static_coolant_strategy("static u_min / maximum cooling", u_min_phys))
    static_stats.append(evaluate_static_coolant_strategy("static midpoint of bounds", u_mid_phys))
    static_stats.append(evaluate_static_coolant_strategy("static u_max / minimum cooling", u_max_phys))
    static_stats.append(evaluate_static_coolant_strategy("static mean of loaded Tcool profile", u_mean_loaded_phys))

    # Save diagnostics for later inspection
    diagnostics_path = RESULTS_DIR / "static_coolant_strategy_diagnostics.json"
    with open(diagnostics_path, "w") as f:
        json.dump(static_stats, f, indent=2)

    print(f"\nSaved static coolant diagnostics to: {diagnostics_path}")

    # =========================================================================
    # 6. SYSTEM VALIDATION (FINAL SAFETY CHECK)
    # =========================================================================
    print("\n" + "-"*40)
    print("6. FINAL PRE-OPTIMIZATION CHECK")
    print("-"*40)

    # Replay 1 hour to ensure model physics are still valid
    VAL_HORIZON = 3600.0
    mask_val = time_arr_raw <= VAL_HORIZON
    time_val = time_arr_raw[mask_val]

    # Construct Inputs for Replay
    u_load_val = load_interp(time_val)
    Tcool_interp = interp1d(time_arr_raw, Tcool_profile_raw, kind='linear',
                            fill_value="extrapolate")
    u_cool_val = Tcool_interp(time_val) / scale_Tcool
    U_prof_val = input_matrix(u_load_val, u_cool_val)

    # Simulate
    X_sim_val = forward_sim_reduced(y0, time_val, U_prof_val, A, H, B, C, n_red)

    # Calculate RMSE against Ground Truth
    T_rom_traj = []
    T_gt_traj = []
    for k in range(len(time_val)):
        full = reduced_to_full_numpy_wrapper(X_sim_val[:, k])
        T_rom_traj.append(np.max(full[n_F_full:]))
        T_gt_traj.append(np.max(get_slice_at_k(T_data_full, start_idx + k)))

    rmse = np.sqrt(np.mean((np.array(T_rom_traj) - np.array(T_gt_traj))**2))
    print(f"  Validation RMSE (1h): {rmse:.2f} K")

    if rmse > 20.0:
        print("  ⚠ WARNING: High validation error. Optimization results might be inaccurate.")
    else:
        print("  ✓ Validation successful. Proceeding to optimization.")

    # =========================================================================
    # 7. INITIAL GUESS GENERATION
    # =========================================================================
    print("\n" + "-"*40 + "\n7. INITIAL GUESS\n" + "-"*40)
    print("Generating initial trajectory from simulation...")

    # 1. Simulate constant control (u_min) on fine grid
    X_sim_fine = forward_sim_wrapper_fine(u_min_scaled)

    # 2. Downsample to Optimization Grid
    Xs_init = X_sim_fine[:, ::stride]

    # 3. Create constant control guess
    U_init = np.ones(N) * u_min_scaled

    # 4. Correct shape mismatch
    if Xs_init.shape[1] != Nt:
        Xs_init = Xs_init[:, :Nt]

    # 5. Check Residuals
    res_init = check_dynamics_residuals(
        Xs_init, U_init, integrator, time_arr, load_interp, input_vector
    )
    print(f"  Initial guess dynamics residual: {res_init:.2e}")

    # =========================================================================
    # 8. OPTIMIZATION SETUP & SOLVE
    # =========================================================================
    print("\n" + "-"*40 + "\n8. OPTIMIZATION\n" + "-"*40)

    # Initial control value for soft start constraint
    u_cool_start_val = Tcool_profile_raw[0] / scale_Tcool
    print(f"Target Initial Control: {u_cool_start_val:.4f} (Soft Constraint)")

    # Prepare result containers
    Xs_opt_final = None
    U_opt_final = None
    S_opt_final = None
    sol_full_reconstructed = None # Physical state trajectory
    loops_used = 1
    ensemble_stats = None

    # -------------------------------------------------------------------------
    # OPTION A: MULTI-FIDELITY (Iterative Back-off with Vector Param)
    # Strategy: Build Graph ONCE -> Parameter Update -> Warm Start Loop
    # -------------------------------------------------------------------------
    if USE_MULTI_FIDELITY:
        print(">> STRATEGY: Multi-Fidelity Iterative Back-off (Vectorized).")

        # --- BUILD OPTIMIZATION PROBLEM (ONCE!) ---
        print("\nConstructing Optimization Graph (One-time setup)...")

        opti, Xs, U, S = setup_optimization_problem(
            n_red, Nt, N, y0, np.ones(n_red),
            u_min_scaled, u_max_scaled,
            ramp_up_step_scaled, ramp_down_step_scaled, dt
        )

        # Define Parameter (Placeholder for the varying limit profile)
        p_T_limit = opti.parameter(1, Nt)

        add_dynamics_constraints(opti, Xs, U, integrator, time_arr, load_interp, input_vector, N)
        if env_lo is not None:
            add_state_envelope(opti, Xs, env_lo, env_hi, Nt)
        if CONVERSION_CAP is not None:
            add_conversion_cap(opti, Xs, reduced_to_full_linear_opt_wrapper,
                               Nt, n_F_full, CONVERSION_CAP)

        # Pass the parameter vector instead of a scalar limit
        add_temperature_constraints(
            opti, Xs, S, reduced_to_full_linear_opt_wrapper,
            p_T_limit,
            Nt, n_F_full,
            sample_rate=opt_config.hotspot_sample_rate
        )

        cost = build_objective(
            Xs, U, S, reduced_to_full_linear_opt_wrapper,
            n_F_full, N, dt,
            opt_config.alpha_u, opt_config.slack_penalty,
            u_start_val=u_cool_start_val
        )
        opti.minimize(cost)

        # Solver Settings
        opti.solver('ipopt', ipopt_opts)

        # --- ITERATION LOOP ---
        T_limit_absolute = constraints.T_hot_max

        # Start Defensive
        initial_margin = 2.0
        current_limit_profile = np.full((1, Nt), T_limit_absolute - initial_margin)
        safety_margin_profile = np.full((1, Nt), initial_margin)

        # State Machine Variables
        polishing_active = False
        polishing_steps = 0
        MAX_POLISH_STEPS = 5  # Give it 5 extra rounds to improve

        # Best Solution Tracking
        best_solution = {
            'conversion': -1.0, 'violation': 999.0,
            'Xs': None, 'U': None, 'S': None, 'cnn_traj': None
        }

        # Initial guesses
        curr_Xs_init = ca.DM(Xs_init)
        curr_U_init = ca.DM(U_init)
        curr_S_init = ca.DM(np.zeros((1, Nt)))

        prev_conversion = -1.0

        for loop_idx in range(MAX_LOOPS):
            loops_used = loop_idx + 1
            min_lim = np.min(current_limit_profile)
            max_lim = np.max(current_limit_profile)
            print(f"\n[LOOP {loop_idx+1}/{MAX_LOOPS}] Optimizing... (Limit: {min_lim:.1f}..{max_lim:.1f} K)")

            opti.set_value(p_T_limit, current_limit_profile)
            opti.set_initial(Xs, curr_Xs_init)
            opti.set_initial(U, curr_U_init)
            opti.set_initial(S, curr_S_init)

            try:
                sol = opti.solve()
                curr_Xs_init, curr_U_init, curr_S_init = sol.value(Xs), sol.value(U), sol.value(S)
                Xs_val = np.array(curr_Xs_init)
            except RuntimeError:
                sol = opti.debug
                Xs_val = np.array(sol.value(Xs))

            sol_full_cnn = reduced_to_full_cnn_check_wrapper(Xs_val)
            T_hot_cnn_vec = np.max(sol_full_cnn[n_F_full : 2*n_F_full, :], axis=0).reshape(1, -1)
            sol_full_lin = reduced_to_full_linear_check_wrapper(Xs_val)
            T_hot_lin_vec = np.max(sol_full_lin[n_F_full : 2*n_F_full, :], axis=0).reshape(1, -1)

            mismatch_vec = T_hot_cnn_vec - T_hot_lin_vec
            violation_vec = T_hot_cnn_vec - T_limit_absolute
            max_violation = np.max(violation_vec)
            # Mean outlet conversion over the horizon, the quantity the
            # objective maximises and the result is reported in.
            current_conv = float(np.mean(sol_full_cnn[n_F_full - 1, :]))

            print(f"     Max Mismatch: {np.max(mismatch_vec):.2f} K")
            print(f"     Max Violation: {max_violation:.2f} K")
            print(f"     Conversion:    {current_conv:.4f}")

            # D) Track Best Solution
            is_safer = max_violation < best_solution['violation'] - 0.5
            is_better_conv = (max_violation <= 1.5) and (current_conv > best_solution['conversion'])

            if is_safer or is_better_conv:
                print(f"     ★ New Best Solution stored!")
                best_solution.update({
                    'conversion': current_conv, 'violation': max_violation,
                    'Xs': Xs_val, 'U': np.array(sol.value(U)).flatten() * scale_Tcool,
                    'S': np.array(sol.value(S)).flatten(), 'cnn_traj': sol_full_cnn
                })

            # E) STATE MACHINE & TERMINATION
            if not polishing_active and max_violation <= 0.5:
                print("  >> STABLE STATE REACHED. Switching to Polishing Mode (max 5 loops).")
                polishing_active = True
                polishing_steps = 0

            if polishing_active:
                polishing_steps += 1

                if polishing_steps > MAX_POLISH_STEPS:
                    print("  ✅ SUCCESS: Polishing complete (Max steps reached).")
                    break

                if max_violation > 2.0:
                    print("  🛑 STOP: Polishing caused violation > 2.0 K. Reverting.")
                    break

                if abs(current_conv - prev_conversion) < 0.00005 and polishing_steps > 1:
                     print("  ✅ SUCCESS: Polishing complete (Convergence stagnated).")
                     break

            prev_conversion = current_conv

            # F) UPDATE PROFILE (Adaptive Logic)
            print("  ⚠ UPDATING PROFILE: Adjusting local constraints...")

            from scipy.ndimage import gaussian_filter1d
            smoothed_mismatch = gaussian_filter1d(mismatch_vec, sigma=5)
            step_vector = np.zeros_like(smoothed_mismatch)

            # 1. Tightening
            mask_tighten = smoothed_mismatch > 0

            if polishing_active and max_violation < 0.2:
                step_vector[mask_tighten] = 0.0
                print("    -> Polishing: Tightening FROZEN (Violation < 0.2 K).")
            else:
                step_vector[mask_tighten] = 1.0

            # 2. Relaxing
            mask_relax = smoothed_mismatch < 0

            if not polishing_active:
                step_vector[mask_relax] = 0.0
                print("    -> Mode: STABILIZING. Relaxation blocked.")

            else:
                if max_violation > 0.8:
                    step_vector[mask_relax] = 0.0
                    print("    -> Mode: POLISHING (HOLD). Slight drift detected.")
                else:
                    step_vector = np.where(
                        (smoothed_mismatch < -5.0) & mask_relax, 0.4, step_vector
                    )
                    step_vector = np.where(
                        (smoothed_mismatch >= -5.0) & (smoothed_mismatch < -2.0) & mask_relax, 0.2, step_vector
                    )
                    step_vector = np.where(
                        (smoothed_mismatch >= -2.0) & (smoothed_mismatch < 0.0) & mask_relax, 0.05, step_vector
                    )
                    print("    -> Mode: POLISHING (ACTIVE). Adaptive relaxing.")

            # Apply
            safety_margin_profile += step_vector * smoothed_mismatch

            # Ceiling/Floor
            T_ceiling = T_limit_absolute + 25.0
            T_floor = T_limit_absolute - 60.0
            current_limit_profile = np.clip(T_limit_absolute - safety_margin_profile, T_floor, T_ceiling)
            safety_margin_profile = T_limit_absolute - current_limit_profile

        # --- END OF LOOP: RESTORE BEST ---
        print("\n" + "="*30)
        if best_solution['Xs'] is not None:
            print(f"RESTORING BEST SOLUTION (Viol: {best_solution['violation']:.2f} K, Conv: {best_solution['conversion']:.4f})")
            Xs_opt_final = best_solution['Xs']
            U_opt_final = best_solution['U']
            S_opt_final = best_solution['S']
            sol_full_reconstructed = best_solution['cnn_traj']
        else:
            print("⚠ WARNING: No feasible solution found!")
            Xs_opt_final = Xs_val
            U_opt_final = np.array(sol.value(U)).flatten() * scale_Tcool
            S_opt_final = np.array(sol.value(S)).flatten()
            sol_full_reconstructed = sol_full_cnn

    # -------------------------------------------------------------------------
    # OPTION C: MODIFIER ADAPTATION (linear constraints, decoder offsets)
    # -------------------------------------------------------------------------
    elif STRATEGY == "modifier":
        print(f">> STRATEGY: modifier adaptation, limit lowered by "
              f"{MARGIN_K:.1f} K, damping {MODIFIER_DAMPING:.2f}.")
        opti, Xs, U, S = setup_optimization_problem(
            n_red, Nt, N, y0, np.ones(n_red),
            u_min_scaled, u_max_scaled,
            ramp_up_step_scaled, ramp_down_step_scaled, dt
        )
        p_off = opti.parameter(n_F_full, Nt)
        add_dynamics_constraints(opti, Xs, U, integrator, time_arr,
                                 load_interp, input_vector, N)
        if env_lo is not None:
            add_state_envelope(opti, Xs, env_lo, env_hi, Nt)
        if CONVERSION_CAP is not None:
            add_conversion_cap(opti, Xs, reduced_to_full_linear_opt_wrapper,
                               Nt, n_F_full, CONVERSION_CAP)
        T_limit = constraints.T_hot_max - MARGIN_K
        add_temperature_constraints_modified(
            opti, Xs, S, reduced_to_full_linear_opt_wrapper,
            float(T_limit), p_off, Nt, n_F_full)
        cost = build_objective(
            Xs, U, S, reduced_to_full_linear_opt_wrapper,
            n_F_full, N, dt,
            opt_config.alpha_u, opt_config.slack_penalty,
            u_start_val=u_cool_start_val
        )
        opti.minimize(cost)
        opti.solver('ipopt', ipopt_opts)

        offset = np.zeros((n_F_full, Nt))
        curr_X, curr_U = ca.DM(Xs_init), ca.DM(U_init)
        curr_S = ca.DM(np.zeros((1, Nt)))
        best = None
        prev_conv = None
        for it in range(MODIFIER_MAX_ITER):
            loops_used = it + 1
            opti.set_value(p_off, offset)
            opti.set_initial(Xs, curr_X)
            opti.set_initial(U, curr_U)
            opti.set_initial(S, curr_S)
            try:
                sol = opti.solve()
            except RuntimeError:
                sol = opti.debug
            X_val = np.array(sol.value(Xs))
            U_val = np.array(sol.value(U)).flatten()
            S_val = np.array(sol.value(S)).flatten()
            curr_X, curr_U, curr_S = ca.DM(X_val), ca.DM(U_val), ca.DM(S_val)

            T_dec = np.asarray(reduced_to_full_cnn_check_wrapper(X_val))[
                n_F_full:2 * n_F_full, :]
            T_lin = np.asarray(reduced_to_full_linear_check_wrapper(X_val))[
                n_F_full:2 * n_F_full, :]
            conv = float(np.mean(np.asarray(
                reduced_to_full_cnn_check_wrapper(X_val))[n_F_full - 1, :]))
            violation = float(np.max(T_dec) - constraints.T_hot_max)
            print(f"[MODIFIER {it + 1}/{MODIFIER_MAX_ITER}] decoder "
                  f"hotspot {np.max(T_dec):.1f} K (violation {violation:+.1f}"
                  f" K), linear {np.max(T_lin):.1f} K, conversion "
                  f"{conv:.4f}, largest offset {np.max(offset):.1f} K")

            feasible = violation <= MODIFIER_TOL_K
            if feasible and (best is None or conv > best["conv"]):
                best = {"conv": conv, "X": X_val, "U": U_val, "S": S_val,
                        "violation": violation}
                print("     ★ best feasible solution so far")
            if feasible and prev_conv is not None \
                    and abs(conv - prev_conv) < 1e-4:
                print("  ✅ converged: feasible and the conversion no "
                      "longer changes.")
                break
            prev_conv = conv

            # New offsets: the decoder's correction to the linear
            # reconstruction at every node and time point, relaxed.
            offset = offset + MODIFIER_DAMPING * ((T_dec - T_lin) - offset)

        if best is None:
            print("⚠ no iterate met the limit with the decoder; keeping the "
                  "last one.")
            best = {"conv": conv, "X": X_val, "U": U_val, "S": S_val,
                    "violation": violation}
        Xs_opt_final = best["X"]
        U_opt_final = best["U"] * scale_Tcool
        S_opt_final = best["S"]

    # -------------------------------------------------------------------------
    # OPTION D: SUCCESSIVE LINEARIZATION OF THE DECODER, TRUST REGION;
    # with OCP_ENSEMBLE for several models at once (shared coolant)
    # -------------------------------------------------------------------------
    elif STRATEGY == "slp":
        T_limit = constraints.T_hot_max - MARGIN_K
        sub = np.arange(0, n_F_full, AGREE_STRIDE)
        n_sub = sub.size
        h_fd = 1e-3 * y_rng

        # Linear reconstruction of the temperature: an affine map, the same
        # for every member (identical basis and shift, checked below).
        basis_pts = np.hstack([np.zeros((n_red, 1)), np.eye(n_red)])
        T_basis = np.asarray(reduced_to_full_linear_check_wrapper(
            basis_pts))[n_F_full:2 * n_F_full]
        d_T = T_basis[:, :1]
        C_T = T_basis[:, 1:] - d_T

        def T_linear(X):
            return C_T @ X + d_T

        def make_decoder_eval(res):
            dec = res.get("decoder").to(device).eval()
            sc = (res.get("input_scaler"), res.get("target_scaler_F"),
                  res.get("target_scaler_T"))

            def ev(X):
                full = np.asarray(reduced_to_full_numpy(
                    X, basis_type, r_F, r_T, sF, sT, basis,
                    reference_states_shifting, state_max_F=max_F,
                    state_max_T=max_T, V_reduced_nonlin=V_reduced_nonlin,
                    Xi=Xi, poly_degree=3, has_decoder=True, decoder=dec,
                    scalers=sc, n_F_full=n_F_full, device=device))
                return full[n_F_full:2 * n_F_full], full[n_F_full - 1]
            return ev

        # Members: the primary model and the ensemble models.
        members = [{"name": file_name, "res": results,
                    "integ": integrator, "X_init": np.asarray(Xs_init)}]
        U_prof = input_matrix(load_interp(time_arr_raw), u_min_scaled)
        for m in ENSEMBLE_MODELS:
            res = load_results(Path("data") / m, decoder_class=decoder_class)
            same = (np.allclose(np.asarray(res["basis"]), basis)
                    and np.allclose(
                        np.asarray(res["reference_states_shifting"]),
                        reference_states_shifting)
                    and list(res["input_names"]) == list(input_names))
            if not same:
                raise ValueError(f"{m} has another basis, shift or input "
                                 f"set than {file_name}")
            X_sim = forward_sim_reduced(y0, time_arr_raw, U_prof, res["A"],
                                        res["H"], res["B"], res["C"], n_red)
            members.append({"name": m, "res": res,
                            "integ": build_casadi_integrator(
                                res["A"], res["H"], res["B"], res["C"],
                                n_red, dt),
                            "X_init": X_sim[:, ::stride][:, :Nt]})
        for mb in members:
            mb["dec"] = make_decoder_eval(mb["res"])
        n_mem = len(members)

        def decoder_linearization(dec_eval, X):
            """Decoder temperatures and outlet conversion with their
            derivatives in the reduced state, by central differences, in
            one batched decoder call."""
            M = X.shape[1]
            cols_ = [X]
            for i in range(n_red):
                e = np.zeros((n_red, 1)); e[i] = h_fd[i]
                cols_ += [X + e, X - e]
            T_all, c_all = dec_eval(np.concatenate(cols_, axis=1))
            T0, c0 = T_all[:, :M], c_all[:M]
            J = np.empty((M, n_F_full, n_red))
            g = np.empty((n_red, M))
            for i in range(n_red):
                a, b = M * (1 + 2 * i), M * (2 + 2 * i)
                J[:, :, i] = ((T_all[:, a:b] - T_all[:, b:b + M])
                              / (2 * h_fd[i])).T
                g[i] = (c_all[a:b] - c_all[b:b + M]) / (2 * h_fd[i])
            return T0, J, c0, g

        # Agreement bound per node (primary model's decoder).
        eps_sub = None
        if AGREE_QUANTILE is not None:
            T_dec_tr, _ = members[0]["dec"](Y_tr)
            D_tr = T_dec_tr - T_linear(Y_tr)
            eps_all = np.maximum(
                np.quantile(np.abs(D_tr), AGREE_QUANTILE, axis=1), 1.0)
            eps_sub = eps_all[sub]
            print(f"  agreement bound ({AGREE_QUANTILE:.2f} quantile over "
                  f"the training states): median {np.median(eps_all):.1f} K,"
                  f" largest {np.max(eps_all):.1f} K")

        def add_members(opti_, U_, X_first):
            """Reduced states of all members, with dynamics and envelope;
            the first member uses the states of setup_optimization_problem."""
            X_list = []
            for j, mb in enumerate(members):
                if j == 0:
                    X_m = X_first
                else:
                    X_m = opti_.variable(n_red, Nt)
                    opti_.subject_to(X_m[:, 0] == ca.DM(y0))
                add_dynamics_constraints(opti_, X_m, U_, mb["integ"],
                                         time_arr, load_interp, input_vector,
                                         N)
                if env_lo is not None:
                    add_state_envelope(opti_, X_m, env_lo, env_hi, Nt)
                X_list.append(X_m)
            return X_list

        def extra_cost(U_, slacks):
            c = (opt_config.alpha_u / N) * ca.sumsqr(U_[:, 1:] - U_[:, :-1])
            for s_ in slacks:
                c += (opt_config.slack_penalty / N) * ca.sumsqr(s_)
            return c + 1e5 * ca.sumsqr(U_[:, 0] - u_cool_start_val)

        # --- Start: linear reconstruction with a lowered limit (or cold) ---
        X0s = [mb["X_init"].astype(float) for mb in members]
        U0 = np.asarray(U_init, dtype=float).reshape(1, -1)
        if SLP_START == "linear":
            print(f">> SLP start: linear reconstruction, limit "
                  f"{constraints.T_hot_max - SLP_START_MARGIN_K:.1f} K, "
                  f"{n_mem} model(s).")
            opti0, Xs_, U_, S_ = setup_optimization_problem(
                n_red, Nt, N, y0, np.ones(n_red),
                u_min_scaled, u_max_scaled,
                ramp_up_step_scaled, ramp_down_step_scaled, dt)
            X_list0 = add_members(opti0, U_, Xs_)
            cost0 = 0
            for X_m in X_list0:
                if CONVERSION_CAP is not None:
                    add_conversion_cap(opti0, X_m,
                                       reduced_to_full_linear_opt_wrapper,
                                       Nt, n_F_full, CONVERSION_CAP)
                add_temperature_constraints(
                    opti0, X_m, S_, reduced_to_full_linear_opt_wrapper,
                    float(constraints.T_hot_max - SLP_START_MARGIN_K), Nt,
                    n_F_full, sample_rate=opt_config.hotspot_sample_rate)
                cost0 += build_objective(X_m, U_, S_,
                                         reduced_to_full_linear_opt_wrapper,
                                         n_F_full, N, dt, 0.0, 0.0, None)
            opti0.minimize(cost0 / n_mem + extra_cost(U_, [S_]))
            for X_m, X_i in zip(X_list0, X0s):
                opti0.set_initial(X_m, ca.DM(X_i))
            opti0.set_initial(U_, ca.DM(U_init))
            opti0.set_initial(S_, ca.DM(np.zeros((1, Nt))))
            opti0.solver('ipopt', ipopt_opts)
            try:
                sol0 = opti0.solve()
                X0s = [np.asarray(sol0.value(X_m), dtype=float)
                       for X_m in X_list0]
                U0 = np.asarray(sol0.value(U_), dtype=float).reshape(1, -1)
            except RuntimeError as e:
                print(f"  ⚠ start problem failed ({e}); cold start instead")

        # --- The linearized problem, built once, parameters per iterate ---
        opti, Xs, U, S = setup_optimization_problem(
            n_red, Nt, N, y0, np.ones(n_red),
            u_min_scaled, u_max_scaled,
            ramp_up_step_scaled, ramp_down_step_scaled, dt)
        X_list = add_members(opti, U, Xs)
        S2 = opti.variable(1, Nt)
        opti.subject_to(S2 >= 0)
        P_Delta = opti.parameter(n_red, 1)
        C_sub = ca.DM(C_T[sub])
        conv_sum = 0
        for mb, X_m in zip(members, X_list):
            mb["P_X0"] = opti.parameter(n_red, Nt)
            mb["P_T0"] = opti.parameter(n_F_full, Nt)
            mb["P_c0"] = opti.parameter(1, Nt)
            mb["P_g"] = opti.parameter(n_red, Nt)
            mb["P_D0"] = opti.parameter(n_sub, Nt)
            mb["P_J"] = [opti.parameter(n_F_full, n_red) for _ in range(Nt)]
            for k in range(1, Nt):
                dx = X_m[:, k] - mb["P_X0"][:, k]
                opti.subject_to(mb["P_T0"][:, k] + mb["P_J"][k] @ dx
                                <= T_limit + S[0, k])
                opti.subject_to(opti.bounded(-P_Delta, dx, P_Delta))
                lin_c = mb["P_c0"][0, k] + ca.dot(mb["P_g"][:, k], dx)
                conv_sum += lin_c
                if CONVERSION_CAP is not None:
                    opti.subject_to(lin_c <= CONVERSION_CAP + 0.01 * S2[0, k])
                if eps_sub is not None:
                    d_k = mb["P_D0"][:, k] + (mb["P_J"][k][list(sub), :]
                                              - C_sub) @ dx
                    opti.subject_to(d_k <= ca.DM(eps_sub) + S2[0, k])
                    opti.subject_to(-d_k <= ca.DM(eps_sub) + S2[0, k])
        opti.minimize(-conv_sum / (N * n_mem) + extra_cost(U, [S, S2]))
        opti.solver('ipopt', ipopt_opts)

        def member_terms(T, c, X):
            # Violation of the limit the iterates are held to, i.e. after the
            # back-off OCP_MARGIN_K, so that "feasible" means the same in
            # the merit, the acceptance and the choice of the best iterate.
            conv = float(np.mean(c[1:]))
            viol_T = max(0.0, float(np.max(T[:, 1:]) - T_limit))
            viol_A = 0.0
            if eps_sub is not None:
                D = (T - T_linear(X))[sub][:, 1:]
                viol_A = max(0.0, float(np.max(np.abs(D) - eps_sub[:, None])))
            return conv, viol_T, viol_A

        def combine(terms):
            conv = float(np.mean([t[0] for t in terms]))
            vT = max(t[1] for t in terms)
            vA = max(t[2] for t in terms)
            return conv, vT, vA, conv - SLP_RHO * (vT + vA)

        print(f">> STRATEGY: successive linearization of the decoder, "
              f"{n_mem} model(s), limit {T_limit:.1f} K, trust radius "
              f"{TR_INIT:.2f} of the training range, agreement "
              f"{'on' if eps_sub is not None else 'off'}.")
        Delta = TR_INIT * y_rng
        conv0, vT0, vA0, merit0 = combine(
            [member_terms(*mb["dec"](X0), X0)
             for mb, X0 in zip(members, X0s)])
        print(f"  start: decoder hotspot violation {vT0:+.1f} K, "
              f"conversion {conv0:.4f}")
        best = None
        if vT0 <= MODIFIER_TOL_K and vA0 <= MODIFIER_TOL_K:
            best = {"conv": conv0, "X": X0s, "U": U0, "S": np.zeros(Nt)}
        slp_iters = 0
        stop_msg = "stop: iteration limit"
        for it in range(SLP_MAX_ITER):
            slp_iters = it + 1
            lins = []
            for mb, X0 in zip(members, X0s):
                T0, J, c0, g = decoder_linearization(mb["dec"], X0)
                lins.append((T0, J, c0, g))
                opti.set_value(mb["P_X0"], X0)
                opti.set_value(mb["P_T0"], T0)
                opti.set_value(mb["P_c0"], c0.reshape(1, -1))
                opti.set_value(mb["P_g"], g)
                opti.set_value(mb["P_D0"], (T0 - T_linear(X0))[sub])
                for k in range(Nt):
                    opti.set_value(mb["P_J"][k], J[k])
            opti.set_value(P_Delta, Delta.reshape(-1, 1))
            for X_m, X0 in zip(X_list, X0s):
                opti.set_initial(X_m, X0)
            opti.set_initial(U, U0)
            opti.set_initial(S, np.zeros((1, Nt)))
            opti.set_initial(S2, np.zeros((1, Nt)))
            # A solve that stops early (iteration limit, restoration) still
            # returns its last iterate; the ratio test below decides whether
            # it is a usable step, exactly as for a converged solve.
            try:
                sol = opti.solve()
                status = ""
            except RuntimeError:
                sol = opti.debug
                try:
                    status = f" [{opti.stats().get('return_status', 'failed')}]"
                except Exception:
                    status = " [failed]"
            X1s = [np.asarray(sol.value(X_m), dtype=float) for X_m in X_list]
            U1 = np.asarray(sol.value(U), dtype=float).reshape(1, -1)
            if not all(np.all(np.isfinite(X1)) for X1 in X1s):
                print(f"[SLP {it + 1}/{SLP_MAX_ITER}]{status} non-finite "
                      f"iterate; radius halved")
                Delta = 0.5 * Delta
                if np.max(Delta / y_rng) < 1e-3:
                    stop_msg = "stop: trust radius below 1e-3 of the range."
                    break
                continue

            pred, act = [], []
            for mb, X0, X1, (T0, J, c0, g) in zip(members, X0s, X1s, lins):
                dX = X1 - X0
                T_pred = T0 + np.einsum('kji,ik->jk', J, dX)
                c_pred = c0 + np.sum(g * dX, axis=0)
                pred.append(member_terms(T_pred, c_pred, X1))
                act.append(member_terms(*mb["dec"](X1), X1))
            _, _, _, merit_pred = combine(pred)
            ca_, vTa, vAa, merit_act = combine(act)
            denom = merit_pred - merit0
            ratio = (merit_act - merit0) / denom if denom > 1e-9 else -1.0
            step = max(float(np.max(np.abs(X1 - X0) / Delta[:, None]))
                       for X0, X1 in zip(X0s, X1s))
            at_edge = step > 0.9
            hot = max(float(np.max(mb["dec"](X1)[0][:, 1:]))
                      for mb, X1 in zip(members, X1s))
            print(f"[SLP {it + 1}/{SLP_MAX_ITER}]{status} decoder hotspot "
                  f"{hot:.1f} K (violation {vTa:+.1f} K), agreement excess "
                  f"{vAa:.1f} K, conversion {ca_:.4f}, ratio {ratio:+.2f}, "
                  f"radius {np.mean(Delta / y_rng):.3f}")

            if ratio > 0.1:
                X0s, U0, merit0 = X1s, U1, merit_act
                if ratio > 0.75 and at_edge:
                    Delta = np.minimum(2.0 * Delta, 0.5 * y_rng)
                if vTa <= MODIFIER_TOL_K and vAa <= MODIFIER_TOL_K and \
                        (best is None or ca_ > best["conv"]):
                    best = {"conv": ca_, "X": X1s, "U": U1,
                            "S": np.asarray(sol.value(S)).flatten()}
                    print("     ★ best feasible solution so far")
                rel = max(float(np.max(np.abs(X1 - X0) / y_rng[:, None]))
                          for X0, X1 in zip(X0s, X1s))
                if not at_edge and rel < 1e-3:
                    stop_msg = "✅ converged: the step no longer moves."
                    break
            else:
                Delta = 0.5 * Delta
            if np.max(Delta / y_rng) < 1e-3:
                stop_msg = "stop: trust radius below 1e-3 of the range."
                break
        print(f"  {stop_msg}")

        if best is None:
            print("⚠ no accepted iterate met both limits; keeping the last "
                  "accepted one.")
            best = {"conv": None, "X": X0s, "U": U0, "S": np.zeros(Nt)}
        Xs_opt_final = best["X"][0]
        U_opt_final = best["U"].flatten() * scale_Tcool
        S_opt_final = best["S"]
        loops_used = slp_iters
        if n_mem > 1:
            ensemble_stats = []
            for mb, X_b in zip(members, best["X"]):
                T_b, c_b = mb["dec"](X_b)
                ensemble_stats.append({
                    "model": mb["name"],
                    "T_max_decoder_K": float(np.max(T_b[:, 1:])),
                    "mean_conversion_decoder": float(np.mean(c_b[1:]))})
                print(f"  {mb['name']}: decoder hotspot "
                      f"{ensemble_stats[-1]['T_max_decoder_K']:.1f} K, "
                      f"conversion "
                      f"{ensemble_stats[-1]['mean_conversion_decoder']:.4f}")

    # -------------------------------------------------------------------------
    # OPTION E: ENSEMBLE OF MODELS, LINEAR RECONSTRUCTION, LOWERED LIMIT
    # -------------------------------------------------------------------------
    elif STRATEGY == "ensemble":
        if not ENSEMBLE_MODELS:
            raise ValueError("strategy ensemble needs OCP_ENSEMBLE")
        others = [load_results(Path("data") / m, decoder_class=decoder_class)
                  for m in ENSEMBLE_MODELS]
        for name, res in zip(ENSEMBLE_MODELS, others):
            same = (np.allclose(np.asarray(res["basis"]), basis)
                    and np.allclose(
                        np.asarray(res["reference_states_shifting"]),
                        reference_states_shifting)
                    and list(res["input_names"]) == list(input_names))
            if not same:
                raise ValueError(f"{name} has another basis, shift or input "
                                 f"set than {file_name}")
        T_limit = constraints.T_hot_max - MARGIN_K
        print(f">> STRATEGY: ensemble of {1 + len(others)} models, linear "
              f"reconstruction, limit {T_limit:.1f} K.")
        opti, Xs, U, S = setup_optimization_problem(
            n_red, Nt, N, y0, np.ones(n_red),
            u_min_scaled, u_max_scaled,
            ramp_up_step_scaled, ramp_down_step_scaled, dt
        )
        add_dynamics_constraints(opti, Xs, U, integrator, time_arr,
                                 load_interp, input_vector, N)
        X_vars = [Xs]
        X_inits = [Xs_init]
        for res in others:
            integ_m = build_casadi_integrator(res["A"], res["H"], res["B"],
                                              res["C"], n_red, dt)
            X_m = opti.variable(n_red, Nt)
            opti.subject_to(X_m[:, 0] == ca.DM(y0))
            add_dynamics_constraints(opti, X_m, U, integ_m, time_arr,
                                     load_interp, input_vector, N)
            X_vars.append(X_m)
            U_prof = input_matrix(load_interp(time_arr_raw), u_min_scaled)
            X_sim = forward_sim_reduced(y0, time_arr_raw, U_prof, res["A"],
                                        res["H"], res["B"], res["C"], n_red)
            X_inits.append(X_sim[:, ::stride][:, :Nt])
        cost = 0
        for X_m in X_vars:
            if env_lo is not None:
                add_state_envelope(opti, X_m, env_lo, env_hi, Nt)
            if CONVERSION_CAP is not None:
                add_conversion_cap(opti, X_m,
                                   reduced_to_full_linear_opt_wrapper, Nt,
                                   n_F_full, CONVERSION_CAP)
            add_temperature_constraints(
                opti, X_m, S, reduced_to_full_linear_opt_wrapper,
                float(T_limit), Nt, n_F_full,
                sample_rate=opt_config.hotspot_sample_rate)
            cost += build_objective(X_m, U, S,
                                    reduced_to_full_linear_opt_wrapper,
                                    n_F_full, N, dt, 0.0, 0.0, None)
        cost = cost / len(X_vars)
        cost += (opt_config.alpha_u / N) * ca.sumsqr(U[:, 1:] - U[:, :-1])
        cost += (opt_config.slack_penalty / N) * ca.sumsqr(S)
        cost += 1e5 * ca.sumsqr(U[:, 0] - u_cool_start_val)
        opti.minimize(cost)
        for X_m, X_i in zip(X_vars, X_inits):
            opti.set_initial(X_m, ca.DM(X_i))
        opti.set_initial(U, ca.DM(U_init))
        opti.set_initial(S, ca.DM(np.zeros((1, Nt))))
        opti.solver('ipopt', ipopt_opts)
        try:
            sol = opti.solve()
            print("\nSOLVER SUCCESS")
        except RuntimeError as e:
            print(f"\n⚠ SOLVER FAILED: {e}")
            sol = opti.debug
        Xs_opt_final = np.array(sol.value(Xs))
        U_opt_final = np.array(sol.value(U)).flatten() * scale_Tcool
        S_opt_final = np.array(sol.value(S)).flatten()

        # Every model with its own decoder.
        ensemble_stats = []
        for name, res, X_m in zip([file_name] + ENSEMBLE_MODELS,
                                  [results] + others, X_vars):
            dec_m = res.get("decoder")
            if dec_m is not None:
                dec_m = dec_m.to(device).eval()
            sc_m = (res.get("input_scaler"), res.get("target_scaler_F"),
                    res.get("target_scaler_T"))
            X_val = np.array(sol.value(X_m))
            full_m = np.asarray(reduced_to_full_numpy(
                X_val, basis_type, r_F, r_T, sF, sT, basis,
                reference_states_shifting, state_max_F=max_F,
                state_max_T=max_T, V_reduced_nonlin=V_reduced_nonlin, Xi=Xi,
                poly_degree=3, has_decoder=dec_m is not None, decoder=dec_m,
                scalers=sc_m, n_F_full=n_F_full, device=device))
            stat = {"model": name,
                    "T_max_decoder_K": float(np.max(
                        full_m[n_F_full:2 * n_F_full])),
                    "mean_conversion_decoder": float(np.mean(
                        full_m[n_F_full - 1]))}
            ensemble_stats.append(stat)
            print(f"  {name}: decoder hotspot {stat['T_max_decoder_K']:.1f} "
                  f"K, conversion {stat['mean_conversion_decoder']:.4f}")

    # -------------------------------------------------------------------------
    # OPTION B: SINGLE PASS, with the decoder or with the linear
    # reconstruction and a lowered limit
    # -------------------------------------------------------------------------
    else:
        if STRATEGY == "decoder":
            print(">> STRATEGY: single pass, decoder in constraints and "
                  "objective.")
            constraint_wrapper = reduced_to_full_casadi_wrapper
            T_limit = constraints.T_hot_max
        else:
            print(f">> STRATEGY: single pass, linear reconstruction, limit "
                  f"lowered by {MARGIN_K:.1f} K.")
            constraint_wrapper = reduced_to_full_linear_opt_wrapper
            T_limit = constraints.T_hot_max - MARGIN_K

        # Setup
        opti, Xs, U, S = setup_optimization_problem(
            n_red, Nt, N, y0, np.ones(n_red),
            u_min_scaled, u_max_scaled,
            ramp_up_step_scaled, ramp_down_step_scaled, dt
        )

        add_dynamics_constraints(opti, Xs, U, integrator, time_arr, load_interp, input_vector, N)
        if env_lo is not None:
            add_state_envelope(opti, Xs, env_lo, env_hi, Nt)
        if CONVERSION_CAP is not None:
            add_conversion_cap(opti, Xs, constraint_wrapper, Nt, n_F_full,
                               CONVERSION_CAP)
        add_temperature_constraints(
            opti, Xs, S, constraint_wrapper,
            float(T_limit), Nt, n_F_full,
            sample_rate=opt_config.hotspot_sample_rate
        )

        # Objective
        cost = build_objective(
            Xs, U, S, constraint_wrapper,
            n_F_full, N, dt,
            opt_config.alpha_u, opt_config.slack_penalty,
            u_start_val=u_cool_start_val
        )
        opti.minimize(cost)

        # Initialize & Solve
        opti.set_initial(Xs, ca.DM(Xs_init))
        opti.set_initial(U, ca.DM(U_init))
        opti.set_initial(S, ca.DM(np.zeros((1, Nt))))
        opti.solver('ipopt', ipopt_opts)

        try:
            sol = opti.solve()
            print("\nSOLVER SUCCESS")
        except RuntimeError as e:
            print(f"\n⚠ SOLVER FAILED: {e}")
            sol = opti.debug

        # Extract Results
        Xs_opt_final = np.array(sol.value(Xs))
        U_opt_final = np.array(sol.value(U)).flatten() * scale_Tcool
        S_opt_final = np.array(sol.value(S)).flatten()

    # =========================================================================
    # 9. POST-PROCESSING & SAVING
    # =========================================================================

    # --- RECONSTRUCTION (Linear vs. CNN) ---
    print("\nReconstructing full physical solutions...")

    # 1. Decoder -> the metrics.
    sol_full_cnn = reduced_to_full_cnn_check_wrapper(Xs_opt_final)
    T_hot_cnn = np.max(sol_full_cnn[n_F_full : 2*n_F_full, :], axis=0)
    conv_outlet = sol_full_cnn[:n_F_full, :][-1, :]

    # 2. Linear reconstruction -> what the optimisation model saw.
    sol_full_linear = reduced_to_full_linear_check_wrapper(Xs_opt_final)
    T_hot_linear = np.max(sol_full_linear[n_F_full : 2*n_F_full, :], axis=0)
    conv_linear = sol_full_linear[:n_F_full, :][-1, :]

    # --- METRICS (decoder) ---
    T_max_opt = float(np.max(T_hot_cnn))
    mean_conv = np.mean(conv_outlet)
    max_slack = np.max(S_opt_final)

    # Share of the optimised trajectory outside the state envelope.
    share_outside = None
    if env_lo is not None:
        X_opt = np.asarray(Xs_opt_final)
        outside = np.maximum(env_lo[:, None] - X_opt, 0) + \
            np.maximum(X_opt - env_hi[:, None], 0)
        share_outside = float(np.mean(np.any(outside > 1e-6, axis=0)))

    # --- INTERPOLATION & PADDING (Restore full timeline) ---
    print("Preparing final control trajectory for export...")
    u_interp_func = interp1d(time_arr[:-1], U_opt_final, kind='previous', fill_value="extrapolate")
    U_opt_fine_sliced = u_interp_func(time_arr_raw)

    if start_idx > 0:
        pad_value = U_opt_fine_sliced[0]
        U_pad = np.full(start_idx, pad_value)
        U_opt_full_export = np.concatenate([U_pad, U_opt_fine_sliced])
    else:
        U_opt_full_export = U_opt_fine_sliced

    # --- SAVE STATIC CONTROL CANDIDATES FOR MECHANISTIC BASELINE CHECK ---
    static_dir = RESULTS_DIR / "static_baselines"
    static_dir.mkdir(exist_ok=True, parents=True)

    u_mid = 0.5 * (constraints.u_min + constraints.u_max)
    candidate_Tcool_values = {
        f"umin_{constraints.u_min:.0f}K": constraints.u_min,
        "500K": 500.0,
        "520K": 520.0,
        "540K": 540.0,
        "560K": 560.0,
        f"umid_{u_mid:.0f}K": u_mid,
        f"umax_{constraints.u_max:.0f}K": constraints.u_max,
    }

    for label, u_val in candidate_Tcool_values.items():
        u_static = np.full_like(U_opt_full_export, u_val)
        np.save(static_dir / f"u_static_{label}.npy", u_static)
        print(f"Saved static baseline {label}: Tcool = {u_val:.2f} K")

    print(f"  Final Export Shape: {U_opt_full_export.shape}")

    # --- PRINT FINAL STATS ---
    print(f"\n" + "="*40)
    print(f"FINAL OPTIMIZATION RESULTS")
    print(f"="*40)
    print(f"  Strategy:         {STRATEGY}")
    print(f"  Max Reactor Temp: {T_max_opt:.2f} K decoder, "
          f"{float(np.max(T_hot_linear)):.2f} K linear "
          f"(Limit: {constraints.T_hot_max} K)")
    print(f"  Mean Conversion:  {mean_conv:.4f} decoder, "
          f"{float(np.mean(conv_linear)):.4f} linear")
    print(f"  Max Slack Usage:  {max_slack:.2e}")
    print(f"  Max Model Error:  {np.max(np.abs(T_hot_cnn - T_hot_linear)):.2f} K")
    if share_outside is not None:
        print(f"  Outside envelope: {share_outside:.1%} of the time points")

    # --- SAVE RESULTS ---
    print(f"\nSaving results to: {RESULTS_DIR}")

    # Save Full Trajectory for Mechanistic Check
    np.save(RESULTS_DIR / 'u_opt.npy', U_opt_full_export)

    # Save Intermediate Coarse Results
    np.save(RESULTS_DIR / 'u_opt_coarse.npy', U_opt_final)
    np.save(RESULTS_DIR / 'time_arr_coarse.npy', time_arr)
    np.save(RESULTS_DIR / 'slack_opt.npy', S_opt_final)

    # SAVE BOTH SOLUTIONS (CNN & LINEAR)
    np.save(RESULTS_DIR / 'sol_full_coarse_cnn.npy', sol_full_cnn)
    np.save(RESULTS_DIR / 'sol_full_coarse_linear.npy', sol_full_linear)

    # One summary per run, for comparing variants.
    summary = {
        "model": file_name, "strategy": STRATEGY,
        "envelope_margin": ENVELOPE_MARGIN, "conversion_cap": CONVERSION_CAP,
        "margin_K": (MARGIN_K if STRATEGY in ("linear_margin", "modifier",
                                              "slp") else None),
        "damping": MODIFIER_DAMPING if STRATEGY == "modifier" else None,
        "u_bounds_K": [float(constraints.u_min), float(constraints.u_max)],
        "T_limit_K": float(constraints.T_hot_max),
        "T_max_decoder_K": T_max_opt,
        "T_max_linear_K": float(np.max(T_hot_linear)),
        "mean_conversion_decoder": float(mean_conv),
        "mean_conversion_linear": float(np.mean(conv_linear)),
        "max_mismatch_K": float(np.max(np.abs(T_hot_cnn - T_hot_linear))),
        "max_slack": float(max_slack),
        "share_outside_envelope": share_outside,
        "loops": int(loops_used),
        "agree_quantile": AGREE_QUANTILE if STRATEGY == "slp" else None,
        "ensemble": ensemble_stats,
        "runtime_s": float(time.time() - t_start),
    }
    with open(RESULTS_DIR / "ocp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved the run summary to {RESULTS_DIR / 'ocp_summary.json'}")

    # --- PLOTTING ---
    print("Generating summary plot...")
    plot_dir = RESULTS_DIR / 'plots'
    plot_dir.mkdir(exist_ok=True, parents=True)

    create_summary_plot(
        time_arr, U_opt_final,
        T_hot_cnn, conv_outlet,
        load_profile * scale_load,
        constraints.T_hot_max, constraints.u_min, constraints.u_max,
        save_path=plot_dir / 'summary.png',
        T_hot_linear=T_hot_linear,
        conv_linear=conv_linear
    )

if __name__ == "__main__":
    main()