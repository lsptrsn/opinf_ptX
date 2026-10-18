"""Operator inference by gradient descent.

The training loop is unchanged in its numerical path. What was added is
measurement, and what was corrected is machinery that did not do what its name
said:

  * The learning-rate ramp is sized in optimizer steps, not epochs. The
    scheduler advances once per batch, so an epoch-sized ramp only matches the
    intent while there are several batches per epoch. With a full batch the
    ramp ran longer than the whole run and training ended at the highest
    learning rate it ever used, with no annealing.

  * The derivative residual handed back to the caller is computed over the
    whole dataset. It used to be whatever the final loop iteration left in
    scope, which is the full dataset only while the batch is full. It is
    reported as LAST_DERIVATIVE_RESIDUAL, one value per call; see the note at
    that constant before reading it after a run with memory terms.

  * Gradient clipping was called once before training, when no gradient
    exists, and therefore never clipped anything. The gradient norm is now
    reported so that a clipping threshold can be chosen from evidence; the
    clipping itself is off unless CLIP_GRAD_NORM is set.

  * The two terms of the hybrid loss are reported separately together with
    their effective weights. The auto-balancing scales are estimated once, on
    the first batch, from randomly initialised weights and then frozen, so the
    weighting that acts at the end of training is not the configured one
    unless the two terms happen to fall at the same rate.

  * The one-step term integrates with explicit Runge-Kutta at the sampling
    interval, which carries a stability limit the model class does not
    respect. A = J - RR^T places the eigenvalues in the closed left half
    plane, so the continuous model is bounded, but J is skew-symmetric and
    nothing bounds the magnitude of its imaginary part. Explicit RK4 is stable
    on the imaginary axis only up to |lambda| dt = 2*sqrt(2); at dt = 10 s
    that is an oscillation period of 22 s. A mode faster than that makes the
    training term amplify rather than damp, and the same spectrum forces the
    delay solver into very small steps afterwards. The spectrum is now
    reported so this can be seen instead of inferred, and RK4_SUBSTEPS buys
    headroom proportionally.

Defaults reproduce the previous behaviour exactly.
"""

__all__ = ["train_model", "learned_model"]

from copy import deepcopy

import numpy as np
import torch
import torch.utils.data

import opinf

# Load global params pointer (Singleton)
Params = opinf.parameters.Params()
device = opinf.utils.get_device()


# --- switches -----------------------------------------------------------------
# Maximum gradient norm, or None for no clipping. None reproduces the previous
# behaviour: the old call site could not clip because it ran before any
# backward pass. Choose a value from the reported gradient norms, not from
# habit.
CLIP_GRAD_NORM = None

# Re-estimate the hybrid auto-balancing scales every n epochs, or None to
# estimate them once on the first batch and freeze them. None is the previous
# behaviour.
LOSS_SCALE_REFRESH_EPOCHS = None

# Epochs without improvement before training stops. None follows
# Params.num_epochs, which makes early stopping inactive, as before.
EARLY_STOPPING_PATIENCE = None

# An epoch counts as an improvement when the average loss falls by more than
# this.
IMPROVEMENT_THRESHOLD = 1e-9

# Where the auto-balancing scales come from.
#   "init" : the loss values of the first batch, that is of the randomly
#            initialised model. Previous behaviour, and the default.
#   "data" : the ratio of the two scales from the data alone, with the
#            magnitude pinned to LOSS_SCALE_MAGNITUDE below.
#
# Under "init" the seed sets the objective and not merely the starting point.
# On the thermal case the constant that normalises the derivative term comes
# out at 2.9e+01 while the mean square of the derivative target is 1.2e-07:
# the normaliser is 2.8e5 times the size of the data it normalises, and is
# fixed by how far the random weights happen to sit from the target. The state
# scale then varies by a factor of 6.8 across three seeds, so lambda_deriv
# trades off two distances from a random starting point rather than two data
# terms.
#
# Under "data" the ratio is mean((x_{k+1} - x_k)^2) / mean(ddt^2), both
# properties of the record. On the same case that is 570, against 215 for the
# scales in use, so this changes the ratio by 2.7x and not the magnitude by
# 1e5 as an earlier attempt did. That attempt failed for a reason worth
# recording: the regularization terms are not scaled with the data term, so
# multiplying the data term by 1e5 divided the effective strength of lambda_H
# by the same factor and switched the penalty off.
LOSS_SCALE_REFERENCE = "init"

# Geometric mean the two scales are pinned to under "data". It is redundant
# with the regularization strengths: multiplying the data term by c and
# lambda_H by c is the same optimisation problem, so only one of the two is
# worth searching. Held at the value the "init" reference produced on the
# reference run, sqrt(2.90e+01 * 1.35e-01), so that switching the reference
# changes the ratio and nothing else.
LOSS_SCALE_MAGNITUDE = 1.978

# Substeps per sampling interval in the one-step term. 1 is the previous
# behaviour: a single RK4 step across the whole interval. Each substep divides
# the effective step size and multiplies the stable range of |lambda| by the
# same factor, at a proportional cost in time. The prediction target is
# unchanged, so the loss stays comparable across settings.
RK4_SUBSTEPS = 1

# Report the spectrum of the learned linear operator after training, against
# the RK4 stability limit that the one-step term implies.
REPORT_SPECTRUM = True

# Horizons, in sampling intervals, over which the state term is evaluated.
# (1,) is the previous behaviour and is bit-identical to it.
#
# The reason to look further is that the objective and the reported metric do
# not agree. Measured on five runs of the thermal case, the best-fitting model
# was among the worst rollouts: one run reached a derivative loss three times
# the trivial predictor and a state loss 1.3 times it, both the best of the
# set, and rolled out eleven times worse than the reference, whose two terms
# are 38 and 4.4 times worse than trivial. A term that looks one sampling
# interval ahead sees the state move by under one percent of its range and
# can say nothing about what accumulates over 2703 of them.
#
# Several horizons are averaged. Keep the list short: the cost is the sum of
# the horizons in forward passes per batch, so (1, 10, 50) costs 61 passes
# where (1,) costs one.
MULTISTEP_HORIZONS = (50,)

# How the horizons are combined.
#   "none"    : plain mean of the losses. The longest horizon dominates,
#               because its error is largest. Previous behaviour for a single
#               horizon, and the default.
#   "trivial" : each horizon is divided by the loss of predicting no change
#               over that horizon, so every term is one for a model that does
#               nothing and the horizons enter comparably.
MULTISTEP_WEIGHTING = "none"

# How the fields of the reduced state are weighted in both loss terms.
#   "none"  : plain mean over all coordinates, the previous behaviour. Each
#             field then weighs in proportion to its number of modes: with
#             r_F = 1 and r_T = 4 the flow field is a fifth of the loss.
#   "equal" : the mean squared error of each field separately, then the mean
#             over fields, so flow and temperature count equally whatever
#             their ranks. This is the weighting of the reported metric,
#             which averages the two fields.
FIELD_WEIGHTING = "none"

# Sizes of the consecutive field blocks of the reduced state, set by the main
# script before training, e.g. (r_F, r_T). Coordinates beyond their sum form
# one more block. Without it, "equal" falls back to the plain mean.
FIELD_SIZES = None

# Epochs during which only the shortest horizon is used, whatever
# MULTISTEP_HORIZONS says. Beyond one horizon this is not optional: at
# initialisation the operators are drawn at randn/10, which puts |lambda|*h
# well above the RK4 stability limit, so a fifty-step rollout from those
# weights overflows on the third epoch while a single step survives it. The
# warm-up lets the spectrum settle before the long horizons are switched on.
# Ignored when MULTISTEP_HORIZONS has one entry.
MULTISTEP_WARMUP_EPOCHS = 150

# Report the spectrum every n epochs during training, 0 for never. The report
# after training looks at the best model, which is exactly the one that did
# not diverge; a periodic one shows where the eigenvalues go while the fit is
# running, which is where the interesting part happens.
SPECTRUM_EVERY_EPOCHS = 0

# Horizons at which the rollout error is reported after training and recorded
# on Params. Diagnostic only: it changes no gradient. The point is to find a
# training-side quantity that predicts the reported metric, which no quantity
# currently does; without one, neither the search nor a multi-start selection
# can work. Empty tuple to switch off.
ROLLOUT_DIAGNOSTIC_HORIZONS = (1, 10, 50, 200)

# What a divergence does.
#   "raise" : abort the run. Previous behaviour and the default.
#   "stop"  : keep the best weights seen so far, stop training, and carry on
#             to the evaluation. A diverged run then still produces a rollout
#             and an error, which is the difference between a data point and
#             four wasted minutes. The message says plainly that the numbers
#             come from an interrupted fit.
ON_DIVERGENCE = "stop"

# Reset the best loss when the active horizons change. Before the switch the
# loss is the warm-up objective, after it a different one, and comparing the
# two can return a warm-up checkpoint as the trained model: in the
# (1, 50, 200) run with "none" weighting the best loss, 1.64e-03, came from
# before epoch 151, and every loss after the switch stayed above it. True
# returns the same weights as before for every run whose best epoch lies
# after the switch, which includes every v4 and v5 run.
RESET_BEST_ON_HORIZON_CHANGE = True

# Horizons dropped from the state term because their rollout was not finite,
# counted per horizon over one fit.
_DROPPED_HORIZONS = {}

ABORT_ON_DIVERGENCE = True
MAX_NONFINITE_BATCHES = 3
DIVERGENCE_LOSS_FACTOR = 1e6

# Relative residual of the derivative fit, left here by the most recent call to
# train_model. It is named for what it computes: the target is the derivative
# estimate, not the state one sampling interval ahead. The one-step quantity is
# the state term of the hybrid loss and is a different number; calling this one
# a one-step residual would put the wrong name in the manuscript.
#
# Every call overwrites it. The memory-term regression calls train_model once
# per lag, so after a run with lag_max > 0 what stands here belongs to the last
# lag rather than to the Markovian fit. A caller that wants the Markovian value
# has to read it directly after its own call, which is what the main script
# does.
#
# Read it as ``module.LAST_DERIVATIVE_RESIDUAL``. A from-import binds the NaN
# below once and never sees an update.
LAST_DERIVATIVE_RESIDUAL = float('nan')

# Rollout error by horizon, left here by the most recent call. Same overwrite
# rule as above: read it directly after the call whose model you mean. The
# caller finds the same values on Params.rollout.
LAST_ROLLOUT = {}


def verify_H_energy_preservation(model, r, n_samples=200, verbose=True):
    """Numerical check that q^T H (q ⊗ q) = 0 for random q near 0.

    The antisymmetrised H should be energy-preserving. The residual is
    normalised by ||q||^3 because H(q⊗q) is cubic in q.
    """
    m = model.module if hasattr(model, 'module') else model
    H = getattr(m, 'H', None)
    if H is None:
        if verbose:
            print("  H energy-preservation check: no H operator, skipping")
        return None
    with torch.no_grad():
        H_np = H.detach().cpu().numpy()
        rng = np.random.default_rng(0)
        q = rng.standard_normal((n_samples, r)) * 0.1
        qkron = np.einsum('bi,bj->bij', q, q).reshape(n_samples, r * r)
        Hq = qkron @ H_np.T
        val = np.einsum('bi,bi->b', q, Hq)
        ref = np.linalg.norm(q, axis=1) ** 3
        ratio = np.abs(val) / np.maximum(ref, 1e-30)
    if verbose:
        print(f"  H energy-preservation: max |q^T H(q⊗q)|/||q||^3 = "
              f"{float(ratio.max()):.3e}, max |q^T H(q⊗q)| = "
              f"{float(np.abs(val).max()):.3e}")
    return float(ratio.max())


def train_model(states, derivatives, time, entries, rom, seed=None,
                scale_reference=None):
    """Fit the reduced operators to the latent states and their derivatives.

    Parameters
    ----------
    states, derivatives : (r, k) ndarray
        Latent trajectories and their time derivatives.
    time : (k,) ndarray or scalar
        Time vector, or a scalar step size.
    entries : (m, k) ndarray
        Input trajectories.
    rom : torch.nn.Module
        Model produced by ``opinf.models.create_rom``.
    seed : int, optional
        Seed for the data loader and for any stochastic step.
    scale_reference : str, optional
        Overrides LOSS_SCALE_REFERENCE for this call. The memory-term
        regressions pass "init": their target is the residual left by the
        previous fit rather than a measured derivative, so mean(ddt^2) there
        describes the last fit and not the record. Measured, the data
        reference collapsed the ratio from 570 to 0.0 on those lags and
        switched the derivative term off.
    """
    current_seed = seed if seed is not None else 0
    opinf.utils.set_deterministic(seed=current_seed)

    # Cleared here so that a failed or aborted fit cannot leave the previous
    # call's residual standing, where a caller would read it as its own.
    globals()['LAST_DERIVATIVE_RESIDUAL'] = float('nan')
    globals()['LAST_ROLLOUT'] = {}
    Params.ddt_residual = float('nan')
    Params.rollout = {}

    # 1. Tensors, one snapshot per row.
    states_torch = torch.tensor(states, device=device, dtype=torch.float32).T
    derivatives_torch = torch.tensor(
        derivatives, device=device, dtype=torch.float32).T
    entries_torch = torch.tensor(entries, device=device, dtype=torch.float32).T

    if isinstance(time, np.ndarray):
        time_torch = torch.tensor(
            time, device=device, dtype=torch.float32).reshape(-1, 1)
    else:
        dt = time if np.isscalar(time) else 1.0
        t_vec = torch.arange(states.shape[1], device=device) * dt
        time_torch = t_vec.reshape(-1, 1)
    if Params.output and isinstance(time, np.ndarray):
        n_seams = int(np.count_nonzero(np.diff(np.asarray(time).ravel()) <= 0))
        print(f"  {n_seams} trajectory boundaries in the training data; "
              f"windows across them are dropped")

    # 2. Batching.
    n_samples = states.shape[1]

    if Params.batch_size > n_samples:
        if Params.output:
            print(f"Info: batch size ({Params.batch_size}) > samples "
                  f"({n_samples}). Using full batch.")
        effective_batch_size = n_samples
        drop_last = False
    else:
        effective_batch_size = Params.batch_size
        drop_last = True

    train_dataset = torch.utils.data.TensorDataset(
        states_torch, time_torch, derivatives_torch, entries_torch
    )

    # The state loss steps from one sample to the next, so the batch has to
    # stay in chronological order whenever that term is active.
    is_sequential_loss = Params.ROM_loss in ['states', 'hybrid']
    do_shuffle = not is_sequential_loss

    g = torch.Generator()
    g.manual_seed(current_seed)

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=do_shuffle,
        drop_last=drop_last,
        generator=g,
    )
    dataloaders = {"train": train_dl}

    # 3. Optimizer.
    optimizer = torch.optim.AdamW(
        rom.parameters(),
        lr=Params.adam_lr,
        betas=Params.adam_betas,
        eps=Params.adam_eps,
        weight_decay=Params.adam_weight_decay,
    )

    # 4. Cyclic learning rate.
    #
    # The scheduler advances once per optimizer step, that is once per batch.
    # Sizing the ramp in epochs only works while the number of batches per
    # epoch is fixed; with a full batch there is one step per epoch, and a ramp
    # of lr_schedule_step_factor epochs is longer than the run. Training then
    # ends part-way up the ramp, at the highest learning rate it ever used,
    # with no annealing at all. The ramp is therefore capped so that the
    # configured number of epochs covers a whole number of cycles and the run
    # finishes near base_lr.
    steps_per_epoch = max(1, len(train_dl))
    total_steps = max(1, Params.num_epochs * steps_per_epoch)
    step_epochs = int(Params.lr_schedule_step_factor)
    step_size_up = max(1, min(step_epochs * steps_per_epoch, total_steps // 2))

    if Params.output:
        n_cycles = total_steps / (2 * step_size_up)
        print(f"  LR schedule: {step_size_up} steps up and down, "
              f"{total_steps} steps total, {n_cycles:.2f} cycles, "
              f"base {Params.lr_schedule_base_lr:.2e}, "
              f"max {Params.lr_schedule_max_lr:.2e}")

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        step_size_up=step_size_up,
        step_size_down=step_size_up,
        mode=Params.lr_schedule_mode,
        cycle_momentum=False,
        base_lr=Params.lr_schedule_base_lr,
        max_lr=Params.lr_schedule_max_lr,
    )

    # 5. Fit.
    full_batch = (states_torch, time_torch, derivatives_torch, entries_torch)
    model, loss_track, error_reduced_states = _fit(
        rom, dataloaders, optimizer, scheduler=scheduler,
        full_batch=full_batch, scale_reference=scale_reference
    )

    return model, loss_track, error_reduced_states


def learned_model(model):
    """Extract the learned operators as numpy arrays."""
    m = model.module if hasattr(model, 'module') else model

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor is not None else None

    sys_order = Params.ROM_order

    A = (to_numpy(m.A)
         if 'A' in Params.model_structure and m.A is not None
         else np.zeros((sys_order, sys_order)))
    B = (to_numpy(m.B)
         if 'B' in Params.model_structure and m.B is not None
         else np.zeros((sys_order, Params.input_dim)))
    C = (to_numpy(m.C).reshape(-1)
         if 'C' in Params.model_structure and m.C is not None
         else np.zeros((sys_order,)))
    H = (to_numpy(m.H)
         if 'H' in Params.model_structure and m.H is not None
         else np.zeros((sys_order, sys_order ** 2)))

    return A, B, C, H


def _rk4th_onestep(model, x, u, t, timestep, substeps=None):
    """Advance one sampling interval with explicit Runge-Kutta of order four.

    The input is held constant over the interval, which is what the data
    support: the inputs are sampled on the same grid as the states.

    ``substeps`` splits the interval into that many equal RK4 steps. One step
    is the previous behaviour. The point is stability, not accuracy: explicit
    RK4 is stable on the imaginary axis only for |lambda| h < 2*sqrt(2), and
    the reduced operator is free to carry modes faster than that because the
    parameterisation bounds the sign of the eigenvalues and not their size.
    Halving h doubles the admissible |lambda|.
    """
    if timestep.ndim == 1:
        timestep = timestep.view(-1, 1)

    n = max(1, int(RK4_SUBSTEPS if substeps is None else substeps))
    h = timestep / n

    for i in range(n):
        t_i = t + i * h
        k1 = model(x, t_i, u)
        k2 = model(x + 0.5 * h * k1, t_i + 0.5 * h, u)
        k3 = model(x + 0.5 * h * k2, t_i + 0.5 * h, u)
        k4 = model(x + 1.0 * h * k3, t_i + 1.0 * h, u)
        x = x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h

    return x


def _handle_divergence(reason, epoch, best_loss):
    """True when the caller should stop training and keep the best weights.

    Raising throws away the best model seen so far, which is often perfectly
    usable and is the only thing a four-minute run produced. Under
    ON_DIVERGENCE = "stop" it is kept instead and the evaluation goes ahead,
    with the interruption stated so that no number from such a run is
    mistaken for a completed fit.
    """
    if ON_DIVERGENCE != "stop":
        return False
    print(f"  training diverged and was stopped: {reason}")
    print(f"    keeping the best weights, from before epoch {epoch + 1}, "
          f"loss {best_loss:.3e}. Every number below comes from an "
          f"interrupted fit.")
    return True


def _report_step_scale(model, learning_rate):
    """Adam step size against the size of the weights it moves.

    Adam moves each parameter by roughly the learning rate per step,
    whatever the gradient is, so what matters is the ratio to the parameter
    itself. At the reference setting the weights are drawn at 0.1 and the
    rate is 0.041, so a step is 40 % of a weight: aggressive but survivable.
    Shrink the initialisation tenfold without touching the rate and a single
    step becomes four times the weight, which is not survivable; the loss
    then rises instead of falling and the run dies while the cyclic schedule
    ramps up. The two are coupled, and the coupling is invisible unless it is
    printed.
    """
    with torch.no_grad():
        sizes = [float(p.detach().pow(2).mean().sqrt().item())
                 for p in model.parameters() if p.requires_grad
                 and p.numel() and float(p.detach().abs().max().item()) > 0]
    if not sizes:
        return
    rms = float(np.mean(sizes))
    ratio = learning_rate / rms if rms > 0 else float('inf')
    print(f"  step size: lr {learning_rate:.3e} against weights of rms "
          f"{rms:.3e}, ratio {ratio:.2f}")
    if ratio > 1.0:
        print(f"    WARNING: one optimiser step moves a weight by more than "
              f"its own magnitude. Lower the learning rate by about "
              f"{ratio / 0.4:.0f}x, or raise the initial weight scale. The "
              f"two are coupled: changing one alone breaks the other.")


def _rk4_amplification(eigenvalues, step):
    """Amplification of one RK4 step of size ``step``, per eigenvalue.

    The stability function of the classical fourth-order scheme is
    R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24. A mode is damped by the step when
    |R(lambda*step)| < 1 and amplified when it exceeds one. On the imaginary
    axis the boundary sits at |z| = 2*sqrt(2), so at a sampling interval of
    10 s any oscillation faster than about 22 s is amplified by the very term
    that is supposed to teach the model to reproduce it.
    """
    z = np.asarray(eigenvalues, dtype=complex) * float(step)
    return np.abs(1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0)


def _report_spectrum(model, timestep, substeps):
    """Print the spectrum of the learned linear operator against RK4.

    The parameterisation A = J - RR^T fixes the sign of the real parts and
    says nothing about the magnitude of the imaginary ones, so a model can be
    perfectly bounded in continuous time and still sit outside the stability
    region of the explicit step the loss uses. When that happens the one-step
    term stops being a fit and starts being a divergence, and the delay solver
    afterwards pays for the same spectrum in step count.
    """
    m = model.module if hasattr(model, 'module') else model
    A = getattr(m, 'A', None)
    if A is None:
        return

    A_np = A.detach().cpu().numpy()
    try:
        ev = np.linalg.eigvals(A_np)
    except np.linalg.LinAlgError:
        print("  spectrum: eigenvalue computation failed")
        return

    step = float(timestep) / max(1, int(substeps))
    amp = _rk4_amplification(ev, step)
    worst = int(np.argmax(amp))

    slowest = np.max(np.abs(ev))
    period = 2 * np.pi / np.abs(ev[worst].imag) if abs(ev[worst].imag) > 0 \
        else float('inf')

    print(f"  spectrum: max |lambda| {slowest:.4e} 1/s, "
          f"|lambda|*h up to {np.max(np.abs(ev)) * step:.3f} "
          f"(RK4 limit 2.83), {int(substeps)} substep(s) of {step:.2f} s")
    print(f"    worst mode {ev[worst].real:+.3e}{ev[worst].imag:+.3e}j, "
          f"period {period:.1f} s, one-step amplification "
          f"{amp[worst]:.4f}")

    n_unstable = int(np.count_nonzero(amp > 1.0))
    if n_unstable:
        need = int(np.ceil(np.max(np.abs(ev)) * float(timestep) / 2.8284))
        print(f"    WARNING: {n_unstable} of {ev.size} modes are amplified by "
              f"the one-step term rather than damped. The fit is being asked "
              f"to match a target through a step that is unstable for it. "
              f"RK4_SUBSTEPS >= {need} would bring the whole spectrum inside "
              f"the stability region.")


def _rollout_diagnostic(model, state, time, entries, horizons):
    """Rollout error at several horizons, against the no-change predictor.

    Returns a dict horizon -> relative error, where the denominator is the
    norm of the change over that horizon. One hundred percent is exactly the
    error of asserting that nothing happens, so a number above it means the
    model is worth less than that assertion over that span.

    This is the candidate the search and a multi-start need. Every quantity
    tried so far comes from the fit and none of them ranks the reported
    metric; this one is a rollout, which is what the metric is, only shorter.
    """
    out = {}
    n = state.shape[0]
    dt_vec = time[1:] - time[:-1]

    model.eval()
    with torch.no_grad():
        for h in horizons:
            h = int(h)
            if h < 1 or n <= h:
                continue
            valid = _valid_windows(dt_vec, n, h)
            if valid.sum() == 0:
                continue
            pred, target = _rollout(model, state, time, entries, dt_vec,
                                    valid, h)
            start = state[:n - h][valid]
            denom = float(torch.linalg.norm(target - start).item())
            if denom <= 0:
                continue
            out[h] = float(torch.linalg.norm(target - pred).item()) / denom
    return out


def _gradient_norm(model):
    """Euclidean norm of the full gradient, without touching it."""
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(total))


def _field_mse(pred, target, criterion):
    """Mean squared error under FIELD_WEIGHTING; see there."""
    if FIELD_WEIGHTING != "equal" or not FIELD_SIZES:
        return criterion(pred, target)
    losses, start = [], 0
    for size in FIELD_SIZES:
        size = int(size)
        if size > 0:
            diff = pred[:, start:start + size] - target[:, start:start + size]
            losses.append(torch.mean(diff ** 2))
        start += size
    if start < pred.shape[1]:
        losses.append(torch.mean((pred[:, start:] - target[:, start:]) ** 2))
    return sum(losses) / len(losses)

def _derivative_loss(model, state, time, entries, ddt, criterion):
    """Residual against the derivative target, pointwise in time."""
    y_pred = model(x=state, t=time, u=entries)
    return _field_mse(y_pred, ddt, criterion)


def _data_scale_ratio(state, time, ddt):
    """Ratio of the two loss scales, determined by the data alone.

    Each term is referred to the loss a model of identically zero would
    produce: mean(ddt^2) for the derivative term, and mean((x_{k+1} - x_k)^2)
    for the one-step term, since a Runge-Kutta step of a zero right-hand side
    returns the state unchanged. Their quotient is the ratio the two scales
    need for lambda_deriv to weigh two comparable quantities.

    The same non-positive time steps that ``_state_loss`` drops are dropped
    here, so the two look at the same transitions.
    """
    with torch.no_grad():
        d_zero = float(torch.mean(ddt ** 2).item())

        dt_vec = time[1:] - time[:-1]
        valid = (dt_vec > 0).reshape(-1)
        if valid.sum() == 0:
            return float('nan')
        dx = (state[1:] - state[:-1])[valid]
        s_zero = float(torch.mean(dx ** 2).item())

    if not (d_zero > 0.0 and s_zero > 0.0):
        return float('nan')
    return s_zero / d_zero


def _valid_windows(dt_vec, n, horizon):
    """Mask of start indices whose whole window has positive time steps.

    A non-positive step marks the boundary between two trajectories in a
    condition split. A window that straddles one would have the model
    integrate across the seam, so the whole window is dropped rather than
    only the offending step.
    """
    ok = (dt_vec > 0).reshape(-1)
    valid = ok[:n - horizon]
    for i in range(1, horizon):
        valid = valid & ok[i:n - horizon + i]
    return valid


def _rollout(model, state, time, entries, dt_vec, valid, horizon):
    """Integrate every valid start index forward by ``horizon`` intervals.

    All windows advance together, so the cost is ``horizon`` forward passes
    over the batch rather than one per window. The input is taken from the
    record at each step, which is what the data support and what the reported
    simulation also does.
    """
    n = state.shape[0]
    x = state[:n - horizon][valid]
    for i in range(horizon):
        u_i = entries[i:n - horizon + i][valid]
        t_i = time[i:n - horizon + i][valid]
        dt_i = dt_vec[i:n - horizon + i][valid]
        x = _rk4th_onestep(model, x, u_i, t_i, dt_i)
    return x, state[horizon:n][valid]


def _horizon_loss(model, state, time, entries, criterion, horizon):
    """Integration residual over one horizon, in sampling intervals."""
    n = state.shape[0]
    horizon = int(horizon)
    if horizon < 1 or n <= horizon:
        return None

    dt_vec = time[1:] - time[:-1]
    valid = _valid_windows(dt_vec, n, horizon)
    if valid.sum() == 0:
        return None

    pred, target = _rollout(model, state, time, entries, dt_vec, valid,
                            horizon)

    # A rollout that left the stability region poisons the whole loss with a
    # single infinity, and the run then aborts on a diverged-training error
    # that names the epoch but not the horizon. Dropping the offending
    # horizon keeps the shorter ones informative and lets training continue.
    if not torch.isfinite(pred).all():
        _DROPPED_HORIZONS[horizon] = _DROPPED_HORIZONS.get(horizon, 0) + 1
        return None
    
    loss = _field_mse(pred, target, criterion)
    if not torch.isfinite(loss):
        _DROPPED_HORIZONS[horizon] = _DROPPED_HORIZONS.get(horizon, 0) + 1
        return None

    if MULTISTEP_WEIGHTING == "trivial":
        # Divided by the loss of predicting no change over the same windows,
        # so that a long horizon does not simply outweigh a short one by
        # having a larger error.
        with torch.no_grad():
            start = state[:n - horizon][valid]
            if FIELD_WEIGHTING == "equal" and FIELD_SIZES:
                denom = float(_field_mse(start, target, criterion).item())
            else:
                denom = float(torch.mean((target - start) ** 2).item())
        if denom > 0:
            loss = loss / denom
    return loss


def _active_horizons(epoch):
    """Horizons in use at this epoch, honouring the warm-up."""
    horizons = tuple(MULTISTEP_HORIZONS)
    if len(horizons) <= 1 or epoch >= MULTISTEP_WARMUP_EPOCHS:
        return horizons
    return (min(horizons),)


def _state_loss(model, state, time, entries, criterion, horizons=None):
    """Integration residual, averaged over the active horizons.

    With the default (1,) this is the single-step residual between
    consecutive samples and is bit-identical to the previous implementation.
    """
    horizons = MULTISTEP_HORIZONS if horizons is None else horizons
    losses = [l for l in (_horizon_loss(model, state, time, entries,
                                        criterion, h)
                          for h in horizons) if l is not None]
    if not losses:
        return torch.tensor(0.0, device=device)
    if len(losses) == 1:
        return losses[0]
    return sum(losses) / len(losses)


def _fit(model, dataloaders, optimizer, scheduler=None, full_batch=None,
         scale_reference=None):
    if Params.output:
        print("_" * 75)
        print(f"Matrix training started (loss: {Params.ROM_loss})")

    loss_track = []
    best_loss = float('inf')
    best_model_state_dict = deepcopy(model.state_dict())

    patience = (Params.num_epochs if EARLY_STOPPING_PATIENCE is None
                else int(EARLY_STOPPING_PATIENCE))
    counter = 0
    nonfinite = 0

    criterion = torch.nn.MSELoss()

    # Auto-balancing scales for the hybrid loss.
    loss_scale_deriv = 1.0
    loss_scale_state = 1.0
    initialized_scales = False

    # Diagnostics carried to the progress line.
    last_d = last_s = float('nan')
    max_grad_norm_seen = 0.0

    report_every = max(1, int(Params.num_epochs / 10))
    previous_horizons = None
    _DROPPED_HORIZONS.clear()

    # The spectrum before the first step. The report after training looks at
    # the model that survived; this one shows what the rollout has to cope
    # with at the start, which is where a multi-step term overflows.
    if Params.output:
        _report_step_scale(model, float(optimizer.param_groups[0]['lr']))

    if Params.output and REPORT_SPECTRUM and full_batch is not None:
        _dt = (full_batch[1][1:] - full_batch[1][:-1]).reshape(-1)
        _pos = _dt[_dt > 0]
        if _pos.numel():
            print("  at initialisation:")
            _report_spectrum(model, float(torch.median(_pos).item()),
                             RK4_SUBSTEPS)
            verify_H_energy_preservation(model, Params.ROM_order)

    for epoch in range(Params.num_epochs):
        model.train()
        total_loss_epoch = 0.0
        n_batches = 0
        epoch_d = epoch_s = 0.0

        active_horizons = _active_horizons(epoch)
        horizons_changed = active_horizons != previous_horizons
        previous_horizons = active_horizons

        refresh = (LOSS_SCALE_REFRESH_EPOCHS is not None
                   and epoch > 0
                   and epoch % int(LOSS_SCALE_REFRESH_EPOCHS) == 0)
        if horizons_changed and epoch > 0:
            # Switching horizons on changes the size of the state term, so
            # the frozen scale no longer balances anything. Re-estimated with
            # the magnitude preserved, which is what the refresh path does.
            refresh = True
            if RESET_BEST_ON_HORIZON_CHANGE:
                # A loss from before the switch belongs to another objective
                # and must not be compared with the losses after it.
                best_loss = float('inf')
                counter = 0
            if Params.output:
                reset_note = (", best loss reset"
                              if RESET_BEST_ON_HORIZON_CHANGE else "")
                print(f"  horizons now {list(active_horizons)} "
                      f"(epoch {epoch + 1}), rebalancing{reset_note}")
        if refresh:
            initialized_scales = False

        for batch in dataloaders["train"]:
            optimizer.zero_grad()
            state, time, ddt, entries = batch

            # --- 1. derivative term ---
            if Params.ROM_loss in ['ddt', 'hybrid']:
                loss_d = _derivative_loss(
                    model, state, time, entries, ddt, criterion)
            else:
                loss_d = torch.tensor(0.0, device=device)

            # --- 2. state term ---
            if Params.ROM_loss in ['states', 'hybrid']:
                loss_s = _state_loss(model, state, time, entries, criterion,
                                     horizons=active_horizons)
            else:
                loss_s = torch.tensor(0.0, device=device)

            # --- 3. auto-balancing ---
            # Two references, see LOSS_SCALE_REFERENCE. Under "init" the
            # scales are 1/loss on the first batch, so they depend on where
            # the random weights happened to land. Under "data" only the
            # ratio is taken, from the record, and the magnitude is pinned.
            if not initialized_scales and Params.ROM_loss == 'hybrid':
                with torch.no_grad():
                    d_val = loss_d.item()
                    s_val = loss_s.item()

                    reference = (LOSS_SCALE_REFERENCE
                                 if scale_reference is None
                                 else scale_reference)
                    source = reference

                    # A refresh always uses the current losses: the point of
                    # refreshing is to follow how the two terms have moved,
                    # which the data reference cannot see.
                    ratio = float('nan')
                    if reference == "data" and not refresh:
                        ratio = _data_scale_ratio(state, time, ddt)

                    if np.isfinite(ratio) and ratio > 0.0:
                        root = float(np.sqrt(ratio))
                        loss_scale_deriv = LOSS_SCALE_MAGNITUDE * root
                        loss_scale_state = LOSS_SCALE_MAGNITUDE / root
                    else:
                        if reference == "data" and not refresh:
                            source = "init (no usable data ratio)"

                        # Setting the scales to 1/loss normalises each term to
                        # one. On the first batch that is what we want. On a
                        # refresh it is not: the loss has fallen since, so
                        # renormalising to one multiplies the gradient by
                        # exactly the factor the loss decreased, which is a
                        # self-amplifying increase of the step size. Preserve
                        # the current magnitude of the combined loss and let
                        # the refresh set only the ratio of the two terms.
                        keep = 1.0
                        if epoch > 0:
                            keep = (Params.lambda_deriv * loss_scale_deriv
                                    * d_val
                                    + Params.lambda_state * loss_scale_state
                                    * s_val)

                        if d_val > 1e-9:
                            loss_scale_deriv = keep / d_val
                        if s_val > 1e-9:
                            loss_scale_state = keep / s_val

                    if Params.output:
                        extra = (f", ratio {ratio:.1f} from the data"
                                 if np.isfinite(ratio) else "")
                        tail = f"  (epoch {epoch + 1})" if epoch else ""
                        print(f"  auto-balancing ({source}): scale D "
                              f"{loss_scale_deriv:.2e}, scale S "
                              f"{loss_scale_state:.2e}{extra}{tail}")
                    initialized_scales = True

            # --- 4. combine ---
            if Params.ROM_loss == 'ddt':
                loss_val = loss_d
            elif Params.ROM_loss == 'states':
                loss_val = loss_s
            else:
                ld = Params.lambda_deriv
                ls = Params.lambda_state
                loss_val = (ld * (loss_d * loss_scale_deriv)
                            + ls * (loss_s * loss_scale_state))

            # --- 5. regularization ---
            if 'H' in Params.model_structure and Params.regularization_H > 0:
                m = model.module if hasattr(model, 'module') else model
                loss_val = loss_val + Params.regularization_H * m.H.abs().mean()

            if 'A' in Params.model_structure and Params.regularization_A > 0:
                m = model.module if hasattr(model, 'module') else model
                loss_val = loss_val + Params.regularization_A * (m.A ** 2).mean()

            # A non-finite loss would corrupt every weight through the backward
            # pass, so the step is skipped entirely. The scheduler is skipped
            # with it, which shifts the cycle; the progress line reports the
            # learning rate actually in use.
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                nonfinite += 1
                # Which term went first, and what the spectrum looked like
                # when it did. "diverged at epoch n" on its own says only
                # that something happened, and the model that is inspected
                # afterwards is the best one, which by definition is not the
                # one that failed.
                if Params.output and nonfinite == 1:
                    which = []
                    if not torch.isfinite(loss_d):
                        which.append("derivative term")
                    if not torch.isfinite(loss_s):
                        which.append("state term")
                    if not which:
                        which.append("regularization or the sum")
                    print(f"  non-finite loss at epoch {epoch + 1}: "
                          f"{', '.join(which)}; horizons "
                          f"{list(active_horizons)}")
                    _dt = (time[1:] - time[:-1]).reshape(-1)
                    _pos = _dt[_dt > 0]
                    if _pos.numel():
                        _report_spectrum(
                            model, float(torch.median(_pos).item()),
                            RK4_SUBSTEPS)
                if ABORT_ON_DIVERGENCE and nonfinite >= MAX_NONFINITE_BATCHES:
                    raise RuntimeError(
                        f"training diverged: non-finite loss at epoch "
                        f"{epoch + 1}")
                continue
            
            loss_val.backward()
            
            grad_norm = _gradient_norm(model)
            if not np.isfinite(grad_norm):
                nonfinite += 1
                optimizer.zero_grad()
                if ABORT_ON_DIVERGENCE and nonfinite >= MAX_NONFINITE_BATCHES:
                    raise RuntimeError(
                        f"training diverged: non-finite gradient at epoch "
                        f"{epoch + 1}")
                continue
            
            max_grad_norm_seen = max(max_grad_norm_seen, grad_norm)
            if CLIP_GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(CLIP_GRAD_NORM))

            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss_epoch += loss_val.item()
            loss_track.append(loss_val.item())
            n_batches += 1
            epoch_d += float(loss_d.item())
            epoch_s += float(loss_s.item())

        if n_batches == 0:
            # Every batch of the epoch was skipped as non-finite. Reporting
            # this as a loss of infinity sends the check below into a
            # division that says nothing; the cause was printed above.
            reason = (f"every batch of epoch {epoch + 1} was non-finite and "
                      f"skipped, so the epoch produced no step")
            if _handle_divergence(reason, epoch, best_loss):
                break
            raise RuntimeError(f"training diverged: {reason}")

        avg_loss = total_loss_epoch / n_batches

        if (ABORT_ON_DIVERGENCE and np.isfinite(best_loss)
                and avg_loss > DIVERGENCE_LOSS_FACTOR * max(best_loss, 1e-30)):
            reason = (f"loss {avg_loss:.3e} at epoch {epoch + 1} is "
                      f"{avg_loss / best_loss:.1e} times the best "
                      f"{best_loss:.3e}")
            if _handle_divergence(reason, epoch, best_loss):
                break
            raise RuntimeError(f"training diverged at epoch {epoch + 1}: "
                               f"{reason}")
    
        if n_batches:
            last_d = epoch_d / n_batches
            last_s = epoch_s / n_batches

        if (epoch + 1) % report_every == 0 and Params.output:
            curr_lr = optimizer.param_groups[0]['lr']
            line = (f"Epoch {epoch + 1}/{Params.num_epochs} "
                    f"| loss {avg_loss:.4e} | lr {curr_lr:.2e} "
                    f"| grad {max_grad_norm_seen:.2e}")
            if Params.ROM_loss == 'hybrid':
                # Contribution of each term to the combined loss, after the
                # frozen scales and the configured weights.
                cd = Params.lambda_deriv * last_d * loss_scale_deriv
                cs = Params.lambda_state * last_s * loss_scale_state
                tot = cd + cs
                share = cd / tot if tot > 0 else float('nan')
                line += (f" | ddt {last_d:.3e} states {last_s:.3e} "
                         f"| ddt share {share:.2f} "
                         f"(configured {Params.lambda_deriv:.2f})")
            print(line)
            max_grad_norm_seen = 0.0

            if SPECTRUM_EVERY_EPOCHS and full_batch is not None:
                if (epoch + 1) % int(SPECTRUM_EVERY_EPOCHS) == 0:
                    _dt = (full_batch[1][1:] - full_batch[1][:-1]).reshape(-1)
                    _pos = _dt[_dt > 0]
                    if _pos.numel():
                        _report_spectrum(
                            model, float(torch.median(_pos).item()),
                            RK4_SUBSTEPS)

        # Early stopping. The counter resets on any improvement worth the name
        # and counts up otherwise; the previous version also counted up on
        # small improvements, which made the criterion mean something other
        # than its name.
        if avg_loss < best_loss:
            improvement = best_loss - avg_loss
            best_loss = avg_loss
            best_model_state_dict = deepcopy(model.state_dict())
            counter = 0 if improvement > IMPROVEMENT_THRESHOLD else counter + 1
        else:
            counter += 1

        if counter >= patience:
            if Params.output:
                print(f"Early stopping at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_model_state_dict)

    if Params.output:
        print("Training completed.")
        print(f"Best loss: {best_loss:.5f}")

    # Recorded unconditionally, so that a search can reject a trial whose
    # state term silently lost a horizon.
    Params.dropped_horizons = dict(sorted(_DROPPED_HORIZONS.items()))
    if Params.output and _DROPPED_HORIZONS:
        print(f"  WARNING: state-term horizons dropped as non-finite "
              f"(horizon: batches): {Params.dropped_horizons}. In those "
              f"batches the state term did not contain them.")

    # One-step residual over the whole dataset, in the caller's layout (r, k).
    # This feeds the non-Markovian regression in the main script, which needs
    # every sample; taking whatever the last loop iteration left in scope is
    # correct only while the batch covers the full dataset.
    model.eval()
    with torch.no_grad():
        state, time, ddt, entries = full_batch
        y_pred = model(x=state, t=time, u=entries)
        residual = (ddt - y_pred)
        error = residual.detach().cpu().numpy().T
    
        # Recorded unconditionally: this is a candidate selection criterion for
        # a multi-start, and a search has to be able to read it with output
        # switched off.
        denom = float(torch.linalg.norm(ddt).item())
        rel = (float(torch.linalg.norm(residual).item()) / denom
               if denom > 0 else float('nan'))
        globals()['LAST_DERIVATIVE_RESIDUAL'] = rel

        # Also placed on the parameter singleton. Reaching the module global
        # from the main script would mean importing the submodule by its
        # private name and relying on the package re-exporting it; the
        # singleton is the same object everywhere and needs neither. This
        # attribute always holds the most recent call, which is why the caller
        # copies it out before the memory-term loop runs.
        Params.ddt_residual = rel
    
        if Params.output:
            print(f"  relative residual against the derivative target: "
                  f"{rel:.2%}")

            if REPORT_SPECTRUM:
                dt_vec = (time[1:] - time[:-1]).reshape(-1)
                positive = dt_vec[dt_vec > 0]
                if positive.numel():
                    _report_spectrum(model,
                                     float(torch.median(positive).item()),
                                     RK4_SUBSTEPS)
                    verify_H_energy_preservation(model, Params.ROM_order)

    # Outside the output guard: a search has to be able to read this with
    # printing switched off.
    if ROLLOUT_DIAGNOSTIC_HORIZONS:
        state, time, ddt, entries = full_batch
        rollout = _rollout_diagnostic(model, state, time, entries,
                                      ROLLOUT_DIAGNOSTIC_HORIZONS)
        globals()['LAST_ROLLOUT'] = dict(rollout)
        Params.rollout = dict(rollout)
        if Params.output and rollout:
            cells = " ".join(f"{h}:{v:.2%}" for h, v in sorted(rollout.items()))
            print(f"  rollout error by horizon (100% = no change assumed): "
                  f"{cells}")

    return model, loss_track, error