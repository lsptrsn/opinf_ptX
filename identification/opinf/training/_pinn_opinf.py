"""PINN-assisted operator inference.

A network is fitted to the measured latent trajectories as a differentiable
surrogate; its analytic time derivative then supplies the regression target for
the reduced operators, which removes the need for a finite-difference estimate.

  A  fit the surrogate to the data alone,
  B  fit the operators against the surrogate, surrogate frozen,
  C  optional joint phase.

Why this file was reworked
--------------------------
The operators used to be fitted by derivative matching alone. On this record
that is the wrong objective: the reduced derivative is noise dominated, the
model with the best rollout has a one-step residual of several hundred percent
against it, and smoothing the derivative target away from the data improves the
rollout rather than harming it. A method that makes the derivative target
*cleaner* therefore fits harder to the quantity that does not govern
prediction. Stage B reached a validation loss of 3e-05 and a rollout error
above one hundred percent, which is that mechanism and not a coincidence.

The operator loss is now the same hybrid used by the gradient-based path:
instantaneous derivative matching plus a single Runge-Kutta step between
consecutive samples. The trajectory term needs the batch in chronological
order, so the batch is sorted by time and the step size is taken from the
timestamps, which also survives the gaps left by a shuffled validation split.

The residual handed to the non-Markovian stage is now computed here, from the
model's own forward pass, so it includes the quadratic operator and uses the
same derivative the operators were fitted against. The caller used to rebuild
it as ``Qdot - (A q + B u)``, dropping H entirely and mixing two different
derivative estimates in one difference.
"""

__all__ = ["train_pinn_opinf"]

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

import opinf

device = opinf.utils.get_device()


# --- switches -----------------------------------------------------------------

# The surrogate maps (t, u) -> x, so time is an input and the network is an
# interpolant over the fitted interval, never evaluated outside it. A
# chronological split asks it to extrapolate in time, which this architecture
# cannot do by construction; the validation loss then measures a capability the
# model does not need and never improves. A shuffled split measures
# interpolation, which is the task. This is the opposite of the right choice
# for a state-to-field decoder, and the difference is the role of time.
PINN_VAL_SPLIT = "random"
PINN_VAL_FRACTION = 0.1

# Objective for the operators in stages B and C.
#   "hybrid" : derivative matching and one-step integration, weighted by
#              Params.lambda_deriv and Params.lambda_state, as in the
#              gradient-based path.
#   "ddt"    : derivative matching only, the previous behaviour.
PINN_OPERATOR_LOSS = "hybrid"

# States used in the trajectory term.
#   "surrogate" : the denoised states produced by the network, which is where
#                 the derivative target also comes from.
#   "data"      : the measured latent states.
PINN_STATE_TARGET = "surrogate"

OPINF_SUBSTEPS = 3                 # operator updates per surrogate update
PHYS_WEIGHT_SMOOTHING = 0.1        # moving average of the adaptive rule
W_PHYS_MAX = 1e3                   # ceiling on the adaptive physics weight
STATE_SMOOTHING_WINDOW = 21
STATE_SMOOTHING_ORDER = 3
GRAD_CLIP_NORM = 1.0


# =============================================================================
# Helpers
# =============================================================================

def select_data_loss(name):
    """Data misfit for the surrogate, from ``Params.PINN_loss``."""
    name = (name or "mse").lower()
    if name == "huber":
        return F.smooth_l1_loss
    if name == "mse":
        return F.mse_loss
    raise ValueError(f"unknown PINN_loss: {name}")


def smooth_states(states, window_length=STATE_SMOOTHING_WINDOW,
                  polyorder=STATE_SMOOTHING_ORDER):
    """Savitzky-Golay smoothing along time, no differentiation."""
    if window_length % 2 == 0:
        window_length += 1
    return scipy.signal.savgol_filter(
        states, window_length=window_length, polyorder=polyorder,
        deriv=0, axis=-1, mode='nearest'
    )


def compute_derivatives(net, inp, t_scale, create_graph=False, use_vmap=False):
    """Time derivative of the surrogate.

    The first input column is the normalised time, so the derivative with
    respect to it is divided by the normalisation to return to seconds. The
    input must carry ``requires_grad`` on the autograd path, and this function
    must not be called inside ``torch.no_grad``.
    """
    if use_vmap:
        from torch.func import vmap, jacrev

        def func(x):
            return net(x.unsqueeze(0))[0].squeeze(0)

        return vmap(jacrev(func))(inp)[:, :, 0] / t_scale

    if not inp.requires_grad:
        inp = inp.detach().requires_grad_(True)

    x_pred, _ = net(inp)
    grads = []
    for i in range(x_pred.shape[1]):
        g = torch.autograd.grad(outputs=x_pred[:, i].sum(), inputs=inp,
                                create_graph=create_graph,
                                retain_graph=True)[0][:, 0:1]
        grads.append(g)
    return torch.cat(grads, dim=1) / t_scale


def _operator_module(opinf_model):
    return opinf_model.module if hasattr(opinf_model, 'module') else opinf_model


def _rk4_onestep(model, x, u, t, timestep):
    """One explicit Runge-Kutta step of order four, input held over the step."""
    if timestep.ndim == 1:
        timestep = timestep.view(-1, 1)
    k1 = model(x, t, u)
    k2 = model(x + 0.5 * timestep * k1, t + 0.5 * timestep, u)
    k3 = model(x + 0.5 * timestep * k2, t + 0.5 * timestep, u)
    k4 = model(x + 1.0 * timestep * k3, t + 1.0 * timestep, u)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep


def _operator_terms(opinf_model, x_states, t_b, u_b, ddt_target):
    """Derivative and trajectory residuals of the operators on one batch.

    The trajectory term steps from one sample to the next, so the batch has to
    be in chronological order and the step size has to come from the
    timestamps: a shuffled validation split removes samples from the middle of
    the record, which leaves gaps of more than one sampling interval.
    """
    loss_d = F.mse_loss(opinf_model(x_states, t_b, u_b), ddt_target)

    order = torch.argsort(t_b.reshape(-1))
    t_s = t_b[order]
    u_s = u_b[order]
    x_s = x_states[order]

    dt_vec = t_s[1:] - t_s[:-1]
    valid = (dt_vec > 0).reshape(-1)
    if valid.sum() == 0:
        return loss_d, torch.tensor(0.0, device=x_states.device)

    pred_next = _rk4_onestep(opinf_model, x_s[:-1][valid], u_s[:-1][valid],
                             t_s[:-1][valid], dt_vec[valid])
    loss_s = F.mse_loss(pred_next, x_s[1:][valid])

    return loss_d, loss_s


def _combine(loss_d, loss_s, scales, Params):
    """Weighted sum of the two operator terms."""
    if PINN_OPERATOR_LOSS == "ddt":
        return loss_d
    sd, ss = scales
    return (Params.lambda_deriv * loss_d * sd
            + Params.lambda_state * loss_s * ss)


def _regularize(loss, opinf_model, Params):
    m = _operator_module(opinf_model)
    if getattr(m, 'H', None) is not None and Params.regularization_H > 0:
        loss = loss + Params.regularization_H * m.H.abs().mean()
    return loss


@torch.no_grad()
def validate_stage_A(net, loader, t_min, t_scale, data_loss):
    acc = 0.0
    for t, u, x, _ in loader:
        tn = (t - t_min) / t_scale
        xp, _ = net(torch.cat([tn, u], 1))
        acc += data_loss(xp, x).item()
    return acc / max(1, len(loader))


def validate_operators(net, opinf_model, loader, t_min, t_scale, scales,
                       Params, use_vmap=False):
    """Validation under the same objective the operators are trained on."""
    acc = 0.0
    for t, u, x, _ in loader:
        tn = (t - t_min) / t_scale
        inp = torch.cat([tn, u], 1).requires_grad_(True)
        ddt = compute_derivatives(net, inp, t_scale, create_graph=False,
                                  use_vmap=use_vmap).detach()
        with torch.no_grad():
            xp, _ = net(inp)
            x_states = xp if PINN_STATE_TARGET == "surrogate" else x
            loss_d, loss_s = _operator_terms(opinf_model, x_states, t, u, ddt)
            acc += float(_combine(loss_d, loss_s, scales, Params).item())
    return acc / max(1, len(loader))


def _report_surrogate_derivative(net, t_raw, u_all, x_all, t_min, t_scale,
                                 use_vmap):
    """Compare the surrogate derivative with a central difference of the data.

    The premise of the method is that the surrogate supplies a cleaner
    derivative than a finite difference. A ratio far from one means the
    surrogate has invented dynamics the record does not contain, or has damped
    what it does; either way the regression target is not what it claims to be.
    """
    inp = torch.cat([(t_raw - t_min) / t_scale, u_all], 1).requires_grad_(True)
    ddt = compute_derivatives(net, inp, t_scale, create_graph=False,
                              use_vmap=use_vmap).detach().cpu().numpy().T

    t_np = t_raw.detach().cpu().numpy().ravel()
    x_np = x_all.detach().cpu().numpy().T
    order = np.argsort(t_np)
    fd = np.gradient(x_np[:, order], t_np[order], axis=1)
    est = ddt[:, order]

    m = slice(5, -5)
    print("  surrogate derivative vs central difference of the latent data")
    for j in range(est.shape[0]):
        a, b = est[j, m], fd[j, m]
        rms_a = float(np.sqrt(np.mean(a ** 2)))
        rms_b = float(np.sqrt(np.mean(b ** 2)))
        ratio = rms_a / rms_b if rms_b > 0 else float('nan')
        corr = (float(np.corrcoef(a, b)[0, 1])
                if rms_a > 0 and rms_b > 0 else float('nan'))
        print(f"    mode {j}: amplitude ratio {ratio:.3f}, "
              f"correlation {corr:.3f}")


# =============================================================================
# Training
# =============================================================================

def train_pinn_opinf(states, time, entries, opinf_model, r_X=None, seed=0):
    """Fit the surrogate and the reduced operators.

    Returns
    -------
    opinf_model : torch.nn.Module
    history : dict
    error_reduced_states : (r, k) ndarray
        One-step residual of the fitted model over the whole training set,
        against the surrogate derivative. This is what the non-Markovian stage
        regresses on.
    """
    local_seed = int(seed)
    opinf.utils.set_deterministic(local_seed)
    Params = opinf.parameters.Params()

    try:
        from torch.func import vmap, jacrev  # noqa: F401
        use_vmap = True
    except ImportError:
        use_vmap = False
    if Params.output:
        print(f"  derivatives via "
              f"{'vmap/jacrev' if use_vmap else 'autograd loop'}")
        print(f"  operator loss: {PINN_OPERATOR_LOSS}"
              + (f", trajectory states from the {PINN_STATE_TARGET}"
                 if PINN_OPERATOR_LOSS == "hybrid" else ""))

    data_loss = select_data_loss(Params.PINN_loss)

    if Params.PINN_state_smoothing:
        states = smooth_states(states)
        if Params.output:
            print(f"  state smoothing on: window {STATE_SMOOTHING_WINDOW}, "
                  f"order {STATE_SMOOTHING_ORDER}")

    # --- data ---------------------------------------------------------------
    t_raw = torch.tensor(time, dtype=torch.float32,
                         device=device).reshape(-1, 1)
    u_all = torch.tensor(entries.T, dtype=torch.float32, device=device)
    x_all = torch.tensor(states.T, dtype=torch.float32, device=device)

    n_samples = len(time)
    dataset = TensorDataset(t_raw, u_all, x_all,
                            torch.arange(n_samples, device=device))

    t_min, t_max = t_raw.min(), t_raw.max()
    t_scale = (t_max - t_min).clamp(min=1e-6)

    n_val = max(1, int(round(PINN_VAL_FRACTION * n_samples)))
    n_train = n_samples - n_val

    if PINN_VAL_SPLIT == "chronological":
        train_idx = torch.arange(n_train)
        val_idx = torch.arange(n_train, n_samples)
    elif PINN_VAL_SPLIT == "random":
        g_perm = torch.Generator()
        g_perm.manual_seed(local_seed)
        perm = torch.randperm(n_samples, generator=g_perm)
        train_idx, val_idx = perm[:n_train], perm[n_train:]
    else:
        raise ValueError(f"unknown PINN_VAL_SPLIT: {PINN_VAL_SPLIT}")

    # A batch larger than the training split combined with drop_last leaves no
    # batches at all, and the epoch average then divides by zero.
    batch_size = min(int(Params.PINN_batch_size), n_train)
    drop_last = batch_size < n_train

    g_loader = torch.Generator()
    g_loader.manual_seed(local_seed + 1)

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=batch_size, shuffle=True,
                              generator=g_loader, drop_last=drop_last)
    val_loader = DataLoader(Subset(dataset, val_idx),
                            batch_size=batch_size, shuffle=False)

    if Params.output:
        print(f"  validation split: {PINN_VAL_SPLIT}, {n_train} train / "
              f"{n_val} validation samples, batch {batch_size}, "
              f"{len(train_loader)} batches per epoch")
    if len(train_loader) == 0:
        raise RuntimeError("the training loader yields no batches")

    # --- network and optimizers ---------------------------------------------
    net = opinf.models.create_network(int(entries.shape[0]) + 1,
                                      list(Params.PINN_hidden_layers),
                                      int(states.shape[0])).to(device)

    opt_net = torch.optim.AdamW(net.parameters(), lr=Params.PINN_lr_net,
                                weight_decay=1e-5)
    opt_opinf = torch.optim.AdamW(opinf_model.parameters(),
                                  lr=Params.PINN_lr_opinf, weight_decay=1e-5)
    sched_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_net, factor=0.8, patience=50)
    sched_opinf = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_opinf, factor=0.8, patience=50)

    history = {k: [] for k in ("train_data", "val_data",
                               "train_phys", "val_phys",
                               "train_ddt", "train_state",
                               "train_total", "val_total", "w_phys")}
    best_states = {"A": None, "B": None, "C": None}
    best_vals = {"A": float('inf'), "B": float('inf'), "C": float('inf')}

    total_epochs = Params.PINN_epochs
    epochs_A = int(Params.PINN_stage_config_A * total_epochs)
    epochs_B = int(Params.PINN_stage_config_B * total_epochs)
    epochs_C = total_epochs - epochs_A - epochs_B
    if Params.output:
        print(f"  stages: A {epochs_A}, B {epochs_B}, C {epochs_C} epochs")

    # --- stage A: the surrogate ---------------------------------------------
    if Params.output:
        print("--- stage A: fitting the surrogate ---")
    net.train()
    opinf_model.eval()

    for epoch in range(epochs_A):
        loss_acc = 0.0
        for t_b, u_b, x_b, _ in train_loader:
            opt_net.zero_grad()
            x_pred, _ = net(torch.cat([(t_b - t_min) / t_scale, u_b], dim=1))
            loss = data_loss(x_pred, x_b)
            loss.backward()
            opt_net.step()
            loss_acc += loss.item()

        avg_loss = loss_acc / len(train_loader)
        val_loss = validate_stage_A(net, val_loader, t_min, t_scale, data_loss)
        sched_net.step(val_loss)

        history["train_data"].append(avg_loss)
        history["val_data"].append(val_loss)
        if val_loss < best_vals["A"]:
            best_vals["A"] = val_loss
            best_states["A"] = deepcopy(net.state_dict())
        if Params.output and epoch % max(1, epochs_A // 5) == 0:
            print(f"  epoch {epoch}: train {avg_loss:.2e}, val {val_loss:.2e}")

    if best_states["A"]:
        net.load_state_dict(best_states["A"])

    if Params.output and epochs_A > 0:
        _report_surrogate_derivative(net, t_raw, u_all, x_all, t_min, t_scale,
                                     use_vmap)

    # --- stage B: the operators ---------------------------------------------
    if Params.output:
        print("--- stage B: fitting the operators ---")
    net.eval()
    opinf_model.train()

    # Auto-balancing scales, estimated once on the first batch so that both
    # terms enter at order one, then frozen. Re-estimating them later
    # renormalises a shrinking loss back to one and amplifies the gradient by
    # exactly the factor the loss fell, which diverges.
    scales = (1.0, 1.0)
    scales_set = (PINN_OPERATOR_LOSS == "ddt")

    for epoch in range(epochs_B):
        loss_acc, ddt_acc, state_acc = 0.0, 0.0, 0.0
        for t_b, u_b, x_b, _ in train_loader:
            opt_opinf.zero_grad()
            inp = torch.cat([(t_b - t_min) / t_scale, u_b],
                            dim=1).requires_grad_(True)

            ddt_target = compute_derivatives(net, inp, t_scale,
                                             create_graph=False,
                                             use_vmap=use_vmap).detach()
            with torch.no_grad():
                x_pred, _ = net(inp)
            x_states = (x_pred.detach() if PINN_STATE_TARGET == "surrogate"
                        else x_b)

            loss_d, loss_s = _operator_terms(opinf_model, x_states, t_b, u_b,
                                             ddt_target)

            if not scales_set:
                with torch.no_grad():
                    d_val, s_val = loss_d.item(), loss_s.item()
                    sd = 1.0 / d_val if d_val > 1e-12 else 1.0
                    ss = 1.0 / s_val if s_val > 1e-12 else 1.0
                    scales = (sd, ss)
                    scales_set = True
                    if Params.output:
                        print(f"  auto-balancing: scale D {sd:.2e}, "
                              f"scale S {ss:.2e}")

            loss = _regularize(_combine(loss_d, loss_s, scales, Params),
                               opinf_model, Params)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opinf_model.parameters(),
                                           GRAD_CLIP_NORM)
            opt_opinf.step()

            loss_acc += loss.item()
            ddt_acc += float(loss_d.item())
            state_acc += float(loss_s.item())

        n_b = len(train_loader)
        avg_loss = loss_acc / n_b
        val_loss = validate_operators(net, opinf_model, val_loader, t_min,
                                      t_scale, scales, Params, use_vmap)
        sched_opinf.step(val_loss)

        history["train_phys"].append(avg_loss)
        history["val_phys"].append(val_loss)
        history["train_ddt"].append(ddt_acc / n_b)
        history["train_state"].append(state_acc / n_b)

        if val_loss < best_vals["B"]:
            best_vals["B"] = val_loss
            best_states["B"] = deepcopy(opinf_model.state_dict())
        if Params.output and epoch % max(1, epochs_B // 5) == 0:
            print(f"  epoch {epoch}: train {avg_loss:.2e}, val {val_loss:.2e},"
                  f" ddt {ddt_acc / n_b:.2e}, states {state_acc / n_b:.2e}")

    if best_states["B"]:
        opinf_model.load_state_dict(best_states["B"])

    # --- stage C: joint ------------------------------------------------------
    # Stage C lets the surrogate move towards the operators. Measured on this
    # record it undid stage B by two orders of magnitude, and the adaptive
    # weight sat permanently at its ceiling, which means the rule had failed.
    # Leaving epochs_C at zero is the tested configuration.
    adaptive_weight = (Params.PINN_phys_weight is None
                       or float(Params.PINN_phys_weight) <= 0)
    w_phys = 1.0 if adaptive_weight else float(Params.PINN_phys_weight)

    if epochs_C > 0:
        if Params.output:
            print("--- stage C: joint training ---")
            print("  physics weight: "
                  + ("adaptive from gradient magnitudes, starting at 1.0"
                     if adaptive_weight else f"fixed at {w_phys:.3e}"))
        net.train()
        opinf_model.train()

    for epoch in range(epochs_C):
        loss_acc = 0.0
        for t_b, u_b, x_b, _ in train_loader:
            inp = torch.cat([(t_b - t_min) / t_scale, u_b],
                            dim=1).requires_grad_(True)

            # 1. surrogate update against data and physics
            opt_net.zero_grad()
            x_pred, _ = net(inp)
            ddt_nn = compute_derivatives(net, inp, t_scale, create_graph=True,
                                         use_vmap=use_vmap)
            with torch.no_grad():
                ddt_opinf = opinf_model(x_pred, t_b, u_b)

            l_data = data_loss(x_pred, x_b)
            l_phys = F.mse_loss(ddt_nn, ddt_opinf)

            if adaptive_weight:
                # Two extra backward passes per batch, on a graph that carries
                # second derivatives. This is the dominant cost of stage C.
                opt_net.zero_grad()
                l_data.backward(retain_graph=True)
                max_grad_data = torch.max(torch.abs(torch.cat(
                    [p.grad.view(-1) for p in net.parameters()
                     if p.grad is not None])))

                opt_net.zero_grad()
                l_phys.backward(retain_graph=True)
                mean_grad_phys = torch.mean(torch.abs(torch.cat(
                    [p.grad.view(-1) for p in net.parameters()
                     if p.grad is not None])))

                with torch.no_grad():
                    w_hat = (max_grad_data / (mean_grad_phys + 1e-8)).item()
                    w_phys = ((1.0 - PHYS_WEIGHT_SMOOTHING) * w_phys
                              + PHYS_WEIGHT_SMOOTHING * w_hat)
                    w_phys = min(w_phys, W_PHYS_MAX)

            opt_net.zero_grad()
            loss_net = l_data + w_phys * l_phys
            loss_net.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
            opt_net.step()
            loss_acc += loss_net.item()

            # 2. operator substeps against the updated surrogate
            inp_det = inp.detach().requires_grad_(True)
            for _ in range(OPINF_SUBSTEPS):
                opt_opinf.zero_grad()
                ddt_target = compute_derivatives(net, inp_det, t_scale,
                                                 create_graph=False,
                                                 use_vmap=use_vmap).detach()
                with torch.no_grad():
                    x_fresh, _ = net(inp_det)
                x_states = (x_fresh if PINN_STATE_TARGET == "surrogate"
                            else x_b)

                loss_d, loss_s = _operator_terms(opinf_model, x_states, t_b,
                                                 u_b, ddt_target)
                loss_opinf = _regularize(
                    _combine(loss_d, loss_s, scales, Params),
                    opinf_model, Params)
                loss_opinf.backward()
                torch.nn.utils.clip_grad_norm_(opinf_model.parameters(),
                                               GRAD_CLIP_NORM)
                opt_opinf.step()

        avg_loss = loss_acc / len(train_loader)
        val_loss = validate_operators(net, opinf_model, val_loader, t_min,
                                      t_scale, scales, Params, use_vmap)

        history["train_total"].append(avg_loss)
        history["val_total"].append(val_loss)
        history["w_phys"].append(w_phys)

        if val_loss < best_vals["C"]:
            best_vals["C"] = val_loss
            best_states["C"] = {'net': deepcopy(net.state_dict()),
                                'opinf': deepcopy(opinf_model.state_dict())}
        if Params.output and epoch % max(1, epochs_C // 5) == 0:
            print(f"  epoch {epoch}: train {avg_loss:.2e}, val {val_loss:.2e},"
                  f" w_phys {w_phys:.2e}")

    if best_states["C"]:
        net.load_state_dict(best_states["C"]['net'])
        opinf_model.load_state_dict(best_states["C"]['opinf'])

    # --- residual for the non-Markovian stage --------------------------------
    # Computed from the model's own forward pass over the whole training set,
    # so it contains every operator including the quadratic one, and against
    # the same derivative the operators were fitted to.
    net.eval()
    opinf_model.eval()
    inp_full = torch.cat([(t_raw - t_min) / t_scale, u_all],
                         1).requires_grad_(True)
    ddt_full = compute_derivatives(net, inp_full, t_scale, create_graph=False,
                                   use_vmap=use_vmap).detach()
    with torch.no_grad():
        x_full, _ = net(inp_full)
        x_states_full = (x_full if PINN_STATE_TARGET == "surrogate" else x_all)
        residual = ddt_full - opinf_model(x_states_full, t_raw, u_all)
        error_reduced_states = residual.cpu().numpy().T

        if Params.output:
            print(f"  best validation: A {best_vals['A']:.3e}, "
                  f"B {best_vals['B']:.3e}, C {best_vals['C']:.3e}")
            denom = float(torch.linalg.norm(ddt_full).item())
            if denom > 0:
                rel = float(torch.linalg.norm(residual).item()) / denom
                print(f"  one-step residual against the surrogate derivative:"
                      f" {rel:.2%}")

    # The surrogate's states and derivatives on the training grid, in the
    # caller's layout (r, k), for use as smoothed training data.
    history["surrogate_states"] = x_full.detach().cpu().numpy().T
    history["surrogate_ddt"] = ddt_full.detach().cpu().numpy().T

    if Params.output:
        plot_pinn_training(history)

    return opinf_model, history, error_reduced_states


# =============================================================================
# Reporting
# =============================================================================

def _smooth(x, k=5):
    x = np.asarray(x, dtype=float)
    return x if x.size < k else np.convolve(x, np.ones(k) / k, mode='same')


def plot_pinn_training(history, filename=None):
    """Training history. Panels without recorded data are left out."""
    panels = [
        ("Stage A: data loss", ["train_data", "val_data"], True),
        ("Stage B: operator loss", ["train_phys", "val_phys"], True),
        ("Stage B: loss terms", ["train_ddt", "train_state"], True),
        ("Stage C: total loss", ["train_total", "val_total"], True),
        ("Stage C: physics weight", ["w_phys"], False),
    ]
    panels = [p for p in panels
              if any(len(history.get(k, [])) > 0 for k in p[1])]
    if not panels:
        return

    rows = (len(panels) + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axs = np.atleast_1d(axs).ravel()

    for ax, (title, keys, logy) in zip(axs, panels):
        for key in keys:
            series = history.get(key, [])
            if len(series) == 0:
                continue
            y = _smooth(series) if logy else np.asarray(series, dtype=float)
            x = np.arange(1, len(y) + 1)
            (ax.semilogy if logy else ax.plot)(x, y, label=key)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for ax in axs[len(panels):]:
        ax.axis('off')

    fig.tight_layout()
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)