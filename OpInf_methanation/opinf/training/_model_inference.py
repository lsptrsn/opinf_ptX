__all__ = ["train_model", "learned_model"]

# operators/training.py (Updated)
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.utils.data

import opinf

# Load global params pointer (Singleton)
Params = opinf.parameters.Params()
device = opinf.utils.get_device()


def train_model(states, derivatives, time, entries, rom, seed=None):
    """
    Trains a model using the operator inference method (Gradient Descent).
    Jetzt mit seed-Argument für volle Kontrolle durch Optuna.
    """
    # 1. Force Determinism
    current_seed = seed if seed is not None else 0
    opinf.utils.set_deterministic(seed=current_seed)

    # 1. Prepare Tensors
    states_torch = torch.tensor(states, device=device, dtype=torch.float32).T
    derivatives_torch = torch.tensor(derivatives, device=device, dtype=torch.float32).T
    entries_torch = torch.tensor(entries, device=device, dtype=torch.float32).T

    # Handle time vector robustly
    if isinstance(time, np.ndarray):
        time_torch = torch.tensor(time, device=device, dtype=torch.float32).reshape(-1, 1)
    else:
        dt = time if np.isscalar(time) else 1.0
        t_vec = torch.arange(states.shape[1], device=device) * dt
        time_torch = t_vec.reshape(-1, 1)

    # 2. Dataset & Batch Sizing
    n_samples = states.shape[1]

    if Params.batch_size > n_samples:
        if Params.output:
            print(f"Info: Batch size ({Params.batch_size}) > Samples ({n_samples}). Using Full Batch.")
        effective_batch_size = n_samples
        drop_last = False
    else:
        effective_batch_size = Params.batch_size
        drop_last = True

    train_dataset = torch.utils.data.TensorDataset(
        states_torch, time_torch, derivatives_torch, entries_torch
    )

    is_sequential_loss = Params.ROM_loss in ['states', 'hybrid']
    do_shuffle = not is_sequential_loss

    g = torch.Generator()
    g.manual_seed(current_seed)

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=do_shuffle,
        drop_last=drop_last,
        generator=g
    )
    dataloaders = {"train": train_dl}

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        rom.parameters(),
        lr=Params.adam_lr,
        betas=Params.adam_betas,
        eps=Params.adam_eps,
        weight_decay=Params.adam_weight_decay
    )

    # 4. Scheduler (Safe Init)
    steps_per_epoch = len(train_dl)
    if steps_per_epoch == 0: steps_per_epoch = 1

    step_epochs = int(Params.lr_schedule_step_factor)
    step_size_up = max(1, step_epochs * steps_per_epoch)

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        step_size_up=step_size_up,
        step_size_down=step_size_up,
        mode=Params.lr_schedule_mode,
        cycle_momentum=False,
        base_lr=Params.lr_schedule_base_lr,
        max_lr=Params.lr_schedule_max_lr,
    )

    torch.nn.utils.clip_grad_norm_(rom.parameters(), max_norm=1.0)

    # 5. Run Training
    model, loss_track, error_reduced_states = _fit(
        rom, dataloaders, optimizer, scheduler=scheduler
    )

    return model, loss_track, error_reduced_states


def learned_model(model):
    """Extracts learned parameters as numpy arrays."""
    # Access underlying module if wrapped in DataParallel
    m = model.module if hasattr(model, 'module') else model

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor is not None else None

    sys_order = Params.ROM_order

    A = to_numpy(m.A) if 'A' in Params.model_structure and m.A is not None else np.zeros((sys_order, sys_order))
    B = to_numpy(m.B) if 'B' in Params.model_structure and m.B is not None else np.zeros((sys_order, Params.input_dim))
    C = to_numpy(m.C).reshape(-1) if 'C' in Params.model_structure and m.C is not None else np.zeros((sys_order,))
    H = to_numpy(m.H) if 'H' in Params.model_structure and m.H is not None else np.zeros((sys_order, sys_order**2))

    # Reshape B specifically if needed, though usually (order, inputs) is correct
    return A, B, C, H


def _rk4th_onestep(model, x, u, t, timestep):
    """Runge-Kutta 4th order integration step."""
    # Ensure timestep is (Batch, 1) for correct broadcasting
    if timestep.ndim == 1:
        timestep = timestep.view(-1, 1)

    k1 = model(x, t, u)
    k2 = model(x + 0.5 * timestep * k1, t + 0.5 * timestep, u)
    k3 = model(x + 0.5 * timestep * k2, t + 0.5 * timestep, u)
    k4 = model(x + 1.0 * timestep * k3, t + 1.0 * timestep, u)

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep


def _fit(model, dataloaders, optimizer, scheduler=None):
    if Params.output:
        print("_" * 75)
        print(f"Matrix training started (Loss: {Params.ROM_loss})")

    loss_track = []
    best_loss = float('inf')
    best_model_state_dict = deepcopy(model.state_dict())

    patience = 1000
    counter = 0
    improvement_threshold = 1e-9

    # Use reduction='mean' because we will handle masking manually via slicing
    criterion = torch.nn.MSELoss()

    # AUTO-BALANCING: Scale factors for hybrid loss
    loss_scale_deriv = 1.0
    loss_scale_state = 1.0
    initialized_scales = False

    for epoch in range(Params.num_epochs):
        model.train()
        total_loss_epoch = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloaders["train"]):
            optimizer.zero_grad()

            # Unpack batch: (Batch_Size, Dim)
            state, time, ddt, entries = batch

            # --- 1. Derivative Loss (ddt) ---
            # This is safe to calculate on all points, as it doesn't use time-stepping
            if Params.ROM_loss in ['ddt', 'hybrid']:
                y_pred = model(x=state, t=time, u=entries)
                loss_d = criterion(y_pred, ddt)
            else:
                loss_d = torch.tensor(0.0, device=device)

            # --- 2. State Loss (Integration) ---
            # This requires valid time steps (t -> t+1)
            loss_s = torch.tensor(0.0, device=device)

            if Params.ROM_loss in ['states', 'hybrid']:
                # Calculate dt between consecutive samples in the batch
                # Assumes batch is sequential (shuffle=False or carefully handled)
                dt_vec = time[1:] - time[:-1]

                # --- CRITICAL FIX: FILTER JUMPS ---
                # We identify valid transitions.
                # A transition is valid if dt > 0 (time moves forward).
                # Negative dt indicates a jump between trajectories (e.g., 3600 -> 0).
                valid_mask = (dt_vec > 0).reshape(-1)

                # Only proceed if we have valid transitions in this batch
                if valid_mask.sum() > 0:
                    # slice inputs to keep ONLY valid transitions
                    # We do this BEFORE the solver to prevent 'inf' generation
                    curr_state = state[:-1][valid_mask]
                    curr_entries = entries[:-1][valid_mask]
                    curr_time = time[:-1][valid_mask]
                    valid_dt = dt_vec[valid_mask]

                    target_next_state = state[1:][valid_mask]

                    # Run Solver ONLY on valid chunks
                    pred_next = _rk4th_onestep(model, curr_state, curr_entries, curr_time, valid_dt)

                    # Calculate MSE only on valid predictions
                    loss_s = criterion(pred_next, target_next_state)
                else:
                    # If a batch happens to contain ONLY a jump (unlikely but possible), skip
                    loss_s = torch.tensor(0.0, device=device)

            # --- 3. Auto-Balancing (First valid batch only) ---
            if not initialized_scales and Params.ROM_loss == 'hybrid':
                with torch.no_grad():
                    # Calculate scales to bring both losses to range ~1.0
                    # Safety check against zero or tiny losses
                    d_val = loss_d.item()
                    s_val = loss_s.item()

                    if d_val > 1e-9: loss_scale_deriv = 1.0 / d_val
                    if s_val > 1e-9: loss_scale_state = 1.0 / s_val

                    if Params.output:
                        print(f"  Auto-Balancing: Scale D={loss_scale_deriv:.2e}, Scale S={loss_scale_state:.2e}")
                    initialized_scales = True

            # --- 4. Combine Losses ---
            if Params.ROM_loss == 'ddt':
                loss_val = loss_d
            elif Params.ROM_loss == 'states':
                loss_val = loss_s
            else: # hybrid
                # Apply weights and auto-scaling
                ld = Params.lambda_deriv
                ls = Params.lambda_state
                loss_val = ld * (loss_d * loss_scale_deriv) + ls * (loss_s * loss_scale_state)

            # --- REGULARIZATION ---
            if 'H' in Params.model_structure and Params.regularization_H > 0:
                m = model.module if hasattr(model, 'module') else model
                loss_val += Params.regularization_H * m.H.abs().mean()

            if 'A' in Params.model_structure and Params.regularization_A > 0:
                m = model.module if hasattr(model, 'module') else model
                loss_val += Params.regularization_A * (m.A ** 2).mean()

            # --- SAFETY CHECK ---
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                # Skip backward pass to avoid corrupting weights
                continue

            # Optimization Step
            loss_val.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            total_loss_epoch += loss_val.item()
            loss_track.append(loss_val.item())
            n_batches += 1

        # --- End of Epoch Calculation ---
        if n_batches > 0:
            avg_loss = total_loss_epoch / n_batches
        else:
            avg_loss = float('inf')

        if (epoch + 1) % max(1, int(Params.num_epochs/10)) == 0:
            if Params.output:
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{Params.num_epochs} | Loss: {avg_loss:.4e} | LR: {curr_lr:.2e}")

        # Early Stopping
        if avg_loss < best_loss:
            improvement = best_loss - avg_loss
            best_loss = avg_loss
            best_model_state_dict = deepcopy(model.state_dict())

            if improvement > improvement_threshold:
                counter = 0
            else:
                counter += 1
        else:
            counter += 1

        if counter >= patience:
            if Params.output: print(f"Early stopping at epoch {epoch + 1}.")
            break

    # Finish
    model.load_state_dict(best_model_state_dict)
    if Params.output:
        print("Training completed.")
        print(f"Best loss: {best_loss:.5f}")

    # Calculate Final Error (approximate on last batch for shape consistency)
    model.eval()
    with torch.no_grad():
        y_pred = model(x=state, t=time, u=entries)
        error = (ddt - y_pred).detach().cpu().numpy().T

    return model, loss_track, error
