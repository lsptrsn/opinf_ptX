# train_pinn_opinf_stages.py
# Optimized for Robustness & Memory Efficiency

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

import opinf

device = opinf.utils.get_device()
Params = opinf.parameters.Params()


def train_pinn_opinf(states, time, entries, opinf_model, r_X):
    """
    Train PINN + OpInf with stabilized Stage C (Suab-stepping).
    """
    opinf.utils.set_deterministic(0)
    try:
        from torch.func import vmap, jacrev
        USE_VMAP = True
        if Params.output:
            print("INFO: torch.func available. Using vmap/jacrev for fast derivatives.")
    except ImportError:
        USE_VMAP = False
        if Params.output:
            print("WARNING: torch.func not found. Using loop-based derivatives (slower).")

    # 1. Setup Data
    t_raw = torch.tensor(time, dtype=torch.float32, device=device).reshape(-1, 1)
    u_all = torch.tensor(entries.T, dtype=torch.float32, device=device)
    x_all = torch.tensor(states.T, dtype=torch.float32, device=device)

    n_samples = len(time)
    indices = torch.arange(n_samples, device=device)
    dataset = TensorDataset(t_raw, u_all, x_all, indices)

    # 2. Time Normalization
    t_min, t_max = t_raw.min(), t_raw.max()
    t_scale = (t_max - t_min).clamp(min=1e-6)

    # 3. Split Train/Val
    train_ratio = 0.9
    n_train = int(train_ratio * n_samples)

    g = torch.Generator()
    g.manual_seed(0)
    perm = torch.randperm(n_samples, generator=g)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # Use slightly larger batch size for stability if using sub-steps
    eff_batch_size = Params.PINN_batch_size
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=eff_batch_size,
        shuffle=True,
        generator=g
        )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=eff_batch_size,
        shuffle=False
        )

    # 4. Network Setup
    n_in = int(entries.shape[0]) + 1
    n_out = int(states.shape[0])
    net = opinf.models.create_network(n_in, list(Params.PINN_hidden_layers), n_out).to(device)

    # 5. Optimizers & Schedulers
    # Slower decay (factor=0.8) to keep learning longer
    opt_net = torch.optim.AdamW(net.parameters(), lr=Params.PINN_lr_net, weight_decay=1e-6)
    opt_opinf = torch.optim.AdamW(opinf_model.parameters(), lr=Params.PINN_lr_opinf, weight_decay=1e-6)

    sched_net = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_net, factor=0.8, patience=50, verbose=False)
    sched_opinf = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_opinf, factor=0.8, patience=50, verbose=False)

    # History
    history = {k: [] for k in ["train_data", "val_data", "train_phys", "val_phys", "train_total", "val_total"]}
    best_states = {"A": None, "B": None, "C": None}
    best_vals = {"A": float('inf'), "B": float('inf'), "C": float('inf')}

    # Stage Config
    total_epochs = Params.PINN_epochs
    epochs_A = int(Params.PINN_stage_config_A * total_epochs)
    epochs_B = int(Params.PINN_stage_config_B * total_epochs)
    epochs_C = total_epochs - epochs_A - epochs_B

    if Params.output:
        print(f"PINN Schedule: A={epochs_A}, B={epochs_B}, C={epochs_C}")

    # =========================================================================
    # STAGE A: NN Training (Denoising)
    # =========================================================================
    if Params.output: print("--- Stage A: Training NN (Denoising) ---")
    net.train(); opinf_model.eval()

    for epoch in range(epochs_A):
        loss_acc = 0.0
        for t_b, u_b, x_b, _ in train_loader:
            opt_net.zero_grad()
            t_norm = (t_b - t_min) / t_scale
            inp = torch.cat([t_norm, u_b], dim=1)
            x_pred, _ = net(inp)
            loss = F.mse_loss(x_pred, x_b)
            loss.backward()
            opt_net.step()
            loss_acc += loss.item()

        avg_loss = loss_acc / len(train_loader)
        val_loss = validate_stage_A(net, val_loader, t_min, t_scale)
        sched_net.step(val_loss)

        history["train_data"].append(avg_loss); history["val_data"].append(val_loss)
        if val_loss < best_vals["A"]:
            best_vals["A"] = val_loss
            best_states["A"] = deepcopy(net.state_dict())

        if epoch % max(1, epochs_A//5) == 0 and Params.output:
            print(f"  Ep {epoch}: Train={avg_loss:.2e}, Val={val_loss:.2e}")

    if best_states["A"]: net.load_state_dict(best_states["A"])

    # =========================================================================
    # STAGE B: OpInf Training (Physics Fit)
    # =========================================================================
    if Params.output: print("--- Stage B: Training OpInf (Derivative Matching) ---")
    net.eval(); opinf_model.train()

    for epoch in range(epochs_B):
        loss_acc = 0.0
        for t_b, u_b, _, _ in train_loader:
            opt_opinf.zero_grad()
            t_norm = (t_b - t_min) / t_scale
            inp = torch.cat([t_norm, u_b], dim=1)
            inp.requires_grad_(True)

            # Get Teacher Targets (No Net update)
            ddt_target = compute_derivatives(net, inp, t_scale, create_graph=False, use_vmap=USE_VMAP)
            with torch.no_grad(): x_pred, _ = net(inp)

            # OpInf Update
            ddt_pred = opinf_model(x_pred.detach(), t_b, u_b)
            loss = F.mse_loss(ddt_pred, ddt_target)

            # Small Reg on H to keep it clean
            if hasattr(opinf_model, 'module'): m = opinf_model.module
            else: m = opinf_model
            if m.H is not None: loss += 1e-6 * m.H.abs().mean()

            loss.backward()
            opt_opinf.step()
            loss_acc += loss.item()

        avg_loss = loss_acc / len(train_loader)
        val_loss = validate_stage_B(net, opinf_model, val_loader, t_min, t_scale, USE_VMAP)
        sched_opinf.step(val_loss)

        history["train_phys"].append(avg_loss); history["val_phys"].append(val_loss)
        if val_loss < best_vals["B"]:
            best_vals["B"] = val_loss
            best_states["B"] = deepcopy(opinf_model.state_dict())

        if epoch % max(1, epochs_B//5) == 0 and Params.output:
            print(f"  Ep {epoch}: Train={avg_loss:.2e}, Val={val_loss:.2e}")

    if best_states["B"]: opinf_model.load_state_dict(best_states["B"])

    # =========================================================================
    # STAGE C: Joint Training (Stabilized)
    # =========================================================================
    if Params.output: print("--- Stage C: Joint Training (Sub-stepped) ---")

    # Heuristic weighting or manual override
    w_phys = Params.PINN_phys_weight
    if best_vals["B"] > 0 and w_phys == 1.0: # Only auto-tune if default
        w_phys = best_vals["A"] / best_vals["B"]

    # CRITICAL FIX: Sub-steps for OpInf
    # Train OpInf 5 times for every 1 Net update to ensure Physics keeps up
    opinf_steps_per_net = 5

    for epoch in range(epochs_C):
        net.train(); opinf_model.train()
        loss_acc = 0.0

        for t_b, u_b, x_b, _ in train_loader:
            t_norm = (t_b - t_min) / t_scale
            inp = torch.cat([t_norm, u_b], dim=1)

            # --- 1. Update NN (Data + Physics) ---
            inp.requires_grad_(True)
            opt_net.zero_grad()

            x_pred, _ = net(inp)
            ddt_nn = compute_derivatives(net, inp, t_scale, create_graph=True, use_vmap=USE_VMAP)

            # Re-eval OpInf with graph connected to x_pred
            ddt_opinf_connected = opinf_model(x_pred, t_b, u_b)

            l_data = F.mse_loss(x_pred, x_b)
            l_phys = F.mse_loss(ddt_nn, ddt_opinf_connected)

            loss_net = l_data + w_phys * l_phys
            loss_net.backward()
            opt_net.step()

            # --- 2. Update OpInf (Stabilization Steps) ---
            # We update OpInf multiple times to fit the NEW NN manifold
            for _ in range(opinf_steps_per_net):
                opt_opinf.zero_grad()

                # Get fresh detached data from updated NN
                inp_det = inp.detach().requires_grad_(True)
                # We need target derivative from NN (Teacher)
                ddt_target = compute_derivatives(net, inp_det, t_scale, create_graph=False, use_vmap=USE_VMAP)
                with torch.no_grad():
                    x_pred_new, _ = net(inp_det)

                # OpInf prediction
                ddt_opinf_new = opinf_model(x_pred_new, t_b, u_b)

                loss_opinf = F.mse_loss(ddt_opinf_new, ddt_target)
                loss_opinf.backward()
                opt_opinf.step()

            loss_acc += (loss_net.item() + loss_opinf.item())

        # Validation
        avg_loss = loss_acc / len(train_loader)
        val_loss = validate_stage_C(net, opinf_model, val_loader, t_min, t_scale, w_phys, USE_VMAP)

        history["train_total"].append(avg_loss); history["val_total"].append(val_loss)

        if val_loss < best_vals["C"]:
            best_vals["C"] = val_loss
            best_states["C"] = {'net': deepcopy(net.state_dict()), 'opinf': deepcopy(opinf_model.state_dict())}

        if epoch % max(1, epochs_C//5) == 0 and Params.output:
            print(f"  Ep {epoch}: Train={avg_loss:.2e}, Val={val_loss:.2e}")

    if best_states["C"]:
        net.load_state_dict(best_states["C"]['net'])
        opinf_model.load_state_dict(best_states["C"]['opinf'])

    if Params.output:
        plot_pinn_training(history, r_X)

    return opinf_model, history

# --------------------------
# Helpers
# --------------------------
def select_data_loss(name: str):
    """Return data loss function (mse or huber)."""
    if name is None:
        name = "mse"
    name = name.lower()
    if name == "huber":
        return lambda p, t: F.smooth_l1_loss(p, t)
    else:
        return lambda p, t: F.mse_loss(p, t)


def grad_norm(params):
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.detach().norm(2).cpu().item())**2
    return (tot**0.5)


def compute_derivatives(net, inp, t_scale, create_graph=False, use_vmap=False):
    """Robust derivative computation using vmap or fallback."""
    if use_vmap:
        try:
            from torch.func import vmap, jacrev
            def func(x): return net(x.unsqueeze(0))[0].squeeze(0)
            # vmap over batch (dim 0)
            jac = vmap(jacrev(func))(inp)
            ddt = jac[:, :, 0] # Derivative w.r.t first input (time)
            return ddt / t_scale
        except ImportError:
            # Fallback to loop if vmap fails
            use_vmap = False

    # Fallback loop
    x_pred, _ = net(inp)
    grads = []
    for i in range(x_pred.shape[1]):
        g = torch.autograd.grad(x_pred[:, i].sum(), inp, create_graph=create_graph, retain_graph=True)[0]
        grads.append(g[:, 0:1])
    ddt = torch.cat(grads, dim=1)
    return ddt / t_scale


def _compute_derivatives_loop(net, inp_req, t_scale, create_graph=False):
    """
    Computes derivatives dx/dt using the original, reliable for-loop method.
    CORRECTED VERSION: Differentiates w.r.t. 'inp_req' instead of 'coords'.
    """
    x_pred, _ = net(inp_req) # We don't need the 'coords' output here anymore
    n_out = x_pred.shape[1]
    ddt_list = []

    # This is your trusted loop for calculating derivatives
    for i in range(n_out):
        g = torch.autograd.grad(
            outputs=x_pred[:, i],
            # --- THE FIX IS HERE ---
            # We differentiate w.r.t. the input that requires grad, which is 'inp_req'.
            inputs=inp_req,
            # --- END FIX ---
            grad_outputs=torch.ones_like(x_pred[:, i]),
            create_graph=create_graph,
            retain_graph=True if create_graph or (i < n_out - 1) else False,
            only_inputs=True
        )[0][:, 0:1] # Select derivative w.r.t. the first coordinate (time)
        ddt_list.append(g)

    ddt = torch.cat(ddt_list, dim=1)
    return ddt / t_scale

# Validation functions remain the same as previous clean version...
@torch.no_grad()
def validate_stage_A(net, loader, t_min, t_scale):
    acc = 0.0
    for t, u, x, _ in loader:
        tn = (t - t_min)/t_scale
        xp, _ = net(torch.cat([tn, u], 1))
        acc += F.mse_loss(xp, x).item()
    return acc / len(loader)

@torch.no_grad()
def validate_stage_B(net, opinf, loader, t_min, t_scale, use_vmap=False):
    acc = 0.0
    for t, u, _, _ in loader:
        tn = (t - t_min)/t_scale
        inp = torch.cat([tn, u], 1)
        with torch.enable_grad():
            inp.requires_grad_(True)
            ddt_nn = compute_derivatives(net, inp, t_scale, create_graph=False, use_vmap=use_vmap)
        xp, _ = net(inp)
        acc += F.mse_loss(opinf(xp, t, u), ddt_nn).item()
    return acc / len(loader)

@torch.no_grad()
def validate_stage_C(net, opinf, loader, t_min, t_scale, w, use_vmap=False):
    acc = 0.0
    for t, u, x, _ in loader:
        tn = (t - t_min)/t_scale
        inp = torch.cat([tn, u], 1)
        xp, _ = net(inp)
        l_d = F.mse_loss(xp, x)
        with torch.enable_grad():
            inp.requires_grad_(True)
            ddt_nn = compute_derivatives(net, inp, t_scale, create_graph=False, use_vmap=use_vmap)
        l_p = F.mse_loss(ddt_nn, opinf(xp, t, u))
        acc += (l_d + w * l_p).item()
    return acc / len(loader)


class EarlyStopping:
    """Simple early stopping (lower metric is better)."""
    def __init__(self, patience=100, verbose=False):
        self.patience = int(patience)
        self.verbose = verbose
        self.counter = 0
        self.best_score = float('inf') # Initialize with infinity
        self.early_stop = False

    def reset(self):
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False

    def __call__(self, metric):
        # A lower metric is better
        if metric < self.best_score:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                if Params.output: print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


def smooth_states(states, window_length=21, polyorder=3):
    """
    Applies Savitzky-Golay smoothing to the states.
    This function ONLY smooths, it does not differentiate.
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    smoothed = scipy.signal.savgol_filter(
        states,
        window_length=window_length,
        polyorder=polyorder,
        deriv=0,  # deriv=0 means just smoothing
        axis=-1,
        mode='nearest'
    )
    return smoothed

def smooth(x, k=5):
    if len(x) < k:
        return np.array(x)
    return np.convolve(x, np.ones(k)/k, mode='same')


def first_valid_index(arr):
    """Return first index with finite value. If none, return 0."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0
    mask = np.isfinite(arr)
    if not mask.any():
        return 0
    return int(np.argmax(mask))  # first True index


def first_valid_index_any(history, keys):
    """
    Return the minimum first-valid index across multiple history keys.
    If no key has any finite values, returns 0.
    """
    idxs = []
    for k in keys:
        if k in history:
            arr = np.asarray(history[k], dtype=float)
            if arr.size == 0:
                continue
            mask = np.isfinite(arr)
            if mask.any():
                idxs.append(int(np.argmax(mask)))
    return int(min(idxs)) if len(idxs) > 0 else 0


def plot_pinn_training(history, r_X, filename="training_enhanced.svg"):
    """
    Plot training history.
    Start plotting at the earliest epoch where at least one monitored metric is finite.
    """
    # choose keys to determine the earliest valid epoch
    keys_to_check = [
        "val_total", "val_data", "val_phys",
        "train_total", "train_data", "train_phys",
        "learning_rates_net", "learning_rates_opinf",
        "gradnorm_net", "gradnorm_opinf"
    ]
    start_idx = first_valid_index_any(history, keys_to_check)

    # ensure consistent length handling
    it = max(len(history.get(k, [])) for k in ["train_total", "val_total", "train_data", "val_data", "train_phys", "val_phys", "learning_rates_net", "learning_rates_opinf", "gradnorm_net", "gradnorm_opinf"])
    epochs = np.arange(1, it + 1)

    sl = slice(start_idx, None)

    # safe accessor that returns zeros if key missing or too short
    def safe(arr_key):
        arr = np.asarray(history.get(arr_key, []), dtype=float)
        if arr.size < it:
            # pad with nan so slicing keeps alignment
            pad = np.full(it - arr.size, np.nan, dtype=float)
            arr = np.concatenate([arr, pad])
        return arr

    train_data = smooth(safe("train_data"))
    val_data = smooth(safe("val_data"))
    train_phys = smooth(safe("train_phys"))
    val_phys = smooth(safe("val_phys"))
    train_total = smooth(safe("train_total"))
    val_total = smooth(safe("val_total"))
    lr_net = safe("learning_rates_net")
    lr_opinf = safe("learning_rates_opinf")
    grad_net = safe("gradnorm_net")
    grad_opinf = safe("gradnorm_opinf")

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))

    # Data loss
    axs[0,0].semilogy(epochs[sl], train_data[sl], label='Train Data (sm)')
    axs[0,0].semilogy(epochs[sl], val_data[sl], '--', label='Val Data (sm)')
    axs[0,0].set_title('Data Loss (log)')
    axs[0,0].legend(); axs[0,0].grid(True, alpha=0.3)

    # Phys loss
    axs[0,1].semilogy(epochs[sl], train_phys[sl], label='Train Phys (sm)')
    axs[0,1].semilogy(epochs[sl], val_phys[sl], '--', label='Val Phys (sm)')
    axs[0,1].set_title('Physics Loss (log)')
    axs[0,1].legend(); axs[0,1].grid(True, alpha=0.3)

    # Total
    axs[1,0].semilogy(epochs[sl], train_total[sl], label='Train Total (sm)')
    axs[1,0].semilogy(epochs[sl], val_total[sl], '--', label='Val Total (sm)')
    axs[1,0].set_title('Total Loss (log)')
    axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)

    # Learning rates
    axs[1,1].semilogy(epochs[sl], lr_net[sl], label='LR Net')
    axs[1,1].semilogy(epochs[sl], lr_opinf[sl], label='LR OpInf')
    axs[1,1].set_title('LRs (log)')
    axs[1,1].legend(); axs[1,1].grid(True, alpha=0.3)

    # Gradnorms
    axs[2,0].semilogy(epochs[sl], grad_net[sl], label='GradNorm Net')
    axs[2,0].semilogy(epochs[sl], grad_opinf[sl], label='GradNorm OpInf')
    axs[2,0].set_title('Gradient norms (log)')
    axs[2,0].legend(); axs[2,0].grid(True, alpha=0.3)

    # empty panel
    axs[2,1].axis('off')

    plt.tight_layout()
    plt.show()
