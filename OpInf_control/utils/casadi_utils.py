"""CasADi utility functions for reduced-order model optimization."""
import casadi as ca
import numpy as np
import torch
import hashlib
from typing import Optional, Callable, Tuple


# ==============================================================================
# 1. HELPER: POLYNOMIAL FORMS
# ==============================================================================

def polynomial_form_casadi(x: ca.MX, p: int = 3) -> ca.MX:
    terms = [ca.power(x, power) for power in range(2, p + 1)]
    return ca.vertcat(*terms)

def polynomial_form_numpy(x: np.ndarray, p: int = 3) -> np.ndarray:
    terms = [np.power(x, power) for power in range(2, p + 1)]
    if x.ndim == 1: return np.concatenate(terms, axis=0)
    return np.vstack(terms)


# ==============================================================================
# 2. CNN INFERENCE LOGIC
# ==============================================================================

def evaluate_cnn_decoder_numpy(
    x_red_unscaled_flat: np.ndarray,
    decoder: torch.nn.Module,
    input_scaler,     # Single scaler for latent space
    target_scaler_F,
    target_scaler_T,
    n_F_full: int,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Evaluates CNN decoder. Matches logic of 'apply_cnn_inference'.
    Returns PHYSICAL state (already unscaled/shifted by target scalers).
    """
    # 1. Prepare Input (1, n_latent)
    # x_red_unscaled_flat comes in shape (n_red,)
    x_red = x_red_unscaled_flat.reshape(1, -1)

    # 2. Scale Input (Latent Space Scaling)
    # Note: input_scaler expects shape (n_samples, n_features)
    input_CNN = input_scaler.transform(x_red)

    # 3. PyTorch Inference
    input_tensor = torch.tensor(input_CNN, dtype=torch.float32, device=device)
    with torch.no_grad():
        sol_decoder = decoder(input_tensor).cpu().numpy()  # (1, n_full_phys)

    # 4. Inverse Transform Outputs (Physical Space Unscaling)
    # sol_decoder has shape (1, 776) -> [F_part, T_part]

    # Split
    pred_F_scaled = sol_decoder[:, :n_F_full]
    pred_T_scaled = sol_decoder[:, n_F_full:]

    # Unscale
    pred_F = target_scaler_F.inverse_transform(pred_F_scaled)
    pred_T = target_scaler_T.inverse_transform(pred_T_scaled)

    # 5. Return flattened full vector
    return np.hstack((pred_F, pred_T)).flatten()


class CNNCallbackCached(ca.Callback):
    """CasADi callback for CNN with caching."""
    def __init__(self, name, evaluate_func, n_red, n_full, cache_size=1000):
        ca.Callback.__init__(self)
        self.evaluate_func = evaluate_func
        self.n_red = n_red
        self.n_full = n_full
        self.cache_size = cache_size
        self.cache = {}
        self.construct(name, {'enable_fd': True})

    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i): return ca.Sparsity.dense(self.n_red, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(self.n_full, 1)

    def eval(self, arg):
        x_red_num = np.array(arg[0]).flatten()
        # Hash for cache
        x_rounded = np.round(x_red_num, decimals=6)
        key = hashlib.md5(x_rounded.tobytes()).hexdigest()

        if key in self.cache:
            return [self.cache[key]]

        full_state = self.evaluate_func(x_red_num)
        result = ca.DM(full_state.reshape(-1, 1))

        if len(self.cache) < self.cache_size:
            self.cache[key] = result
        return [result]


def create_cnn_casadi_function(
    decoder: torch.nn.Module,
    scalers: Tuple,
    r_F: int,
    r_T: int,
    n_F_full: int,
    n_full: int,
    device: str = 'cpu',
    enable_caching: bool = True,
    cache_size: int = 1000,
) -> ca.Callback:
    """Factory for CasADi CNN callback."""
    # Unpack scalers (Tuple from load_results)
    # Order must match main.py packing: (input, target_F, target_T)
    input_scaler, target_scaler_F, target_scaler_T = scalers

    def evaluate_func(x_red_flat):
        return evaluate_cnn_decoder_numpy(
            x_red_flat, decoder, input_scaler, target_scaler_F, target_scaler_T,
            n_F_full, device
        )

    # Actual output size of decoder is 2 * n_F_full (Physical F + T)
    # The 'n_full' passed here should ideally match 776, not 1552.
    # But let's rely on what evaluate_func returns.
    output_dim = 2 * n_F_full

    return CNNCallbackCached('cnn_decoder', evaluate_func, r_F + r_T, output_dim, cache_size)


# ==============================================================================
# 3. RECONSTRUCTION LOGIC (Consolidated)
# ==============================================================================

# ==============================================================================
# 4. RECONSTRUCTION LOGIC (LINEAR vs CNN)
# ==============================================================================

def reduced_to_full_casadi(
    x_red_scaled: ca.MX,
    basis_type: str,
    r_F: int,
    r_T: int,
    scaling_fac_F: float,
    scaling_fac_T: float,
    basis: np.ndarray,
    reference_states_shifting: np.ndarray,
    state_max_F: float,
    state_max_T: float,
    V_reduced_nonlin: Optional[np.ndarray] = None,
    Xi: Optional[np.ndarray] = None,
    poly_degree: int = 3,
    has_decoder: bool = False,
    cnn_func: Optional[ca.Callback] = None,
    n_full: Optional[int] = None,
    n_F_full: Optional[int] = None,
) -> ca.MX:
    """
    Reconstructs full physical state (CasADi).
    Switches between Linear (POD) and Non-linear (CNN) automatically.
    """
    # 1. Latent Unscaling (Always needed first)
    # Converts optimizer variables (approx 0..1) to Latent State Space
    X_lat = x_red_scaled[:r_F] * float(scaling_fac_F)
    T_lat = x_red_scaled[r_F:r_F + r_T] * float(scaling_fac_T)
    x_lat = ca.vertcat(X_lat, T_lat)

    # --- BRANCH A: CNN DECODER ---
    if has_decoder and cnn_func is not None:
        # CNN takes Latent State -> Returns Physical State [F; T]
        # Includes Inverse Scaling + Shifting internally via TargetScalers
        return cnn_func(x_lat)

    # --- BRANCH B: LINEAR POD ---
    # Project to Full Normalized Space (includes Lifting w1, w2 if present)
    basis_ca = ca.DM(basis)
    x_full_norm = basis_ca @ x_lat

    # NL-POD Correction
    if basis_type in ['NL-POD', 'AM'] and V_reduced_nonlin is not None:
        poly = polynomial_form_casadi(x_lat, p=poly_degree)
        V_nl = ca.DM(V_reduced_nonlin)
        Xi_ca = ca.DM(Xi)
        x_full_norm += V_nl @ Xi_ca @ poly

    # Truncation & Physical Unscaling
    if n_F_full is None: n_F_full = basis.shape[0] // 4

    F_norm = x_full_norm[:n_F_full]
    T_norm = x_full_norm[n_F_full : 2*n_F_full]

    F_phys_delta = F_norm * state_max_F
    T_phys_delta = T_norm * state_max_T
    x_phys_delta = ca.vertcat(F_phys_delta, T_phys_delta)

    # Shifting
    ref_full = ca.DM(reference_states_shifting).reshape((-1, 1))
    shift_trunc = ca.vertcat(ref_full[:n_F_full], ref_full[n_F_full : 2*n_F_full])

    return x_phys_delta + shift_trunc


def reduced_to_full_numpy(
    x_red_scaled: np.ndarray,
    basis_type: str,
    r_F: int,
    r_T: int,
    scaling_fac_F: float,
    scaling_fac_T: float,
    basis: np.ndarray,
    reference_states_shifting: np.ndarray,
    state_max_F: float,
    state_max_T: float,
    V_reduced_nonlin: Optional[np.ndarray] = None,
    Xi: Optional[np.ndarray] = None,
    poly_degree: int = 3,
    has_decoder: bool = False,
    decoder: torch.nn.Module = None,
    scalers: Tuple = None,
    n_F_full: Optional[int] = None,
    device: str = 'cpu',
) -> np.ndarray:
    """Numpy equivalent of reduced_to_full_casadi."""
    flatten = False
    if x_red_scaled.ndim == 1:
        x_red_scaled = x_red_scaled.reshape(-1, 1)
        flatten = True

    # 1. Latent Unscaling
    X_lat = x_red_scaled[:r_F, :] * scaling_fac_F
    T_lat = x_red_scaled[r_F:r_F + r_T, :] * scaling_fac_T
    x_lat = np.vstack((X_lat, T_lat))

    # --- BRANCH A: CNN DECODER ---
    if has_decoder and decoder is not None:
        input_scaler, target_scaler_F, target_scaler_T = scalers

        # Batch inference logic
        # Transpose to (N_samples, Features) for Scaler/Torch
        inputs_T = x_lat.T
        inputs_scaled = input_scaler.transform(inputs_T)
        inputs_torch = torch.tensor(inputs_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            pred = decoder(inputs_torch).cpu().numpy() # (N, 776)

        # Inverse Scale
        # pred has shape (N, 776). Split cols.
        pred_F = target_scaler_F.inverse_transform(pred[:, :n_F_full])
        pred_T = target_scaler_T.inverse_transform(pred[:, n_F_full:])

        # Stack back to (776, N)
        full = np.hstack((pred_F, pred_T)).T
        return full.flatten() if flatten else full

    # --- BRANCH B: LINEAR POD ---
    x_full_norm = basis @ x_lat

    if basis_type in ['NL-POD', 'AM'] and V_reduced_nonlin is not None:
        poly = polynomial_form_numpy(x_lat, p=poly_degree)
        x_full_norm += V_reduced_nonlin @ Xi @ poly

    if n_F_full is None: n_F_full = basis.shape[0] // 4

    F_norm = x_full_norm[:n_F_full, :]
    T_norm = x_full_norm[n_F_full : 2*n_F_full, :]

    F_phys_delta = F_norm * state_max_F
    T_phys_delta = T_norm * state_max_T
    x_phys_delta = np.vstack((F_phys_delta, T_phys_delta))

    shift_F = reference_states_shifting[:n_F_full].reshape((-1, 1))
    shift_T = reference_states_shifting[n_F_full : 2*n_F_full].reshape((-1, 1))
    shift_trunc = np.vstack((shift_F, shift_T))

    full = x_phys_delta + shift_trunc

    return full.flatten() if flatten else full
