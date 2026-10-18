#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruction of the full state and the neural decoder.

Three things in here are worth stating plainly, because the previous version
implied otherwise.

``reconstruct_solution`` is the deterministic map s_ref + V q + V_bar Xi g(q).
It needs no training data and no random seed; the arguments that suggested
otherwise were never used and have been removed.

The decoder is fitted on the *simulated* latent trajectory against the *true*
field. It therefore absorbs part of the dynamics error, not only the
reconstruction error, and describing it as resolving remaining spatial
nonlinearities understates what it does. It is also what lets it score below
the projection error of the linear basis, which for a linear reconstruction is
a floor: on the thermal case that floor is 7.5 % of the centred temperature
field, and the decoder is reported below it.

The decoder's validation set is now carved out chronologically. A random split
of a time series puts a sample and its neighbour ten seconds later on opposite
sides, so the validation loss measures interpolation between adjacent samples
rather than generalisation, and early stopping is driven by that number. Set
DECODER_VAL_SPLIT to "random" to reproduce the previous behaviour.
"""

__all__ = [
    "reconstruct_solution",
    "train_global_cnn",
    "apply_cnn_inference",
]

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data

import opinf
import opinf.parameters
import opinf.utils

device = opinf.utils.get_device()
Params = opinf.parameters.Params()


# --- switches -----------------------------------------------------------------
# "chronological": the last VAL_FRACTION of the samples form the validation set.
# "random"       : the previous behaviour, a shuffled split.
DECODER_VAL_SPLIT = "chronological"
VAL_FRACTION = 0.2

# Basis names that carry a manifold correction. The same set is spelled out in
# basis/_pod.py; it cannot be imported from there, see _manifold_degree below.
MANIFOLD_BASES = ('NL-POD', 'AM', 'MPOD', 'MAM')


def _manifold_degree():
    """Degree of the polynomial manifold, from the module that fitted Xi.

    Resolved when it is needed rather than at import. The package imports
    basis before post, and basis/_pod.py used to import opinf.post while still
    executing, so a module-level ``opinf.basis.MANIFOLD_DEGREE`` here ran
    before basis existed. That import has been removed at the other end as
    well, which is the actual repair; this function makes the order of imports
    irrelevant rather than merely currently harmless.
    """
    return opinf.basis.MANIFOLD_DEGREE


# =============================================================================
# Reconstruction
# =============================================================================

def reconstruct_solution(sol, sol_reduced, V_reduced_nonlin, Xi,
                         degree=None):
    """Add the polynomial manifold correction to a linear reconstruction.

    Parameters
    ----------
    sol : (n, k) ndarray
        Linear reconstruction V q, in the shifted and scaled space. The
        correction belongs in that space: Xi was regressed against the shifted,
        scaled snapshots, so adding it after the state has been scaled back
        applies it in the wrong units.
    sol_reduced : (r, k) ndarray
        Latent trajectory, unscaled in the reduced sense.
    V_reduced_nonlin : (n_full, q) ndarray
        Correction basis, block diagonal over the physical fields.
    Xi : (q, (degree - 1) r) ndarray
        Coefficient matrix.
    degree : int, optional
        Degree of the polynomial embedding. Defaults to the value the basis
        module fitted Xi with, so the two cannot drift apart.

    Returns
    -------
    sol : (n, k) ndarray
        Corrected reconstruction, or the input unchanged when the configured
        basis carries no manifold.
    """
    if Params.basis not in MANIFOLD_BASES:
        return sol
    if V_reduced_nonlin is None or Xi is None or V_reduced_nonlin.shape[1] == 0:
        return sol

    degree = _manifold_degree() if degree is None else degree

    poly = np.concatenate(
        opinf.basis.polynomial_form(sol_reduced, p=degree), axis=0
    )
    correction = V_reduced_nonlin @ Xi @ poly

    # The correction basis spans every physical field, including any that carry
    # no retained modes; sol covers only the fields that are reconstructed.
    # Both are ordered the same way, so the leading rows and columns line up.
    if correction.shape[1] > sol.shape[1]:
        correction = correction[:, :sol.shape[1]]
    if correction.shape[0] < sol.shape[0]:
        raise ValueError(
            f"correction has {correction.shape[0]} rows, "
            f"reconstruction has {sol.shape[0]}"
        )

    return sol + correction[:sol.shape[0], :]


# =============================================================================
# Decoder
# =============================================================================

def train_global_cnn(sol_reduced_list, F_train_true, T_train_true, seed=None):
    """Fit the decoder from latent trajectories to the full fields.

    The inputs are the latent trajectories produced by integrating the reduced
    model, not projections of the data, so the decoder learns to map a
    simulated latent state onto the measured field.
    """
    current_seed = seed if seed is not None else 0
    if Params.output:
        print("\n--- Training the decoder ---")

    r_F, r_T = Params.r_F, Params.r_T

    # Latent inputs, one sample per row.
    if isinstance(sol_reduced_list, list):
        X_input = np.vstack([s[:r_F, :].T for s in sol_reduced_list])
        T_input = np.vstack([s[r_F:r_F + r_T, :].T for s in sol_reduced_list])
    else:
        X_input = sol_reduced_list[:r_F, :].T
        T_input = sol_reduced_list[r_F:r_F + r_T, :].T
    cnn_input = np.hstack((X_input, T_input))

    # Physical targets.
    if isinstance(F_train_true, list):
        F_true = np.hstack(F_train_true)
        T_true = np.hstack(T_train_true)
    else:
        F_true, T_true = F_train_true, T_train_true

    target_scaler_F = preprocessing.StandardScaler().fit(F_true.T)
    target_scaler_T = preprocessing.StandardScaler().fit(T_true.T)
    cnn_target = np.hstack((target_scaler_F.transform(F_true.T),
                            target_scaler_T.transform(T_true.T)))

    # Validation split.
    if DECODER_VAL_SPLIT == "chronological":
        n_val = max(1, int(round(VAL_FRACTION * cnn_input.shape[0])))
        split = cnn_input.shape[0] - n_val
        X_tr, X_val = cnn_input[:split], cnn_input[split:]
        y_tr, y_val = cnn_target[:split], cnn_target[split:]
    elif DECODER_VAL_SPLIT == "random":
        X_tr, X_val, y_tr, y_val = train_test_split(
            cnn_input, cnn_target, test_size=VAL_FRACTION,
            random_state=current_seed
        )
    else:
        raise ValueError(f"unknown DECODER_VAL_SPLIT: {DECODER_VAL_SPLIT}")

    if Params.output:
        print(f"  validation split: {DECODER_VAL_SPLIT}, "
              f"{X_tr.shape[0]} train / {X_val.shape[0]} validation samples")

    # Input noise on the training half only.
    rng = np.random.RandomState(current_seed)
    std_feat = X_tr.std(axis=0, keepdims=True)
    X_tr_noisy = X_tr + Params.CNN_input_noise * std_feat * rng.randn(*X_tr.shape)

    input_scaler = preprocessing.StandardScaler().fit(X_tr_noisy)
    X_tr_final = input_scaler.transform(X_tr_noisy)
    X_val_final = input_scaler.transform(X_val)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_tr_final, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val_final, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=Params.CNN_batch_size, shuffle=True, drop_last=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=Params.CNN_batch_size, shuffle=False
    )

    # The decoder needs the width of the field it reconstructs, nothing more.
    # This used to be handed over as a zero matrix of that many rows, which
    # train_decoder then also used to compute what it printed as a POD
    # baseline; multiplying by zeros, that baseline was the target variance.
    # Both ends now say what they mean.
    output_dim = F_true.shape[0] + T_true.shape[0]

    decoder, _, _ = opinf.post.train_decoder(
        train_dl=train_dl, val_dl=val_dl,
        output_dim=output_dim,
        latent_dim=r_F + r_T,
        seed=current_seed,
    )

    return {
        'decoder': decoder.float().to(device),
        'input_scaler': input_scaler,
        'target_scaler_F': target_scaler_F,
        'target_scaler_T': target_scaler_T,
    }


def apply_cnn_inference(sol_reduced, cnn_pack):
    """Map a latent trajectory to the full physical state with the decoder."""
    decoder = cnn_pack['decoder']
    input_scaler = cnn_pack['input_scaler']
    ts_F, ts_T = cnn_pack['target_scaler_F'], cnn_pack['target_scaler_T']

    inputs_scaled = input_scaler.transform(sol_reduced.T)
    inputs_torch = torch.tensor(inputs_scaled, dtype=torch.float32,
                                device=device)

    decoder.eval()
    with torch.no_grad():
        # The decoder may live on another device than the one the inputs
        # were built on, for instance when a control script loads it on the
        # GPU; move the inputs to wherever its weights are.
        inputs_torch = inputs_torch.to(next(decoder.parameters()).device)        
        pred = decoder(inputs_torch).cpu().numpy()

    n_F = ts_F.mean_.shape[0]
    pred_F = ts_F.inverse_transform(pred[:, :n_F]).T
    pred_T = ts_T.inverse_transform(pred[:, n_F:]).T

    return np.vstack((pred_F, pred_T))