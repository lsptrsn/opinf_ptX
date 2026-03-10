#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Post-Processing Module
Separates CNN Training and Inference.
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import opinf
device = opinf.utils.get_device()
Params = opinf.parameters.Params()

# --- 1. CLEAN RECONSTRUCTION (Linear / NL-POD only) ---
def reconstruct_solution(sol, sol_reduced, X_train_true, T_train_true,
                         V_reduced, V_reduced_nonlin, Xi, seed=None):
    """
    Standard reconstruction (Linear or NL-POD).
    """
    # NL-POD Correction
    if Params.basis in ['NL-POD', 'AM']:
        sol_linear = sol
        poly_update = np.concatenate(opinf.basis.polynomial_form(sol_reduced, p=3), axis=0)
        sol_correction_nonlin = V_reduced_nonlin @ Xi @ poly_update
        # Ensure shapes match
        if sol_correction_nonlin.shape[1] > sol.shape[1]:
             sol_correction_nonlin = sol_correction_nonlin[:, :sol.shape[1]]
        sol = sol_linear + sol_correction_nonlin[:sol.shape[0], :]

    return sol

# --- 2. GLOBAL CNN TRAINING ---
def train_global_cnn(sol_reduced_list, F_train_true, T_train_true, V_reduced, seed=None):
    """
    Trains the CNN decoder globally on the provided training trajectories.
    Expects lists of arrays for sol_reduced (latent) and F/T_true (physical).
    """
    print("\n--- Training Global CNN Decoder ---")
    current_seed = seed if seed is not None else 0

    r_F, r_T = Params.r_F, Params.r_T

    # 1. Stack Data (List -> Big Array)
    # Inputs (Latent): Transpose to (N_samples, Features)
    if isinstance(sol_reduced_list, list):
        X_input = np.vstack([s[:r_F, :].T for s in sol_reduced_list])
        T_input = np.vstack([s[r_F:r_F+r_T, :].T for s in sol_reduced_list])
    else:
        # Single array case (Time Split)
        X_input = sol_reduced_list[:r_F, :].T
        T_input = sol_reduced_list[r_F:r_F+r_T, :].T

    CNN_input = np.hstack((X_input, T_input))

    # Targets (Physical): Transpose to (N_samples, Features)
    if isinstance(F_train_true, list):
        F_true = np.hstack(F_train_true)
        T_true = np.hstack(T_train_true)
    else:
        F_true = F_train_true
        T_true = T_train_true

    # Fit Target Scalers (Standard Scaling for CNN)
    target_scaler_F = preprocessing.StandardScaler().fit(F_true.T)
    target_scaler_T = preprocessing.StandardScaler().fit(T_true.T)

    y_F = target_scaler_F.transform(F_true.T)
    y_T = target_scaler_T.transform(T_true.T)
    CNN_target = np.hstack((y_F, y_T))

    # 2. Train/Val Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        CNN_input, CNN_target, test_size=0.2, random_state=current_seed
    )

    # 3. Input Scaling & Noise
    rng = np.random.RandomState(current_seed)
    noise = Params.CNN_input_noise

    # Add noise to train only
    std_feat = X_tr.std(axis=0, keepdims=True)
    X_tr_noisy = X_tr + noise * std_feat * rng.randn(*X_tr.shape)

    input_scaler = preprocessing.StandardScaler().fit(X_tr_noisy)
    X_tr_final = input_scaler.transform(X_tr_noisy)
    X_val_final = input_scaler.transform(X_val)

    # 4. Train Decoder
    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_tr_final, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val_final, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=Params.CNN_batch_size, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=Params.CNN_batch_size, shuffle=False)

    # Dummy basis for dimension check inside train_decoder
    dummy_basis = np.zeros((F_true.shape[0] + T_true.shape[0], r_F + r_T))

    decoder, _, _ = opinf.post.train_decoder(
        train_dl=train_dl, val_dl=val_dl,
        basis=dummy_basis,
        latent_dim=r_F + r_T,
        seed=current_seed
    )
    decoder = decoder.float().to(device)

    return {
        'decoder': decoder,
        'input_scaler': input_scaler,
        'target_scaler_F': target_scaler_F,
        'target_scaler_T': target_scaler_T
    }

def apply_cnn_inference(sol_reduced, cnn_pack):
    """
    Applies the trained CNN to new reduced data.
    Returns Unscaled Physical State.
    """
    decoder = cnn_pack['decoder']
    input_scaler = cnn_pack['input_scaler']
    ts_F = cnn_pack['target_scaler_F']
    ts_T = cnn_pack['target_scaler_T']

    # Transpose (t, r)
    inputs = sol_reduced.T

    # Scale
    inputs_scaled = input_scaler.transform(inputs)
    inputs_torch = torch.tensor(inputs_scaled, dtype=torch.float32, device=device)

    # Predict
    decoder.eval()
    with torch.no_grad():
        pred = decoder(inputs_torch).cpu().numpy()

    # Unscale
    n_F = ts_F.mean_.shape[0]
    pred_F = ts_F.inverse_transform(pred[:, :n_F]).T
    pred_T = ts_T.inverse_transform(pred[:, n_F:]).T

    return np.vstack((pred_F, pred_T))
