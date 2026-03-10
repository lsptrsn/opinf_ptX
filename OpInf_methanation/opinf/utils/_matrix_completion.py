#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:54:59 2025

@author: luisa
"""
import numpy as np

import opinf
Params = opinf.parameters.Params()  # call parameters from dataclass

def matrix_completion_F(F_in, F_out, T):
    """
    Reconstructs the spatial flow profile F(z) based on boundary values and physics.

    Args:
        F_in (array): Vector of inlet flows over time (1D).
        F_out (array): Vector of outlet flows over time (1D).
        T (array): Matrix of Temperatures (space x time).

    Returns:
        F (array): Reconstructed Flow Matrix.
                   Shape (n_z, n_t) for 'arrhenius'/'linear'.
                   Shape (2, n_t) for 'none' (only boundaries).
    """
    method = Params.completion
    n_z, n_t = T.shape

    # Ensure inputs are flat vectors
    F_in = np.ravel(F_in)
    F_out = np.ravel(F_out)

    # 1. Linear Interpolation (Baseline)
    # Assumption: Constant conversion over length (dF/dz = const)
    if method == 'linear':
        F_linear = np.zeros((n_z, n_t))
        # np.linspace generates (n_z, n_t) if start/stop are vectors
        # Simple loop for readability and safety:
        for i in range(n_t):
            F_linear[:, i] = np.linspace(F_in[i], F_out[i], n_z)
        return F_linear

    # 2. Arrhenius-weighted (Physics-based)
    # Assumption: Reaction rate (and thus dF/dz) is proportional to exp(-Ea/RT).
    # Flow drops most significantly where temperature is high.
    elif method == 'knowledge-based':
        F_arrhenius = np.zeros((n_z, n_t))

        # Activation energy / R [K].
        # 10000K corresponds to approx. 83 kJ/mol, typical for methanation.
        Ea_R = 10000.0

        for i in range(n_t):
            T_profile = T[:, i]

            # Calculate local "reaction strength"
            # Clipping at 273K prevents division by zero or unrealistic values
            rate_profile = np.exp(-Ea_R / np.maximum(T_profile, 273.15))

            # Integrate rate -> Cumulative conversion progress
            cumulative_reaction = np.cumsum(rate_profile)

            # Normalize to 0..1 (Progress along the reactor)
            total_int = cumulative_reaction[-1]

            if total_int > 1e-12:
                progress_z = cumulative_reaction / total_int
            else:
                # If T is too low for reaction (cold reactor) -> Fallback to linear
                progress_z = np.linspace(0, 1, n_z)

            # Scale to actual boundary values:
            # F(z) starts at F_in and drops by (F_in - F_out) * progress
            drop = F_in[i] - F_out[i]
            F_arrhenius[:, i] = F_in[i] - drop * progress_z

        return F_arrhenius

    # 3. No Completion (Boundaries only)
    # Returns inlet and outlet rows stacked. Shape will be (2, n_t).
    elif method == 'none':
        return np.vstack((F_in, F_out))

    else:
        print(f"Warning: Unknown completion method '{method}'. Defaulting to linear.")
        F_linear = np.zeros((n_z, n_t))
        for i in range(n_t):
            F_linear[:, i] = np.linspace(F_in[i], F_out[i], n_z)
        return F_linear


def matrix_completion(X_z_0, X_z_end, T):
    # Flatten conversion values at z=0 and z=end to 1D arrays
    X0_series = np.ravel(X_z_0)   # Conversion at reactor inlet
    Xend_series = np.ravel(X_z_end)  # Conversion at reactor outlet

    # Reactor length (normalized 0..1) and array shapes
    n_z, n_t = T.shape
    z = np.linspace(0, 1, n_z)

    # --- Linear interpolation between inlet and outlet conversions ---
    X_linear = np.array([
        np.linspace(X0_series[i], Xend_series[i], n_z)
        for i in range(n_t)
    ]).T

    # --- Exponential saturation shape ---
    decay = 3.0  # Controls steepness of saturation curve
    X_exp = np.array([
        Xend_series[i] - (Xend_series[i] - X0_series[i]) * np.exp(-decay * z)
        for i in range(n_t)
    ]).T

    # --- Knowledge-based completion using temperature gradient ---
    X_temp = np.zeros_like(T)
    for i in range(n_t):
        T_profile = T[:, i]

        # Spatial gradient of temperature
        dT_dz = np.gradient(T_profile, z)

        # Normalize gradient to range 0..1
        grad_norm = (dT_dz - np.min(dT_dz)) / (np.max(dT_dz) - np.min(dT_dz) + 1e-12)

        # Integrate normalized gradient → monotonic profile
        cum_grad = np.cumsum(grad_norm)
        cum_grad_norm = cum_grad / (np.max(cum_grad) + 1e-12)

        # Scale to actual inlet → outlet conversion range
        X_temp[:, i] = X0_series[i] + (Xend_series[i] - X0_series[i]) * cum_grad_norm

    if Params.completion == 'linear':
        X_completed = X_linear
    elif Params.completion == 'exp_saturation':
        X_completed = X_exp
    elif Params.completion == 'knowledge-based':
        X_completed = X_temp
    else:
        X_completed = np.vstack((X_z_0, X_z_end))
    return X_completed
