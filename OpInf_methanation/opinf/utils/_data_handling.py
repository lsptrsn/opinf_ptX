#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:22:56 2024

@author: peterson
"""
__all__ = [
    "reduced_state",
    "full_state",
    "scaled_states",
    "unscaled_states",
    "initial_values_error",
    "save_results",
]

import datetime
import numpy as np
import os
import pickle

import opinf.parameters
import opinf.post
Params = opinf.parameters.Params()


def scaled_states(states, derivatives=None, scaling_fac=None):
    """
    This function scales the input states and their derivatives by the maximum
    absolute value of the states.

    Parameters:
    ----------
    states (numpy.ndarray): The input states to be scaled.
    states_derivative (numpy.ndarray, optional): The derivatives of the states
    to be scaled. Defaults to None.

    Returns:
    --------
    scaling_fac (numpy.ndarray): Factor used for scaling.
    scaled_states (numpy.ndarray) Returns scaled states and their derivatives.
    scaled_derivatives (numpy.ndarray) If states_derivative is None, returns
    the scaled derivatives.
    """
    if scaling_fac is None:
        # Compute the scaling factor as the maximum absolute value of the states
        scaling_fac = np.max(np.abs(states))

    # Scale the states
    scaled_states = states / scaling_fac

    if derivatives is not None:
        # If states_derivative is provided, scale it as well
        scaled_states_derivative = derivatives / scaling_fac
        return scaling_fac, scaled_states, scaled_states_derivative

    # If states_derivative is not provided, return only the scaled states
    return scaling_fac, scaled_states


def unscaled_states(scaling_fac, scaled_states):
    """
    This function unscales the input states by the maximum absolute value of
    the states (=scaling_fac).

    Parameters:
    ----------
    scaling_fac (numpy.ndarray): Factor used for scaling.
    scaled_states (numpy.ndarray) Scaled states.

    Returns:
    --------
    states (numpy.ndarray): Unscaled states.
    """
    states = scaled_states * scaling_fac
    return states


def reduced_state(states_full, basis):
    """
    Function to reduce the dimensionality of a set of states using a given
    basis.

    Parameters:
    ----------
    states_full (numpy.ndarray): The original states to be reduced. Each column
    should represent a state.
    basis (numpy.ndarray): The basis used for reduction. This should be a
    matrix where each column is a basis vector.

    Returns:
    --------
    states_reduced (numpy.ndarray): The reduced states. Each column represents
    a state in the reduced space.
    """
    # Perform the reduction by projecting the states onto the basis
    states_reduced = basis.T @ states_full

    return states_reduced


def full_state(states_reduced, basis):
    """
    Function to reduce the dimensionality of a set of states using a given
    basis.

    Parameters:
    ----------f
    states_reduced (numpy.ndarray): The reduced states. Each column represents
    a state in the reduced space.
    basis (numpy.ndarray): The basis used for reduction. This should be a
    matrix where each column is a basis vector.

    Returns:
    --------
    states_full (numpy.ndarray): The original states. Each column should
    represent a state.
    """
    # Perform the reduction by projecting the states onto the basis
    states_full = basis @ states_reduced

    return states_full


def initial_values_error(y0_rom, y0_true, scaling_fac, shift_fac):
    """
    Compute the mean absolute and mean relative errors for initial values.

    Parameters:
    - y0_rom (np.ndarray): Initial values predicted by the ROM.
    - y0_true (np.ndarray): True initial values.
    - scaling_fac (float): Scaling factor applied to the ROM predictions.
    - shift_fac (np.ndarray): Shift factor applied to the ROM predictions.

    Returns:
    - mean_absolute_error (np.ndarray): Absolute error between true and
    predicted initial values
    - mean_relative_error (np.ndarray): Relative  error between true and
    predicted initial values
    """

    if Params.scaling:
        # unscale data
        y0_rom = y0_rom * scaling_fac

    # unshift data
    y0_rom = y0_rom + shift_fac.flatten()

    mean_absolute_error = np.mean(np.abs(y0_rom - y0_true))
    #print("Mean absolute error for initial values:", mean_absolute_error)

    mean_relative_error = np.mean(np.abs(y0_rom - y0_true) / np.abs(y0_true))
    #print("Mean relative error for initial values:", mean_relative_error)
    return mean_absolute_error, mean_relative_error


def save_results(model, Params, save_dir=None, decoder=None,
                 input_scaler=None,
                 target_scaler_F=None, target_scaler_T=None):
    """
    Save model operators, parameters, scalers, and optional decoder.
    """
    # --- create timestamped save directory ---
    if save_dir is None:
        current_date = datetime.datetime.now().strftime("%Y%m%d%H%M")
        save_dir = os.path.join("results", f"model_{current_date}")
    os.makedirs(save_dir, exist_ok=True)

    # --- prepare dictionary for pickle ---
    save_dict = {}

    # store params information
    save_dict["basis_type"] = Params.basis  # "POD", "NL-POD", etc.

    # store decoder
    if decoder is not None:
        save_dict["decoder_state_dict"] = decoder.state_dict()
        save_dict["decoder_init_params"] = {
            "latent_dim": getattr(decoder, "latent_dim", None),
            "output_dim": getattr(decoder, "output_dim", None),
            "hidden_dims": getattr(decoder, "hidden_dims", None),
            "conv_channels": getattr(decoder, "conv_channels", None),
            "kernel_sizes": getattr(decoder, "kernel_sizes", None),
            "strides": getattr(decoder, "strides", None),
            "paddings": getattr(decoder, "paddings", None),
            "dropout_linear": getattr(decoder, "dropout_linear", None),
            "dropout_conv": getattr(decoder, "dropout_conv", None),
            "norm_type": getattr(decoder, "norm_type", None),
            "activation_fn": getattr(decoder, "activation_fn_name", None),
        }

    # store scalers
    if input_scaler is not None:
        save_dict["input_scaler"] = input_scaler
    if target_scaler_F is not None:
        save_dict["target_scaler_F"] = target_scaler_F
    if target_scaler_T is not None:
        save_dict["target_scaler_T"] = target_scaler_T

    # --- save pickle (contains scalers, decoder, metadata) ---
    with open(os.path.join(save_dir, "results.pkl"), "wb") as f:
        pickle.dump(save_dict, f)

    # --- save npz (large numpy arrays from model) ---
    np.savez_compressed(os.path.join(save_dir, "results.npz"), **model)

    # --- save Params (as numpy for reproducibility) ---
    np.save(os.path.join(save_dir, "Params.npy"), Params)

    return save_dir
