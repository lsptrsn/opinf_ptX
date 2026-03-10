#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:19:23 2024

@author: peterson
"""

__all__ = [
            "sampled_data",
            "train_test_split_time",
            "train_test_split_conditions",
            "apply_smoothing",
            "apply_minmax"
          ]

import numpy as np
import opinf.parameters
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

Params = opinf.parameters.Params()  # call parameters from dataclass


def sampled_data(t, z, states, derivatives, entries):
    """
    Reduces the input data matrix by taking every xth value for the row
    (spatial coordinate) and every yth axis for the column (time coordinate).
    """
    start_value_t = 0
    end_value_t = int(1 * states.shape[-1])
    end_value_z = int(1 * states.shape[-2])
    z = z[0:end_value_z+2*Params.step_z_sampling:Params.step_z_sampling]
    t = t[start_value_t:end_value_t:Params.step_t_sampling]
    if len(states.shape) == 3:
        states = states[:, :end_value_z+2*Params.step_z_sampling:Params.step_z_sampling,
                        start_value_t:end_value_t:Params.step_t_sampling]
        derivatives = derivatives[:, :end_value_z:Params.step_z_sampling,
                                  start_value_t:end_value_t:Params.step_t_sampling]
        entries = entries[:, start_value_t:end_value_t:Params.step_t_sampling]
    else:
        states = states[:end_value_z+2*Params.step_z_sampling:Params.step_z_sampling,
                        start_value_t:end_value_t:Params.step_t_sampling]
        derivatives = derivatives[:end_value_z:Params.step_z_sampling,
                                  start_value_t:end_value_t:Params.step_t_sampling]
        entries = entries[:, start_value_t:end_value_t:Params.step_t_sampling]
    return t, z, states, derivatives, entries


def train_test_split_time(matrix, training_split):
    """
    Splits the input matrix into training and test sets based on the time axis.

    Parameters:
    - matrix (np.ndarray): Input array to be split.
    - training_split (float): Fraction of the data to use for training.

    Returns:
    - tuple: (train_set, test_set) split arrays.
    """
    # Determine the number of columns (for 2D) or elements (for 1D)
    num_cols = matrix.shape[-1]
    train_cols = int(num_cols * training_split)

    # Perform the split
    train_set = matrix[..., :train_cols]
    test_set = matrix[..., train_cols:]

    return train_set, test_set


def train_test_split_conditions(matrix, traj_starts, traj_ends, train_indices, test_indices):
    """
    Splits a horizontally stacked matrix based on PRE-CALCULATED trajectory boundaries and indices.

    Arguments:
    matrix -- The data matrix (n_features, n_time) or (n_time,)
    traj_starts -- Array of start indices for each trajectory
    traj_ends -- Array of end indices for each trajectory
    train_indices -- List/Array of trajectory IDs belonging to the Train set
    test_indices -- List/Array of trajectory IDs belonging to the Test set
    """

    is_1d = matrix.ndim == 1
    train_chunks = []
    test_chunks = []

    # Helper for stacking
    def robust_hstack(chunks, original_mat):
        if not chunks:
            if is_1d:
                return np.array([])
            else:
                return np.empty((original_mat.shape[0], 0))
        return np.hstack(chunks)

    # --- 1. Collect Training Chunks ---
    for idx in train_indices:
        start = traj_starts[idx]
        end = traj_ends[idx]

        # Slice based on dimension
        chunk = matrix[start:end] if is_1d else matrix[:, start:end]
        train_chunks.append(chunk)

    # --- 2. Collect Test Chunks ---
    for idx in test_indices:
        start = traj_starts[idx]
        end = traj_ends[idx]

        chunk = matrix[start:end] if is_1d else matrix[:, start:end]
        test_chunks.append(chunk)

    # --- 3. Re-assemble ---
    train_set = robust_hstack(train_chunks, matrix)
    test_set = robust_hstack(test_chunks, matrix)

    return train_set, test_set


def apply_smoothing(states, window_len, poly_ord):
    """
    Applies Median filtering (despiking) followed by Savitzky-Golay smoothing.
    Expects states shape: (n_features, n_time)
    """
    if states.ndim != 2:
        raise ValueError(f"States must be 2D (features, time), got {states.shape}")

    # 1. Despiking
    states_clean = median_filter(states, size=(1, 5))

    # 2. Smoothing
    states_smooth = savgol_filter(states_clean,
                                  window_length=window_len,
                                  polyorder=poly_ord,
                                  axis=1)
    return states_smooth


def apply_minmax(data, params, inverse=False):
    """
    Applies Min-Max scaling WITHOUT clipping.
    params is a tuple/list: (min_val, max_val).
    """
    if params is None:
        return data

    min_v, max_v = params[0].reshape(-1, 1), params[1].reshape(-1, 1)
    rng = max_v - min_v
    rng[rng == 0] = 1.0 # Protect against division by zero

    if inverse:
        return data * rng + min_v
    else:
        return (data - min_v) / rng
