#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subsampling, splitting and elementwise preprocessing of snapshot matrices."""

__all__ = [
    "sampled_data",
    "train_test_split_time",
    "train_test_split_conditions",
    "apply_smoothing",
    "apply_minmax",
]

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

import opinf.parameters

Params = opinf.parameters.Params()  # call parameters from dataclass


def sampled_data(t, z, states, derivatives, entries):
    """Take every ``step_z_sampling``-th axial node and every
    ``step_t_sampling``-th time sample.

    The axial slice is applied identically to the axis vector, the states and
    the derivatives. The previous version gave the states a wider stop index
    than the derivatives, which is invisible while the step is one, because
    both stops then fall past the end of the array, and produces blocks with
    different row counts as soon as the axial step is increased.

    The caller must pass the unsampled time vector and inputs together with
    the unsampled states; passing an already-sampled pair subsamples it a
    second time.
    """
    step_z = Params.step_z_sampling
    step_t = Params.step_t_sampling

    n_z = states.shape[-2]
    n_t = states.shape[-1]

    if len(z) < n_z:
        raise ValueError(
            f"axis vector has {len(z)} nodes, states have {n_z} rows"
        )

    z_s = z[:n_z:step_z]
    t_s = t[:n_t:step_t]

    if states.ndim == 3:
        states_s = states[:, :n_z:step_z, :n_t:step_t]
        derivatives_s = derivatives[:, :n_z:step_z, :n_t:step_t]
    else:
        states_s = states[:n_z:step_z, :n_t:step_t]
        derivatives_s = derivatives[:n_z:step_z, :n_t:step_t]

    entries_s = entries[:, :n_t:step_t]

    return t_s, z_s, states_s, derivatives_s, entries_s


def train_test_split_time(matrix, training_split):
    """Split along the last axis at ``training_split`` of its length.

    Parameters
    ----------
    matrix : ndarray
        Array to split; the last axis is time.
    training_split : float
        Fraction of the samples used for training.

    Returns
    -------
    (train_set, test_set) : tuple of ndarray
    """
    num_cols = matrix.shape[-1]
    train_cols = int(num_cols * training_split)

    return matrix[..., :train_cols], matrix[..., train_cols:]


def train_test_split_conditions(matrix, traj_starts, traj_ends,
                                train_indices, test_indices):
    """Split a horizontally stacked matrix along precomputed trajectory bounds.

    Parameters
    ----------
    matrix : (n, k) or (k,) ndarray
        Data, trajectories concatenated along the last axis.
    traj_starts, traj_ends : sequence of int
        Start and end index of every trajectory.
    train_indices, test_indices : sequence of int
        Trajectory identifiers assigned to each set.
    """
    is_1d = matrix.ndim == 1

    def collect(indices):
        chunks = []
        for idx in indices:
            start, end = traj_starts[idx], traj_ends[idx]
            chunks.append(matrix[start:end] if is_1d
                          else matrix[:, start:end])
        if not chunks:
            return np.array([]) if is_1d else np.empty((matrix.shape[0], 0))
        return np.hstack(chunks)

    return collect(train_indices), collect(test_indices)


# Width of the median despiking kernel, in samples. Fixed at 5 because the
# manuscript quotes it; named here rather than buried in the call.
MEDIAN_KERNEL = 5


def apply_smoothing(states, window_len, poly_ord):
    """Median despiking followed by Savitzky-Golay smoothing along time.

    Expects ``states`` with shape (features, time).

    The window is coerced to something savgol_filter accepts: odd, longer than
    the polynomial order, and no longer than the record. The search draws
    filter widths up to 700, and a draw that violates any of the three used to
    raise inside the trial, where the driver records it as a bad region of the
    search space rather than as the out-of-range draw it is.
    """
    if states.ndim != 2:
        raise ValueError(
            f"states must be 2D (features, time), got {states.shape}"
        )

    n_t = states.shape[1]
    window = int(window_len)
    if window % 2 == 0:
        window += 1
    limit = n_t if n_t % 2 else n_t - 1
    window = max(int(poly_ord) + 1 + int(poly_ord) % 2, min(window, limit))

    if window != int(window_len) and Params.output:
        print(f"  smoothing: window {int(window_len)} -> {window} "
              f"(odd, above order {poly_ord}, within {n_t} samples)")

    states_clean = median_filter(states, size=(1, MEDIAN_KERNEL))
    return savgol_filter(states_clean, window_length=window,
                         polyorder=poly_ord, axis=1)


def apply_minmax(data, params, inverse=False):
    """Affine min-max mapping without clipping.

    Parameters
    ----------
    data : ndarray
    params : (2,) tuple or None
        Lower and upper bound, either scalars or per-row arrays. None passes
        the data through unchanged.
    inverse : bool
        Map back to the original range instead.
    """
    if params is None:
        return data

    min_v = np.asarray(params[0], dtype=float).reshape(-1, 1)
    max_v = np.asarray(params[1], dtype=float).reshape(-1, 1)

    rng = max_v - min_v
    rng[rng == 0] = 1.0  # a constant row has no range to map

    return data * rng + min_v if inverse else (data - min_v) / rng