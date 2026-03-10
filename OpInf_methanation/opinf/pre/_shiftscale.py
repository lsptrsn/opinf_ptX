# pre/_shiftscale.py

__all__ = [
    "shift",
    "unshift",
    "scale",
    "remove_spikes"
]

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy import interpolate
from scipy.stats import median_abs_deviation


def shift(states, shift_by=None):
    """Shift the columns of `states` by a vector.

    Parameters
    ----------
    states : (n, k) ndarray
        Matrix of k snapshots. Each column is a single snapshot.
    shift_by : (n,) or (n, 1) ndarray
        Vector that is the same size as a single snapshot.

    Returns
    -------
    states_shifted : (n, k) ndarray
        Shifted state matrix, i.e.,
        states_shifted[:, j] = states[:, j] - shift_by for j = 0, ..., k-1.
    shift_by : (n,) ndarray
        Shift factor, returned only if shift_by=None.
        Since this is a one-dimensional array, it must be reshaped to be
        applied to a matrix (e.g., states_shifted + shift_by.reshape(-1, 1)).
    """
    # Check dimensions.
    if states.ndim == 2:
        # If shift_by factor is not provided, compute the steady state.
        learning = shift_by is None
        if learning:
            shift_by_ = np.mean(states[:, -5:], axis=1)[:, np.newaxis]
        else:
            shift_by_ = shift_by

        # Shift the columns by the steady state
        states_shifted = states - shift_by_
    # Check dimensions.
    if states.ndim == 3:
        # Shift the columns by the steady state
        states_shifted = np.zeros_like(states)
        for i in range(states.shape[0]):
            # If not shift_by factor is provided, compute the steady state.
            learning = shift_by is None
            if learning:
                shift_by_ = np.mean(states[i, :, -5:], axis=1)[:, np.newaxis]
            else:
                shift_by = shift_by
            states_shifted[i] = states[i] - shift_by_
    return states_shifted, shift_by_


def unshift(shifted_states, shift_by):
    """Shift the columns of `states` by a vector.

    Parameters
    ----------
    states_shifted : (n, k) ndarray
        Shifted state matrix, i.e.,
        states_shifted[:, j] = states[:, j] - shift_by for j = 0, ..., k-1.
    shift_by : (n,) ndarray
        Shift factor
    Returns
    -------
    unshifted_states : (n, k) ndarray
        Matrix of k snapshots. Each column is a single snapshot.
    """
    # Shift the columns by the steady state
    unshifted_states = shifted_states + shift_by
    return unshifted_states


def scale(states, scale_to, scale_from=None):
    """Scale the entries of the snapshot matrix `states` from the interval
    [scale_from[0], scale_from[1]] to [scale_to[0], scale_to[1]].
    Scaling algorithm follows sklearn.preprocessing.MinMaxScaler.

    Parameters
    ----------
    states : (n, k) ndarray
        Matrix of k snapshots to be scaled. Each column is a single snapshot.
    scale_to : (2,) tuple
        Desired minimum and maximum of the scaled data.
    scale_from : (2,) tuple
        Minimum and maximum of the snapshot data. If None, learn the scaling:
        scale_from[0] = min(states); scale_from[1] = max(states).

    Returns
    -------
    states_scaled : (n, k) ndarray
        Scaled snapshot matrix.
    scaled_to : (2,) tuple
        Bounds that the snapshot matrix was scaled to, i.e.,
        scaled_to[0] = min(states_scaled); scaled_to[1] = max(states_scaled).
        Only returned if scale_from = None.
    scaled_from : (2,) tuple
        Minimum and maximum of the snapshot data, i.e., the bounds that
        the data was scaled from. Only returned if scale_from = None.
    """
    # If no scale_from bounds are provided, learn them.
    learning = scale_from is None
    if learning:
        scale_from = np.min(states), np.max(states)

    # Check scales.
    if len(scale_to) != 2:
        raise ValueError("scale_to must have exactly 2 elements")
    if len(scale_from) != 2:
        raise ValueError("scale_from must have exactly 2 elements")

    # Do the scaling.
    mini, maxi = scale_to
    xmin, xmax = scale_from
    scl = (maxi - mini) / (xmax - xmin)
    states_scaled = states * scl + (mini - xmin * scl)

    return (states_scaled, scale_to, scale_from) if learning else states_scaled


def remove_spikes(trajectories, z_threshold=2.5, savgol_window=51, savgol_order=2):
    """
    Remove extreme spikes from trajectories using a combination of techniques

    Parameters:
    trajectories: 2D array of shape (n_trajectories, n_points)
    z_threshold: Threshold for spike detection (higher = more conservative)
    savgol_window: Window size for Savitzky-Golay filter (must be odd) - make bigger for smoothness
    savgol_order: Polynomial order for Savitzky-Golay filter

    Returns:
    Cleaned trajectories with spikes removed
    """
    cleaned_trajectories = trajectories.copy()

    for i in range(trajectories.shape[0]):
        # First apply a strong median filter to get a baseline
        baseline = medfilt(trajectories[i, :], kernel_size=31)

        # Calculate robust Z-scores using Median Absolute Deviation (MAD)
        deviation = trajectories[i, :] - baseline
        mad = median_abs_deviation(deviation)
        if mad > 0:  # Avoid division by zero
            robust_z_scores = np.abs(deviation) / (1.4826 * mad)  # 1.4826 makes MAD comparable to STD
        else:
            robust_z_scores = np.zeros_like(deviation)

        # Identify extreme spikes
        spike_indices = np.where(robust_z_scores > z_threshold)[0]

        if len(spike_indices) > 0:
            # Create a mask of valid (non-spike) points
            valid_mask = np.ones_like(trajectories[i, :], dtype=bool)
            valid_mask[spike_indices] = False

            # Interpolate over the spikes
            x = np.arange(trajectories.shape[1])
            f = interpolate.interp1d(x[valid_mask], trajectories[i, valid_mask],
                                    kind='linear', fill_value='extrapolate')

            # Replace spikes with interpolated values
            cleaned_trajectories[i, spike_indices] = f(spike_indices)

        # Apply Savitzky-Golay filter for final smoothing
        cleaned_trajectories[i, :] = savgol_filter(cleaned_trajectories[i, :],
                                                  savgol_window, savgol_order)

    return cleaned_trajectories
