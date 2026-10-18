# pre/_shiftscale.py
"""Shifting, scaling and spike removal.

Two notes on what the functions in here actually do, because the names
understate it.

``shift`` learns, when no shift vector is supplied, the mean of the last five
snapshots rather than the mean of the record. The main script computes its own
shift vector for the state fields and passes it in, so the learned default only
applies where nothing is passed; keeping the two definitions apart matters when
reading the pipeline.

``remove_spikes`` does more than remove spikes. It replaces every sample whose
robust z-score exceeds a threshold, and then applies a Savitzky-Golay filter to
the whole trajectory unconditionally. On the reduced derivatives this is a
second smoothing stage after the derivative estimate itself, and it acts
directly on the regression target. How much it changes is now reported, and on
the thermal case it is not small: about a fifth of the samples are replaced and
the final filter leaves roughly half of the amplitude. Its settings used to be
module constants and were therefore the only preprocessing on the target that
no hyperparameter search could reach. They are now read from the configuration,
with the previous constants as the defaults, so that a search can decide
whether the stage earns its place.
"""

__all__ = [
    "shift",
    "unshift",
    "scale",
    "remove_spikes",
]

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy import interpolate
from scipy.stats import median_abs_deviation

import opinf


# --- spike removal settings ---------------------------------------------------
# The defaults reproduce the previous behaviour exactly. Each one is taken from
# the matching attribute on Params when that attribute exists, so the stage can
# be searched without this module knowing anything about a search, and a
# configuration that never sets them behaves as before.
SPIKE_MEDIAN_KERNEL = 31     # baseline median filter, in samples
SPIKE_Z_THRESHOLD = 2.5      # robust z above which a sample is replaced
SPIKE_SAVGOL_WINDOW = 51     # final smoothing window, in samples
SPIKE_SAVGOL_ORDER = 2
SPIKE_FINAL_SMOOTHING = True  # apply the final Savitzky-Golay pass

# Argument name -> (attribute consulted on Params, module default).
_SPIKE_SETTINGS = {
    "z_threshold": ("spike_z_threshold", SPIKE_Z_THRESHOLD),
    "savgol_window": ("spike_savgol_window", SPIKE_SAVGOL_WINDOW),
    "savgol_order": ("spike_savgol_order", SPIKE_SAVGOL_ORDER),
    "median_kernel": ("spike_median_kernel", SPIKE_MEDIAN_KERNEL),
    "final_smoothing": ("spike_final_smoothing", SPIKE_FINAL_SMOOTHING),
}


def _setting(name, explicit):
    """Resolve one setting: the explicit argument, else Params, else default."""
    if explicit is not None:
        return explicit
    attribute, default = _SPIKE_SETTINGS[name]
    return getattr(opinf.parameters.Params(), attribute, default)


def _odd_at_most(window, n_samples, minimum=3):
    """Nearest odd window that both filters will accept.

    Both ``savgol_filter`` and ``medfilt`` require an odd window, and
    ``savgol_filter`` additionally requires it to be no longer than the signal.
    A search that draws a window violating either raises inside a trial, which
    a driver then records as a bad region of the search space rather than as
    the out-of-range draw it is. Coercing here keeps that from happening; the
    value actually used is printed.
    """
    window = int(window)
    if window % 2 == 0:
        window += 1
    limit = int(n_samples)
    if limit % 2 == 0:
        limit -= 1
    return max(minimum, min(window, limit))


def shift(states, shift_by=None):
    """Shift the columns of ``states`` by a vector.

    Parameters
    ----------
    states : (n, k) or (m, n, k) ndarray
        Snapshot matrix, one snapshot per column.
    shift_by : (n, 1) ndarray or None
        Vector to subtract. When None it is learned as the mean of the last
        five snapshots, which is a steady-state estimate and not the mean of
        the record.

    Returns
    -------
    states_shifted : ndarray
        ``states`` with the shift subtracted from every column.
    shift_by : (n, 1) ndarray
        The shift that was applied.
    """
    if states.ndim == 2:
        if shift_by is None:
            shift_by_ = np.mean(states[:, -5:], axis=1)[:, np.newaxis]
        else:
            shift_by_ = shift_by
        return states - shift_by_, shift_by_

    if states.ndim == 3:
        states_shifted = np.zeros_like(states)
        shift_by_ = shift_by
        for i in range(states.shape[0]):
            if shift_by is None:
                # Learned per trajectory; the returned value is then the shift
                # of the last one, which is what the two-dimensional branch
                # does for its single trajectory.
                shift_by_ = np.mean(states[i, :, -5:], axis=1)[:, np.newaxis]
            states_shifted[i] = states[i] - shift_by_
        return states_shifted, shift_by_

    raise ValueError(
        f"states must be two- or three-dimensional, got {states.ndim}"
    )


def unshift(shifted_states, shift_by):
    """Add the shift vector back to every column."""
    return shifted_states + shift_by


def scale(states, scale_to, scale_from=None):
    """Affinely map the entries of ``states`` from one interval to another.

    The bounds are learned over the whole matrix, not per row, so every state
    shares one scaling. Follows sklearn.preprocessing.MinMaxScaler.

    Parameters
    ----------
    states : (n, k) ndarray
        Snapshot matrix.
    scale_to : (2,) tuple
        Target minimum and maximum.
    scale_from : (2,) tuple or None
        Source minimum and maximum. Learned from the data when None.

    Returns
    -------
    states_scaled : (n, k) ndarray
    scale_to : (2,) tuple
        Only when the bounds were learned.
    scale_from : (2,) tuple
        Only when the bounds were learned.
    """
    learning = scale_from is None
    if learning:
        scale_from = np.min(states), np.max(states)

    if len(scale_to) != 2:
        raise ValueError("scale_to must have exactly 2 elements")
    if len(scale_from) != 2:
        raise ValueError("scale_from must have exactly 2 elements")

    mini, maxi = scale_to
    xmin, xmax = scale_from

    # A constant field has no range to scale from. Returning zeros is the
    # meaningful answer; dividing produces infinities that the caller then has
    # to clean up with nan_to_num, which hides the degenerate case instead of
    # handling it.
    span = xmax - xmin
    if not np.isfinite(span) or abs(span) < 1e-30:
        states_scaled = np.full_like(np.asarray(states, dtype=float),
                                     float(mini))
    else:
        scl = (maxi - mini) / span
        states_scaled = states * scl + (mini - xmin * scl)

    return (states_scaled, scale_to, scale_from) if learning else states_scaled


def remove_spikes(trajectories,
                  z_threshold=None,
                  savgol_window=None,
                  savgol_order=None,
                  median_kernel=None,
                  final_smoothing=None,
                  label="reduced derivatives"):
    """Replace outliers in each trajectory and smooth the result.

    The procedure is: a median filter supplies a baseline, samples whose
    robust z-score against that baseline exceeds ``z_threshold`` are replaced
    by linear interpolation from their neighbours, and a Savitzky-Golay filter
    is applied to the whole trajectory.

    The final filter runs on every sample, not only the replaced ones. Applied
    to reduced derivatives it is therefore a second smoothing stage after the
    derivative estimate, and it acts on the target of the regression. The
    fraction of replaced samples and the amplitude removed by the final filter
    are reported so the size of that effect is on record.

    Parameters
    ----------
    trajectories : (n, k) ndarray
        One trajectory per row.
    z_threshold, savgol_window, savgol_order, median_kernel, final_smoothing
        Override the configuration. None takes the value from Params when that
        attribute exists and the module default otherwise.
    label : str
        Name used in the report.
    """
    z_threshold = _setting("z_threshold", z_threshold)
    savgol_order = int(_setting("savgol_order", savgol_order))
    final_smoothing = bool(_setting("final_smoothing", final_smoothing))

    n_samples = trajectories.shape[1]
    savgol_window_req = _setting("savgol_window", savgol_window)
    median_kernel_req = _setting("median_kernel", median_kernel)

    # The Savitzky-Golay window has to exceed the polynomial order as well, or
    # the fit is underdetermined and scipy raises.
    savgol_window = _odd_at_most(savgol_window_req, n_samples,
                                 minimum=savgol_order + 1 + savgol_order % 2)
    median_kernel = _odd_at_most(median_kernel_req, n_samples)

    output = opinf.parameters.Params().output
    if output and (savgol_window != int(savgol_window_req)
                   or median_kernel != int(median_kernel_req)):
        print(f"  spike removal: windows adjusted to fit "
              f"{n_samples} samples and order {savgol_order}: "
              f"savgol {savgol_window_req} -> {savgol_window}, "
              f"median {median_kernel_req} -> {median_kernel}")

    cleaned = trajectories.copy()
    n_replaced_total = 0

    for i in range(trajectories.shape[0]):
        baseline = medfilt(trajectories[i, :], kernel_size=median_kernel)

        deviation = trajectories[i, :] - baseline
        mad = median_abs_deviation(deviation)
        if mad > 0:
            # 1.4826 puts the MAD on the scale of a standard deviation.
            robust_z = np.abs(deviation) / (1.4826 * mad)
        else:
            robust_z = np.zeros_like(deviation)

        spike_indices = np.where(robust_z > z_threshold)[0]
        n_replaced_total += int(spike_indices.size)

        if spike_indices.size > 0:
            valid = np.ones(trajectories.shape[1], dtype=bool)
            valid[spike_indices] = False
            x = np.arange(trajectories.shape[1])
            f = interpolate.interp1d(x[valid], trajectories[i, valid],
                                     kind='linear', fill_value='extrapolate')
            cleaned[i, spike_indices] = f(spike_indices)

        if final_smoothing:
            cleaned[i, :] = savgol_filter(cleaned[i, :],
                                          savgol_window, savgol_order)

    if output:
        n_total = trajectories.size
        frac = 100.0 * n_replaced_total / n_total if n_total else 0.0
        rms_in = float(np.sqrt(np.mean(trajectories ** 2)))
        rms_out = float(np.sqrt(np.mean(cleaned ** 2)))
        ratio = rms_out / rms_in if rms_in > 0 else float('nan')
        print(f"  spike removal on {label}: replaced {n_replaced_total} of "
              f"{n_total} samples ({frac:.2f}%), "
              f"z > {z_threshold}, median kernel {median_kernel}")
        if final_smoothing:
            print(f"    final Savitzky-Golay: window {savgol_window} samples, "
                  f"order {savgol_order}, amplitude retained {ratio:.3f}")
        else:
            print(f"    final smoothing off, amplitude retained {ratio:.3f}")

    return cleaned