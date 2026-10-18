#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projection, reduced scaling and result persistence."""

__all__ = [
    "reduced_state",
    "full_state",
    "scaled_states",
    "unscaled_states",
    "initial_values_error",
    "save_results",
]

import datetime
import hashlib
import json
import os
import pickle

import numpy as np

# Attributes the pipeline writes back onto Params after a run. They are
# results, not configuration, and are kept out of params.json so that the
# fingerprint below identifies a configuration rather than a run.
RESULT_ATTRS = ("last_global_error", "last_ddt_residual", "ddt_residual")

import opinf.parameters

Params = opinf.parameters.Params()


# =============================================================================
# Reduced scaling
# =============================================================================

def scaled_states(states, derivatives=None, scaling_fac=None):
    """Divide a block of latent states by one scalar.

    The factor is the largest absolute value over the whole block, so every
    mode of a field shares it. Scaling each mode by its own maximum would give
    the noise-dominated trailing modes the same weight as the leading ones.

    Parameters
    ----------
    states : ndarray
        Latent states to scale.
    derivatives : ndarray, optional
        Scaled by the same factor when given.
    scaling_fac : float, optional
        Factor to apply. Learned from ``states`` when None.

    Returns
    -------
    scaling_fac : float
    scaled_states : ndarray
    scaled_derivatives : ndarray
        Only when ``derivatives`` was given.
    """
    if scaling_fac is None:
        scaling_fac = np.max(np.abs(states))
        if scaling_fac == 0:
            # A block that is identically zero has no scale; dividing by zero
            # would turn it into NaN and propagate through the whole model.
            scaling_fac = 1.0

    scaled = states / scaling_fac

    if derivatives is not None:
        return scaling_fac, scaled, derivatives / scaling_fac
    return scaling_fac, scaled


def unscaled_states(scaling_fac, scaled_states):
    """Multiply a block of latent states by its scaling factor."""
    return scaled_states * scaling_fac


# =============================================================================
# Projection
# =============================================================================

def reduced_state(states_full, basis):
    """Project full states onto a basis: ``basis.T @ states_full``.

    This is the encoder of a linear subspace method. It is not the encoder of
    an alternating-minimisation manifold, where the latent coordinates are
    defined implicitly by a nonlinear least-squares problem.
    """
    return basis.T @ states_full


def full_state(states_reduced, basis):
    """Expand latent states back to the full space: ``basis @ states_reduced``."""
    return basis @ states_reduced


def initial_values_error(y0_rom, y0_true, scaling_fac, shift_fac,
                         unscale=True):
    """Mean absolute and mean relative error of an initial condition.

    Parameters
    ----------
    y0_rom, y0_true : ndarray
        Predicted and true initial values.
    scaling_fac : float
        Reduced scaling factor, applied when ``unscale`` is True.
    shift_fac : ndarray
        Shift vector added back before comparing.
    unscale : bool
        Whether the prediction still carries the reduced scaling. The previous
        version read this from a ``Params.scaling`` attribute that does not
        exist, so any call raised AttributeError.
    """
    if unscale:
        y0_rom = y0_rom * scaling_fac

    y0_rom = y0_rom + shift_fac.flatten()

    mean_absolute_error = float(np.mean(np.abs(y0_rom - y0_true)))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(y0_rom - y0_true) / np.abs(y0_true)
    mean_relative_error = float(np.nanmean(np.where(np.isfinite(rel), rel,
                                                    np.nan)))

    return mean_absolute_error, mean_relative_error


# =============================================================================
# Persistence
# =============================================================================
def _jsonable(value):
    """Convert a value into something json.dumps accepts, without losing it.

    json.dumps takes np.float64 but not np.int64, np.bool_ or np.float32, and
    the repr fallback below turns those into strings such as "np.int64(1)".
    A table filled from the result then carries text where it needs numbers,
    which is a poor property for the file that is supposed to be the evidence
    behind it.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _params_as_dict(params, skip=RESULT_ATTRS):
    """Plain dictionary of the configuration, for a readable record.

    The pickled dataclass can only be read back with the class definition at
    hand; a JSON copy survives a refactor and can be diffed between runs.
    """
    out = {}
    for key in dir(params):
        if key.startswith('_') or key in skip:
            continue
        value = getattr(params, key, None)
        if callable(value):
            continue
        out[key] = _jsonable(value)
    return out


def _config_fingerprint(config):
    """Short hash of a configuration. Two runs configured alike share it.

    The result attributes are excluded before hashing; with them in, every run
    would fingerprint differently and the hash would identify nothing.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def save_results(model, Params, save_dir=None, decoder=None,
                 input_scaler=None, target_scaler_F=None, target_scaler_T=None,
                 cnn_pack=None, metrics=None, seed=None, **legacy):
    """Persist the learned operators, the configuration and the decoder.

    Parameters
    ----------
    model : dict
        Arrays to store in the npz archive.
    Params : dataclass
        Configuration of the run.
    save_dir : str, optional
        Target directory. A timestamped one under ``results`` when None.
    decoder : torch.nn.Module, optional
        Trained decoder. Its weights and constructor arguments are stored.
    input_scaler, target_scaler_F, target_scaler_T : optional
        Scalers belonging to the decoder. Without them the decoder cannot be
        applied to new latent trajectories after loading.
    cnn_pack : dict, optional
        The dictionary returned by ``train_global_cnn``. Supplying it fills in
        the decoder and all three scalers at once, which is what a reloadable
        model needs.
    **legacy
        Accepted and ignored, with a note. The main script passes
        ``input_scaler_F`` and ``input_scaler_T``, which this function never
        had; without this catch-all the call raises TypeError the first time
        saving is switched on.
    """
    if legacy:
        print(f"  save_results: ignoring unknown arguments "
              f"{sorted(legacy.keys())}")

    if cnn_pack is not None:
        decoder = decoder if decoder is not None else cnn_pack.get('decoder')
        input_scaler = (input_scaler if input_scaler is not None
                        else cnn_pack.get('input_scaler'))
        target_scaler_F = (target_scaler_F if target_scaler_F is not None
                           else cnn_pack.get('target_scaler_F'))
        target_scaler_T = (target_scaler_T if target_scaler_T is not None
                           else cnn_pack.get('target_scaler_T'))

    if save_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        save_dir = os.path.join("results", f"model_{stamp}")
    os.makedirs(save_dir, exist_ok=True)

    save_dict = {"basis_type": Params.basis}

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

    for name, scaler in (("input_scaler", input_scaler),
                         ("target_scaler_F", target_scaler_F),
                         ("target_scaler_T", target_scaler_T)):
        if scaler is not None:
            save_dict[name] = scaler

    with open(os.path.join(save_dir, "results.pkl"), "wb") as f:
        pickle.dump(save_dict, f)

    # Lists of operators become object arrays and then need allow_pickle on
    # load; converting them to numbered entries keeps the archive plain.
    flat_model = {}
    for key, value in model.items():
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                flat_model[f"{key}_{i}"] = np.asarray(item)
            flat_model[f"{key}_count"] = np.asarray(len(value))
        else:
            flat_model[key] = value
    np.savez_compressed(os.path.join(save_dir, "results.npz"), **flat_model)

    config = _params_as_dict(Params)
    fingerprint = _config_fingerprint(config)
    config["config_sha256"] = fingerprint
    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    # The numbers the configuration produced, in their own file and carrying
    # the same fingerprint. A manuscript entry can then cite one short string
    # that ties a reported value to the configuration that produced it, which
    # is what "no number without a frozen configuration" needs in practice.
    record = {
        "config_sha256": fingerprint,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": _jsonable(seed),
        "metrics": _jsonable(metrics or {}),
        "results_on_params": {
            name: _jsonable(getattr(Params, name))
            for name in RESULT_ATTRS if hasattr(Params, name)
        },
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)

    np.save(os.path.join(save_dir, "Params.npy"), Params)

    if getattr(Params, "output", False):
        print(f"  saved to {os.path.abspath(save_dir)}: results.npz, "
              f"results.pkl, params.json, metrics.json "
              f"(config {fingerprint})")

    return save_dir