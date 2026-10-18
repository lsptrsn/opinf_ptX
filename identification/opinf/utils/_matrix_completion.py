#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial completion of a partially observed axial profile.

Only the reactor outlet is instrumented for the CO2 flow, so the axial profile
between inlet and outlet has to be reconstructed from the two boundary values
and whatever prior is available. This module provides three priors, selected
through ``Params.completion``:

  * ``'linear'``          constant conversion per unit length,
  * ``'knowledge-based'`` conversion weighted by a local Arrhenius rate,
  * ``'none'``            no completion, boundaries only.

Every completed profile must reproduce the two measured boundary values
exactly. That property is the only hard constraint the reconstruction has, and
it is checked and reported rather than assumed: a completion that misses the
outlet silently biases the field the ROM is fitted to, and the reported flow
error is evaluated at exactly that node.

Note on what completion can and cannot add. For ``'linear'`` the profile is an
affine combination of the two boundary series. When the inlet flow is constant,
as in the temperature-ramp campaign, the centred snapshot matrix has rank one
up to sensor noise: the completed field carries no spatial information beyond
the outlet measurement. Only a time-varying weighting, such as the
Arrhenius-based one, puts genuine structure into the interior.
"""

import numpy as np

import opinf
Params = opinf.parameters.Params()  # call parameters from dataclass

# Activation energy over the gas constant, in K. 10000 K is about 83 kJ/mol,
# a typical value for methanation. The same constant appears in the Arrhenius
# lifting of the main script; keep the two in step if either is changed.
EA_OVER_R = 10000.0

# Lower bound on the temperature entering the rate expression, in K.
T_FLOOR = 273.15

# Tolerance for the boundary check, in the units of the completed field.
BOUNDARY_TOL = 1e-9


def _report_boundaries(F, F_in, F_out, label):
    """Verify that the completed profile reproduces both boundary values.

    A completion prior may only redistribute the drop between inlet and outlet;
    it may not move the endpoints. Deviations are reported rather than raised,
    so that a run is never silently invalidated but also never aborted at the
    end of a long pipeline.
    """
    err_in = float(np.max(np.abs(F[0, :] - F_in)))
    err_out = float(np.max(np.abs(F[-1, :] - F_out)))

    if max(err_in, err_out) > BOUNDARY_TOL and Params.output:
        print(f"   [completion] WARNING: {label} violates a boundary "
              f"condition: max |inlet error| {err_in:.4g}, "
              f"max |outlet error| {err_out:.4g}")
    elif Params.output:
        print(f"   [completion] {label}: boundary conditions hold "
              f"(max error {max(err_in, err_out):.2e})")


def _linear_profile(F_in, F_out, n_z):
    """Constant drop per unit length between the two boundary values."""
    F = np.zeros((n_z, F_in.size))
    for i in range(F_in.size):
        F[:, i] = np.linspace(F_in[i], F_out[i], n_z)
    return F


def _arrhenius_progress(T_profile, n_z):
    """Normalised reaction progress along the axis, from 0 at the inlet to 1
    at the outlet.

    The local rate is weighted by exp(-Ea/RT), so the flow drops fastest where
    the reactor is hottest. The progress variable is the cumulative integral of
    that rate, normalised to the unit interval.

    The integral is taken with the trapezoidal rule and starts at zero. A plain
    cumulative sum starts at the first sample instead, which puts the profile
    at a nonzero progress already at the inlet node and therefore violates the
    inlet boundary condition by the first rate value divided by the total.
    """
    rate = np.exp(-EA_OVER_R / np.maximum(T_profile, T_FLOOR))

    # Cumulative trapezoidal integral with a leading zero, so progress[0] = 0.
    increments = 0.5 * (rate[1:] + rate[:-1])
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))

    total = cumulative[-1]
    if total > 1e-12:
        return cumulative / total

    # Reactor too cold for the rate expression to carry information.
    return np.linspace(0.0, 1.0, n_z)


def matrix_completion_F(F_in, F_out, T):
    """Reconstruct the axial flow profile F(z) from its boundary values.

    Parameters
    ----------
    F_in : array
        Inlet flow over time, 1-D.
    F_out : array
        Outlet flow over time, 1-D.
    T : (n_z, n_t) array
        Axial temperature field, used by the Arrhenius prior only.

    Returns
    -------
    F : ndarray
        Completed flow field, shape (n_z, n_t), or (2, n_t) for
        ``Params.completion == 'none'``.
    """
    method = Params.completion
    n_z, n_t = T.shape

    F_in = np.ravel(np.asarray(F_in, dtype=float))
    F_out = np.ravel(np.asarray(F_out, dtype=float))

    if F_in.size != n_t or F_out.size != n_t:
        raise ValueError(
            f"boundary series and temperature field disagree: "
            f"F_in {F_in.size}, F_out {F_out.size}, T {T.shape}"
        )

    if method == 'none':
        # Boundaries only. Shape differs from the other branches by design.
        return np.vstack((F_in, F_out))

    if method == 'knowledge-based':
        F = np.zeros((n_z, n_t))
        for i in range(n_t):
            progress = _arrhenius_progress(T[:, i], n_z)
            # F falls from the inlet value by the full drop, distributed
            # according to the local reaction progress.
            F[:, i] = F_in[i] - (F_in[i] - F_out[i]) * progress
        _report_boundaries(F, F_in, F_out, "Arrhenius-weighted completion")
        return F

    if method != 'linear' and Params.output:
        print(f"   [completion] unknown method '{method}', using linear")

    F = _linear_profile(F_in, F_out, n_z)
    _report_boundaries(F, F_in, F_out, "linear completion")
    return F


def matrix_completion(X_z_0, X_z_end, T):
    """Reconstruct an axial conversion profile from its boundary values.

    Kept for the conversion field of the synthetic case, where an exponential
    saturation prior is also available. The main experimental path uses
    ``matrix_completion_F``.

    Parameters
    ----------
    X_z_0, X_z_end : array
        Conversion at the reactor inlet and outlet over time, 1-D.
    T : (n_z, n_t) array
        Axial temperature field.

    Returns
    -------
    X : ndarray
        Completed conversion field, shape (n_z, n_t), or (2, n_t) for
        ``Params.completion == 'none'``.
    """
    method = Params.completion
    n_z, n_t = T.shape

    X0 = np.ravel(np.asarray(X_z_0, dtype=float))
    Xend = np.ravel(np.asarray(X_z_end, dtype=float))

    if X0.size != n_t or Xend.size != n_t:
        raise ValueError(
            f"boundary series and temperature field disagree: "
            f"X_z_0 {X0.size}, X_z_end {Xend.size}, T {T.shape}"
        )

    if method == 'none':
        return np.vstack((X0, Xend))

    z = np.linspace(0.0, 1.0, n_z)

    if method == 'exp_saturation':
        # Saturating rise from inlet to outlet. The shape function is
        # normalised so that it reaches exactly one at z = 1; the unnormalised
        # form exp(-decay*z) leaves a residual of exp(-decay) at the outlet,
        # which for decay = 3 misses the measured outlet value by about 5 %.
        decay = 3.0
        shape = (1.0 - np.exp(-decay * z)) / (1.0 - np.exp(-decay))
        X = X0[None, :] + (Xend - X0)[None, :] * shape[:, None]

    elif method == 'knowledge-based':
        # Progress weighted by the axial temperature gradient. Same leading
        # zero as in the Arrhenius variant, so that the inlet value is met.
        X = np.zeros((n_z, n_t))
        for i in range(n_t):
            dT_dz = np.gradient(T[:, i], z)
            span = np.max(dT_dz) - np.min(dT_dz)
            grad_norm = (dT_dz - np.min(dT_dz)) / (span + 1e-12)

            increments = 0.5 * (grad_norm[1:] + grad_norm[:-1])
            cumulative = np.concatenate(([0.0], np.cumsum(increments)))
            total = cumulative[-1]
            progress = (cumulative / total if total > 1e-12
                        else np.linspace(0.0, 1.0, n_z))

            X[:, i] = X0[i] + (Xend[i] - X0[i]) * progress

    else:
        if method != 'linear' and Params.output:
            print(f"   [completion] unknown method '{method}', using linear")
        X = np.zeros((n_z, n_t))
        for i in range(n_t):
            X[:, i] = np.linspace(X0[i], Xend[i], n_z)

    _report_boundaries(X, X0, Xend, f"conversion completion ({method})")
    return X