# post/_errors.py
"""Accuracy and error evaluation.

Two groups of functions:

  * ``rel_fro``, ``field_errors`` and ``count_parameters`` are the lightweight
    metrics the pipeline reports. They allocate nothing beyond the residual
    and never plot, so they are safe to call inside a hyperparameter search.

  * ``run_postprocessing`` is the reporting path: it draws the figures and
    prints the per-field errors. It returns the same combined error as before,
    so existing callers are unaffected.

The combined error is the mean of the relative Frobenius errors of the two
fields. That definition is used everywhere, including as the objective of the
hyperparameter search, so it lives in one place.

The reference norms that normalise the two field errors are reported next to
them. A relative error is only comparable across two runs if its denominator
is the same; any change to the flow reconstruction or to the mass-balance
bound moves that denominator, and without it in the log the change is
invisible inside the ratio.

Two denominators are reported, and the difference between them is not a
detail. The headline error divides by the norm of the field itself, which for a
temperature field in kelvin is dominated by the mean operating condition: a
model that returned the constant mean field would already score well on it.
The second divides by the norm of the field after its temporal mean is
removed, so it measures how much of the variation the model actually
reproduces. On that scale 100 % is exactly the error of predicting the mean and
nothing else, which makes it the baseline the manuscript does not otherwise
have.

It is also the only one of the two that can be held against the projection
error of the basis, which is measured against the centred snapshot matrix. The
two live on different scales, and a temperature error below the projection
floor is the sign that they have been compared anyway.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la

from ..utils import (
    plot_PDE_dynamics_3D,
    plot_PDE_dynamics_2D,
    plot_compare_PDE_data,
    plot_1D_comparison,
)

import opinf
import opinf.parameters
Params = opinf.parameters.Params()

# Which combined error run_postprocessing returns, and with it main() and the
# objective of a hyperparameter search.
#   "combined"         : mean of the two relative errors about zero, the
#                        previous behaviour and the headline of the first
#                        submission.
#   "combined_centred" : the same residuals over the variation of each field
#                        about its segment mean, the metric every method
#                        decision of the revision was taken on.
RETURN_METRIC = "combined"


__all__ = [
    "frobenius_error",
    "lp_error",
    "rel_fro",
    "field_errors",
    "count_parameters",
    "physical_errors",
    "run_postprocessing",
]


# =============================================================================
# Core metrics
# =============================================================================

def _fro(a):
    """Frobenius norm for 2-D input, Euclidean norm for 1-D input."""
    a = np.asarray(a)
    return float(la.norm(a, ord="fro" if a.ndim == 2 else None))


def _rel_fro_with_denom(pred, true):
    """Relative Frobenius-norm error together with its denominator.

    Returns ``(rel_err, denom)``. The error is NaN when the reference is
    numerically zero, rather than raising or silently producing an enormous
    number. The denominator is returned in every case so that a caller can
    tell a small error from a large reference apart from a large error from a
    small one.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    denom = _fro(true)
    if denom < 1e-30:
        return float("nan"), denom
    return _fro(true - pred) / denom, denom


def _rel_fro_centred(pred, true):
    """Same residual, divided by the norm of the reference about its own mean.

    The temporal mean is taken along the last axis, so the reference becomes
    the variation the segment actually contains. The result is the error
    relative to the best constant predictor on that segment: at 100 % the model
    is worth exactly as much as returning the mean field, above it less.

    Returns ``(rel_err, denom)``, with NaN for a reference that does not vary.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    denom = _fro(true - true.mean(axis=-1, keepdims=True))
    if denom < 1e-30:
        return float("nan"), denom
    return _fro(true - pred) / denom, denom


def rel_fro(pred, true):
    """Relative Frobenius-norm error, as a fraction.

    Returns NaN when the reference is numerically zero, rather than raising or
    silently producing an enormous number.
    """
    return _rel_fro_with_denom(pred, true)[0]


def field_errors(sol, F_true, T_true, n_F=None, outlet_only=None):
    """Per-field and combined error for one segment.

    Parameters
    ----------
    sol : (n_F + n_T, k) ndarray
        Reconstructed full-order state, flow block stacked above temperature.
    F_true, T_true : ndarray
        Reference fields for the same segment.
    n_F : int, optional
        Height of the flow block. Defaults to ``F_true.shape[0]``.
    outlet_only : bool, optional
        Evaluate the flow error at the outlet node alone rather than over the
        whole reconstructed profile. Defaults to True for a time split, where
        only the outlet is measured, and False for a condition split, where
        the synthetic ground truth is available everywhere. This mirrors what
        ``run_postprocessing`` reports, so the two never disagree.

    Returns
    -------
    dict with keys ``F``, ``T``, ``combined``, ``denom_F``, ``denom_T`` and
    ``outlet_only``, plus ``F_centred``, ``T_centred``, ``combined_centred``,
    ``denom_F_centred`` and ``denom_T_centred``. The first three are unchanged
    in value and meaning, so nothing that reads them moves. The centred
    entries carry the same residuals over the norm of the reference about its
    own temporal mean; see the module docstring for why both are reported.
    """
    n_F = F_true.shape[0] if n_F is None else int(n_F)
    if outlet_only is None:
        outlet_only = (Params.split != "condition")

    F_pred = sol[:n_F, :]
    T_pred = sol[n_F:, :]

    if outlet_only:
        # Time last, so that the centred variant removes the temporal mean of
        # the outlet series rather than the mean over a single sample. The
        # Frobenius norm is the same either way, so the headline error is
        # unchanged by the orientation.
        f_true = F_true[-1, :].reshape(1, -1)
        f_pred = F_pred[-1, :].reshape(1, -1)
    else:
        f_true, f_pred = F_true, F_pred

    err_F, denom_F = _rel_fro_with_denom(f_pred, f_true)
    err_T, denom_T = _rel_fro_with_denom(T_pred, T_true)

    err_F_c, denom_F_c = _rel_fro_centred(f_pred, f_true)
    err_T_c, denom_T_c = _rel_fro_centred(T_pred, T_true)

    return {"F": err_F,
            "T": err_T,
            "combined": float(np.mean([err_F, err_T])),
            "denom_F": denom_F,
            "denom_T": denom_T,
            "F_centred": err_F_c,
            "T_centred": err_T_c,
            "combined_centred": float(np.mean([err_F_c, err_T_c])),
            "denom_F_centred": denom_F_c,
            "denom_T_centred": denom_T_c,
            "outlet_only": bool(outlet_only)}


def count_parameters(A=None, B=None, C=None, H=None, E_list=None, F_list=None):
    """Number of learned entries in the reduced model.

    Reported alongside the error so that a model of order r with L memory
    terms is not presented as being as compact as a memory-free model of the
    same order.
    """
    total = 0
    for op in (A, B, C, H):
        if op is not None:
            total += int(np.asarray(op).size)
    for lst in (E_list, F_list):
        for op in lst or []:
            if op is not None:
                total += int(np.asarray(op).size)
    return int(total)


# =============================================================================
# Legacy helpers, kept for compatibility
# =============================================================================

def _absolute_and_relative_error(Qtrue, Qapprox, norm):
    norm_of_data = norm(Qtrue)
    absolute_error = norm(Qtrue - Qapprox)
    return absolute_error, absolute_error / norm_of_data


def frobenius_error(Qtrue, Qapprox):
    """Absolute and relative Frobenius-norm error between two snapshot sets."""
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim != 2:
        raise ValueError("Qtrue and Qapprox must be two-dimensional")
    return _absolute_and_relative_error(
        Qtrue, Qapprox, lambda Z: la.norm(Z, ord="fro")
    )


def lp_error(t, Qtrue, Qapprox, p=2, normalize=False, plot=False):
    """Absolute and relative lp-norm error per snapshot.

    Plotting is off by default; it used to call ``plt.show()`` unconditionally,
    which blocks any non-interactive run.
    """
    if not np.isscalar(p) or p <= 0:
        raise ValueError("norm order p must be positive (np.inf ok)")
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim not in (1, 2):
        raise ValueError("Qtrue and Qapprox must be one- or two-dimensional")

    norm_of_data = la.norm(Qtrue, ord=p, axis=0)
    if normalize:
        norm_of_data = norm_of_data.max()
    absolute_error = la.norm(Qtrue - Qapprox, ord=p, axis=0)

    if plot and Params.output:
        plt.semilogy(t, absolute_error / norm_of_data)
        plt.title(r"Relative $\ell^{2}$ error over time")
        plt.xlabel("time in s")
        plt.ylabel(r"relative $\ell^{2}$ error")
        plt.show()

    return absolute_error, absolute_error / norm_of_data


def physical_errors(sol, F_true, T_true, z, n_F=None, outlet_only=None):
    """Errors in the units of the record, and the error of the hotspot.

    The relative errors are what the models are compared by. These are what an
    operator reads: the root-mean-square error of the outlet flow in the unit
    of the record, of the temperature field in kelvin, and of the hottest point
    of the profile, whose temperature and axial position the safety constraint
    of the control problem acts on.

    Parameters
    ----------
    sol : (n_F + n_T, k) ndarray
        Reconstructed state, flow block stacked above temperature.
    F_true, T_true : ndarray
        Reference fields for the same segment.
    z : (n_T,) ndarray
        Axial positions of the temperature nodes, in metres.
    n_F : int, optional
        Height of the flow block. Defaults to ``F_true.shape[0]``.
    outlet_only : bool, optional
        As in ``field_errors``.
    """
    n_F = F_true.shape[0] if n_F is None else int(n_F)
    if outlet_only is None:
        outlet_only = (Params.split != "condition")

    F_pred = np.asarray(sol[:n_F, :], dtype=float)
    T_pred = np.asarray(sol[n_F:, :], dtype=float)
    F_true = np.asarray(F_true, dtype=float)
    T_true = np.asarray(T_true, dtype=float)

    f_err = (F_pred[-1, :] - F_true[-1, :]) if outlet_only else (F_pred - F_true)

    # Hotspot: the hottest node of each profile, in prediction and reference
    # separately, so that a shifted front shows up as a position error rather
    # than being folded into the temperature error.
    cols = np.arange(T_true.shape[1])
    i_true = np.argmax(T_true, axis=0)
    i_pred = np.argmax(T_pred, axis=0)
    dT_hot = T_pred[i_pred, cols] - T_true[i_true, cols]
    z = np.asarray(z, dtype=float).reshape(-1)
    dz_hot = z[i_pred] - z[i_true]

    return {
        "F_rmse": float(np.sqrt(np.mean(f_err ** 2))),
        "T_rmse_K": float(np.sqrt(np.mean((T_pred - T_true) ** 2))),
        "hotspot_T_rmse_K": float(np.sqrt(np.mean(dT_hot ** 2))),
        "hotspot_T_maxabs_K": float(np.max(np.abs(dT_hot))),
        "hotspot_z_rmse_cm": float(100.0 * np.sqrt(np.mean(dz_hot ** 2))),
    }

# =============================================================================
# Reporting path
# =============================================================================

def run_postprocessing(sol, parameters, F_test, T_test, z_all, t,
                       r_F, r_T, F_all, T_all, k="",
                       draw_split=True, tiny=False, plotting=True,
                       return_details=False):
    """Evaluate one segment and, when output is enabled, draw the figures.

    Returns the combined error by default. Pass ``return_details=True`` to get
    the per-field breakdown as well; the combined value is identical either
    way, so the search and the tables cannot drift apart.
    """
    z = z_all[1:]
    n_F = F_test.shape[0]
    F_pred = sol[:n_F, :]
    T_pred = sol[n_F:, :]

    if Params.output and plotting:
        # --- outlet flow ---------------------------------------------------
        F_0 = F_all[0, :].reshape(1, -1)
        F_pred_merge = np.vstack((F_0, F_pred))

        if F_pred.shape == T_pred.shape:
            plot_compare_PDE_data(
                F_test, F_pred, z, t,
                f"$F_{{out, CO_2}}$ - OpInf ($r={r_F}$)",
                function_name=r"$F_{\mathrm{out}, \mathrm{CO}_2} \, / \, "
                              r"\mathrm{Ln\,min^{-1}}$",
            )

        plot_PDE_dynamics_2D(
            z_all, t, F_all, F_pred_merge,
            filename_suffix="F_out_CO2",
            val_label=r"$F_{\mathrm{out}, \mathrm{CO}_2} \, / \, "
                      r"\mathrm{Ln\,min^{-1}}$",
            train_ratio=Params.train_ratio,
            draw_split=draw_split, tiny=tiny,
        )
        plot_PDE_dynamics_3D(
            z_all, t, F_all, F_pred_merge,
            [r"$F_{\mathrm{out}, \mathrm{CO}_2}$ - true data",
             f"$F_{{\\mathrm{{out}}, \\mathrm{{CO}}_2}}$ - OpInf ($r={r_F}$)",
             "residual"],
            function_name=r"$F_{\mathrm{out}, \mathrm{CO}_2} \, / \, "
                          r"\mathrm{Ln\,min^{-1}}$",
        )
        plot_1D_comparison(
            t=t,
            y_true=F_test[-1, :].reshape(-1, 1),
            y_pred=F_pred[-1, :].reshape(-1, 1),
            title=r"experimental reference vs. OpInf prediction",
            ylabel=r"flow rate $F_{\mathrm{out}, \mathrm{CO}_2} \,/\, "
                   r"\mathrm{Ln\,min^{-1}}$",
            train_ratio=Params.train_ratio,
            draw_split=draw_split,
        )

        # --- temperature ---------------------------------------------------
        T_0 = T_all[0, :].reshape(1, -1)
        T_pred_merge = np.vstack((T_0, T_pred))

        if F_pred.shape == T_pred.shape:
            plot_compare_PDE_data(
                T_test, T_pred, z, t,
                f"temperature - OpInf ($r={r_T}$)",
                function_name=r"temperature $\, / \, \mathrm{K}$",
            )

        plot_PDE_dynamics_2D(
            z_all, t, T_all, T_pred_merge,
            filename_suffix="Temperature",
            val_label=r"$T \,/\, \mathrm{K}$",
            train_ratio=Params.train_ratio,
            draw_split=draw_split, tiny=tiny,
        )
        plot_PDE_dynamics_3D(
            z_all, t, T_all, T_pred_merge,
            [r"$T \,/\, \mathrm{K}$ - true data",
             rf"$T \,/\, \mathrm{{K}}$ - OpInf ($r={r_T}$)",
             "residual"],
            function_name=r"$T \,/\, \mathrm{K}$",
        )

    # --- errors -------------------------------------------------------------
    # Same definition as field_errors, so the reported numbers and the search
    # objective can never disagree.
    errors = field_errors(sol, F_test, T_test, n_F=n_F)

    if Params.output:
        label = "F_out CO2" if Params.split == "condition" \
            else "final F_out CO2"
        print(f"Relative Frobenius-norm error for {label}: "
              f"{errors['F']:.4%}   (reference norm {errors['denom_F']:.6e}, "
              f"{F_test.shape[1]} samples)")
        print(f"Relative Frobenius-norm error for Temperature: "
              f"{errors['T']:.4%}   (reference norm {errors['denom_T']:.6e}, "
              f"{T_test.shape[0]} nodes)")
        print(f"  about the segment mean: {label} "
              f"{errors['F_centred']:.4%}, Temperature "
              f"{errors['T_centred']:.4%}, combined "
              f"{errors['combined_centred']:.4%}   "
              f"(100% would be the constant mean field)")
        phys = physical_errors(sol, F_test, T_test, z, n_F=n_F)
        print(f"  in physical units: {label} RMSE {phys['F_rmse']:.4g} "
              f"(unit of the record), temperature RMSE "
              f"{phys['T_rmse_K']:.3g} K; hotspot temperature RMSE "
              f"{phys['hotspot_T_rmse_K']:.3g} K (largest "
              f"{phys['hotspot_T_maxabs_K']:.3g} K), position RMSE "
              f"{phys['hotspot_z_rmse_cm']:.3g} cm")

    if return_details:
        errors.update(physical_errors(sol, F_test, T_test, z, n_F=n_F))
        return errors
    return errors[RETURN_METRIC]