# post/_errors.py
"""Tools for accuracy and error evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la

from ..utils import (
    plot_PDE_dynamics_3D,
    plot_PDE_dynamics_2D,
    plot_compare_PDE_data,
    plot_1D_comparison
)

import opinf
Params = opinf.parameters.Params()  # call parameters from dataclass

__all__ = [
            "frobenius_error",
            "run_postprocessing",
            "lp_error"
          ]


def lp_error(t, Qtrue, Qapprox, p=2, normalize=False):
    """Compute the absolute and relative lp-norm errors between the snapshot
    sets Qtrue and Qapprox, where Qapprox approximates to Qtrue:

        absolute_error_j = ||Qtrue_j - Qapprox_j||_p,
        relative_error_j = ||Qtrue_j - Qapprox_j||_p / ||Qtrue_j||_p.

    Parameters
    ----------
    t: (n,) ndayarray
        An array corresponding to the time
    Qtrue : (n, k) or (n,) ndarray
        "True" data. Each column is one snapshot, i.e., Qtrue[:, j] is the data
        at some time t[j]. If one-dimensional, all of Qtrue is a single
        snapshot.
    Qapprox : (n, k) or (n,) ndarray
        An approximation to Qtrue, i.e., Qapprox[:, j] approximates Qtrue[:, j]
        and corresponds to some time t[j]. If one-dimensional, all of Qapprox
        is a single snapshot approximation.
    p : float
        Order of the lp norm (default p=2 is the Euclidean norm). Used as
        the `ord` argument for scipy.linalg.norm(); see options at
        docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html.
    normalize : bool
        If true, compute the normalized absolute error instead of the relative
        error, defined by

            normalized_absolute_error_j
                = ||Qtrue_j - Qapprox_j||_2 / max_k{||Qtrue_k||_2}.

    Returns
    -------
    abs_err : (k,) ndarray or float
        Absolute error of each pair of snapshots Qtrue[:, j] and Qapprox[:, j].
        If Qtrue and Qapprox are one-dimensional, Qtrue and Qapprox are treated
        as single snapshots, so the error is a float.
    rel_err : (k,) ndarray or float
        Relative or normed absolute error of each pair of snapshots Qtrue[:, j]
        and Qapprox[:, j]. If Qtrue and Qapprox are one-dimensional, Qtrue and
        Qapprox are treated as single snapshots, so the error is a float.
    """
    # Check p.
    if not np.isscalar(p) or p <= 0:
        raise ValueError("norm order p must be positive (np.inf ok)")

    # Check dimensions.
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim not in (1, 2):
        raise ValueError("Qtrue and Qapprox must be one- or two-dimensional")

    # Compute the error.
    norm_of_data = la.norm(Qtrue, ord=p, axis=0)
    if normalize:
        norm_of_data = norm_of_data.max()
    absolute_error = la.norm(Qtrue - Qapprox, ord=p, axis=0)

    plt.semilogy(t, absolute_error/ norm_of_data)
    plt.title(r"Relative $\ell^{2}$ error over time")
    plt.xlabel('time in s')
    plt.ylabel('relative $\ell^{2}$ error ')
    plt.show()
    return absolute_error, absolute_error / norm_of_data


def _absolute_and_relative_error(Qtrue, Qapprox, norm):
    """Compute the absolute and relative errors between Qtrue and Qapprox,
    where Qapprox approximates Qtrue:

        absolute_error = ||Qtrue - Qapprox||,
        relative_error = ||Qtrue - Qapprox|| / ||Qtrue||
                       = absolute_error / ||Qtrue||,

    with ||Q|| defined by norm(Q).
    """
    norm_of_data = norm(Qtrue)
    absolute_error = norm(Qtrue - Qapprox)
    return absolute_error, absolute_error / norm_of_data


def frobenius_error(Qtrue, Qapprox):
    """Compute the absolute and relative Frobenius-norm errors between the
    snapshot sets Qtrue and Qapprox, where Qapprox approximates Qtrue:

        absolute_error = ||Qtrue - Qapprox||_F,
        relative_error = ||Qtrue - Qapprox||_F / ||Qtrue||_F.

    Parameters
    ----------
    Qtrue : (n, k)
        "True" data. Each column is one snapshot, i.e., Qtrue[:, j] is the data
        at some time t[j].
    Qapprox : (n, k)
        An approximation to Qtrue, i.e., Qapprox[:, j] approximates Qtrue[:, j]
        and corresponds to some time t[j].

    Returns
    -------
    abs_err : float
        Absolute error ||Qtrue - Qapprox||_F.
    rel_err : float
        Relative error ||Qtrue - Qapprox||_F / ||Qtrue||_F.
    """
    # Check dimensions.
    if Qtrue.shape != Qapprox.shape:
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim != 2:
        raise ValueError("Qtrue and Qapprox must be two-dimensional")

    # Compute the errors.
    return _absolute_and_relative_error(Qtrue, Qapprox,
                                        lambda Z: la.norm(Z, ord="fro"))


def run_postprocessing(sol, parameters, F_test, T_test, z_all, t,
                       r_F, r_T, F_all, T_all, k='',
                       draw_split=True, tiny=False, plotting=True):
    z = z_all[1:]
    F_pred = sol[:F_test.shape[0], :]
    T_pred = sol[F_test.shape[0]:, :]
    F_pred_last = F_pred[-1, :].reshape(-1, 1)
    F_test_last = F_test[-1, :].reshape(-1, 1)

    if Params.output and plotting:
        ## --------------------------------------------------------
        ## 1. F_out CO2 Analysis
        ## --------------------------------------------------------
        F_0 = F_all[0, :].reshape(1, -1)
        F_pred_merge = np.vstack((F_0, F_pred))

        # A) comparison of trajectories
        if F_pred.shape == T_pred.shape:
            plot_compare_PDE_data(
                F_test, F_pred,  z, t,
                f"$F_{{out, CO_2}}$ - OpInf ($r={r_F}$)",
                function_name=r"$F_{\mathrm{out}, \mathrm{CO}_2} \, / \, \mathrm{Ln\,min^{-1}}$"
            )


        # B) 2D Dynamics Heatmap
        plot_PDE_dynamics_2D(z_all, t, F_all, F_pred_merge,
                             [r'$F_{\mathrm{out}, \mathrm{CO}_2}$ - true data',
                              f'$F_{{\mathrm{{out}}, \mathrm{{CO}}_2}}$ - OpInf ($r={{{r_F}}}$)',
                              'absolute error'],
                             train_ratio=Params.train_ratio,
                             draw_split=draw_split, tiny=tiny)

        # C) 3D Surface Plot
        plot_PDE_dynamics_3D(z_all, t, F_all, F_pred_merge,
                             [r'$F_{\mathrm{out}, \mathrm{CO}_2}$ - true data',
                              f'$F_{{\mathrm{{out}}, \mathrm{{CO}}_2}}$ - OpInf ($r={{{r_F}}}$)',
                              'residual'],
                             function_name=r"$F_{\mathrm{out}, \mathrm{CO}_2} \, / \, \mathrm{Ln\,min^{-1}}$")

        # D) 1D Plot for reactor outlet
        plot_1D_comparison(t=t,
                             y_true=F_test_last,
                             y_pred=F_pred_last,
                             title=r"model prediction vs. test data $\left( F_{\mathrm{out}, \mathrm{CO}_2} \right)$",
                             ylabel=r"$F_{\mathrm{out}, \mathrm{CO}_2} \, / \, \mathrm{Ln\,min^{-1}}$",
                             train_ratio=Params.train_ratio,
                             draw_split=draw_split)

        ## --------------------------------------------------------
        ## 2. Temperature Analysis
        ## --------------------------------------------------------
        T_0 = T_all[0, :].reshape(1, -1)
        T_pred_merge = np.vstack((T_0, T_pred))

        # A) Schnitt-Vergleich
        if F_pred.shape == T_pred.shape:
            plot_compare_PDE_data(
                T_test, T_pred,  z, t,
                f"$temperature - $OpInf ($r={r_T}$)",
                function_name=r"$temperature \, / \, \mathrm{K}$"
            )


        # B) 2D Dynamics Heatmap
        plot_PDE_dynamics_2D(
            z_all, t, T_all, T_pred_merge,
            [
                r"$T \,/\, \mathrm{K}$ — true data",
                fr"$T \,/\, \mathrm{{K}}$ — OpInf ($r={r_T}$)",
                "absolute error"
            ],
            train_ratio=Params.train_ratio,
            draw_split=draw_split, tiny=tiny
        )

        # C) 3D Surface Plot
        plot_PDE_dynamics_3D(
            z_all, t, T_all, T_pred_merge,
            [
                r"$T \,/\, \mathrm{K}$ — true data",
                fr"$T \,/\, \mathrm{{K}}$ — OpInf ($r={r_T}$)",
                "residual"
            ],
            function_name=r"$T \,/\, \mathrm{K}$"
        )

    ## --------------------------------------------------------
    ## 3. Error Calculation
    ## --------------------------------------------------------
    if Params.split == 'condition':
        abs_froerr_F, rel_froerr_F = frobenius_error(Qtrue=F_test,
                                                     Qapprox=F_pred)
        print(f"Relative Frobenius-norm error for F_out CO2: {rel_froerr_F:.4%}")
    else:
        abs_froerr_F, rel_froerr_F = frobenius_error(Qtrue=F_test_last,
                                                     Qapprox=F_pred_last)
        print(f"Relative Frobenius-norm error for final F_out CO2: {rel_froerr_F:.4%}")

    abs_froerr_T, rel_froerr_T = frobenius_error(Qtrue=T_test,
                                                 Qapprox=T_pred)
    print(f"Relative Frobenius-norm error for Temperature: {rel_froerr_T:.4%}")

    rel_froerr = (rel_froerr_T + rel_froerr_F) / 2
    return rel_froerr
