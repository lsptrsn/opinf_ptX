#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward integration of the reduced model.

Two entry points: ``integrate`` for the memory-free case, ``integrate_dde``
when non-Markovian terms are present.

The history of a delay differential equation is a function of absolute time.
The previous version built its history interpolant on ``[-max(delay), 0]``
while the integration ran on the record's own time axis, which does not start
at zero once the leading samples are truncated. Every history lookup was then a
linear extrapolation well outside the support of the interpolant. The history
is now anchored on the same axis as the integration; see HISTORY_ANCHOR.
"""

from dde_ivp import solve_ddeivp
import scipy
from scipy.interpolate import interp1d
import numpy as np
import time as _time

import opinf.parameters
import opinf.training
Params = opinf.parameters.Params()

# Wall-clock budget for one delay integration, in seconds, or None for no
# limit. A fit whose spectrum carries fast modes makes RK23 take very small
# steps: the trajectory that costs three seconds for a well-fitted model has
# been measured at 638 s for one that is not. Inside a search that is a trial
# that never returns, and interactively it is an afternoon. The budget turns
# a runaway integration into a failure the caller already knows how to
# handle, at the price of a limit that depends on the machine. Raise it
# rather than remove it if a legitimate case ever hits it.
MAX_INTEGRATION_SECONDS = 120.0


  # History of the delay system before the first output time. "constant" holds
# the initial state, x(t) = x(t_0) for t < t_0, and integrates from t_0, as
# the memory-free model does, so both start from the same information.
# "data" places the first samples of the projected record in the history and
# starts after them, the previous behaviour; the model then receives measured
# states over the whole first delay window, which favours it by exactly that
# window in any comparison with a memory-free model.
HISTORY = "constant"

def integrate(t_span, y0, t, entries, A, B, C, H, E_list, F_list):
    """Integrate the memory-free reduced model.

    The delay operators are accepted for signature compatibility and ignored;
    an initial value solver cannot carry them. A model trained with memory
    terms must go through ``integrate_dde``.
    """
    y0 = y0.flatten()

    if (E_list or F_list) and Params.output:
        print("  Warning: delay operators present but the memory-free "
              "integrator was called; they are ignored.")

    # One interpolator for all input channels; axis 1 is time.
    u_func = interp1d(t, entries, axis=1,
                      bounds_error=False, fill_value="extrapolate")

    def model_quad_OpInf(t_now, x):
        u_val = u_func(t_now)
        if u_val.ndim == 1:
            u_val = u_val.reshape(-1, 1)

        # dx/dt = A x + H (x kron x) + C + B u
        dx = A @ x + H @ np.kron(x, x) + C.flatten()

        if B is not None and B.size > 0:
            dx += (B @ u_val).flatten()

        return dx

    if Params.output:
        print("Starting integration...")
    tic = _time.perf_counter()

    sol_ROM = scipy.integrate.solve_ivp(
        model_quad_OpInf, t_span, y0, method='Radau',
        t_eval=t, rtol=1e-3, atol=1e-6
    )

    toc = _time.perf_counter()
    if Params.output:
        print(f"Integration took {toc - tic:.4f} seconds")
    return sol_ROM.y


def integrate_dde(t_span, y0, t, entries, A, B, C, H, E_list, F_list, delays):
    """Integrate the reduced model with delay terms.

        x'(t) = A x(t) + H (x kron x) + C + B u(t)
                + sum_l [ E_l x(t - tau_l) + F_l u(t - tau_l) ]

    Parameters
    ----------
    t_span : (float, float)
        Requested integration interval. Under HISTORY_ANCHOR = "absolute" the
        start is moved to the last history point, and the samples before it are
        taken from the history itself.
    y0 : (n, L+1) ndarray
        States at the first L+1 output times, earliest column first. These are
        the history: the caller supplies the projected data there, which is the
        only state information available before the model starts running.
    t : (k,) ndarray
        Output times, ascending.
    entries : (p, k) ndarray
        Input trajectories on the same grid.
    A, B, C, H : ndarray
        Markov operators.
    E_list, F_list : list of ndarray
        Delay operators for state and input.
    delays : list of float
        Delays tau_l, ascending.

    Returns
    -------
    Y : (n, k) ndarray
        State at every requested output time.
    """
    dt = t[1] - t[0]
    p = entries.shape[0]

    A, B, H = map(np.array, (A, B, H))
    # Flattened, as the memory-free integrator already does. A constant term
    # of shape (n, 1) added to a vector of shape (n,) broadcasts to an (n, n)
    # matrix instead of raising, and the right-hand side then returns the wrong
    # shape for a reason nothing in the traceback points at.
    C = np.ravel(np.array(C))
    E_list = [np.array(E) for E in E_list]
    F_list = [np.array(F) for F in F_list]

    # Input interpolation, one callable per channel. A channel that is constant
    # to within floating point is replaced by its value, which avoids building
    # an interpolant whose slope is pure sensor noise.
    u_interp = []
    for i in range(p):
        if np.allclose(entries[i], entries[i, 0]):
            val = float(entries[i, 0])
            u_interp.append(lambda tq, v=val: v)
        else:
            u_interp.append(interp1d(
                t, entries[i], kind='linear', fill_value='extrapolate',
                assume_sorted=True
            ))

    def u_at(tq):
        return np.array([f(tq) for f in u_interp])

    # --- history ------------------------------------------------------------
    max_delay = float(max(delays))
    if HISTORY == "constant":
        # x(t) = x(t_0) on [t_0 - max_delay, t_0]; the solver starts at t_0.
        x0 = np.asarray(y0, dtype=float)[:, :1]
        hist_times = np.array([float(t[0]) - max_delay, float(t[0])])
        y_hist = np.hstack([x0, x0])
    else:
        # The columns of y0 are the states at the first n_hist output times.
        # Placing them there, rather than on [-max(delay), 0], is what makes
        # a history lookup an interpolation instead of an extrapolation: the
        # record's time axis does not start at zero. They must reach back
        # over the longest delay, which with spaced delays takes more than
        # len(delays) + 1 samples.
        n_hist = y0.shape[1]
        hist_times = np.asarray(t[:n_hist], dtype=float)
        covered = float(hist_times[-1] - hist_times[0]) if n_hist > 1 else 0.0
        if covered < max_delay * (1.0 - 1e-9):
            raise ValueError(
                f"y0 must cover the longest delay, {max_delay:g} s; its "
                f"{n_hist} time points cover {covered:g} s")
        y_hist = np.asarray(y0, dtype=float)
    t_start = float(hist_times[-1])
    t_span = (t_start, float(t_span[1]))

    interp_hist = interp1d(
        hist_times, y_hist, axis=1,
        kind='linear', fill_value='extrapolate', assume_sorted=True
    )

    def history(tq):
        return interp_hist(tq)

    # --- right-hand side ----------------------------------------------------
    deadline = (None if MAX_INTEGRATION_SECONDS is None
                else _time.perf_counter() + MAX_INTEGRATION_SECONDS)
    n_calls = [0]

    def rhs(tq, Y):
        n_calls[0] += 1
        # Checked every few hundred evaluations rather than every one; the
        # clock costs more than the arithmetic it guards.
        if deadline is not None and n_calls[0] % 512 == 0:
            if _time.perf_counter() > deadline:
                raise RuntimeError(
                    f"DDE integration exceeded {MAX_INTEGRATION_SECONDS:.0f} s "
                    f"after {n_calls[0]} right-hand side evaluations; the fit "
                    f"is almost certainly diverging rather than slow")
        x = np.ravel(Y(tq))
        u_curr = u_at(tq)

        quad = H @ np.kron(x, x)
        markov = A @ x + quad + C + B @ u_curr

        nonmarkov = np.zeros_like(x)
        for l, tau in enumerate(delays):
            t_lag = tq - tau
            x_tau = np.ravel(Y(t_lag))
            u_tau = u_at(t_lag)
            nonmarkov += E_list[l] @ x_tau + F_list[l] @ u_tau

        return markov + nonmarkov

    if Params.output:
        print("Starting integration...")
    tic = _time.perf_counter()

    # The delayed arguments cross the output grid at t - tau, which is where a
    # DDE solver should place steps. solve_ddeivp does not take a
    # discontinuity list: the keyword falls through **options into the
    # Runge-Kutta constructor, which discards it. The list was therefore built
    # and thrown away. max_step below is what actually bounds the step size.
    sol = solve_ddeivp(
        rhs, t_span, history,
        max_step=dt / 2,
        method='RK23',
    )

    toc = _time.perf_counter()

    if sol.status < 0:
        raise RuntimeError(f"DDE integration failed: {sol.message}")

    if Params.output:
        print(f"Integration took {toc - tic:.4f} seconds, "
              f"{n_calls[0]} right-hand side evaluations")

    # --- assemble the output ------------------------------------------------
    # Output times before the start of the integration are the history
    # itself; the remainder comes from the solver. The first n_hist - 1
    # columns are therefore projected data rather than simulated state,
    # which is three of several thousand samples.
    mask_solved = t >= t_start
    Y = np.empty((y0.shape[0], len(t)), dtype=float)

    for j, ti in enumerate(t):
        if not mask_solved[j]:
            Y[:, j] = np.ravel(history(ti))

    if sol.sol is not None:
        for j, ti in enumerate(t):
            if mask_solved[j]:
                Y[:, j] = np.ravel(sol.sol(ti))
    else:
        # Without a dense output the solver returns its own steps, which are
        # adaptive and do not line up with the requested output times. Taking
        # the first n columns of sol.y assumed they did, and silently returned
        # the state at the wrong instants whenever the step count happened to
        # be large enough for the slice to succeed. Interpolate instead, and
        # say so rather than hiding the fallback.
        if Params.output:
            print("  Warning: the solver returned no dense output; "
                  "interpolating its steps onto the output grid.")
        solved = np.asarray(sol.y)
        solved_t = np.asarray(sol.t, dtype=float)
        if solved_t.size < 2:
            raise RuntimeError(
                "DDE integration returned fewer than two steps and no dense "
                "output; nothing can be evaluated on the output grid."
            )
        Y[:, mask_solved] = interp1d(
            solved_t, solved, axis=1, kind='linear',
            bounds_error=False, fill_value='extrapolate',
            assume_sorted=True,
        )(t[mask_solved])
    return Y