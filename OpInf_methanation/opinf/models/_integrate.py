#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:28:13 2024

@author: peterson
"""
from dde_ivp import solve_ddeivp
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import numpy as np
import time

import opinf.training
Params = opinf.parameters.Params()


def integrate(t_span, y0, t, entries, A, B, C, H, E_list, F_list):
    """
    Integrate ROM with Markov and Non-Markov terms.
    """

    # 1. Create a SINGLE interpolator for ALL inputs
    # entries shape: (n_inputs, n_time_steps)
    # axis=1 tells scipy to interpolate along the time axis
    y0 = y0.flatten()
    u_func = interp1d(t, entries, axis=1,
                  bounds_error=False, fill_value="extrapolate")

    dt = t[1] - t[0]

    def model_quad_OpInf(t_now, x):
        # 2. Get all inputs at t_now
        # Result has shape (n_inputs,), we reshape to (n_inputs, 1) for matrix math
        u_val = u_func(t_now)
        if u_val.ndim == 1:
            u_val = u_val.reshape(-1, 1)

        # 3. Compute Dynamics (Generic B @ u)
        # dx/dt = Ax + H(x⊗x) + C + Bu
        # Note: solve_ivp expects return shape (n,), so we flatten result

        dx = A @ x + H @ np.kron(x, x) + C.flatten()

        if B is not None and B.size > 0:
            dx += (B @ u_val).flatten()

        return dx

    # Solve ODE
    # Note: This ignores E_list/F_list (Memory terms) as solve_ivp cannot handle delays.
    if Params.output: print("Starting integration...")
    tic = time.perf_counter() # Start timer

    sol_ROM = scipy.integrate.solve_ivp(
        model_quad_OpInf, t_span, y0, method='Radau',
        t_eval=t, rtol=1e-3, atol=1e-6
    )

    toc = time.perf_counter() # Stop timer
    if Params.output: print(f"Integration took {toc - tic:.4f} seconds")
    return sol_ROM.y


def integrate_dde(t_span, y0, t, entries, A, B, C, H, E_list, F_list, delays):
    """
    Integrates a nonlinear DDE system of the form:
        x'(t) = A x(t) + H(x⊗x) + C + B u(t)
              + Σ_l [E_l x(t-τ_l) + F_l u(t-τ_l)]

    Parameters
    ----------
    t_span : tuple (float, float)
        Integration interval [t0, tf].
    y0 : ndarray (n, m)
        History of the state for times <= 0.
        Axis 0 = state, axis 1 = support points.
    t : ndarray
        Output points of the solution (ascending order).
    entries : ndarray (p, len(t))
        Input signals u(t).
    A,B,C,H : ndarray
        Markov operators.
    E_list,F_list : list of ndarray
        Delay operators for state and input.
    delays : list of float
        Delays τ_l.

    Returns
    -------
    Y : ndarray (n, len(t))
        State matrix at times t.
    """
    dt = t[1] - t[0]
    p = entries.shape[0]

    # Ensure all matrices are numpy arrays
    A, B, C, H = map(np.array, (A, B, C, H))
    E_list = [np.array(E) for E in E_list]
    F_list = [np.array(F) for F in F_list]

    # Input interpolation
    u_interp = []
    for i in range(p):
        if np.allclose(entries[i], entries[i, 0]):
            val = float(entries[i, 0])
            u_interp.append(lambda t, v=val: v)
        else:
            u_interp.append(interp1d(
                t, entries[i], kind='linear', fill_value='extrapolate',
                assume_sorted=True
            ))

    def u_at(time):
        return np.array([f(time) for f in u_interp])

    # History function
    time_points = np.array([-d for d in reversed(delays)] + [0.0])
    if y0.shape[1] != len(time_points):
        raise ValueError(f"y0 must have {len(time_points)} time points, not {y0.shape[1]}")

    interp_hist = interp1d(
        time_points, y0, axis=1,
        kind='linear', fill_value='extrapolate', assume_sorted=True
    )
    history = lambda t: interp_hist(t)

    # Right-hand side of the DDE
    def rhs(time, Y):
        x = np.ravel(Y(time))           # current state
        u_curr = u_at(time)             # current input

        # Markov part
        quad = H @ np.kron(x, x)
        markov = A @ x + quad + C + B @ u_curr

        # Delayed terms
        nonmarkov = np.zeros_like(x)
        for l, tau in enumerate(delays):
            t_lag = time - tau
            x_tau = np.ravel(Y(t_lag))  # fetch from history if t_lag < t0
            u_tau = u_at(t_lag)
            nonmarkov += E_list[l] @ x_tau + F_list[l] @ u_tau

        return markov + nonmarkov

    # Discontinuities: times where delays hit the time grid
    disc_points = sorted(set([ti - tau for tau in delays for ti in t
                              if t_span[0] <= ti - tau <= t_span[1]]))

    if Params.output: print("Starting integration...")
    tic = time.perf_counter() # Start timer
    # Solver
    sol = solve_ddeivp(
        rhs, t_span, history,
        max_step=dt/2,  # smaller steps for stability
        method='RK23',
        discontinuities=disc_points
    )
    toc = time.perf_counter() # Stop timer

    if sol.status < 0:
        raise RuntimeError(f"DDE integration failed: {sol.message}")

    if Params.output: print(f"Integration took {toc - tic:.4f} seconds")

    # Solution at the requested points
    if sol.sol is not None:
        Y = np.array([sol.sol(ti) for ti in t]).T
    else:
        Y = sol.y
    return Y
