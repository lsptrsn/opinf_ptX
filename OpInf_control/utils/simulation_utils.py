"""
Numerical simulation utilities for forward integration and diagnostics.
Includes robust integrators for stiff Reduced Order Models (ROMs).
"""
import numpy as np
import casadi as ca
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional

def forward_sim_reduced(
    y0: np.ndarray,
    time_arr: np.ndarray,
    U_profile: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    B: Optional[np.ndarray],
    C: Optional[np.ndarray],
    n_red: Optional[int] = None
) -> np.ndarray:
    """
    Simulate reduced dynamics forward in time using a robust implicit integrator (Radau).

    This replaces explicit Euler integration, which is often unstable for
    stiff quadratic ROMs (OpInf models).

    Args:
        y0: Initial reduced state vector (n_red,).
        time_arr: Array of time points for the simulation.
        U_profile: Input profile array of shape (n_inputs, Nt).
                   MUST match the time_arr grid.
        A, H, B, C: ROM Operator matrices (Linear, Quadratic, Input, Constant).
        n_red: Reduced state dimension (optional, derived from y0 if None).

    Returns:
        X_sim: Simulated state trajectory of shape (n_red, Nt).
    """
    # 1. Setup dimensions and flattening
    y0 = y0.flatten()
    if C is not None:
        C = C.flatten()

    if n_red is None:
        n_red = len(y0)

    t_start = time_arr[0]
    t_end = time_arr[-1]

    # 2. Input Interpolation
    # Create a function u(t) that interpolates the U_profile over time.
    # axis=1 ensures we interpolate along the time dimension.
    u_func = interp1d(time_arr, U_profile, axis=1,
                      kind='linear', fill_value="extrapolate")

    # 3. Define Dynamics Function (RHS) for solve_ivp
    def dynamics(t, x):
        # x shape: (n_red,)

        # Get control vector at time t
        u_t = u_func(t)

        # Ensure correct shape for matrix multiplication
        if u_t.ndim == 1:
            u_t = u_t.reshape(-1, 1)

        # --- Compute Terms ---

        # Linear Term: A * x
        dx = A @ x

        # Quadratic Term: H * (x ⊗ x)
        # np.kron efficiently computes the Kronecker product
        x_kron = np.kron(x, x)
        dx += H @ x_kron

        # Control Term: B * u
        if B is not None and B.size > 0:
            # Flatten result to ensure it adds correctly to dx (1D array)
            dx += (B @ u_t).flatten()

        # Constant Term: C
        if C is not None:
            dx += C

        return dx

    # 4. Integrate using 'Radau' (Implicit Runge-Kutta)
    # This method is A-stable and handles stiff chemical dynamics well.
    sol = solve_ivp(
        dynamics,
        (t_start, t_end),
        y0,
        t_eval=time_arr,
        method='Radau',
        rtol=1e-3,
        atol=1e-6
    )

    return sol.y


def compute_max_temperature(
    X_sim: np.ndarray,
    reduced_to_full_func: Callable,
) -> float:
    """
    Compute maximum temperature from a reduced state trajectory.

    Args:
        X_sim: Reduced state trajectory (n_red, n_time).
        reduced_to_full_func: Function that lifts reduced state to full physical state.
                              Expected to return [Conversion; Temperature].

    Returns:
        Maximum value found in the full state (assumed to be Temperature).
    """
    # Reconstruct full physical field [F; T]
    full_states = reduced_to_full_func(X_sim)

    # We assume Temperature values (300-1000 K) are larger than Conversion (0-1).
    # Thus, np.max global is sufficient to find the Hotspot.
    return float(np.max(full_states))


def check_dynamics_residuals(
    X_traj: np.ndarray,
    U_traj: np.ndarray,
    integrator: ca.Function,
    time_arr: np.ndarray,
    load_interp: Callable,
    inlet_temperature_interp: Callable,
) -> float:
    """
    Verify if a given trajectory satisfies the CasADi integrator dynamics.
    Used to check the quality of the initial guess.

    Args:
        X_traj: State trajectory (n_red, n_time).
        U_traj: Control trajectory (n_control,).
        integrator: CasADi integrator function used in the optimization.
        time_arr: Time grid.
        load_interp: Interpolator for disturbance (Load).
        inlet_temperature_interp: Interpolator for disturbance (T_in).

    Returns:
        Maximum absolute residual between X[k+1] and Integrator(X[k]).
    """
    n_time = len(time_arr)
    max_res = 0.0

    for k in range(n_time - 1):
        xk = X_traj[:, k]

        # Evaluate external parameters at current time step
        load_k = float(load_interp(time_arr[k]))
        T_inlet_k = float(inlet_temperature_interp(time_arr[k]))

        # Construct parameter vector p: [load, u_cool, T_in]
        p_eval = ca.DM([load_k, float(U_traj[k]), T_inlet_k])

        # Integrate forward one step using the Optimizer's integrator
        xf = integrator(x0=ca.DM(xk), p=p_eval)['xf']

        # Compare integrated result with the actual next point in trajectory
        res = np.max(np.abs(X_traj[:, k + 1] - np.array(xf).flatten()))
        max_res = max(max_res, float(res))

    return max_res


def run_diagnostic_checks(
    u_min: float,
    u_max: float,
    y0: np.ndarray,
    time_arr: np.ndarray,
    forward_sim_func: Callable,
    reduced_to_full_func: Callable,
    input_scale: float = 1.0,
) -> Tuple[float, float]:
    """
    Run open-loop diagnostic checks at control bounds.

    Simulates the system response for constant minimum and maximum cooling
    to verify physical plausibility (monotonicity) and stability.

    Args:
        u_min: Minimum scaled control value.
        u_max: Maximum scaled control value.
        y0: Initial reduced state.
        time_arr: Time grid.
        forward_sim_func: Wrapper function for forward simulation.
        reduced_to_full_func: Wrapper function for state reconstruction.
        input_scale: Scaling factor for printing physical control values.

    Returns:
        Tuple (T_max_at_u_min, T_max_at_u_max).
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC CHECKS (OPEN LOOP)")
    print("="*60)

    # Calculate physical values for display
    u_min_phys = u_min * input_scale
    u_max_phys = u_max * input_scale

    # 1. Simulation at u_min (Low Cooling -> Expected Hot)
    print(f"\nSimulating at u_min (Low Cooling):")
    print(f"  > Physical Input: {u_min_phys:.2f}")
    print(f"  > Scaled Input:   {u_min:.4f}")

    X_sim_min = forward_sim_func(u_min)
    T_max_min = compute_max_temperature(X_sim_min, reduced_to_full_func)

    # 2. Simulation at u_max (High Cooling -> Expected Cold)
    print(f"\nSimulating at u_max (High Cooling):")
    print(f"  > Physical Input: {u_max_phys:.2f}")
    print(f"  > Scaled Input:   {u_max:.4f}")

    X_sim_max = forward_sim_func(u_max)
    T_max_max = compute_max_temperature(X_sim_max, reduced_to_full_func)

    print(f"\nResults (Reactor Hotspot):")
    print(f"  T_max(u_min) = {T_max_min:.2f} K (Drift expected for unstable ROMs)")
    print(f"  T_max(u_max) = {T_max_max:.2f} K")

    # 3. Check Monotonicity
    # Note: Logic depends on definition of u.
    # Usually: u=CoolingTemp. Higher u -> Warmer Coolant -> Less Cooling -> Hotter Reactor.
    if T_max_max > T_max_min:
        print("  ✓ Trend OK: Higher coolant temp leads to higher reactor temp.")
    else:
        print("  ? Trend Check: Reactor is colder at higher u. (Check if u is Flow Rate?)")

    print("="*60 + "\n")

    return T_max_min, T_max_max
