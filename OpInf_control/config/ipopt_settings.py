"""IPOPT solver configuration."""
import os
from pathlib import Path
from typing import Dict, Any, Optional


def get_coinhsl_path() -> Optional[Path]:
    """Get path to COINHSL library.

    Returns:
        Path to libcoinhsl.so or None if not found
    """
    default_path = Path.home() / '.local/coinhsl-2024.05.15/lib/libcoinhsl.so'

    # Check environment variable
    env_path = os.environ.get('COINHSL_PATH')
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Check default location
    if default_path.exists():
        return default_path

    print("Warning: COINHSL library not found. Using default linear solver.")
    return None


def get_ipopt_options(
    hessian_approximation: str = 'limited-memory',
    linear_solver: str = 'mumps',
    max_iter: int = 1000,
    tolerance: float = 1e-3,
    print_level: int = 5,
    use_coinhsl: bool = True,
) -> Dict[str, Any]:
    """Get IPOPT solver options.

    Args:
        hessian_approximation: 'limited-memory' (fast) or 'exact' (accurate)
        linear_solver: 'mumps' (free), 'ma57', 'ma86', 'ma97' (requires HSL)
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance
        print_level: Output verbosity (0-12)
        use_coinhsl: Try to use HSL linear solvers if available

    Returns:
        Dictionary of IPOPT options
    """
    opts = {
        # === Termination criteria ===
        'ipopt.max_iter': max_iter,
        'ipopt.tol': tolerance,
        'ipopt.acceptable_tol': tolerance * 10,
        'ipopt.acceptable_iter': 15,
        'ipopt.max_wall_time': 6000.0,

        # === Initialization ===
        'ipopt.mu_init': 1e-3,
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.bound_mult_init_method': 'mu-based',
        'ipopt.warm_start_init_point': 'yes',

        # === Feasibility tolerances ===
        'ipopt.constr_viol_tol': 1e-5,
        'ipopt.acceptable_constr_viol_tol': 1e-4,

        # === Hessian approximation ===
        'ipopt.hessian_approximation': hessian_approximation,

        # === Scaling ===
        'ipopt.nlp_scaling_method': 'gradient-based',
        'ipopt.nlp_scaling_max_gradient': 100.0,

        # === Restoration phase ===
        'ipopt.expect_infeasible_problem': 'no',
        'ipopt.start_with_resto': 'no',

        # === Output ===
        'ipopt.print_level': print_level,
        'print_time': True,
    }

    # === Linear solver (HSL if available) ===
    if use_coinhsl:
        coinhsl_path = get_coinhsl_path()
        if coinhsl_path is not None:
            opts['ipopt.linear_solver'] = linear_solver
            opts['ipopt.hsllib'] = str(coinhsl_path)
            print(f"Using HSL linear solver: {linear_solver}")
        else:
            opts['ipopt.linear_solver'] = 'mumps'
            print("Using MUMPS linear solver (COINHSL not available)")
    else:
        opts['ipopt.linear_solver'] = 'mumps'

    return opts


def get_fast_ipopt_options() -> Dict[str, Any]:
    """Get IPOPT options optimized for speed.

    Recommended for initial testing and large problems.
    """
    return get_ipopt_options(
        hessian_approximation='limited-memory',
        linear_solver='ma57',  # Fastest HSL solver
        max_iter=2000,
        tolerance=1e-4,
        print_level=3,
    )


def get_accurate_ipopt_options() -> Dict[str, Any]:
    """Get IPOPT options optimized for accuracy.

    Recommended for final optimization runs.
    """
    return get_ipopt_options(
        hessian_approximation='exact',
        linear_solver='ma86',
        max_iter=5000,
        tolerance=1e-6,
        print_level=5,
    )
