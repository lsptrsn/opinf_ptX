"""Source package for optimal control utilities."""
from .casadi_utils import (
    CNNCallbackCached,
    create_cnn_casadi_function,
    reduced_to_full_casadi,
    reduced_to_full_numpy,
    evaluate_cnn_decoder_numpy,
)
from .simulation_utils import (
    forward_sim_reduced,
    compute_max_temperature,
    check_dynamics_residuals,
    run_diagnostic_checks,
)
# from .model_loader import load_model_data

from .plotting_utils import (
    plot_control_trajectory,
    plot_temperature_profile,
    plot_conversion_profile,
    plot_disturbance_profile,
    create_summary_plot
)

from .load_utils import(
    load_results,
    get_device,
    setup_results_dir
)

__all__ = [
    # CasADi utilities
    'CNNCallbackCached',
    'create_cnn_casadi_function',
    'reduced_to_full_casadi',
    'reduced_to_full_numpy',
    'evaluate_cnn_decoder_numpy',
    # Simulation utilities
    'forward_sim_reduced',
    'compute_max_temperature',
    'check_dynamics_residuals',
    'run_diagnostic_checks',
    # Model loading
    'load_model_data',
    'get_device',
    'setup_results_dir'
]
