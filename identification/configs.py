"""Configurations of the final models (Table C.1 of the paper).

Each configuration has four groups of settings:
  params           attributes of opinf.parameters.Params
  model_inference  module constants of opinf.training._model_inference
  nonparametric    module constants of opinf.models._nonparametric
  exp              module constants of opinf_methanation_exp.py
"""

_INPUTS_EXP = ("load_setpoint", "cooling_temperature", "inlet_temperature")

# Shared by all final models.
_COMMON = {
    "params": {
        "model_structure": "AHB", "basis": "POD", "completion": "linear",
        "ROM_loss": "states", "num_epochs": 2000,
        "use_filter": False, "ddt_postprocessing": False, "lag_max": 0,
        "regularization_H": 3.124e-05,
        "use_CNN": True, "save_results": False, "output": True,
    },
    "model_inference": {
        "MULTISTEP_HORIZONS": (1, 50, 200),
        "MULTISTEP_WEIGHTING": "trivial",
        "MULTISTEP_WARMUP_EPOCHS": 150,
        "FIELD_WEIGHTING": "equal",
        "ON_DIVERGENCE": "stop",
    },
    "nonparametric": {"WEIGHT_INIT_SCALE": 0.05},
    "exp": {"LAG_FIT_ON_DERIVATIVE": True},
}

_EXPERIMENT = {
    "params": {"data_dir": "data/exp_data", "split": "time",
               "hidden_states": True, "time_to_leave_out": 100,
               "input_names": _INPUTS_EXP,
               "CNN_learn_rate": 2.4e-4, "CNN_input_noise": 2.5e-4,
               "CNN_activation": "gelu", "CNN_loss_function": "mse"},
    "exp": {"REPAIR_FLATLINES_MIN_SAMPLES": 300,   # 5 min at 1 Hz
            "BAD_TEMPERATURE_NODES": (62, 66),
            "TEST_FORECAST_FROM": "segment"},
}


def _lr(eta, half_cycle):
    """Cyclic learning rate between eta/5 and eta."""
    return {"adam_lr": eta, "lr_schedule_base_lr": eta / 5,
            "lr_schedule_max_lr": eta, "lr_schedule_step_factor": half_cycle}


def _merge(*parts):
    out = {}
    for part in parts:
        for group, values in part.items():
            out.setdefault(group, {}).update(values)
    return out


CONFIGS = {
    # Thermal excitation: trial t56 of the second search.
    "case1": _merge(_COMMON, _EXPERIMENT, {
        "params": {"file_suffix": "_dyn_T_ramps_exp",
                   "energy_threshold_single": 0.9952, **_lr(0.02143, 894)},
        "nonparametric": {"A_MAX_RATE": 0.1127, "H_MAX_RATE": 0.155},
        "exp": {"SAVE_PREDICTIONS_TO": "predictions/case1",
                "PREDICTION_TAG": "case1"},
    }),
    # Multi-variable excitation: trial t59 of the first search.
    "case2": _merge(_COMMON, _EXPERIMENT, {
        "params": {"file_suffix": "_dyn_var_T_var_F_exp",
                   "energy_threshold_single": 0.9903,
                   "ROM_loss": "hybrid", "lambda_deriv": 0.4997,
                   "lambda_state": 0.5003, **_lr(0.01869, 746)},
        "nonparametric": {"A_MAX_RATE": 0.1175, "H_MAX_RATE": 0.08527},
        "exp": {"SAVE_PREDICTIONS_TO": "predictions/case2",
                "PREDICTION_TAG": "case2"},
    }),
    # Synthetic record: method and hyperparameters of Case 1, seven
    # trajectories for training, three held out.
    "synthetic": _merge(_COMMON, {
        "params": {"data_dir": "data/smooth_aprbs", "file_suffix": "_training",
                   "split": "condition", "hidden_states": False,
                   "time_to_leave_out": 0,
                   "input_names": ("load", "cooling_temperature"),
                   "model_structure": "AHBC",
                   "energy_threshold_single": 0.999, **_lr(0.02143, 894),
                   "CNN_learn_rate": 2.365e-4, "CNN_input_noise": 2.516e-4,
                   "CNN_activation": "silu", "CNN_loss_function": "smooth"},
        "nonparametric": {"A_MAX_RATE": 0.1127, "H_MAX_RATE": 0.155},
        "exp": {"REPAIR_FLATLINES_MIN_SAMPLES": None},
    }),
}

# Models of the control study: the synthetic configuration refitted on all
# ten trajectories and saved.
CONFIGS["synthetic_control"] = _merge(CONFIGS["synthetic"], {
    "params": {"train_ratio": 1.0, "save_results": True}})

# Seeds of the reported runs: three independent sets of ten initializations
# for the experimental cases (Table E.3), and the three models of the
# control ensemble.
SEEDS = {
    "case1": range(10, 40), "case2": range(10, 40),
    "synthetic": range(10, 20), "synthetic_control": (18, 10, 15),
}
