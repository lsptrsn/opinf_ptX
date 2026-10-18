# -*- coding: utf-8 -*-
"""Hyperparameter search for the OpInf methanation pipeline.

One script for all three cases and both stages. What to run is set in the
configuration block below; nothing else needs editing.

Two stages
----------
``STAGE = "model"``    searches the preprocessing, basis and learning
                      parameters with the decoder switched off.
``STAGE = "decoder"``  loads the winner of the model stage, freezes every one
                      of its parameters, and searches only the decoder.

Running them separately is what makes the second stage cheap and the first one
interpretable: with the decoder on, its own capacity masks differences in the
reduced model, and every trial pays for a thousand decoder epochs.

Design notes
------------
* Every trial runs in its own interpreter. A leak inside a trial then cannot
  accumulate across a study, the timeout is enforced by killing a process
  rather than by a signal an IDE kernel may swallow, and a crash in one trial
  does not take the study with it.

* The objective is measured on a window the search carves out for itself, not
  on the segment that is later reported. Selecting over a hundred trials on
  the reported held-out error turns that error into a selection statistic.
  The reported number comes from one final run of the chosen configuration
  with holdout_ratio = 0, trained on the full training segment.

* Anything a case needs that is not searched has to be stated in its ``fixed``
  dictionary, or in MODULE_SETTINGS for the constants that live in the
  modules rather than in parameters.py. Leaving a parameter out of the search
  space does not neutralise it; it keeps whatever stands in the files, which
  is the configuration of some earlier experiment.

* Each trial is a small multistart: it trains on every seed in TRIAL_SEEDS
  and scores the seed whose Markovian rollout on the training data is best,
  which is exactly how the final model is chosen. The seed scatter of one
  configuration (13 to 44 % on the thermal case) is larger than most effects
  of the parameters, so a single seed hands the sampler mostly noise.

* A failed trial is recorded with its exception type and counted, and the
  counts are printed at the end. A systematic failure has to look like a bug,
  not like a region of bad hyperparameters.

Run it from a terminal, in the directory that holds the data folders:

    cd /path/to/OpInf_methanation
    python opinf_methanation_optuna.py

Several processes may run the same study at once, one per terminal; they
share the SQLite database and each writes its own trial log.
"""
import glob
import shutil
import copy
import gc
import importlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter

import numpy as np
import optuna
import optuna.visualization as ov

import opinf
from opinf_methanation_exp import main as methanation_main

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# CONFIGURATION
# =============================================================================

STAGE = "model"        # "model" | "decoder"
REGIME = "gradient"    # "gradient" | "pinn"   (model stage only)

SEED = 1
CASE = os.environ.get("OPINF_CASE", "exp1")     # "exp1" | "exp2" | "synthetic"
N_TRIALS = int(os.environ.get("OPINF_N_TRIALS", "16"))   # per process
STARTUP_TRIALS = 15
# Per run of the pipeline, that is per seed of a trial. A run with 2000
# epochs of the multi-step loss takes about 75 min on the full training
# segment; the search trains on less.
TRIAL_TIMEOUT_SEC = 100 * 60
TOTAL_TIMEOUT_SEC = None

# Known good points the search starts from: trial #72 of the first search
# (r_T = 3), and the best configuration of the H-bound study (r_T = 4, H
# bound 0.15).
EXTRA_STARTS = [
    dict(energy_threshold=0.9867, A_MAX_RATE=0.07366, H_MAX_RATE=0.05,
         adam_lr=0.02297, lr_schedule_step_factor=552,
         regularization_H=3.208e-05),
    dict(energy_threshold=0.9932, A_MAX_RATE=0.07366, H_MAX_RATE=0.15,
         adam_lr=0.02297, lr_schedule_step_factor=552,
         regularization_H=3.208e-05),
]

BASE_DIR = "optuna_studies"
RUN_TRIALS_IN_SUBPROCESS = True

# Seeds every trial is evaluated on, and how those results become the single
# number the sampler sees.
#   "select" : the run whose Markovian rollout error at SELECT_HORIZON on the
#              training data is lowest, the multistart rule of the final
#              model. A seed that fails is skipped; the trial fails only if
#              every seed does.
#   "median", "mean", "max" : over the seeds; any failed seed fails the trial.
TRIAL_SEEDS = tuple(range(int(os.environ.get("OPINF_N_SEEDS", "3"))))
TRIAL_SEED_AGGREGATE = os.environ.get("OPINF_AGGREGATE", "median")
# How the seeds of a trial are chosen.
#   "fixed"     : TRIAL_SEEDS in every trial.
#   "per_trial" : fresh seeds in every trial, TRIAL_SEED_BASE + n * k + i.
# Fixed seeds make trials comparable but let the search tune the
# hyperparameters to those particular initialisations: the best exp2
# configuration of the first search scored 16 % on seeds 0-2 and a median of
# 59 % on five fresh seeds.
TRIAL_SEED_MODE = os.environ.get("OPINF_SEED_MODE", "fixed")
TRIAL_SEED_BASE = 1000
# For TRIAL_SEED_AGGREGATE = "ensemble": the forecasts of the ENSEMBLE_TOP_K
# seeds with the lowest training rollout error are averaged, and the error of
# that average is the objective, which is the estimator the final model uses.
ENSEMBLE_TOP_K = 3
SELECT_HORIZON = 200

# A trial whose first seed is far worse than the median of the completed
# trials is not completed: with the seed spread of v8, where most
# initializations lie within a few points of each other, the remaining seeds
# cannot rescue it, and the budget is better spent elsewhere. The trial is
# scored on the seed that ran and marked. None switches this off.
EARLY_ABORT_FACTOR = 1.5
EARLY_ABORT_MIN_TRIALS = 10

# Score for a trial that cannot be evaluated, and ceiling for one that can. A
# failure used to return infinity, which the sampler cannot place next to
# finite values; the ceiling orders it last without distorting the scale.
OBJECTIVE_CEILING = 10.0

# Fraction of the record the search never sees. The reported test error is
# computed on it afterwards, by a final run with holdout_ratio = 0.
HO_HOLDOUT = 0.30
# Split of what remains, into the part the model is fitted on and the part the
# objective is measured on. A larger value gives the search a fitting set of
# nearly the size the final run uses, at the cost of a shorter scoring window.
HO_TRAIN_RATIO = 0.75

# Fixed values that win over the search. A name listed here is removed from the
# search space and set to this value in every trial. Use it to freeze a
# decision that has already been made, or to test one setting in isolation.
OVERRIDES = {
    # "lag_max": 0,
}

# Narrow the numeric ranges around the best trial of an existing study, for a
# second, finer pass. Give the study name; None for a full-range search.
REFINE_FROM = None
REFINE_SPAN = 0.35      # kept fraction of each range, centred on the best
REFINE_FIX_CATEGORICAL = True


# =============================================================================
# MODULE CONSTANTS
# =============================================================================
# Settings that live in the modules rather than in parameters.py. Every trial
# sets all of them, so no trial inherits a value from the file or, in process,
# from the trial before. Same values as the BASE block of run_queue.py; a
# searched parameter of the same name overrides its entry here.
#
# Decided on the night of 24/25 September (v8): state loss only, 2000 epochs,
# fields weighted equally. The last one is neutral at r_T = 3 and keeps the
# share of the flow field in the loss independent of the rank, which the
# search varies through the energy threshold.
MODULE_SETTINGS = {
    "opinf.training._model_inference": {
        "MULTISTEP_HORIZONS": (1, 50, 200),
        "MULTISTEP_WEIGHTING": "trivial",
        "MULTISTEP_WARMUP_EPOCHS": 150,
        "ON_DIVERGENCE": "stop",
        "FIELD_WEIGHTING": "equal",
    },
    "opinf.models._nonparametric": {
        "WEIGHT_INIT_SCALE": 0.05,
        "A_MAX_RATE": 0.05,
        "H_MAX_RATE": 0.05,
    },
    # The objective is the combined error about the segment mean, the metric
    # every method decision of the revision was taken on. The default of
    # run_postprocessing is the error about zero, in which the temperature
    # term is dominated by the mean operating temperature and the ranking is
    # driven almost entirely by the flow.
    "opinf.post._errors": {
        "RETURN_METRIC": "combined_centred",
    },
    "opinf_methanation_exp": {
        "RUN_DIAGNOSTICS_B": False,
        "REPORT_PROJECTION_FLOOR": False,
        "REPORT_STATIC_F": False,
        "REPAIR_FLATLINES_MIN_SAMPLES": 300,
        "LAG_FIT_ON_DERIVATIVE": True,
        "LAG_STEP": 1,
        "LAG_OPERATOR_STRUCTURE": "stable",
        "TEST_FORECAST_FROM": "segment",
    },
}


# =============================================================================
# CASES
# =============================================================================

_EXP_FIXED = dict(
    model_structure="AHB",
    evaluate_train_cases=False,
    # Stated rather than inherited. A parameter that is neither fixed nor
    # searched keeps whatever stands in parameters.py, which is the
    # configuration of one particular case.
    basis="POD",
    ddt_method="savgol",
    ddt_postprocessing=False,
    use_filter=False,
    state_shifting="mean",
    # Load from the flow controller setpoints, temperatures measured.
    input_names=["load_setpoint", "cooling_temperature", "inlet_temperature"],
    # The optimum of the learning rate depends on the training length, so the
    # length is fixed and the rate and its schedule are searched at it.
    num_epochs=2000,
    # Memory terms are evaluated in a dedicated study instead: none of four
    # variants improved on the Markovian model, and two made the delay system
    # unstable.
    lag_max=0,
    ROM_loss="states",
    completion="linear",    
)

_EXP_ENABLED = [
    "energy_threshold", "regularization_H",
    "A_MAX_RATE", "H_MAX_RATE",
    "adam_lr", "lr_schedule_step_factor",
]

CASES = {
    "exp1": dict(
        data_dir="data/exp_data",
        file_suffix="_dyn_T_ramps_exp",
        split="time",
        hidden_states=True,
        step_t_sampling=10,
        time_to_leave_out=100,
        # The l1 penalty on H had no measurable effect in the second search
        # (importance 0.01, best trials spread over its whole range) once H
        # is rate-bounded; fixed at the value of the best trial. The loss is
        # searched again: the first search rejected the hybrid loss at a small
        # H bound and mostly two modes, conditions that no longer hold.
        fixed={**{k: v for k, v in _EXP_FIXED.items() if k != "ROM_loss"},
               "regularization_H": 3.124e-05},
        modules={},
        enabled=["energy_threshold", "A_MAX_RATE", "H_MAX_RATE",
                 "adam_lr", "lr_schedule_step_factor",
                 "ROM_loss", "lambda_deriv"],
        # Refinement around the second search: r_T = 4 or 5, and the H bound
        # beyond the 0.15 its best trials were pressed against.
        space={
            "energy_threshold": dict(choices=[0.9932, 0.9956]),
            "A_MAX_RATE": dict(low=0.04, high=0.12),
            "H_MAX_RATE": dict(low=0.08, high=0.30),
            "adam_lr": dict(low=8e-3, high=4e-2),
            "lr_schedule_step_factor": dict(low=300, high=1000),
        },
        starts=[
            dict(energy_threshold=0.9932, ROM_loss="states",
                 A_MAX_RATE=0.09072, H_MAX_RATE=0.1483, adam_lr=0.02109,
                 lr_schedule_step_factor=790),
            dict(energy_threshold=0.9932, ROM_loss="states",
                 A_MAX_RATE=0.05646, H_MAX_RATE=0.1184, adam_lr=0.0145,
                 lr_schedule_step_factor=498),
            dict(energy_threshold=0.9932, ROM_loss="states",
                 A_MAX_RATE=0.09072, H_MAX_RATE=0.22, adam_lr=0.02109,
                 lr_schedule_step_factor=790),
            dict(energy_threshold=0.9932, ROM_loss="hybrid",
                 lambda_deriv=0.4219, A_MAX_RATE=0.09072,
                 H_MAX_RATE=0.1483, adam_lr=0.02109,
                 lr_schedule_step_factor=790),
        ],
    ),
    "exp2": dict(
        data_dir="data/exp_data",
        file_suffix="_dyn_var_T_var_F_exp",
        split="time",
        hidden_states=True,
        step_t_sampling=10,
        time_to_leave_out=100,
        # Loss and flow completion are searched: in the pre-study the hybrid
        # loss was clearly ahead on this campaign, where the feed steps give
        # the derivative a real signal, and the large H bound hurt the
        # state-only loss. The l1 penalty on H is fixed as for exp1.
        fixed={**{k: v for k, v in _EXP_FIXED.items()
                  if k not in ("ROM_loss", "completion")},
               "regularization_H": 3.124e-05},
        modules={},
        enabled=["energy_threshold", "completion", "ROM_loss",
                 "lambda_deriv", "A_MAX_RATE", "H_MAX_RATE",
                 "adam_lr", "lr_schedule_step_factor"],
        space={
            # r_T = 2, 3, 4, 5: midpoints of each rank's energy interval on
            # the exp2 search record; every value keeps the flow at two modes.
            "energy_threshold": dict(choices=[0.9874, 0.9951, 0.9975]),
            "A_MAX_RATE": dict(low=0.03, high=0.12),
            "H_MAX_RATE": dict(low=0.02, high=0.30),
            "adam_lr": dict(low=5e-3, high=5e-2),
            "lr_schedule_step_factor": dict(low=100, high=1000),
        },
        starts=[
            dict(energy_threshold=0.9874, completion="linear",
                 ROM_loss="hybrid", lambda_deriv=0.4219,
                 A_MAX_RATE=0.09072, H_MAX_RATE=0.1483, adam_lr=0.02109,
                 lr_schedule_step_factor=790),
            dict(energy_threshold=0.9874, completion="linear",
                 ROM_loss="states",
                 A_MAX_RATE=0.09072, H_MAX_RATE=0.05, adam_lr=0.02109,
                 lr_schedule_step_factor=790),
            dict(energy_threshold=0.9951, completion="linear",
                 ROM_loss="hybrid", lambda_deriv=0.4219,
                 A_MAX_RATE=0.09072, H_MAX_RATE=0.1483, adam_lr=0.02109,
                 lr_schedule_step_factor=790),
        ],
    ),
    "synthetic": dict(
        data_dir="data/smooth_aprbs",
        file_suffix="_training",
        split="condition",
        hidden_states=False,
        step_t_sampling=100,
        time_to_leave_out=0,
        # The synthetic model learns a constant term as well, the condition
        # split evaluates its training trajectories, and the record is fully
        # observed, so there is nothing for the memory terms to absorb. None of
        # that follows from leaving a parameter out of the search space.
        fixed=dict(model_structure="AHBC", evaluate_train_cases=True,
                   lag_max=0, completion="linear",
                   input_names=["load", "cooling_temperature"]),
        # Simulated fields have no frozen sensors, and a true steady state
        # can repeat values exactly.
        modules={"opinf_methanation_exp":
                 {"REPAIR_FLATLINES_MIN_SAMPLES": None}},
        # The smoothing stays in the search: the manuscript reports a filter
        # width of 179 for this case, so it was searched, and that number only
        # means something if it still is.
        enabled=[
            "use_filter", "filter_window", "ddt_method", "ddt_window",
            "ddt_postprocessing", "spike_z_threshold",
            "spike_final_smoothing", "spike_savgol_window",
            "energy_threshold", "basis", "regularization_H",
            "adam_lr", "ROM_loss", "lambda_deriv",
        ],
    ),
    # Fine phase around the converged exp1 searches: rank 4, state loss and
    # linear completion fixed; only the rate bounds and the learning-rate
    # schedule are searched, in the region the best trials occupy.
    "exp1_fine": dict(
        data_dir="data/exp_data",
        file_suffix="_dyn_T_ramps_exp",
        split="time",
        hidden_states=True,
        step_t_sampling=10,
        time_to_leave_out=100,
        fixed=dict(_EXP_FIXED, regularization_H=3.124e-05,
                   energy_threshold_single=0.9932),
        modules={},
        enabled=["A_MAX_RATE", "H_MAX_RATE", "adam_lr",
                 "lr_schedule_step_factor"],
        space={
            "A_MAX_RATE": dict(low=0.07, high=0.14),
            "H_MAX_RATE": dict(low=0.11, high=0.19),
            "adam_lr": dict(low=0.015, high=0.032),
            "lr_schedule_step_factor": dict(low=600, high=1000),
        },
        starts=[
            dict(A_MAX_RATE=0.1009, H_MAX_RATE=0.1434, adam_lr=0.0242,
                 lr_schedule_step_factor=884),
            dict(A_MAX_RATE=0.09072, H_MAX_RATE=0.1483, adam_lr=0.02109,
                 lr_schedule_step_factor=790),
            dict(A_MAX_RATE=0.09303, H_MAX_RATE=0.1413, adam_lr=0.02521,
                 lr_schedule_step_factor=704),
            dict(A_MAX_RATE=0.1127, H_MAX_RATE=0.155, adam_lr=0.02143,
                 lr_schedule_step_factor=894),
        ],
    ),
    # Fine phase around the first exp2 search: rank 3, hybrid loss and linear
    # completion fixed; the weight of the derivative term, the rate bounds and
    # the schedule are searched. The A bound and the ramp were pressed against
    # their upper ends, so both ranges reach further.
    "exp2_fine": dict(
        data_dir="data/exp_data",
        file_suffix="_dyn_var_T_var_F_exp",
        split="time",
        hidden_states=True,
        step_t_sampling=10,
        time_to_leave_out=100,
        fixed=dict(_EXP_FIXED, ROM_loss="hybrid", regularization_H=3.124e-05,
                   energy_threshold_single=0.9874),
        modules={},
        enabled=["lambda_deriv", "A_MAX_RATE", "H_MAX_RATE", "adam_lr",
                 "lr_schedule_step_factor"],
        space={
            "lambda_deriv": dict(low=0.3, high=0.7),
            "A_MAX_RATE": dict(low=0.06, high=0.15),
            "H_MAX_RATE": dict(low=0.06, high=0.20),
            "adam_lr": dict(low=0.01, high=0.03),
            "lr_schedule_step_factor": dict(low=600, high=1000),
        },
        starts=[
            dict(lambda_deriv=0.4219, A_MAX_RATE=0.09072, H_MAX_RATE=0.1483,
                 adam_lr=0.02109, lr_schedule_step_factor=790),
            dict(lambda_deriv=0.5394, A_MAX_RATE=0.08275, H_MAX_RATE=0.1029,
                 adam_lr=0.02187, lr_schedule_step_factor=921),
            dict(lambda_deriv=0.5117, A_MAX_RATE=0.1111, H_MAX_RATE=0.08608,
                 adam_lr=0.01653, lr_schedule_step_factor=777),
            dict(lambda_deriv=0.4997, A_MAX_RATE=0.1175, H_MAX_RATE=0.08527,
                 adam_lr=0.01869, lr_schedule_step_factor=746),
        ],
    ),
    # exp2 with fresh seeds in every trial and the ensemble of the three best
    # seeds as the objective, after the first exp2 search turned out to have
    # tuned its hyperparameters to seeds 0-2. Rank, loss and its weight, the
    # rate bounds and the schedule are searched again, over ranges that
    # contain every configuration that did well so far.
    "exp2_robust": dict(
        data_dir="data/exp_data",
        file_suffix="_dyn_var_T_var_F_exp",
        split="time",
        hidden_states=True,
        step_t_sampling=10,
        time_to_leave_out=100,
        fixed={**{k: v for k, v in _EXP_FIXED.items() if k != "ROM_loss"},
               "regularization_H": 3.124e-05},
        modules={},
        enabled=["energy_threshold", "ROM_loss", "lambda_deriv",
                 "A_MAX_RATE", "H_MAX_RATE", "adam_lr",
                 "lr_schedule_step_factor"],
        space={
            "energy_threshold": dict(choices=[0.9874, 0.9951]),
            "lambda_deriv": dict(low=0.2, high=0.7),
            "A_MAX_RATE": dict(low=0.04, high=0.15),
            "H_MAX_RATE": dict(low=0.02, high=0.20),
            "adam_lr": dict(low=0.008, high=0.03),
            "lr_schedule_step_factor": dict(low=400, high=1000),
        },
        starts=[
            dict(energy_threshold=0.9874, ROM_loss="hybrid",
                 lambda_deriv=0.5394, A_MAX_RATE=0.08275,
                 H_MAX_RATE=0.1029, adam_lr=0.02187,
                 lr_schedule_step_factor=921),
            dict(energy_threshold=0.9874, ROM_loss="states",
                 A_MAX_RATE=0.09072, H_MAX_RATE=0.05, adam_lr=0.02109,
                 lr_schedule_step_factor=790),
        ],
    ),        
}

DECODER_PARAMS = ["CNN_learn_rate", "CNN_activation", "CNN_input_noise",
                  "CNN_loss_function"]
PINN_PARAMS = ["PINN_lr_net", "PINN_lr_opinf", "PINN_stage_A"]

# =============================================================================
# SEARCH SPACE
# =============================================================================
# The ranges follow the table in the manuscript, so that the table describes
# what was actually searched. An entry with a "module" key sets a module
# constant instead of a parameter attribute. A "requires" pair makes the
# parameter conditional: (name, value) for equality, (name, "positive") for a
# number above zero.

SPACE = {
    # --- preprocessing ---
    "use_filter": dict(attr="use_filter", kind="cat",
                       choices=[True, False]),
    "filter_window": dict(attr="filter_window", kind="int",
                          low=7, high=700, log=True, odd=True,
                          requires=("use_filter", True)),
    # The quadratic term is expanded about the reference state, so where the
    # shift puts the origin changes what H can represent.
    "state_shifting": dict(attr="state_shifting", kind="cat",
                           choices=["mean", "steady-state"]),
    # --- derivative estimate ---
    # The manuscript states that the synthetic case falls back to sixth-order
    # central differences because the data are noise-free. They are not quite:
    # the trajectories carry the modelling noise of the mechanistic solver, and
    # ord6 is a plain seven-point stencil with no smoothing, so it amplifies
    # whatever is there. Rather than assert either, the synthetic case searches
    # over both and the manuscript reports what won. The experimental cases
    # fix savgol, where the choice is not in doubt.
    "ddt_method": dict(attr="ddt_method", kind="cat",
                       choices=["savgol", "ord6"]),
    "ddt_window": dict(attr="ddt_window", kind="int",
                       low=5, high=51, odd=True,
                       requires=("ddt_method", "savgol")),

    # --- derivative post-processing ---
    # Off for the experimental cases, where it made no difference to the
    # result. Kept searchable for the synthetic case.
    "ddt_postprocessing": dict(attr="ddt_postprocessing", kind="cat",
                               choices=[True, False]),
    "spike_z_threshold": dict(attr="spike_z_threshold", kind="float",
                              low=2.0, high=10.0,
                              requires=("ddt_postprocessing", True)),
    "spike_final_smoothing": dict(attr="spike_final_smoothing", kind="cat",
                                  choices=[True, False],
                                  requires=("ddt_postprocessing", True)),
    "spike_savgol_window": dict(attr="spike_savgol_window", kind="int",
                                low=7, high=101, odd=True,
                                requires=("spike_final_smoothing", True)),

    # --- basis ---
    # One choice per temperature rank r_T = 3, 4, 5: each value is the
    # midpoint of that rank's energy interval on the exp1 search record.
    # Drawn uniformly over the threshold, ranks 4 and 5 received only a
    # sixth of the trials of the first search, their intervals being narrow.
    "energy_threshold": dict(attr="energy_threshold_single", kind="cat",
                             choices=[0.9867, 0.9932, 0.9956]),
    "basis": dict(attr="basis", kind="cat",
                  choices=["POD", "NL-POD", "AM"]),
    # Reconstruction of the unmeasured flow profile. The experimental cases
    # score only the outlet, so the kernel shapes the latent flow
    # coordinates, not the reference the models are scored against.
    "completion": dict(attr="completion", kind="cat",
                       choices=["linear", "knowledge-based"]),

    # --- model structure ---
    "regularization_H": dict(attr="regularization_H", kind="float",
                             low=5e-6, high=1e-3, log=True),

    # --- rate bounds ---
    # Upper ends from the stability region of the Runge-Kutta step used in
    # training: (rho_A + rho_H) * dt <= 2.5 keeps every linearised rate
    # inside it (limit 2.83). At rho_H = 0.40, 4 of 9 runs diverged; the
    # H bound binds in every run, so its value is a real hyperparameter.
    "A_MAX_RATE": dict(module="opinf.models._nonparametric",
                       attr="A_MAX_RATE", kind="float",
                       low=0.03, high=0.10, log=True),
    "H_MAX_RATE": dict(module="opinf.models._nonparametric",
                       attr="H_MAX_RATE", kind="float",
                       low=0.02, high=0.15, log=True),

    # --- memory terms ---
    "lag_max": dict(attr="lag_max", kind="int", low=0, high=3),
    "LAG_STEP": dict(module="opinf_methanation_exp", attr="LAG_STEP",
                     kind="cat", choices=[1, 5, 10, 30],
                     requires=("lag_max", "positive")),
    "LAG_OPERATOR_STRUCTURE": dict(module="opinf_methanation_exp",
                                   attr="LAG_OPERATOR_STRUCTURE",
                                   kind="cat", choices=["stable", "free"],
                                   requires=("lag_max", "positive")),

    # --- gradient-based learning ---
    # The cyclic schedule peaks at this value. At the reference 0.0203 one of
    # ten v8 seeds already spiked at the peak (loss x40 at epoch 800) and
    # finished in the poor mode; above 0.03 most trials would spend their
    # budget recovering from such spikes.
    "adam_lr": dict(attr="adam_lr", kind="float",
                    low=5e-3, high=5e-2, log=True),
    # Length of the learning-rate ramp in epochs, capped at half the run. At
    # 2000 epochs a value of 1000 or more gives one symmetric cycle, which
    # spends half the run ramping up; shorter ramps give several cycles whose
    # amplitude halves each time (triangular2), so more of the run anneals.
    # Measured: 4000 epochs instead of 2000 gained 2 to 3 points on two of
    # three seeds, which a schedule that anneals earlier may recover.
    "lr_schedule_step_factor": dict(attr="lr_schedule_step_factor",
                                    kind="int", low=100, high=1000,
                                    log=True),
    # "ddt" alone is the single-step derivative fit and is left out: it is
    # the ablation baseline, not a candidate.
    "ROM_loss": dict(attr="ROM_loss", kind="cat",
                     choices=["hybrid", "states"]),
    "lambda_deriv": dict(attr="lambda_deriv", kind="float",
                         low=0.0, high=1.0,
                         requires=("ROM_loss", "hybrid")),

    # --- PINN learning ---
    "PINN_lr_net": dict(attr="PINN_lr_net", kind="float",
                        low=5e-5, high=2e-3, log=True),
    "PINN_lr_opinf": dict(attr="PINN_lr_opinf", kind="float",
                          low=5e-4, high=5e-3, log=True),
    "PINN_stage_A": dict(attr="PINN_stage_config_A", kind="float",
                         low=0.3, high=0.7),

    # --- decoder ---
    "CNN_learn_rate": dict(attr="CNN_learn_rate", kind="float",
                           low=1e-4, high=5e-3, log=True),
    "CNN_activation": dict(attr="CNN_activation", kind="cat",
                           choices=["elu", "gelu", "silu", "softplus"]),
    "CNN_input_noise": dict(attr="CNN_input_noise", kind="float",
                            low=0.0, high=0.1),
    "CNN_loss_function": dict(attr="CNN_loss_function", kind="cat",
                              choices=["mse", "mae", "smooth"]),
}

# Parameter attribute back to the name it carries in the search space, used to
# recognise a case's fixed values as settings that conditions can refer to.
# Module constants are not parameter attributes and stay out.
_ATTR_TO_NAME = {spec["attr"]: name for name, spec in SPACE.items()
                 if "module" not in spec}


# =============================================================================
# PARAMETER SINGLETON AND MODULE CONSTANTS
# =============================================================================

def _snapshot(params):
    out = {}
    for name in dir(params):
        if name.startswith('_'):
            continue
        value = getattr(params, name)
        if callable(value):
            continue
        out[name] = copy.deepcopy(value)
    return out


def _restore(params, snapshot):
    for name, value in snapshot.items():
        setattr(params, name, copy.deepcopy(value))


def _apply_modules(modules):
    """Set module constants, refusing names a module does not have.

    JSON turns tuples into lists on the way to a worker; the code under test
    expects tuples, so they are turned back.
    """
    for module_name, values in modules.items():
        module = importlib.import_module(module_name)
        for key, value in values.items():
            if not hasattr(module, key):
                raise AttributeError(
                    f"{module_name} has no setting '{key}'; apply the patch "
                    f"that introduces it before searching")
            setattr(module, key,
                    tuple(value) if isinstance(value, list) else value)


def _merge_modules(*layers):
    """Deep merge of module dictionaries, later layers winning."""
    out = {}
    for layer in layers:
        for module_name, values in (layer or {}).items():
            out.setdefault(module_name, {}).update(copy.deepcopy(values))
    return out


# Taken before any trial runs, so that it holds the configured defaults and not
# the state some earlier trial left behind.
PRISTINE = _snapshot(opinf.parameters.Params())


# =============================================================================
# HELPERS
# =============================================================================

def _rss_gb():
    """Peak resident set size of this process, in GB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2


def _study_name(case=None, stage=None, regime=None):
    case = CASE if case is None else case
    stage = STAGE if stage is None else stage
    regime = REGIME if regime is None else regime
    tag = regime if stage == "model" else "decoder"
    # A tag distinguishes successive searches on the same case, so a new
    # search space never resumes an old study.
    return f"opinf_{case}_{tag}{os.environ.get('OPINF_STUDY_TAG', '')}"


def _study_dir(name):
    path = os.path.join(BASE_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def _load_best(study_name):
    """Best parameters of an existing study, by name."""
    path = os.path.join(BASE_DIR, study_name, f"{study_name}.db")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no study database at {path}; run that stage first")
    study = optuna.load_study(study_name=study_name,
                              storage=f"sqlite:///{path}")
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None and np.isfinite(t.value)]
    if not completed:
        raise RuntimeError(f"study {study_name} has no usable trial")
    best = min(completed, key=lambda t: t.value)
    params = dict(best.params)
    params.update(best.user_attrs)   # the real window widths, not the halves
    return params, best.value


def _narrow(spec, best_value):
    """Shrink one numeric range around a value, clamped to the original."""
    spec = dict(spec)
    if spec["kind"] == "cat":
        if REFINE_FIX_CATEGORICAL:
            spec["choices"] = [best_value]
        return spec

    low, high = spec["low"], spec["high"]
    if spec.get("log"):
        span = (np.log10(high) - np.log10(low)) * REFINE_SPAN / 2
        centre = np.log10(max(float(best_value), low))
        spec["low"] = float(max(low, 10 ** (centre - span)))
        spec["high"] = float(min(high, 10 ** (centre + span)))
    else:
        span = (high - low) * REFINE_SPAN / 2
        spec["low"] = float(max(low, best_value - span))
        spec["high"] = float(min(high, best_value + span))

    if spec["kind"] == "int":
        spec["low"] = int(np.floor(spec["low"]))
        spec["high"] = max(int(np.ceil(spec["high"])), spec["low"] + 1)
    return spec


def _condition_met(requirement, chosen):
    """Whether a conditional parameter is active for the values drawn so far."""
    if requirement is None:
        return True
    name, wanted = requirement
    have = chosen.get(name)
    if wanted == "positive":
        return isinstance(have, (int, float)) and not isinstance(have, bool) \
            and have > 0
    return have == wanted


def _suggest(trial, name, spec, chosen):
    """Draw one parameter, honouring conditions and the odd-length rule."""
    if not _condition_met(spec.get("requires"), chosen):
        return None

    kind = spec["kind"]

    if kind == "cat":
        if len(spec["choices"]) == 1:
            return spec["choices"][0]
        return trial.suggest_categorical(name, spec["choices"])

    if kind == "int":
        if spec.get("odd"):
            # Savitzky-Golay needs an odd window. The search runs over the half
            # width so that every draw is valid; drawing the window directly
            # would make half the range raise an exception, which a driver then
            # scores as a very bad objective and the sampler learns to avoid
            # for entirely the wrong reason. The window itself is recorded as a
            # user attribute, because a study whose best trial reports the half
            # width would put the wrong number in the manuscript.
            lo = max(1, spec["low"] // 2)
            hi = max(lo + 1, spec["high"] // 2)
            half = trial.suggest_int(f"{name}_half", lo, hi,
                                     log=bool(spec.get("log")))
            value = 2 * half + 1
            trial.set_user_attr(name, value)
            return value
        return trial.suggest_int(name, spec["low"], spec["high"],
                                 log=bool(spec.get("log")))

    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"],
                                   log=bool(spec.get("log")))

    raise ValueError(f"unknown parameter kind '{kind}' for {name}")


def _active_space():
    """Names to search in this stage, in a fixed order."""
    case_cfg = CASES[CASE]
    if STAGE == "decoder":
        names = list(DECODER_PARAMS)
    else:
        names = [n for n in SPACE if n in case_cfg["enabled"]]
        if REGIME == "pinn":
            names = [n for n in names
                     if n not in ("adam_lr", "ROM_loss", "lambda_deriv")]
            names += PINN_PARAMS
    return [n for n in names if n not in OVERRIDES]


def _case_attrs():
    """Settings that identify the case rather than being searched."""
    cfg = CASES[CASE]
    attrs = dict(
        data_dir=cfg["data_dir"],
        file_suffix=cfg["file_suffix"],
        split=cfg["split"],
        hidden_states=cfg["hidden_states"],
        step_t_sampling=cfg["step_t_sampling"],
        time_to_leave_out=cfg["time_to_leave_out"],
    )
    attrs.update(cfg.get("fixed", {}))
    return attrs


def _check_inputs():
    """Fail before the study starts rather than inside the first trial."""
    cfg = CASES[CASE]
    base, suffix = cfg["data_dir"], f"{cfg['file_suffix']}.npy"

    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"case '{CASE}': no directory {base} relative to "
            f"{os.getcwd()}; start the script from the project folder")

    present = sorted(f[:-len(suffix)] for f in os.listdir(base)
                     if f.endswith(suffix))
    print(f"Arrays under {base} with suffix {suffix}:")
    print("  " + (", ".join(present) if present else "none"))

    required = ["z", "time", "center_temperature"]
    required += list(cfg.get("fixed", {}).get("input_names", []))
    missing = [n for n in required if n not in present]
    if not ("flow_rate_out" in present or "conversion" in present):
        missing.append("flow_rate_out or conversion")
    if missing:
        raise FileNotFoundError(f"case '{CASE}': missing arrays {missing}")

    # Every module constant a trial will set has to exist, or the first trial
    # fails inside a worker where the message is easy to miss.
    _apply_modules(_merge_modules(MODULE_SETTINGS, cfg.get("modules")))


# =============================================================================
# OBJECTIVE
# =============================================================================

FAILURES = Counter()
TRIAL_LOG = []
FROZEN = {}
ACTIVE_SPACE = {}
ACTIVE = []


def _set_value(name, value, attrs, modules):
    """Route a value to a parameter attribute or to a module constant."""
    spec = SPACE[name]
    if "module" in spec:
        modules.setdefault(spec["module"], {})[spec["attr"]] = value
    else:
        attrs[spec["attr"]] = value


def _collect_attrs(trial):
    """Every parameter attribute and module constant this trial sets."""
    attrs = _case_attrs()
    attrs["output"] = False
    attrs["save_results"] = False
    attrs["use_CNN"] = (STAGE == "decoder")
    attrs["use_PINN"] = (STAGE == "model" and REGIME == "pinn")
    modules = _merge_modules(MODULE_SETTINGS, CASES[CASE].get("modules"))

    for name, value in FROZEN.items():
        if name in SPACE:
            _set_value(name, value, attrs, modules)
    for name, value in OVERRIDES.items():
        if name in SPACE:
            _set_value(name, value, attrs, modules)

    # Conditions are checked against ``chosen``, so a value the case fixes has
    # to appear there too. Without this, a parameter conditional on a fixed
    # setting is never drawn: ddt_window requires ddt_method == "savgol", and
    # the experimental cases fix that rather than searching it. It also makes
    # the trial log describe the whole configuration instead of only its
    # searched part.
    chosen = {_ATTR_TO_NAME[attr]: value for attr, value in attrs.items()
              if attr in _ATTR_TO_NAME}
    chosen.update(OVERRIDES)
    chosen.update({k: v for k, v in FROZEN.items() if k in SPACE})
    for name in ACTIVE:
        value = _suggest(trial, name, ACTIVE_SPACE[name], chosen)
        if value is None:
            continue
        chosen[name] = value
        _set_value(name, value, attrs, modules)

    lam = attrs.get("lambda_deriv", PRISTINE["lambda_deriv"])
    lr = attrs.get("adam_lr", PRISTINE["adam_lr"])
    attrs["lambda_state"] = 1.0 - lam
    attrs["lr_schedule_base_lr"] = lr / 5.0
    attrs["lr_schedule_max_lr"] = lr
    if attrs["use_PINN"]:
        stage_a = attrs.get("PINN_stage_config_A",
                            PRISTINE["PINN_stage_config_A"])
        attrs["PINN_stage_config_B"] = max(0.0, 1.0 - stage_a)

    attrs["holdout_ratio"] = HO_HOLDOUT
    attrs["train_ratio"] = HO_TRAIN_RATIO
    return attrs, modules, chosen


def _nan(value):
    """A float, with anything unusable turned into NaN rather than an error."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _rollout_of(params):
    """Markovian rollout by horizon, keyed by the horizon as a string."""
    rollout = getattr(params, "last_rollout", {}) or {}
    return {str(h): _nan(v) for h, v in rollout.items()}


def _run_in_process(attrs, modules, seed):
    Params = opinf.parameters.Params()
    _restore(Params, PRISTINE)
    _apply_modules(modules)
    for name, value in attrs.items():
        setattr(Params, name, value)
    value = float(methanation_main(Params, seed=seed))
    return dict(
        value=value,
        peak_gb=None,
        global_err=_nan(getattr(Params, "last_global_error", float("nan"))),
        ddt_residual=_nan(getattr(Params, "last_ddt_residual", float("nan"))),
        rollout=_rollout_of(Params),
    )


def _run_in_subprocess(attrs, modules, seed):
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "config.json")
        out_path = os.path.join(tmp, "result.json")
        with open(cfg_path, "w") as f:
            json.dump({"seed": int(seed), "attrs": attrs,
                       "modules": modules}, f)

        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--worker", cfg_path, out_path],
            cwd=os.getcwd(), capture_output=True, text=True,
            timeout=TRIAL_TIMEOUT_SEC)

        if not os.path.exists(out_path):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise RuntimeError("worker produced no result: "
                               + " | ".join(tail[-4:]))
        with open(out_path) as f:
            result = json.load(f)

    if result.get("error"):
        raise RuntimeError(result["error"])
    return dict(
        value=float(result["value"]),
        peak_gb=result.get("peak_gb"),
        global_err=_nan(result.get("global")),
        ddt_residual=_nan(result.get("ddt_residual")),
        rollout={k: _nan(v) for k, v in (result.get("rollout") or {}).items()},
    )


def _evaluate(attrs, modules, seed):
    """One run of the pipeline, in whichever way is configured.

    Both paths return the same keys. The previous version returned two values
    from one path and three from the other, which raised ValueError on every
    in-process trial; the generic handler then counted it as a bad region of
    the search space rather than as the bug it was.
    """
    if RUN_TRIALS_IN_SUBPROCESS:
        return _run_in_subprocess(attrs, modules, seed)
    return _run_in_process(attrs, modules, seed)


def _aggregate(runs):
    """The number the sampler sees, and the index of the run it came from.

    Under "select" that is the run with the lowest training rollout error at
    SELECT_HORIZON; the index is None for the other rules, which combine all
    runs. Without a usable rollout, "select" falls back to the median.
    """
    values = [r["value"] for r in runs]
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan"), None

    if TRIAL_SEED_AGGREGATE == "select":
        scored = [(r["rollout"].get(str(SELECT_HORIZON), float("nan")), i)
                  for i, r in enumerate(runs) if np.isfinite(r["value"])]
        scored = [(s, i) for s, i in scored if np.isfinite(s)]
        if scored:
            _, index = min(scored)
            return float(values[index]), index
        return float(np.median(finite)), None
    if TRIAL_SEED_AGGREGATE == "mean":
        return float(np.mean(finite)), None
    if TRIAL_SEED_AGGREGATE == "max":
        return float(np.max(finite)), None
    return float(np.median(finite)), None


def _ensemble_error(folder):
    """Error of the mean forecast of the best seeds by training rollout."""
    from opinf.post import field_errors
    runs = []
    for path in glob.glob(os.path.join(folder, "*.npz")):
        with np.load(path) as f:
            run = {k: f[k] for k in f.files}
        if np.all(np.isfinite(run["sol"])):
            runs.append(run)
    if not runs:
        return float("nan")
    roll = np.array([float(r["rollout_200"]) for r in runs])
    order = np.argsort(np.where(np.isfinite(roll), roll, np.inf))
    best = [runs[i] for i in order[:ENSEMBLE_TOP_K]]
    mean = np.mean([np.asarray(r["sol"], dtype=float) for r in best], axis=0)
    e = field_errors(mean, np.asarray(best[0]["F_true"], dtype=float),
                     np.asarray(best[0]["T_true"], dtype=float))
    return float(e["combined_centred"])


def objective(trial):
    # The driver draws nothing random of its own beyond the sampler, which
    # carries its own generator, and the work happens in a fresh interpreter
    # per seed. Reseeding the global generators here would only look like a
    # guarantee.
    attrs, modules, chosen = _collect_attrs(trial)
    started = time.time()
    runs, run_seeds = [], []
    tolerate = TRIAL_SEED_AGGREGATE in ("select", "ensemble")
    pred_dir = None
    if TRIAL_SEED_AGGREGATE == "ensemble":
        pred_dir = tempfile.mkdtemp(prefix=f"opinf_trial{trial.number}_")
        modules = _merge_modules(modules, {"opinf_methanation_exp": {
            "SAVE_PREDICTIONS_TO": pred_dir, "PREDICTION_TAG": "t"}})

    # The decoder stage refits the same reduced model in every trial; doing
    # that for three seeds would triple its cost for nothing, so it runs on
    # the seed the model stage selected for its winner.
    if STAGE == "decoder":
        seeds = (int(FROZEN.get("selected_seed", TRIAL_SEEDS[0])),)
    elif TRIAL_SEED_MODE == "per_trial":
        k = len(TRIAL_SEEDS)
        seeds = tuple(TRIAL_SEED_BASE + k * trial.number + i
                      for i in range(k))
    else:
        seeds = TRIAL_SEEDS

    for seed in seeds:
        try:
            runs.append(_evaluate(attrs, modules, seed))
            run_seeds.append(seed)
            if (EARLY_ABORT_FACTOR and STAGE == "model"
                    and TRIAL_SEED_AGGREGATE != "ensemble"
                    and len(runs) == 1 and len(seeds) > 1):
                done = [t.value for t in trial.study.get_trials(
                            deepcopy=False,
                            states=(optuna.trial.TrialState.COMPLETE,))
                        if t.value is not None]
                if (len(done) >= EARLY_ABORT_MIN_TRIALS
                        and runs[0]["value"]
                        > EARLY_ABORT_FACTOR * float(np.median(done))):
                    trial.set_user_attr("aborted_after_first_seed", True)
                    break
        except subprocess.TimeoutExpired:
            FAILURES["timeout"] += 1
            trial.set_user_attr("error", "timeout")                    
            TRIAL_LOG.append(dict(number=trial.number, value=None, seed=seed,
                                  error="timeout", params=chosen))
            if not tolerate:
                return OBJECTIVE_CEILING
        except Exception as exc:
            FAILURES[type(exc).__name__] += 1
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}"[:500])
            TRIAL_LOG.append(dict(number=trial.number, value=None, seed=seed,
                                  error=f"{type(exc).__name__}: {exc}",
                                  params=chosen))
            if not tolerate:
                return OBJECTIVE_CEILING
        finally:
            gc.collect()

    if not runs:
        return OBJECTIVE_CEILING

    values = [r["value"] for r in runs]
    if pred_dir is not None:
        err, index = _ensemble_error(pred_dir), None
        shutil.rmtree(pred_dir, ignore_errors=True)
    else:
        err, index = _aggregate(runs)

    if not np.isfinite(err):
        FAILURES["non-finite objective"] += 1
        TRIAL_LOG.append(dict(number=trial.number, value=None,
                              seed_values=values,
                              error="non-finite objective", params=chosen))
        return OBJECTIVE_CEILING

    if index is not None:
        glob = runs[index]["global_err"]
        resid = runs[index]["ddt_residual"]
        trial.set_user_attr("selected_seed", int(run_seeds[index]))
    else:
        glob = _aggregate([dict(r, value=r["global_err"]) for r in runs])[0]
        resid = _aggregate([dict(r, value=r["ddt_residual"])
                            for r in runs])[0]

    peaks = [r["peak_gb"] for r in runs if r["peak_gb"] is not None]
    peak = max(peaks) if peaks else None
    spread = (float(np.max(values) - np.min(values))
              if len(values) > 1 else 0.0)
    rollouts = [r["rollout"].get(str(SELECT_HORIZON), float("nan"))
                for r in runs]

    # Recorded whatever they are, so that a finished study can be re-ranked
    # afterwards without re-running anything: against the globally reported
    # metric rather than the search window, or against another aggregate of
    # the seeds.
    if np.isfinite(glob):
        trial.set_user_attr("global_err", glob)
    if np.isfinite(resid):
        trial.set_user_attr("ddt_residual", resid)
    if len(values) > 1:
        trial.set_user_attr("seed_values", values)
        trial.set_user_attr("seed_rollouts", rollouts)
        trial.set_user_attr("seed_spread", spread)

    seconds = round(time.time() - started, 1)
    TRIAL_LOG.append(dict(number=trial.number, value=float(err),
                          seeds=run_seeds, seed_values=values,
                          seed_rollouts=rollouts, seed_spread=spread,
                          global_err=glob, ddt_residual=resid,
                          seconds=seconds, peak_gb=peak,
                          driver_peak_gb=round(_rss_gb(), 2),
                          error=None, params=chosen))

    if trial.number % 5 == 0:
        line = f"  trial {trial.number}: {err:.5f}"
        if len(values) > 1:
            line += f" (spread {spread:.5f} over {len(values)} seeds)"
        line += f" in {seconds:.0f} s"
        if peak is not None:
            line += f", worker peak {peak:.1f} GB"
        print(line, flush=True)

    return float(min(err, OBJECTIVE_CEILING))


# =============================================================================
# RUN
# =============================================================================

def _prepare():
    global FROZEN, ACTIVE_SPACE, ACTIVE

    _check_inputs()

    if STAGE == "decoder":
        model_study = _study_name(stage="model")
        FROZEN, model_value = _load_best(model_study)
        print(f"\nFreezing the winner of {model_study} "
              f"(objective {model_value:.6f}):")
        for key, value in sorted(FROZEN.items()):
            print(f"  {key}: {value}")
    else:
        FROZEN = {}

    ACTIVE = _active_space()
    ACTIVE_SPACE = {n: dict(SPACE[n]) for n in ACTIVE}
    # Per-case ranges and choices. The rank thresholds in particular depend
    # on the energy spectrum of each record.
    for name, spec in CASES[CASE].get("space", {}).items():
        if name in ACTIVE_SPACE:
            ACTIVE_SPACE[name].update(spec)    

    if REFINE_FROM:
        best, value = _load_best(REFINE_FROM)
        print(f"\nRefining around {REFINE_FROM} (objective {value:.6f}), "
              f"keeping {REFINE_SPAN:.0%} of each range.")
        for name in ACTIVE:
            if name in best:
                ACTIVE_SPACE[name] = _narrow(ACTIVE_SPACE[name], best[name])

    print(f"\nCase {CASE}, stage {STAGE}"
          + (f", regime {REGIME}" if STAGE == "model" else ""))
    print(f"Objective: error on the last {1 - HO_TRAIN_RATIO:.0%} of the "
          f"first {1 - HO_HOLDOUT:.0%} of the record; the final "
          f"{HO_HOLDOUT:.0%} stays untouched.")
    if TRIAL_SEED_AGGREGATE == "select":
        print(f"Seeds: {list(TRIAL_SEEDS)} per trial; the seed with the "
              f"lowest training rollout error at {SELECT_HORIZON} steps is "
              f"scored, as in the final multistart.")
    elif len(TRIAL_SEEDS) == 1:
        print(f"Seeds: one per trial, seed {TRIAL_SEEDS[0]}. Differences "
              f"smaller than the seed-to-seed scatter are not resolvable.")
    else:
        print(f"Seeds: {list(TRIAL_SEEDS)} per trial, combined by "
              f"{TRIAL_SEED_AGGREGATE}; the spread is recorded per trial.")
    print(f"Fixed for this case: "
          f"{ {k: v for k, v in CASES[CASE].get('fixed', {}).items()} }")
    print(f"Module constants: "
          f"{_merge_modules(MODULE_SETTINGS, CASES[CASE].get('modules'))}")
    print(f"Searching {len(ACTIVE)} parameters:")
    for name in ACTIVE:
        spec = ACTIVE_SPACE[name]
        cond = (f"  if {spec['requires'][0]} {spec['requires'][1]}"
                if spec.get("requires") else "")
        if spec["kind"] == "cat":
            print(f"  {name}: {spec['choices']}{cond}")
        else:
            print(f"  {name}: [{spec['low']:.4g}, {spec['high']:.4g}]"
                  + ("  log" if spec.get("log") else "")
                  + ("  odd" if spec.get("odd") else "") + cond)
    if OVERRIDES:
        print(f"Overridden: {OVERRIDES}")


def _default_of(spec):
    """The value a parameter has when it is not searched."""
    if "module" in spec:
        modules = _merge_modules(MODULE_SETTINGS, CASES[CASE].get("modules"))
        return modules.get(spec["module"], {}).get(spec["attr"])
    return PRISTINE.get(spec["attr"])


def _seed_defaults(study):
    """Enqueue the configured defaults, so the search has to beat them."""
    seedling = {}
    for name in ACTIVE:
        spec = ACTIVE_SPACE[name]
        value = _default_of(spec)
        if value is None:
            continue

        if spec["kind"] == "cat":
            if value in spec["choices"]:
                seedling[name] = value
            continue

        if spec["kind"] == "int" and spec.get("odd"):
            half = int(value) // 2
            lo = max(1, spec["low"] // 2)
            hi = max(lo + 1, spec["high"] // 2)
            seedling[f"{name}_half"] = int(min(max(half, lo), hi))
            continue

        value = min(max(value, spec["low"]), spec["high"])
        seedling[name] = int(value) if spec["kind"] == "int" else float(value)

    if seedling:
        study.enqueue_trial(seedling)
        # The defaults in parameters.py use the hybrid loss; v8 uses the
        # state loss alone. Both go in, so the search starts from each.
        if "ROM_loss" in ACTIVE_SPACE:
            study.enqueue_trial(dict(seedling, ROM_loss="states"))
        print("\nEnqueued the configured defaults as the first trial.")


def main():
    _prepare()

    name = _study_name()
    directory = _study_dir(name)
    # Several processes write to the same file; without a timeout a write
    # that meets a lock fails at once instead of waiting for it.
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{os.path.join(directory, name)}.db",
        engine_kwargs={"connect_args": {"timeout": 60}})

    # constant_liar keeps parallel processes from proposing the same point
    # while their trials are still running.
    sampler = optuna.samplers.TPESampler(
        seed=SEED, multivariate=True, group=True, constant_liar=True,
        n_startup_trials=STARTUP_TRIALS)

    study = optuna.create_study(study_name=name, storage=storage,
                                direction="minimize", load_if_exists=True,
                                sampler=sampler)

    if study.trials:
        print(f"\nResuming {name}: {len(study.trials)} trials already stored. "
              f"Delete {directory} to start over. A study recorded under a "
              f"different search space must not be resumed: the sampler would "
              f"model parameters that no longer mean the same thing.")
    else:
        _seed_defaults(study)
        for point in CASES[CASE].get("starts", []):
            study.enqueue_trial(point)        

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TOTAL_TIMEOUT_SEC)
    elapsed = time.time() - t0

    # --- report -------------------------------------------------------------
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None and np.isfinite(t.value)]

    print("\n" + "=" * 70)
    print(f"{len(study.trials)} trials in the study, this process ran for "
          f"{elapsed / 60:.1f} min; {len(completed)} usable")
    if FAILURES:
        print("Failures by type in this process, which are bugs until shown "
              "otherwise:")
        for kind, count in FAILURES.most_common():
            print(f"  {kind}: {count}")

    # One log per process: several processes may share the study, and each
    # knows only its own trials.
    log_path = os.path.join(directory, f"trials_{os.getpid()}.jsonl")
    with open(log_path, "w") as f:
        for record in TRIAL_LOG:
            f.write(json.dumps(record, default=str) + "\n")

    if not completed:
        print("No usable trial. Nothing to report.")
        return

    best = min(completed, key=lambda t: t.value)
    # The user attributes carry the real window widths; the raw parameters
    # carry their halves, which is not what belongs in the manuscript.
    best_all = dict(best.params)
    best_all.update(best.user_attrs)

    print(f"\nBest objective: {best.value:.6f}  (trial {best.number})")
    for key, value in sorted(best_all.items()):
        print(f"  {key}: {value}")

    peaks = [r["peak_gb"] for r in TRIAL_LOG if r.get("peak_gb")]
    if peaks:
        print(f"\nWorker memory: median {np.median(peaks):.1f} GB, "
              f"max {np.max(peaks):.1f} GB")

    export = {
        "case": CASE,
        "stage": STAGE,
        "regime": REGIME if STAGE == "model" else None,
        "sampler_seed": SEED,
        "trial_seeds": list(TRIAL_SEEDS),
        "trial_seed_aggregate": TRIAL_SEED_AGGREGATE,
        "select_horizon": SELECT_HORIZON,
        "objective_ceiling": OBJECTIVE_CEILING,
        "holdout_ratio": HO_HOLDOUT,
        "train_ratio": HO_TRAIN_RATIO,
        "case_fixed": CASES[CASE].get("fixed", {}),
        "module_settings": _merge_modules(MODULE_SETTINGS,
                                          CASES[CASE].get("modules")),
        "frozen_from_model_stage": FROZEN,
        "overrides": OVERRIDES,
        "searched": {n: ACTIVE_SPACE[n] for n in ACTIVE},
        "best_value": best.value,
        "best_params": best_all,
        "best_params_raw": best.params,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "failures_this_process": dict(FAILURES),
    }

    def _convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(os.path.join(directory, "best_params.json"), "w") as f:
        json.dump(export, f, indent=4, default=_convert)
    print(f"\nWrote best_params.json and {os.path.basename(log_path)} "
          f"to {directory}")

    # --- plots --------------------------------------------------------------
    # Each figure embeds the whole plotly library by default, which is several
    # megabytes per file; loading it from a CDN instead makes them small enough
    # to keep alongside the manuscript. The slice and parallel-coordinate plots
    # are restricted to the searched parameters, and the latter is skipped for
    # large studies, where it is unreadable anyway.
    plot_params = [p for p in ACTIVE if p in best.params]
    html = dict(include_plotlyjs="cdn", full_html=True)
    try:
        ov.plot_optimization_history(study).write_html(
            os.path.join(directory, "history.html"), **html)
        if len(completed) > 1:
            ov.plot_param_importances(study).write_html(
                os.path.join(directory, "importance.html"), **html)
        if plot_params:
            ov.plot_slice(study, params=plot_params).write_html(
                os.path.join(directory, "slice.html"), **html)
            if len(completed) <= 200:
                ov.plot_parallel_coordinate(
                    study, params=plot_params).write_html(
                    os.path.join(directory, "parallel_coords.html"), **html)
        print(f"Wrote the plots to {directory}")
    except Exception as exc:
        print(f"Could not generate the plots: {type(exc).__name__}: {exc}")


# =============================================================================
# WORKER
# =============================================================================

def _worker_main(config_path, result_path):
    """One run of the pipeline, at one seed, in its own interpreter."""
    payload = {}
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        attrs = cfg["attrs"]
        seed = int(cfg.get("seed", 0))

        _apply_modules(cfg.get("modules", {}))
        Params = opinf.parameters.Params()
        for name, value in attrs.items():
            setattr(Params, name, value)
        payload["value"] = float(methanation_main(Params, seed=seed))
        payload["global"] = float(
            getattr(Params, "last_global_error", float("nan")))
        payload["ddt_residual"] = float(
            getattr(Params, "last_ddt_residual", float("nan")))
        payload["rollout"] = _rollout_of(Params)
    except Exception as exc:
        import traceback
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()[-2000:]
    payload["peak_gb"] = round(_rss_gb(), 2)
    with open(result_path, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3])
    else:
        main()