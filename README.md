# opinf_ptX

Code for the paper

> L. Peterson, A. Geschke, I. V. Gosea, P. Goyal, J. Bremer, R. T. Zimmermann,
> P. Benner, K. Sundmacher. *Data-driven reduced-order model for optimal control
> of dynamically operated reactors in Power-to-X processes.* Chemical Engineering
> Journal (submitted, CEJ-D-26-16749).

The repository identifies quadratic reduced-order models from reactor measurements by operator inference and uses them for optimal control.

## Layout

```
identification/   model identification
  opinf/            operator-inference library
  configs.py        settings of the final models
  run.py            trains the final models
  final_eval.py     reported test errors from the saved forecasts
  opinf_methanation_exp.py     identification and evaluation of one model
  opinf_methanation_Optuna.py  hyperparameter search
control/          optimal control with the identified model
```

## Data

The measured and simulated records are published on EDMOND: DOI *to be added*.
Unpack them so that the identification finds them under

```
identification/data/exp_data/      measured records (Cases 1 and 2)
identification/data/smooth_aprbs/  synthetic record
```

## Installation

```
conda env create -f environment.yml
conda activate opinf_ptx
```

The environment covers identification and control.

## Reproducing the models

From `identification/`:

```
python run.py case1                     # Case 1, seeds 10-39
python run.py case2 --seeds 10-19       # Case 2, one seed set
python run.py synthetic                 # synthetic record
python run.py synthetic_control         # models used in the control study
```

Each seed runs in a fresh process with single-threaded numerics on the CPU.
Forecasts are written to `predictions/<case>/`. The reported test errors of
Cases 1 and 2 (ensembles of the three seeds with the lowest training rollout
error, per seed set) follow from

```
python final_eval.py case1
python final_eval.py case2
```

For the synthetic record, the reported model is the seed with the lowest
training rollout error (seed 18); its test error is printed by
`python run.py synthetic --seeds 18`.

The reported models were computed on macOS (Apple silicon) with the package
versions in `environment.yml`, which reproduce them bitwise on that platform.
Other versions or builds of NumPy and SciPy change the integrated trajectories
at the level of the solver tolerance (relative differences of order 1e-5 in our
tests). Training itself is unaffected.

## Optimal control

The control study uses the three synthetic models of `run.py synthetic_control`
(seeds 18, 10 and 15, refitted on all ten trajectories) as an ensemble. Its
scenario (load profile, coolant profile and initial state) is part of the
EDMOND data set and goes to `control/data/control_scenario_0/`.

```
cd identification
python run.py synthetic_control
cd ../control
mkdir -p data
cp -r ../identification/results/model_* data/
```

`run.py` writes one folder `model_<timestamp>` per seed, in the order 18, 10, 15.
With these three folder names, the reported control run is

```
env OPINF_DEVICE=cpu OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OCP_TAG=ensemble OCP_MODEL=<model seed 18> \
    OCP_ENSEMBLE=<model seed 10>,<model seed 15> \
    OCP_TRAIN_TRAJ=all OCP_STRATEGY=slp OCP_MAX_ITER=0 OCP_SLP_START=linear \
    OCP_SLP_START_MARGIN=20 OCP_MARGIN_K=0 OCP_ENVELOPE=0.30 \
    OCP_CONV_CAP=1.0 OCP_AGREE_Q=off \
    python optimal_control_main.py
```

It solves the problem with the linear reconstruction of all three models, a
temperature limit lowered by 20 K and the reduced states kept within 30 % of
their training range. Results are written to `control/results/ensemble/`.

## Not included

The mechanistic reactor model used for the plant-side verification of the
controller is not part of this repository.

## License

See `LICENSE`.
