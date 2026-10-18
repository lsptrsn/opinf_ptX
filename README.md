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
  opinf_methanation_exp.py     identification and evaluation of one model
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

The hyperparameters were selected by an Optuna search; the search space and
the selected values are reported in the paper (Table C.1), and the values are
set in `configs.py`. The search script is not included.

Each seed runs in a fresh process with single-threaded numerics on the CPU.
Forecasts are written to `predictions/<case>/`.

The reported models were computed on macOS (Apple silicon) with the package
versions in `environment.yml`, which reproduce them bitwise on that platform.
Other versions or builds of NumPy and SciPy change the integrated trajectories
at the level of the solver tolerance (relative differences of order 1e-5 in our
tests). Training itself is unaffected.

## Not included

The mechanistic reactor model used for the plant-side verification of the
controller is not part of this repository.

## License

See `LICENSE`.
