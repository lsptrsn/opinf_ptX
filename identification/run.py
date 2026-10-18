"""Train the final models of the paper.

    python run.py case1                 all seeds of Case 1, one process each
    python run.py case2 --seeds 10-19   a range of seeds
    python run.py synthetic_control     the three models of the control study

Run from this directory. Every seed runs in a fresh process, so that no
setting of one run carries over into the next.
"""
import argparse
import os
import subprocess
import sys

# Single-threaded numerics, set before numpy or torch is imported: with
# several BLAS threads the order of summations, and with it the result,
# depends on the machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("OPINF_DEVICE", "cpu")

from configs import CONFIGS, SEEDS  # noqa: E402


def apply(config):
    """Set the configured values, refusing names that do not exist."""
    import opinf
    import opinf_methanation_exp as exp
    targets = {
        "params": opinf.parameters.Params(),
        "model_inference": opinf.training._model_inference,
        "nonparametric": opinf.models._nonparametric,
        "exp": exp,
    }
    for group, values in config.items():
        target = targets[group]
        for name, value in values.items():
            if group != "params" and not hasattr(target, name):
                raise AttributeError(f"{group}.{name} does not exist")
            setattr(target, name, value)
    return exp, targets["params"]


def run_one(case, seed):
    exp, params = apply(CONFIGS[case])
    error = exp.main(params, seed=seed)
    print(f"{case} seed {seed}: test error {error:.4%}")


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        seeds += range(int(a), int(b or a) + 1)
    return seeds


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case", choices=sorted(CONFIGS))
    ap.add_argument("--seeds", help="e.g. 10-19 or 10,12,15")
    ap.add_argument("--single", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single is not None:
        run_one(args.case, args.single)
        sys.exit(0)
    seeds = parse_seeds(args.seeds) if args.seeds else SEEDS[args.case]
    for seed in seeds:
        subprocess.run([sys.executable, __file__, args.case,
                        "--single", str(seed)], check=True)
