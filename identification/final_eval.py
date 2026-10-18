"""Reported test errors of Case 1 and Case 2, from the saved forecasts.

    python final_eval.py case1
    python final_eval.py case2
    python final_eval.py case1 path/to/forecasts    (another folder)

Run after ``python run.py case1`` (or case2). For each seed set the estimator
is the mean of the three seeds with the lowest training rollout error.
Reported are the linear and the decoder reconstruction: their relative errors
(about the segment mean and about zero) and errors in physical units, on the
whole test segment, inside and outside the envelope of the training data,
with the feed on and off, and before and after the first extinction.

A test sample is outside the envelope when its hotspot lies above the hottest
or below the coldest hotspot of the training record, or when its input
combination lies farther than INPUT_DISTANCE from every input combination of
the training record (inputs scaled by their training maxima).
"""
import glob
import os
import sys

import numpy as np

import opinf
from opinf.post import field_errors, physical_errors
from configs import CONFIGS
from opinf_methanation_exp import replace_nodes

TOP_K = 3
INPUT_DISTANCE = 0.1
SEED_SETS = {"A": range(10, 20), "B": range(20, 30), "C": range(30, 40)}
# Fields reconstructed by the decoder; the flow of Case 2 stays linear.
DECODED_FIELDS = {"case1": ("F", "T"), "case2": ("T",)}
# Extinction: the fifth-hottest node drops by EXTINCTION_K within
# EXTINCTION_WINDOW samples while the feed is on.
EXTINCTION_K, EXTINCTION_WINDOW = 150.0, 60


def load_record(cfg):
    """Temperatures, inputs and time of the record, as the pipeline sees them."""
    p, exp = cfg["params"], cfg["exp"]
    defaults = opinf.parameters.Params()

    def load(name):
        return np.load(os.path.join(p["data_dir"],
                                    f"{name}{p['file_suffix']}.npy"))

    T = np.asarray(load("center_temperature"), dtype=float)
    if T.shape[0] > T.shape[1]:
        T = T.T
    T = replace_nodes(T, exp.get("BAD_TEMPERATURE_NODES", ()))
    U = np.vstack([np.asarray(load(n), dtype=float).ravel()
                   for n in p["input_names"]])
    t = np.asarray(load("time"), dtype=float).ravel()
    z = np.asarray(load("z"), dtype=float).ravel()
    sl = slice(p["time_to_leave_out"], None, defaults.step_t_sampling)
    n_train = int(T[:, sl].shape[1] * p.get("train_ratio",
                                             defaults.train_ratio))
    return T[:, sl], U[:, sl], t[sl], z, n_train


def ensemble(runs, decoded):
    """Mean of the TOP_K runs with the lowest training rollout error."""
    roll = np.array([float(r["rollout_200"]) for r in runs])
    best = [runs[i] for i in np.argsort(np.where(np.isfinite(roll), roll,
                                                 np.inf))[:TOP_K]]

    def field(r):
        sol = np.array(r["sol"], dtype=float)
        if decoded:
            n_F = r["F_true"].shape[0]
            dec = np.asarray(r["sol_decoder"], dtype=float)
            if "F" in decoded:
                sol[:n_F] = dec[:n_F]
            if "T" in decoded:
                sol[n_F:] = dec[n_F:]
        return sol

    return np.mean([field(r) for r in best], axis=0), [int(r["seed"])
                                                       for r in best]


def report(label, sol, F_true, T_true, z, cols):
    if not cols.any():
        print(f"    {label}: no samples")
        return
    s, Ft, Tt = sol[:, cols], F_true[:, cols], T_true[:, cols]
    e = field_errors(s, Ft, Tt)
    p = physical_errors(s, Ft, Tt, z)
    print(f"    {label} ({cols.sum()} samples): combined "
          f"{100 * e['combined_centred']:.1f} % about the mean "
          f"(F {100 * e['F_centred']:.1f}, T {100 * e['T_centred']:.1f}), "
          f"{100 * e['combined']:.2f} % about zero")
    print(f"        outlet flow RMSE {p['F_rmse']:.3g}, temperature RMSE "
          f"{p['T_rmse_K']:.1f} K, hotspot RMSE {p['hotspot_T_rmse_K']:.1f} K "
          f"(largest {p['hotspot_T_maxabs_K']:.1f} K), position RMSE "
          f"{p['hotspot_z_rmse_cm']:.2f} cm")


def main():
    case = sys.argv[1]
    cfg = CONFIGS[case]
    folder = (sys.argv[2] if len(sys.argv) > 2
              else cfg["exp"]["SAVE_PREDICTIONS_TO"])
    tag = cfg["exp"]["PREDICTION_TAG"]

    T_s, U_s, t_s, z, n_train = load_record(cfg)

    # Envelope of the training record.
    hot_train = np.nanmax(T_s[:, :n_train], axis=0)
    hot_max, hot_min = float(hot_train.max()), float(hot_train.min())
    scale = np.max(np.abs(U_s[:, :n_train]), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    U_train, U_test = (U_s[:, :n_train] / scale).T, (U_s[:, n_train:] / scale).T
    d_min = np.array([np.min(np.linalg.norm(U_train - u, axis=1))
                      for u in U_test])
    feed_on = U_s[0, n_train:] > 0.05 * np.nanmax(U_s[0, :n_train])

    runs = []
    for path in sorted(glob.glob(os.path.join(folder, f"{tag}_seed*.npz"))):
        with np.load(path) as f:
            runs.append({k: f[k] for k in f.files})
    if not runs:
        sys.exit(f"no forecasts {tag}_seed*.npz in {folder}")

    F_true = np.asarray(runs[0]["F_true"], dtype=float)
    T_true = np.asarray(runs[0]["T_true"], dtype=float)
    if T_true.shape[1] != U_test.shape[0]:
        sys.exit(f"{T_true.shape[1]} forecast samples against "
                 f"{U_test.shape[0]} test samples of the record")
    zz = z[1:] if z.size == T_true.shape[0] + 1 else z

    hot_true = T_true.max(axis=0)
    out_hot, out_cold = hot_true > hot_max, hot_true < hot_min
    out_inp = d_min > INPUT_DISTANCE
    outside = out_hot | out_cold | out_inp
    print(f"{case}: training hotspot {hot_min:.1f} to {hot_max:.1f} K, test "
          f"hotspot {hot_true.min():.1f} to {hot_true.max():.1f} K")
    print(f"  outside by hotspot: above {out_hot.mean():.1%}, below "
          f"{out_cold.mean():.1%}; by inputs {out_inp.mean():.1%} (distance "
          f"> {INPUT_DISTANCE}; 95th percentile "
          f"{np.percentile(d_min, 95):.3f}); either {outside.mean():.1%}")

    hot5 = np.sort(T_true, axis=0)[-5]
    w = EXTINCTION_WINDOW
    k = next((i for i in range(hot5.size - w)
              if hot5[i] - hot5[i + w] > EXTINCTION_K
              and feed_on[i:i + w].all()), None)
    dt_min = float(np.median(np.diff(t_s))) / 60.0
    print("  no extinction in the test segment" if k is None else
          f"  first extinction {k * dt_min:.0f} min into the test segment")

    subsets = [("whole test segment", np.ones(hot_true.size, dtype=bool)),
               ("inside the envelope", ~outside),
               ("outside the envelope", outside),
               ("feed on", feed_on), ("feed off", ~feed_on)]
    if k is not None:
        before = np.arange(hot_true.size) < k
        subsets += [("before the first extinction", before),
                    ("from the first extinction on", ~before)]

    for set_name, seeds in SEED_SETS.items():
        members = [r for r in runs if int(r["seed"]) in seeds]
        if not members:
            continue
        for decoded in ((), DECODED_FIELDS[case]):
            sol, best = ensemble(members, decoded)
            kind = (f"decoder for {' and '.join(decoded)}" if decoded
                    else "linear reconstruction")
            print(f"\nseed set {set_name} ({len(members)} seeds), ensemble "
                  f"of seeds {best}, {kind}")
            for label, cols in subsets:
                report(label, sol, F_true, T_true, zz, cols)


if __name__ == "__main__":
    main()
