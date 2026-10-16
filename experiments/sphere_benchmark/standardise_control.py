#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input-scale control for Supplementary Table S2.

Question: does the advantage of the direction-cosine (MCR) embedding survive
when the raw angular inputs are standardised?

The control repeats the *raw* runs of ``sphere_benchmark.py`` with the angle
inputs z-scored on the training split; the training subsets come from the same
random stream as in the main script, so the two experiments see exactly the
same data.  The published Table S2 already contains the MCR column, which does
not change here.

Reported (five runs, ~15 min on a CPU):

    raw + standardised R^2
    100  samples  0.6544 +/- 0.0311
    200  samples  0.7568 +/- 0.0212
    500  samples  0.7369 +/- 0.0710
    1000 samples  0.7916 +/- 0.0372
    2000 samples  0.8033 +/- 0.0252

against 0.9143 +/- 0.0358 at 100 samples and 0.9920 +/- 0.0050 at 2000 samples
for the MCR.  Standardising removes much of the run-to-run spread of the raw
model but does not close the accuracy gap, so the advantage of the embedding is
not an artefact of the input scale.

Usage
-----
    python standardise_control.py
    python standardise_control.py --n-runs 2 --quick       # smoke test
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sphere_benchmark as sb  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--n-total", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sizes", type=int, nargs="+",
                        default=list(sb.TRAIN_SIZES))
    parser.add_argument("--quick", action="store_true",
                        help="60 epochs, for a smoke test only")
    parser.add_argument("--out", default="table_s2_standardised_inputs.md")
    args = parser.parse_args(argv)

    if args.quick:
        args.epochs = 60

    rng, data = sb.make_dataset(args.n_total, args.seed)
    pool = args.n_total - args.n_test
    x_test, y_test = data["raw"][pool:], data["y"][pool:]

    results = {n: [] for n in args.train_sizes}
    for run in range(1, args.n_runs + 1):
        torch.manual_seed(run)
        print("run %d/%d" % (run, args.n_runs))
        sys.stdout.flush()
        for n_train in args.train_sizes:
            idx = rng.choice(pool, n_train, replace=False)
            results[n_train].append(sb.train_and_eval(
                data["raw"][idx], data["y"][idx], x_test, y_test, 2,
                epochs=args.epochs, standardize=True))

    lines = ["| Training samples | Raw (theta, phi), standardised, R^2 |",
             "| :--- | :--- |"]
    for n_train in args.train_sizes:
        values = np.asarray(results[n_train])
        lines.append("| %d | %.4f ± %.4f |" % (n_train, values.mean(), values.std()))
        print("%5d | %.4f +/- %.4f" % (n_train, values.mean(), values.std()))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())