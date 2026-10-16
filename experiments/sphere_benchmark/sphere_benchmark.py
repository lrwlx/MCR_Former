#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary section S4: the 2D -> 3D minimal complete representation (MCR)
on a spherical parameter manifold.  This script reproduces Table S2.

Task
----
Predict the directional Young's modulus E(theta, phi) of a cubic single crystal
from the two spherical angles of the loading direction, and compare two input
parameterisations under an otherwise identical protocol:

    raw : (theta, phi)   the angles themselves, 2 inputs
    MCR : (l, m, n)      the direction cosines, 3 inputs (Proposition 2)

The target is evaluated in closed form, so the dataset carries no numerical
noise and the two learning curves differ only in how the same function is
presented to the model.

Physics
-------
For a cubic crystal with compliances S11, S12, S44,

    1 / E = S11 - 2 (S11 - S12 - S44 / 2) (l^2 m^2 + m^2 n^2 + n^2 l^2),

    l = sin(theta) cos(phi),  m = sin(theta) sin(phi),  n = cos(theta).

The benchmark uses the representative dimensionless compliances
(S11, S12, S44) = (1.0, 0.3, 2.0), that is, a Zener anisotropy ratio
A = 2 (S11 - S12) / S44 = 0.7.  ``--selftest`` checks the closed form against
its [100] and [111] special cases and against the moduli quoted for Al, Cu, W
and Si.

Protocol (identical for the two parameterisations)
--------------------------------------------------
* 3000 directions drawn uniformly in (theta, phi); the last 500 of them form
  the held-out test set and are never used for training.
* training sets of 100, 200, 500, 1000 and 2000 directions; both
  parameterisations see exactly the same indices in every run.
* one 2 x 64 x 64 x 1 tanh MLP per run, Adam (lr = 0.01), batch size 32,
  800 epochs, mean-squared error.
* five runs; run r calls torch.manual_seed(r) before its first model is built,
  which fixes the weight initialisation and the mini-batch order.
* every random draw comes from numpy's legacy stream, seeded once with 42: the
  3000 directions first, then the training subsets in the order
  run 1 (100, 200, 500, 1000, 2000), run 2 (...), and so on.  This is the scheme
  behind Table S2, so running this script unchanged regenerates the table.

Output
------
Mean +/- standard deviation of the test R^2 over the runs, printed as a
markdown table and written to --out.

Usage
-----
    python sphere_benchmark.py                 # Table S2, ~30 min on a CPU
    python sphere_benchmark.py --quick         # smoke test, 60 epochs
    python sphere_benchmark.py --selftest      # closed form vs literature
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# representative dimensionless compliances used for Table S2
S11, S12, S44 = 1.0, 0.3, 2.0

TRAIN_SIZES = (100, 200, 500, 1000, 2000)


# ---------------------------------------------------------------------------
# target function
# ---------------------------------------------------------------------------
def direction_cosines(theta, phi):
    """Direction cosines (l, m, n) of the direction (theta, phi)."""
    return (np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta))


def directional_modulus(theta, phi, s11=S11, s12=S12, s44=S44):
    """Closed-form Young's modulus along (theta, phi) of a cubic crystal."""
    l, m, n = direction_cosines(theta, phi)
    inv_e = s11 - 2.0 * (s11 - s12 - 0.5 * s44) * (
        l ** 2 * m ** 2 + m ** 2 * n ** 2 + n ** 2 * l ** 2)
    return 1.0 / inv_e


def selftest(verbose=True):
    """Check the closed form against [100]/[111] and against four crystals.

    The compliances are the room-temperature values in 1/TPa, the moduli are
    the values usually quoted for these crystals in GPa.
    """
    literature = {
        "Al": ((15.98, -5.78, 35.24), (63.2, 75.6)),
        "Cu": ((14.98, -6.28, 13.22), (66.7, 191.1)),
        "W": ((2.60, -0.75, 6.81), (384.6, 379.0)),
        "Si": ((7.68, -2.14, 12.60), (130.2, 187.5)),
    }
    ok = True
    if verbose:
        print("closed form against quoted moduli (GPa)")
        print("  %-4s %8s %8s %8s %8s" % ("", "E[100]", "*", "E[111]", "*"))
    for name, ((s11, s12, s44), (e100, e111)) in literature.items():
        got100 = 1000.0 * float(directional_modulus(np.pi / 2.0, 0.0, s11, s12, s44))
        got111 = 1000.0 * float(directional_modulus(
            np.arccos(1.0 / np.sqrt(3.0)), np.pi / 4.0, s11, s12, s44))
        if verbose:
            print("  %-4s %8.1f %8.1f %8.1f %8.1f"
                  % (name, got100, e100, got111, e111))
        ok = ok and abs(got100 / e100 - 1.0) < 0.02 and abs(got111 / e111 - 1.0) < 0.02

    # the special cases of the closed form itself
    ok = ok and abs(float(directional_modulus(np.pi / 2.0, 0.0)) - 1.0 / S11) < 1e-12
    ok = ok and abs(float(directional_modulus(np.arccos(1.0 / np.sqrt(3.0)),
                                              np.pi / 4.0))
                    - 3.0 / (S11 + 2.0 * S12 + S44)) < 1e-12
    if verbose:
        print("[100] and [111] special cases: %s" % ("passed" if ok else "FAILED"))
    return ok


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def make_dataset(n_total, seed):
    """Directions, both parameterisations and the target.

    Returns the random state as well, because the training subsets are drawn
    from the same stream (this reproduces the published table).
    """
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0.0, np.pi, n_total)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_total)
    l, m, n = direction_cosines(theta, phi)
    data = {
        "raw": np.column_stack((theta, phi)),
        "mcr": np.column_stack((l, m, n)),
        "y": directional_modulus(theta, phi).reshape(-1, 1),
    }
    return rng, data


# ---------------------------------------------------------------------------
# model and training
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    """2/3 -> 64 -> 64 -> 1, tanh; identical for both parameterisations."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def train_and_eval(x_train, y_train, x_test, y_test, input_dim,
                   epochs=800, lr=0.01, batch_size=32, standardize=False):
    """Train one model from scratch and return the test R^2."""
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    if standardize:                       # optional robustness variant
        mu = x_train.mean(axis=0, keepdims=True)
        sd = x_train.std(axis=0, keepdims=True)
        sd[sd == 0.0] = 1.0
        x_train = (x_train - mu) / sd
        x_test = (x_test - mu) / sd

    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                      torch.tensor(np.asarray(y_train, dtype=np.float32))),
        batch_size=batch_size, shuffle=True)
    model = SimpleMLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float32)).numpy()
    return r2(y_test, pred)


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------
def run_benchmark(train_sizes=TRAIN_SIZES, n_runs=5, n_total=3000, n_test=500,
                  epochs=800, lr=0.01, batch_size=32, seed=42,
                  standardize=False, verbose=True):
    rng, data = make_dataset(n_total, seed)
    pool = n_total - n_test
    test = {k: v[pool:] for k, v in data.items()}

    results = {enc: {n: [] for n in train_sizes} for enc in ("raw", "mcr")}
    for run in range(1, n_runs + 1):
        torch.manual_seed(run)            # weight init and mini-batch order
        if verbose:
            print("run %d/%d" % (run, n_runs))
            sys.stdout.flush()
        for n_train in train_sizes:
            idx = rng.choice(pool, n_train, replace=False)
            for enc, dim in (("raw", 2), ("mcr", 3)):
                results[enc][n_train].append(train_and_eval(
                    data[enc][idx], data["y"][idx], test[enc], test["y"], dim,
                    epochs=epochs, lr=lr, batch_size=batch_size,
                    standardize=standardize))
    return results


def format_table(results):
    lines = ["| Training samples | Raw (theta, phi) R^2 | MCR (l, m, n) R^2 |",
             "| :--- | :--- | :--- |"]
    for n in sorted(results["raw"]):
        raw = np.asarray(results["raw"][n])
        mcr = np.asarray(results["mcr"][n])
        lines.append("| %d | %.4f ± %.4f | %.4f ± %.4f |"
                     % (n, raw.mean(), raw.std(), mcr.mean(), mcr.std()))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-total", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sizes", type=int, nargs="+",
                        default=list(TRAIN_SIZES))
    parser.add_argument("--standardize-inputs", action="store_true",
                        help="z-score both parameterisations on the training "
                             "split (optional robustness variant, not used for "
                             "Table S2)")
    parser.add_argument("--selftest", action="store_true",
                        help="check the closed form and exit")
    parser.add_argument("--quick", action="store_true",
                        help="60 epochs, for a smoke test only")
    parser.add_argument("--out", default=None, help="write the table here")
    args = parser.parse_args(argv)

    if args.selftest:
        return 0 if selftest() else 1
    if args.quick:
        args.epochs = 60
    if not selftest(verbose=False):
        print("the closed form failed the self-test", file=sys.stderr)
        return 1

    print("comparing %d runs per training-set size on %d held-out directions "
          "(epochs = %d%s)"
          % (args.n_runs, args.n_test, args.epochs,
             ", inputs standardised" if args.standardize_inputs else ""))
    results = run_benchmark(train_sizes=tuple(args.train_sizes),
                            n_runs=args.n_runs, n_total=args.n_total,
                            n_test=args.n_test, epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, seed=args.seed,
                            standardize=args.standardize_inputs)
    table = format_table(results)
    print("\n" + table + "\n")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())