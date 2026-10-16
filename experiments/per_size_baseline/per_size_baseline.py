#!/usr/bin/env python
"""
Per-thickness (per-size) baseline for the data-efficiency claim.

For every laminate thickness N the conventional approach trains a separate model on
that thickness alone. This script trains those models over a grid of training-set
sizes and records the accuracy, so that two questions can be answered with data:

  1. threshold mode  - how many samples does each per-thickness model need to reach
                       R2 >= --threshold?  Summing those numbers gives the total budget
                       the conventional approach needs, to be compared with the unified
                       model's budget of 200 samples per layer.
  2. equal-budget mode - at the same budget per thickness as the unified model, how
                       accurate is a per-thickness model?

Training recipe, encoding and model class are the ones used for the unified
variable-length model, so that the comparison isolates the modelling choice rather
than the architecture or the optimiser.

Usage
-----
    python per_size_baseline.py --layers 3-20 --budgets 25,50,100,150,200,300,400 --seeds 3
    python per_size_baseline.py --layers 16 --budgets 25,50,100 --seeds 1 --epochs 5   # smoke test

Outputs (written next to this script, or to --out-dir)
-----------------------------------------------------
    per_size_results.csv        long table: layer, budget, seed, test R2, test RMSE
    per_size_thresholds.csv     smallest budget reaching the threshold, per layer
    per_size_summary.json       totals, comparison with the unified budget, run record
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))


def add_src_to_path():
    """Find the shared model definitions (src/MCR_Former.py) walking upwards."""
    d = HERE
    for _ in range(4):
        cand = os.path.join(d, "src")
        if os.path.isfile(os.path.join(cand, "MCR_Former.py")):
            sys.path.insert(0, cand)
            return cand
        if os.path.isfile(os.path.join(d, "MCR_Former.py")):
            sys.path.insert(0, d)
            return d
        d = os.path.dirname(d)
    sys.path.insert(0, HERE)
    return HERE


add_src_to_path()
from MCR_Former import MCR_Former  # noqa: E402


def parse_layers(spec):
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            layers.extend(range(int(a), int(b) + 1))
        elif part:
            layers.append(int(part))
    return sorted(set(layers))


def encode(angles):
    """Minimal complete representation: (sin 2t, cos 2t) per ply."""
    rad = np.radians(angles.astype(np.float64))
    feats = np.zeros((angles.shape[0], angles.shape[1], 2), dtype=np.float32)
    feats[:, :, 0] = np.sin(2 * rad)
    feats[:, :, 1] = np.cos(2 * rad)
    return feats


def train_one(x_tr, y_tr, x_te, y_te, cfg, seed, device):
    """Train one per-thickness model and return (r2, rmse, seconds)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MCR_Former(input_dim=2, d_model=cfg.d_model, nhead=cfg.heads,
                       num_layers=cfg.layers, dim_feedforward=cfg.d_ff,
                       dropout=cfg.dropout, max_seq_len=cfg.max_seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    crit = nn.MSELoss()

    # standardise the target on the training split only (the model returns a 1-D batch)
    mu, sd = float(y_tr.mean()), float(y_tr.std() + 1e-12)
    ytr = torch.tensor((y_tr - mu) / sd, dtype=torch.float32)
    xtr = torch.tensor(x_tr, dtype=torch.float32)
    lengths = torch.full((len(x_tr),), x_tr.shape[1], dtype=torch.int64)

    # a small validation slice for early stopping, carved out of the training set
    n_val = max(1, int(0.1 * len(xtr)))
    perm = torch.randperm(len(xtr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    best, best_state, patience = float("inf"), None, 0
    t0 = time.time()
    for _epoch in range(cfg.epochs):
        model.train()
        pred = model(xtr[tr_idx], lengths[tr_idx])
        loss = crit(pred, ytr[tr_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(xtr[val_idx], lengths[val_idx]), ytr[val_idx]).item()
        sched.step(vl)
        if vl < best - 1e-6:
            best, patience = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(x_te, dtype=torch.float32),
                  torch.full((len(x_te),), x_te.shape[1], dtype=torch.int64)).detach().cpu().numpy()
    p = np.asarray(p).reshape(-1) * sd + mu
    return r2_score(y_te, p), math.sqrt(mean_squared_error(y_te, p)), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="3-20", help="thicknesses, e.g. 3-20 or 8,16,20")
    ap.add_argument("--budgets", default="25,50,100,150,200,300,400",
                    help="training-set sizes to sweep per thickness (max 400 for a 500-sample pool)")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds per (layer, budget)")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--data-dir",
                    default=os.path.join(HERE, "..", "cross_scale", "data"),
                    help="folder holding layer_<N>.csv (angle_1..angle_N + v13)")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8,
                    help="match the unified variable-length model for a fair comparison")
    ap.add_argument("--n-encoder-layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=30)
    args = ap.parse_args()

    thicknesses = parse_layers(args.layers)
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    cfg = argparse.Namespace(d_model=args.d_model, heads=args.heads, layers=args.n_encoder_layers,
                             d_ff=args.d_ff, dropout=args.dropout, max_seq_len=args.max_seq_len,
                             lr=args.lr, epochs=args.epochs, patience=args.patience)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    print("device:", device)
    print("thicknesses:", thicknesses)
    print("budgets:", budgets, "seeds per cell:", args.seeds, "threshold:", args.threshold)

    rows, skipped = [], []
    for layer in thicknesses:
        path = os.path.join(args.data_dir, "layer_%d.csv" % layer)
        if not os.path.exists(path):
            skipped.append(layer)
            print("missing %s, skipping thickness %d" % (path, layer))
            continue
        df = pd.read_csv(path)
        angle_cols = [c for c in df.columns if c.lower().startswith("angle")]
        angles = df[angle_cols].to_numpy(dtype=np.float64)[:, :layer]
        y = df["v13"].to_numpy(dtype=np.float64)
        feats = encode(angles)

        for seed in range(args.seeds):
            # one fixed split per (thickness, seed) so budgets form a nested learning curve
            idx = rng.permutation(len(y))
            n_test = int(round(args.test_size * len(y)))
            test_idx, pool_idx = idx[:n_test], idx[n_test:]
            for budget in budgets:
                if budget > len(pool_idx):
                    continue
                tr_idx = pool_idx[:budget]
                r2, rmse, secs = train_one(feats[tr_idx], y[tr_idx], feats[test_idx], y[test_idx],
                                           cfg, seed, device)
                rows.append({"layer": layer, "budget": budget, "seed": seed,
                             "test_r2": r2, "test_rmse": rmse, "seconds": round(secs, 1)})
                print("  layer %2d  budget %3d  seed %d  R2 = %.4f  RMSE = %.5f  (%.0f s)"
                      % (layer, budget, seed, r2, rmse, secs), flush=True)

    if not rows:
        print("no runs completed")
        return 1

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "per_size_results.csv"), index=False)
    agg = (res.groupby(["layer", "budget"])["test_r2"]
              .agg(["mean", "std", "count"]).reset_index()
              .rename(columns={"mean": "r2_mean", "std": "r2_sd", "count": "n_seeds"}))

    thresholds = []
    for layer, grp in agg.groupby("layer"):
        ok = grp[grp["r2_mean"] >= args.threshold].sort_values("budget")
        thresholds.append({
            "layer": int(layer),
            "samples_needed": int(ok.iloc[0]["budget"]) if len(ok) else None,
            "r2_at_threshold": float(ok.iloc[0]["r2_mean"]) if len(ok) else None,
            "r2_at_max_budget": float(grp.sort_values("budget").iloc[-1]["r2_mean"]),
        })
    thr = pd.DataFrame(thresholds)
    thr.to_csv(os.path.join(args.out_dir, "per_size_thresholds.csv"), index=False)

    solved = thr.dropna(subset=["samples_needed"])
    per_size_total = int(solved["samples_needed"].sum())
    n_unified_layers = len(thr)
    unified_budget = 200 * n_unified_layers
    equal = agg[agg["budget"] == 200]
    summary = {
        "protocol": {
            "thicknesses": thicknesses, "budgets": budgets, "seeds": args.seeds,
            "threshold": args.threshold, "test_size": args.test_size,
            "encoder": "sin(2t), cos(2t)", "epochs_cap": args.epochs,
            "architecture": {"d_model": cfg.d_model, "heads": cfg.heads,
                             "encoder_layers": cfg.layers, "d_ff": cfg.d_ff,
                             "dropout": cfg.dropout},
        },
        "layers_solved": int(len(solved)),
        "layers_without_solution": [int(x) for x in thr[thr["samples_needed"].isna()]["layer"]],
        "samples_needed_sum": per_size_total,
        "n_layers": n_unified_layers,
        "unified_model_budget": unified_budget,
        "ratio_unified_over_per_size": (unified_budget / per_size_total) if per_size_total else None,
        "mean_r2_at_200_per_layer": float(equal["r2_mean"].mean()) if len(equal) else None,
        "missing_data_layers": skipped,
    }
    json.dump(summary, open(os.path.join(args.out_dir, "per_size_summary.json"), "w"), indent=1)

    print()
    print("=" * 68)
    print("per-size baseline summary")
    print("  per-thickness samples needed to reach R2 >= %.2f: %s" % (args.threshold, per_size_total))
    print("  unified model budget for the same layers (200 x %d): %d" % (n_unified_layers, unified_budget))
    if per_size_total:
        print("  ratio unified / per-size: %.2f" % (unified_budget / per_size_total))
    if len(equal):
        print("  mean R2 of per-thickness models at 200 samples/layer: %.4f" % equal["r2_mean"].mean())
    print("  outputs: per_size_results.csv, per_size_thresholds.csv, per_size_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
