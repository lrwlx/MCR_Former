#!/usr/bin/env python
"""
Run the flagship protocol with several random seeds and aggregate the result.

This produces the multi-seed version of the headline number reported in Table 1
and Fig. 2 of the manuscript. Each seed changes only the random seed of the
flagship configuration (3 encoder layers, 2 attention heads, d_ff = 512,
dropout 0.09, 80/20 split, 60 epochs, reversal augmentation).

Usage
-----
    python run_five_seed.py --seeds 42,43,44,45,46

Outputs
-------
    seed<k>_mcr_former_best_model_results_with_aug.json   per-seed results
    five_seed_summary.json                                mean and sd
    five_seed_summary.csv                                 same, tabular
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--data", default="composite_v13_dataset.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--train-script", default="train_flagship.py")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed seed")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows, failures = [], []
    t_all = time.time()

    for seed in seeds:
        t0 = time.time()
        print("=" * 60)
        print("seed %d" % seed, flush=True)
        cmd = [sys.executable, os.path.join(HERE, args.train_script),
               "--seed", str(seed), "--data", args.data, "--epochs", str(args.epochs)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.time() - t0
        fn = os.path.join(HERE, "seed%d_mcr_former_best_model_results_with_aug.json" % seed)
        if os.path.exists(fn):
            d = json.load(open(fn, encoding="utf-8"))
            rows.append({"seed": seed, "r2": d["test_r2"], "rmse": d["test_rmse"], "seconds": round(dt, 1)})
            print("seed %d: R2 = %.6f, RMSE = %.6f  (%.0f s)" % (seed, d["test_r2"], d["test_rmse"], dt), flush=True)
        else:
            failures.append(seed)
            print("seed %d failed (no result file). Last output:" % seed, flush=True)
            print((r.stdout or "")[-600:], flush=True)
            print((r.stderr or "")[-600:], flush=True)
            if not args.keep_going:
                break

    if not rows:
        print("no successful runs; nothing to aggregate")
        return 1

    r2 = [x["r2"] for x in rows]
    rm = [x["rmse"] for x in rows]
    summary = {
        "protocol": {
            "layers": 3, "heads": 2, "d_model": 64, "d_ff": 512, "dropout": 0.09,
            "split": "80/20", "epochs": args.epochs, "augmentation": True,
            "data": args.data,
        },
        "seeds": rows,
        "failed_seeds": failures,
        "r2_mean": st.mean(r2),
        "r2_sd": st.stdev(r2) if len(r2) > 1 else 0.0,
        "rmse_mean": st.mean(rm),
        "rmse_sd": st.stdev(rm) if len(rm) > 1 else 0.0,
        "n_seeds": len(rows),
        "total_seconds": round(time.time() - t_all, 1),
    }
    json.dump(summary, open(os.path.join(HERE, "five_seed_summary.json"), "w"), indent=1)
    with open(os.path.join(HERE, "five_seed_summary.csv"), "w") as fh:
        fh.write("seed,test_r2,test_rmse,seconds\n")
        for x in rows:
            fh.write("%d,%.6f,%.6f,%.1f\n" % (x["seed"], x["r2"], x["rmse"], x["seconds"]))
        fh.write("mean,%.6f,%.6f,\n" % (summary["r2_mean"], summary["rmse_mean"]))
        fh.write("sd,%.6f,%.6f,\n" % (summary["r2_sd"], summary["rmse_sd"]))

    print()
    print("=" * 60)
    print("five-seed flagship result over %d runs" % len(rows))
    print("  R2   = %.4f +- %.4f" % (summary["r2_mean"], summary["r2_sd"]))
    print("  RMSE = %.4f +- %.4f" % (summary["rmse_mean"], summary["rmse_sd"]))
    if failures:
        print("  failed seeds:", failures)
    print("  wrote five_seed_summary.json / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
