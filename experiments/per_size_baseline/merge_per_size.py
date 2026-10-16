#!/usr/bin/env python
"""Merge the per-thickness outputs produced by run_per_size_parallel.sh."""
import glob
import json

import pandas as pd


def main():
    frames = [pd.read_csv(p) for p in sorted(glob.glob("out/layer_*/per_size_results.csv"))]
    if not frames:
        raise SystemExit("no per-layer results found under out/")

    res = pd.concat(frames, ignore_index=True).sort_values(["layer", "budget", "seed"])
    res.to_csv("per_size_results.csv", index=False)

    thr = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("out/layer_*/per_size_thresholds.csv"))],
                    ignore_index=True).sort_values("layer")
    thr.to_csv("per_size_thresholds.csv", index=False)

    solved = thr.dropna(subset=["samples_needed"])
    n_layers = int(len(thr))
    unified = 200 * n_layers
    at200 = res[res["budget"] == 200]["test_r2"]
    summary = {
        "n_layers": n_layers,
        "layers": [int(x) for x in thr["layer"]],
        "samples_needed_sum": int(solved["samples_needed"].sum()),
        "layers_without_solution": [int(x) for x in thr[thr["samples_needed"].isna()]["layer"]],
        "unified_model_budget": unified,
        "ratio_unified_over_per_size": (unified / float(solved["samples_needed"].sum())) if len(solved) else None,
        "mean_r2_at_200_per_layer": float(at200.mean()) if len(at200) else None,
    }
    json.dump(summary, open("per_size_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote per_size_results.csv, per_size_thresholds.csv, per_size_summary.json")


if __name__ == "__main__":
    main()
