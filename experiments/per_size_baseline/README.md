# Per-thickness baseline (supports the data-efficiency claim)

## What question this answers

The unified model trains once on layers 1-20 at 200 samples per layer. The conventional
approach trains one model per thickness. To state a data saving honestly we need the
second half of that comparison, which is what this folder produces:

- **threshold mode** - for each thickness, how many training samples does a dedicated
  model need before it reaches R2 >= 0.95? Summing those numbers gives the total budget
  of the per-size approach, to be compared with 200 x (number of layers) for the unified
  model.
- **equal-budget mode** - at 200 samples per layer, how accurate is a per-thickness
  model? The archived `per_size_equal_budget_summary.json` reports that mean from the
  same sweep: per-thickness R2 = 0.658 +- 0.137 across layers against 0.951 (min 0.889)
  for the unified model at the same 4,000-sequence budget.

## Run

```bash
python per_size_baseline.py --layers 3-20 --budgets 25,50,100,150,200,300,400 --seeds 3
```

Cluster: `sbatch slurm_per_size.sh` (edit the two module/conda lines first), or use
`run_per_size_parallel.sh` to run one process per thickness and `merge_per_size.py` to merge
the per-layer outputs afterwards.

Smoke test that finishes in under a minute:

```bash
python per_size_baseline.py --layers 16 --budgets 25 --seeds 1 --epochs 3
```

## Fairness rules this script follows

- Model class, encoding `(sin 2t, cos 2t)`, optimiser, learning rate, weight decay and
  early-stopping rule are the same as for the unified model; only the number of plies the
  model sees changes. Pass `--heads`, `--n-encoder-layers`, `--d-ff`, `--dropout` if your
  unified run used a different configuration, so the comparison stays matched.
- For each (thickness, seed) the test split is drawn once and the budgets are nested
  subsets of the remaining pool, so the sweep is a proper learning curve rather than
  independent runs.
- The target is standardised on the training split only, and a 10 % validation slice of
  the training set drives early stopping, so the test set is never used for selection.

## Defaults and assumptions

- Data: `../cross_scale/data/layer_<N>.csv` (500 samples per thickness, columns
  `angle_1 ... angle_N, v13`). Override with `--data-dir` if you keep the per-thickness
  pools elsewhere; a thickness whose file is missing is skipped and listed in the summary.
- Budgets above 400 are skipped automatically for a 500-sample pool (20 % is held out for
  testing).
- 300 epochs is a cap, not a target; early stopping with patience 30 usually stops far
  earlier, which is why a single 8-core job can cover the whole grid.

## Outputs

| File | Contents |
|---|---|
| `per_size_results.csv` | every run: layer, budget, seed, test R2, test RMSE, seconds (written per layer into `out/layer_<N>/` when using `run_per_size_parallel.sh`) |
| `per_size_thresholds.csv` | per thickness: samples needed for the threshold, R2 at that point, R2 at the largest budget |
| `per_size_summary.json` | protocol record, summed budget, unified budget for the same layers, ratio, mean R2 at 200 samples/layer |
| `per_size_equal_budget_summary.json`, `per_size_equal_budget.csv` | the archived equal-budget comparison at 200 samples per layer (3 seeds, layers 1-20): per-thickness 0.658 +- 0.137 vs unified 0.951 |

The `per_size_results.csv`, `per_size_thresholds.csv` and `per_size_summary.json` files are
regenerated when the sweep is run (they are git-ignored run outputs); the archived
`per_size_equal_budget_*` files are the tables quoted in the manuscript.

## How to use the result in the paper

Tables S3 and S4 report the equal-budget comparison and the threshold sample counts;
`per_size_equal_budget_summary.json` and the threshold table regenerate both. If some
thicknesses never reach the threshold, say so and report the largest budget tested rather
than dropping them: `layers_without_solution` in the summary lists them.