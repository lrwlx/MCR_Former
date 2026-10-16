# Running the revision experiments on a Linux cluster

This file describes how to run the two extra experiments added for the revision — the
five-seed repeat of the flagship protocol and the per-thickness (per-size) baseline — on a
Linux server, using the H100 node or any multi-core CPU machine.

## One-shot run

Copy the repository to the server and run

```bash
cd <folder containing run_all_h100.sh>
USE_CUDA_TORCH=1 JOBS=8 bash run_all_h100.sh
```

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `USE_CUDA_TORCH` | `0` | `1` installs the CUDA build of PyTorch; otherwise the CPU build is installed (the model is small and CPU is sufficient) |
| `JOBS` | `8` | parallel processes for the per-thickness sweep |
| `ENVDIR` | `~/.venvs/mcrformer` | virtual environment used for the run |

The script

1. creates the virtual environment and installs `requirements.txt`;
2. runs three smoke tests (per-thickness baseline, flagship training, sphere benchmark self-test);
3. starts the five-seed sweep in the background, writing `experiments/v13_main/five_seed/five_seed.log`;
4. starts the per-thickness sweep in the background, writing `experiments/per_size_baseline/per_size.log`.

## Watching progress

```bash
tail -f experiments/v13_main/five_seed/five_seed.log
tail -f experiments/per_size_baseline/per_size.log
tail -f experiments/per_size_baseline/out/layer_16.log   # one thickness
nvidia-smi
```

## Results to collect

```
experiments/v13_main/five_seed/five_seed_summary.json   # mean +- sd over the five seeds
experiments/v13_main/five_seed/five_seed_summary.csv
experiments/per_size_baseline/per_size_summary.json     # per-thickness budgets
experiments/per_size_baseline/per_size_thresholds.csv
experiments/per_size_baseline/per_size_results.csv
```

Expected runtime: the five-seed sweep runs five full flagship trainings in sequence (each a
few minutes on a GPU, longer on CPU); the per-thickness sweep is 18 thicknesses x 7 budgets
x 3 seeds — a few seconds to a minute per training, about ten to thirty minutes with
`JOBS=8`.

## Running the two experiments separately

```bash
# five-seed repeat of the flagship protocol
cd experiments/v13_main/five_seed
python run_five_seed.py --seeds 42,43,44,45,46
# or on Slurm (edit the module/conda lines first)
sbatch slurm_five_seed.sh
```

```bash
# per-thickness baseline
cd experiments/per_size_baseline
python per_size_baseline.py --layers 3-20 --budgets 25,50,100,150,200,300,400 --seeds 3
# or on Slurm
sbatch slurm_per_size.sh
```

## Reproducibility notes

All scripts set their random seeds explicitly, and the summaries written to disk record the
protocol used. Small differences in the last digits can appear across PyTorch/NumPy versions;
re-run the smoke tests first to confirm the environment, and quote the mean ± standard
deviation from `five_seed_summary.json` rather than any single run.