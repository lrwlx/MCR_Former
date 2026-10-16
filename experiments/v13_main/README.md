# Flagship experiment — 16-ply nu13 prediction (Table 1, Fig. 2)

## What this folder contains

The flagship protocol of the manuscript: a single MCR-Former trained on 3,000 sixteen-ply
laminate sequences to predict the effective Poisson ratio nu13.

| File | Contents |
| --- | --- |
| `train_evaluate_v13_2.py` | training and evaluation script (the seed is fixed at 42 inside the config) |
| `composite_v13_dataset.csv` | 3,000 sequences, columns `angle_1 ... angle_16, v13` (Sobol sampling, classical lamination theory) |
| `generate_data.py` | regenerates that dataset (`d = 16` ply angles, 3,000 samples, Sobol seed 42); `get_v.py` computes the nu values |
| `MCR_Former.py` | local copy of the model used by the script |
| `mcr_former_best_model_results_with_aug.json` | archived metrics and full config of the reported run |
| `mcr_former_best_model_symmetry_validation_results_with_aug.xlsx` | per-sample predictions, forward and reversed |
| `mcr_former_best_model_training_history*.png/pdf`, `..._prediction_vs_actual*.png/pdf` | training curves and parity plot of the reported run |

## Protocol

3 encoder layers, 2 attention heads, `d_model = 64`, `d_ff = 512`, dropout 0.09, batch size 32,
Adam (lr 1e-3), 60 epochs, 80/20 train-test split (`random_state = 42`), reversal augmentation
on, strict symmetric fusion on. This is the configuration behind Table 1, Fig. 2 and the
surrogate metrics quoted for the manuscript's flagship model.

## Run

```bash
python train_evaluate_v13_2.py
```

The script writes `mcr_former_best_model_*.json/png/pdf/xlsx` and the checkpoint
`mcr_former_best_model_with_aug.pth` into the current folder. Change `random_state` in the
config to run another seed, or use `five_seed/` (below).

Smoke test (about 30 s):

```bash
python -c "import pandas as pd; print(pd.read_csv('composite_v13_dataset.csv').shape)"   # (3000, 17)
```

## Archived result

The run stored in this folder reports **test R^2 = 0.9863, RMSE = 0.0067**
(`mcr_former_best_model_results_with_aug.json`). The JSON also stores the exact config, so any
later run can be compared against it; small differences across PyTorch versions are expected.

## five_seed/ — the same protocol over five seeds

```
five_seed/train_flagship.py        flagship training with --seed on the command line
five_seed/run_five_seed.py         runs seeds 42-46 and aggregates mean +- sd
five_seed/slurm_five_seed.sh       cluster submission
five_seed/README.md                usage notes
```

```bash
cd five_seed
python run_five_seed.py --seeds 42,43,44,45,46
```