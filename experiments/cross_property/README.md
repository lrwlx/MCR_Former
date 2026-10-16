# Cross-property transfer (nu13 -> nu12) — Table 4 and Fig. 5b

## Question

The model pre-trained on the nu13 task is fine-tuned on a *different* effective property, the
in-plane Poisson ratio nu12, using 1 % to 100 % of the available training pool. How much data
does the transfer need before it beats training the same architecture from scratch?
Table 4 and Fig. 5b answer this; the crossover sits at 5 % of the data (20 samples).

## Data

`data_v12/v12_dataset.csv`: 500 sixteen-ply sequences. Columns are `angle_1 ... angle_16` plus
one target column whose header is still named `v13` for historical reasons — the script reads
the last column as nu12.

## Scripts

| Script | Purpose | Writes |
| --- | --- | --- |
| `transfer_v12_full_finetune2.py` | the protocol behind Table 4: fixed 20 % test set (100 samples), data fractions `[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]`, five seeds per fraction, fixed 100 fine-tuning epochs, no validation split | `results_v12/` (JSON with per-seed and aggregated values), `models_v12/` |
| `transfer_v12_full_finetune.py` | first version of the same study (with validation-based early stopping), kept for reference | `results_v12/`, `models_v12/` |

The transformed model is the variable-length `MCR_Former_Concat_2D`; the pre-trained weights
are loaded from

```
models/mcr_former_concat_v13_base.pth
```

(same base model as in `../few_shot/`; see that README for how to produce it). Trained weights
are not tracked in git.

## Run

```bash
mkdir -p models
cp /path/to/mcr_former_concat_v13_base.pth models/
python transfer_v12_full_finetune2.py
```

Fine-tuning uses layer-wise learning rates (embedding 0.1x, early encoder layers 0.3x, late
layers 0.7x, fusion and head 1x of lr 1e-3) for a fixed 100 epochs, so that transfer and
from-scratch runs see exactly the same optimisation budget. The test set is fixed once and
never touched during training.

## Where the manuscript numbers are

Re-running the script writes the per-seed and mean +- s.d. values into `results_v12/`. The
Table 4 rows and the Fig. 5b curves are transcribed in `../figures/Fig5/Fig5.py`, the plotting
script used for the figure.