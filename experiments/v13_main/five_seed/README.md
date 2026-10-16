# Five-seed repeat of the flagship protocol

## Why

Table 1 and Fig. 2 of the manuscript report the flagship protocol from a single run
(seed 42). Table 3 and Supplementary Table S1 report a smaller ablation configuration
(4 encoder layers, 8 heads, d_ff 256) repeated over five seeds, which is why their
absolute accuracies are a little lower. Running this folder gives the flagship number
with dispersion, so the headline result and the ablation can be quoted in the same
way.

## Run

```bash
python run_five_seed.py --seeds 42,43,44,45,46
```

On a cluster: `sbatch slurm_five_seed.sh` (edit the two `module`/`conda` lines first).
Each run takes a few minutes on 8 CPU threads; the five runs are executed in sequence,
so the whole job is roughly five times one run.

## What it does

`train_flagship.py` is the manuscript's flagship training script with two changes:
the random seed and the output prefix are command-line arguments, and the random
seed is applied to numpy and torch as well. Nothing else in the model, the data
handling, the split or the training loop was touched.

`run_five_seed.py` calls it once per seed and aggregates the per-seed JSON files.

## Outputs

| File | Contents |
|---|---|
| `seed<k>_mcr_former_best_model_results_with_aug.json` | per-seed test R2, RMSE, symmetry statistics |
| `five_seed_summary.json` | protocol record, per-seed values, mean and standard deviation |
| `five_seed_summary.csv` | the same table in CSV, ready to paste into the manuscript |

## How to use the result in the paper

Replace the single-run value in Table 1, in the Fig. 2b annotation and in Section 2.3
with `R2 = mean +- sd` over the five seeds, and state the seed range in Methods 4.2.
The protocol note in Methods already describes this configuration, so only the numbers
change. Keep the sentence that distinguishes the flagship protocol from the ablation
configuration.
