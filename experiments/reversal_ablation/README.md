# Reversal-invariance ablation — Table 2

## Question

Under the same MCR encoding `(sin 2theta, cos 2theta)`, how much does the *hard* reversal
constraint of MCR-Former buy compared with reversal augmentation or with no symmetry handling
at all? Three architectures (Transformer, LSTM, CNN) are compared under three symmetry modes:

| Mode | Meaning |
| --- | --- |
| `Hard` | architecture-enforced symmetry: shared-weight dual path + symmetric pooling `(E(X) + E(R(X))) / 2` |
| `Aug` | no architectural constraint; the training set is augmented with reversed sequences |
| `None` | no symmetry handling |

Reported quantities: test R^2 (mean +- s.d. over five repeats) and the mean symmetry violation
`|f(X) - f(R(X))|` over the test set (0 for `Hard` by construction).

## Run

```bash
# the script reads composite_v13_dataset.csv from the working directory
cp ../v13_main/composite_v13_dataset.csv .
python abaltion_reverse.py
```

Configuration: 16 plies, 80 epochs, batch 64, Adam (lr 1e-3), `d_model = 64`, 8 heads,
4 encoder layers, `d_ff = 256`, dropout 0.1, 80/20 split (`random_state = 42`),
5 repeats per (model, mode). The script takes roughly an hour on CPU.

## Outputs

| File | Contents |
| --- | --- |
| `ablation_reversal_detailed.xlsx` | every repeat: model, symmetry mode, seed, R^2, RMSE and symmetry statistics |
| `ablation_reversal_aggregated.xlsx` | the mean +- s.d. table used for Table 2 of the manuscript |

## Notes

- `BASE_SEED = 42` and `N_REPEATS = 5`. The per-repeat seed is
  `BASE_SEED + 10*rep + hash(model + mode) % 1000`; Python randomises `hash()` of strings per
  process, so set `PYTHONHASHSEED` to a fixed value if you want identical seeds across
  processes. The archived XLSX files are the reference output of the reported run.
- R^2 and delta statistics are computed on the same fixed test split for all repeats; only the
  model initialisation and training stochasticity change between repeats.