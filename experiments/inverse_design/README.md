# Inverse design — Table 5 and Section 2.8

## Question

How close can a data-driven surrogate get to the theoretical optimum of the nu13 minimisation
problem, and what does the MCR encoding contribute? A differential-evolution (DE) search runs
on the surrogate's prediction, and the best candidates are re-evaluated with the exact
classical-lamination-theory formula. Table 5 reports, for the 8-ply design, the MCR-Former
surrogate and four baseline surrogates (SVR / MLP / LSTM, each with and without MCR), plus the
theoretical optimum quoted from the referenced theoretical study.

## Folder layout

| Folder | Design | Dataset |
| --- | --- | --- |
| `8Layer/` | 8-ply laminate; the search behind the main row of Table 5 | `composite_v13_dataset_8.csv` (2,000 sequences) |
| `10Layer/` | 10-ply design, same workflow | `composite_v13_dataset_10.csv` (10,000 sequences) |
| `16Layer/` | 16-ply design, same workflow | `composite_v13_dataset_16.csv` (10,000 sequences) |

Each folder contains

| File | Purpose |
| --- | --- |
| `train_evaluate_v13.py` | trains the per-thickness surrogate (8-ply: `d_model = 32`, 2 heads, 3 encoder layers, `d_ff = 512`, dropout 0.08, 80 epochs, 90/10 split, seed 42) and saves `mcr_former_best_model_with_aug.pth` + result JSON/XLSX |
| `Inverse_design_DE.py` | DE search against the surrogate: bounds -90..90 deg per ply, 2 restarts x 50 iterations, `popsize = 50` (population 50 x N_plies), random strategy per restart, tolerance 1e-6, polishing on; the three best solutions per restart are re-evaluated with the exact nu13 formula; logs every restart |
| `Inverse_design_other_models*.py`, `Inverse_designs_OtherModels.py` | the same search budget against SVR / MLP / LSTM surrogates, with MCR and with raw angles |
| `generate_data*.py`, `get_v13.py` | dataset generation and the CLT target |
| `global_optimization_results_*.xlsx`, `global_optimization_details_*.json` | archived search logs (all restarts, best sequences, predicted and true nu13) |
| `model_comparison_*.json` | archived surrogate training metrics for the baseline models |

## Run

```bash
cd 8Layer
python train_evaluate_v13.py     # trains the surrogate (~minutes on CPU)
python Inverse_design_DE.py      # runs the DE search (~minutes; writes global_optimization_*)
```

The DE script needs the surrogate checkpoint `mcr_former_best_model_with_aug.pth` in the same
folder (produced by the training step; not tracked in git).

## Where the manuscript numbers are

The newest archived run in `8Layer/` is `global_optimization_results_20260120_121108.xlsx` with
its `global_optimization_details_20260120_121108.json` companion. The surrogate metrics recorded
in this folder come from the training run archived here; the values tabulated in the manuscript
were assembled from a separate run of the same protocol, so last-digit differences between the
two are possible. The theoretical-optimum row of Table 5 is quoted from the referenced
theoretical study rather than computed here.