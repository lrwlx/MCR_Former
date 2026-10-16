# Cross-scale study — Fig. 4 (one model, 1-20 plies)

## Question

Can a single MCR-Former handle laminates of *any* thickness? The variable-length model
(`MCR_Former_Concat_2D`: MCR encoding + length channel + padding mask) is trained once on all
thicknesses 1-20 pooled together, then evaluated per thickness on held-out data. Fig. 4 shows
the overall R^2/RMSE versus training samples per layer and the per-layer R^2 of the
200 samples/layer run; Supplementary Tables S3 and S4 compare it with per-thickness models.

## Data

`data/layer_<N>.csv`, N = 1...30, 500 sequences each (`angle_1 ... angle_N, v13`).
Layers 21-30 are only used by the transfer scripts (`few_shot_v13_1.py`, `few_shot_v13_3.py`)
and by the per-thickness baseline in `../per_size_baseline/`.

## Scripts

| Script | Purpose | Writes |
| --- | --- | --- |
| `cross_layer_v13.py` | unified variable-length model, evaluated per layer at 100/200/300 samples per layer | `cross_layer_results.csv` |
| `cross_layer_v13_2.py` | sample-size sweep 25/50/100/200/300 per layer (overall + per layer) | `sample_size_ablation_results.csv` |
| `cross_layer_v13_3.py` | same sweep with per-layer evaluation at 100/200/300 | `cross_layer_unified_results.csv` |
| `train_mcr_former_concat_crossLayer_v13.py` | training script for the concat architecture | `results_mcr_concat/*.pth`, histories |
| `train_mcr_former_sym_crossLayer_v13.py`, `train_mcr_former_sym_alpha_crossLayer_v13.py` | symmetric-fusion variants of the unified model | training outputs |
| `Transformer_variable_layer5_best.py` | plain-Transformer variable-length baseline | training outputs |
| `few_shot_v13_1.py` | trains a 1-20 base model, then 10-shot transfer to layers 21-25 | `models/`, `results/` |
| `few_shot_v13_3.py` | transfer variant for layers 21-30 | `models/`, `results/` |
| `Fig_cross_layer_*.py` | plotting scripts for the panels of Fig. 4 | figure files |

## Run

```bash
# from this folder, with data/layer_1.csv ... layer_30.csv present
python cross_layer_v13_3.py
```

The unified model is trained on the pooled 1-20 data at one sample size per invocation; the
25/50/100/200/300 sweep therefore runs the training five times. Run it on a GPU or a
multi-core CPU node for the full sweep.

## Archived results

| File | Contents |
| --- | --- |
| `layer_wise_results.csv` | per-layer R^2/RMSE for 100, 200 and 300 samples per layer (the table used by the figure scripts) |
| `layer_wise_results2.csv` | per-layer values for the full 25-300 sweep |
| `overall_results.csv` | overall R^2/RMSE for 25-300 samples per layer |
| `Nature_Fig_R2.pdf/png/tif`, `figure_r2.*`, `figure_rmse.*`, `figure_b_*`, `figure_c_*`, `Fig4.png` | figure files produced during the study |

The plotting script in `../figures/Fig4/` reads copies of `overall_results.csv` and
`layer_wise_results.csv` stored next to it; the per-layer R^2 of layer 13 at 200 samples/layer
(0.889) quoted in the manuscript is the corresponding entry of `layer_wise_results.csv`.

## Notes

- All results in this folder come from splits fixed with `random_state = 42`; the 20 % test set
  of each thickness is never used for model selection.
- The CSV names written by the three `cross_layer_v13*.py` scripts differ from the archived
  `layer_wise_results*.csv` / `overall_results.csv` names; the archived files are the
  post-processed tables that were used for plotting.