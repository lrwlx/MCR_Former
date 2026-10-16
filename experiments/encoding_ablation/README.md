# Encoding ablation — Table S1 and Fig. 3

## Question

Which periodic encoding of a ply-angle sequence is *sufficient and minimal*? Seven
architectures — Transformer, LSTM, CNN, SVR, MLP, Random Forest, Ridge — are trained under
six encodings and compared:

| # | Encoding | Input dim |
| --- | --- | --- |
| 0 | raw angle | 1 |
| 1 | `(sin theta, cos theta)` | 2 |
| 2 | `(sin 2theta, cos 2theta)` — MCR | 2 |
| 3 | `(sin 4theta, cos 4theta)` (wrong periodicity) | 2 |
| 4 | `(sin theta, cos theta, sin 2theta, cos 2theta)` | 4 |
| 5 | Tsai invariants `[cos t, sin t, cos 2t, sin 2t, cos 4t, sin 4t]` | 6 |

## Scripts

| Script | Produces |
| --- | --- |
| `ablation1_6encoding.py` | the main sweep (six encodings incl. raw angles and Tsai 6D); writes `results/encoding_ablation_with_tsai_and_raw/` |
| `ablation1_5encoding.py` | earlier five-encoding sweep (without raw angles); writes `results/encoding_ablation_with_tsai/` |
| `ablation1_4encoding.py` | earlier four-encoding sweep; writes `results/minimal_complete_test/` |
| `ablation1_mcrVsraw.py` | paired MCR-vs-raw comparison with statistical tests; writes XLSX + PNG summaries into this folder |
| `LayupFormer.py` | LayupFormer-style baseline (Tsai-invariant embedding, 80/10/10 protocol; see header comments) |
| `SE3_Transformer_Baseline.py`, `train_se3_baseline.py` | SE(3)-Transformer baseline (archived results `se3_transformer_best_model_*`) |
| `train_evaluate_v13.py` | single-model training script used by some sweeps |

## Run

```bash
python ablation1_6encoding.py        # tens of minutes to about an hour on CPU
```

The script reads `composite_v13_dataset.csv` (3,000 sixteen-ply sequences; byte-identical to
`../v13_main/composite_v13_dataset.csv`) and writes CSV/XLSX/JSON summaries plus figures into
`results/encoding_ablation_with_tsai_and_raw/`. Configuration: 4 encoder layers, 8 heads,
`d_model = 64`, `d_ff = 256`, dropout 0.1, 80 epochs, 80/20 split, seed 42.

## Archived results

`results/encoding_ablation_with_tsai_and_raw/encoding_comparison_results.csv` holds the 42
model x encoding cells (7 architectures x 6 encodings) with test R^2 and RMSE; the
`statistical_results.json` and `experiment_summary.txt` files of the same results folder hold
the paired comparisons and the human-readable summary. Table S1 of the manuscript is built
from this sweep, and Fig. 3 (`../figures/Fig3/Fig3.py`) is its heatmap.

`fig_delta_r2_one_shot.pdf` and `fig_fewshot_transfer.png/pdf` are auxiliary figures prepared
during the revision; their underlying values are in `results/` and `../few_shot/`.
