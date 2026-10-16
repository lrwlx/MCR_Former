# Figure scripts

All figures are produced with matplotlib (600 dpi PNG plus vector PDF/SVG where applicable,
PDF fonts embedded with `pdf.fonttype = 42` for editability). Run each script from its own
folder; each script reads the data files stored next to it.

| Folder / file | Figure | Reads | Writes |
| --- | --- | --- | --- |
| `make_fig1.py` | Fig. 1, framework schematic | none (drawn from primitives) | `Fig1_redrawn.png/pdf/svg` |
| `Fig2/Fig2_a_b.py` | Fig. 2a,b (training history, parity plot) | `Fig2/mcr_former_best_model_training_history_with_aug.csv` and `Fig2/mcr_former_best_model_prediction_vs_actual_with_aug.csv` | `fig2a.*`, `fig2b.*` |
| `Fig2/Fig2_c_d.py` | Fig. 2c,d (Table 1 benchmark values, log-scale RMSE ratio) | none (values inline, transcribed from Table 1) | `fig2c.*`, `fig2d.*` |
| `Fig2/Fig2_combined.py` | assembled four-panel Fig. 2 | the two `Fig2/` CSVs above | `Figure_2_Combined_Final.*` |
| `Fig3/Fig3.py` | Fig. 3, encoding-ablation heatmap | none (matrix inline, transcribed from `../experiments/encoding_ablation/results/encoding_ablation_with_tsai_and_raw/encoding_comparison_results.csv`) | `Fig3_Ablation_Heatmap_HighContrast.png/pdf/svg` |
| `Fig4/Fig4.py` | Fig. 4, variable-length study | `Fig4/overall_results.csv`, `Fig4/layer_wise_results.csv` (copies of the tables in `../experiments/cross_scale/`) | figure files for panels a-c |
| `Fig5/Fig5.py` | Fig. 5, cross-scale and cross-property generalisation | none (values inline, transcribed from Table 3, Table 4 and the transfer runs) | `Figure_5_Combined_Clean.png/pdf/svg` |

## Notes

- The submitted version of Fig. 1 uses the original artwork; `make_fig1.py` is a vector redraw
  kept as a reference implementation of the schematic.
- `Fig4/Fig4.py` falls back to synthetic mock data (with a printed warning) if the CSV files are
  missing, so run it from the `Fig4/` folder to use the real tables.
- The CSV files in `Fig2/` are the data behind the flagship run in
  `../experiments/v13_main/`; the `*.xlsx` in that folder holds the same per-sample values.
