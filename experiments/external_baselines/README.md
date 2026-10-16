# External and classical baselines (Table 1)

Scripts that produced the non-MCR-Former rows of Table 1. Every model was trained or run
**in this work** on the same 3,000-sequence 16-ply dataset and the same 2,400/600 train-test
split (`test_size = 0.2`, `random_state = 42`) as MCR-Former, each with its own native input
representation. The citations in the manuscript identify the architectures, not the source of
the numbers. All implementations are self-contained (PyTorch / scikit-learn only).

| Script | Model | Test R^2 (Table 1) | Test RMSE |
| --- | --- | --- | --- |
| `Informer.py` | Informer | 0.891 | 0.0188 |
| `ModernTCN.py` | ModernTCN | 0.926 | 0.0155 |
| `TFT.py` | TFT | 0.937 | 0.0143 |
| `DeepAR.py` | DeepAR | 0.946 | 0.0133 |
| `ml_and_neural_baselines_16layers.py` | Transformer / CNN / LSTM / Random Forest / SVR / MLP / Ridge under three encodings | raw-angle rows: 0.725 / 0.715 / 0.711 / 0.328 (see CSV) | see CSV |
| `../encoding_ablation/LayupFormer.py` | LayupFormer-style baseline (Tsai-invariant embedding) | 0.972 | 0.0093 |

## Run

```bash
python Informer.py            # prints test R^2 and RMSE
python ModernTCN.py
python TFT.py
python DeepAR.py
python ml_and_neural_baselines_16layers.py    # writes results/model_comparison_results_16layers.csv
```

Each script reads `composite_v13_dataset.csv` from the working directory; an identical copy is
shipped in this folder so everything runs out of the box. The same file is also in
`../v13_main/` and `../encoding_ablation/` (byte-identical, MD5 `87b8373b...`).

## Notes

- `results/model_comparison_results_16layers.csv` is the archived output of the
  `ml_and_neural_baselines_16layers.py` sweep; the Table 1 rows for Transformer, CNN, LSTM and
  Random Forest are its `raw_angles` entries.
- `LayupFormer.py` follows the LayupFormer training recipe (Tsai-invariant embedding, 80/10/10
  split) as described in its header comments.
- Runs are single-seed, matching the manuscript; the scripts print metrics rather than saving
  them, so re-runs should be compared against the archived CSV or against the manuscript table.