# Zero-shot and few-shot transfer to unseen thicknesses — Table 3 and Fig. 5a

## Question

A model trained on 1-20 ply laminates meets laminates of **unseen** thickness. How far does a
single new sample go? Table 3 reports full fine-tuning and linear-head fine-tuning on 21, 25
and 30 plies with 0 (zero-shot), 1 and 5 target samples, averaged over five repeats.
Fig. 5a is the same experiment with the full learning curve.

## Data

`data/layer_<N>.csv`, N = 1...30, 200 sequences each (`angle_1 ... angle_N, v13`). Layers 1-20
form the pre-training domain, layers 21-30 are the unseen target domains. The larger 500-sample
pools live in `../cross_scale/data/`.

## Scripts

| Script | Purpose | Writes |
| --- | --- | --- |
| `few_shot_v13_3.py` | full fine-tuning across sample sizes `[0, 1, 2, 3, 5, 7, 10, 100]` for target layers 21-30, 5 repeats per point | `results_few_shot_analysis/` (CSV + JSON) |
| `few_shot_v13_4.py` | the same sweep with a frozen backbone and a linear regression head | `results_few_shot_linear_probing/` |
| `few_shot_v13_2.py` | trains the variable-length base model on 1-20 and fine-tunes on 21-25 with 10 samples per layer | `models/`, `results/` |
| `train_mcr_former_fewshot_v13.py` | earlier one-shot transfer variant | `best_pretrained_model_temp.pth` |

## Prerequisite: the pre-trained base model

Both transfer scripts expect the variable-length model pre-trained on layers 1-20 at

```
models/mcr_former_concat_v13_base.pth
```

(path configurable in the script config). Train it with the variable-length training script in
`../cross_scale/` (`MCR_Former_Concat_2D` architecture, layers 1-20 pooled) and save or copy
the checkpoint to that path. Trained weights are not tracked in git.

## Run

```bash
mkdir -p models
cp /path/to/mcr_former_concat_v13_base.pth models/
python few_shot_v13_3.py      # full fine-tuning    -> results_few_shot_analysis/
python few_shot_v13_4.py      # linear probing      -> results_few_shot_linear_probing/
```

Fine-tuning uses lr 1e-4, up to 100 epochs with patience 20, batch 2 for the 1-2 sample
regimes; each (layer, sample size) point is repeated five times with fixed seeds and the
scripts report mean +- s.d.

## Where the manuscript numbers are

The Table 3 values (layers 21/25/30, 0/1/5 samples, full fine-tuning and linear head) and the
Fig. 5a curves are transcribed in `../figures/Fig5/Fig5.py`, which is the plotting script used
for the figure. Re-running the two scripts above regenerates the same sweep and writes the raw
CSVs under the `results_few_shot_analysis/` and `results_few_shot_linear_probing/` folders.