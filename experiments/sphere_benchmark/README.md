# Sphere benchmark -- Supplementary section S4 and Table S2

`sphere_benchmark.py` reproduces Supplementary Table S2: the two-parameter
(spherical) counterpart of the minimal-complete-representation argument, tested
on a closed-form physical target rather than on laminate data.

## What is compared

| encoding | inputs | dimension |
| --- | --- | --- |
| raw | (theta, phi) | 2 |
| MCR | (l, m, n) = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)) | 3 |

Everything else is held fixed: the same 3000 directions, the same 500 held-out
directions, the same training subsets in every run, one 2 x 64 x 64 x 1 tanh
MLP, Adam (lr 0.01), batch size 32, 800 epochs, mean-squared error, five runs.
The two parameterisations are evaluated on exactly the same training indices, so
the comparison isolates the input parameterisation.

## Target

Directional Young's modulus of a cubic single crystal, from the closed form

    1 / E = S11 - 2 (S11 - S12 - S44 / 2) (l^2 m^2 + m^2 n^2 + n^2 l^2),

with the representative dimensionless compliances
(S11, S12, S44) = (1.0, 0.3, 2.0), that is, a Zener anisotropy ratio
A = 2 (S11 - S12) / S44 = 0.7. The target is analytic, so no finite-element
noise enters the comparison: the two learning curves differ only in how the
same function is presented to the model.

`python sphere_benchmark.py --selftest` checks the closed form against its
[100] and [111] special cases (1/S11 and 3/(S11 + 2 S12 + S44)) and against the
moduli usually quoted for Al, Cu, W and Si.

## Run

    python sphere_benchmark.py                       # Table S2, about 30 min on CPU
    python sphere_benchmark.py --out table_s2.md     # also write the table to a file
    python sphere_benchmark.py --quick               # smoke test, 60 epochs
    python sphere_benchmark.py --selftest            # closed form vs literature values
    python sphere_benchmark.py --standardize-inputs  # optional robustness variant

`--n-runs`, `--epochs`, `--lr`, `--batch-size`, `--n-total`, `--n-test` and
`--train-sizes` change the budget; the seeds and the order of the random draws
are fixed, so the default invocation is the one behind the table.

Reference output (`table_s2.md`, five runs):

| Training samples | Raw (theta, phi) R^2 | MCR (l, m, n) R^2 |
| --- | --- | --- |
| 100 | 0.5771 +/- 0.1205 | 0.9143 +/- 0.0358 |
| 200 | 0.7677 +/- 0.0407 | 0.9414 +/- 0.0257 |
| 500 | 0.7566 +/- 0.0640 | 0.9807 +/- 0.0113 |
| 1000 | 0.6780 +/- 0.2301 | 0.9779 +/- 0.0150 |
| 2000 | 0.7170 +/- 0.2519 | 0.9920 +/- 0.0050 |

The MCR column is monotone to within 0.003 and its standard deviation stays
below 0.04, whereas the raw column does not improve beyond 200 samples and its
run-to-run standard deviation grows to 0.25.

## Input-scale control

    python standardise_control.py     # about 15 min on CPU

repeats the raw runs with the two angles z-scored on the training split (the
same training subsets as in the main script) and answers the obvious question
whether the advantage of the embedding is only a matter of input scale.
Reference output (`table_s2_standardised_inputs.md`):

| Training samples | Raw (theta, phi), standardised, R^2 |
| :--- | :--- |
| 100 | 0.6544 +/- 0.0311 |
| 200 | 0.7568 +/- 0.0212 |
| 500 | 0.7369 +/- 0.0710 |
| 1000 | 0.7916 +/- 0.0372 |
| 2000 | 0.8033 +/- 0.0252 |

Standardising removes much of the run-to-run spread of the raw model (its
standard deviation drops from 0.23-0.25 to 0.02-0.07 at the larger budgets) but
it does not close the accuracy gap: the standardised raw model stays at
R^2 = 0.65-0.80, against 0.91-0.99 for the direction cosines in Table S2.

## Reproducibility

All random draws come from a single numpy legacy stream seeded once with 42:
the 3000 directions first, then the training subsets in the fixed order
run 1 (100, 200, 500, 1000, 2000), run 2 (...), ... . Each run calls
`torch.manual_seed(run)` before its first model is built, which fixes the weight
initialisation and the mini-batch order. Running the script unchanged therefore
regenerates the values above; the reproduction was verified on CPU with
PyTorch 2.10 and Python 3.13.