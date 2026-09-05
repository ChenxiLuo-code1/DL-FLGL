# Validation report

Executed 2026-09-05 on CPU.

- Python: 3.8.3
- NumPy: 1.18.5
- SciPy: 1.5.0
- scikit-learn: 0.23.1
- PyTorch: 2.4.1+cpu

Nine unit tests passed: Gini hand calculation, width weighting, relation orientation/clipping, PSD with constant labels, autograd gradient agreement, FISTA closed-form agreement, deterministic end-to-end behavior, known metrics, and parameter validation.

- demo: five disjoint train/test splits cover each sample once; scaler extrema match training-only data.
- flags: five disjoint train/test splits cover each sample once; scaler extrema match training-only data.
- All fingerprinted original source files and manuscript remained unchanged.
- Runtime source has no MLTSK_OFL import/path dependency or wandb dependency.

### demo

Final inner solves meeting tolerance: 5/5.

| Metric | Mean | Sample SD |
|---|---:|---:|
| AP | 0.953611 | 0.031329 |
| HL | 0.217500 | 0.050467 |
| OE | 0.060000 | 0.054772 |
| RL | 0.082448 | 0.047874 |
| CV | 0.305000 | 0.020917 |
### flags

Final inner solves meeting tolerance: 0/5.

| Metric | Mean | Sample SD |
|---|---:|---:|
| AP | 0.815728 | 0.027980 |
| HL | 0.258531 | 0.029899 |
| OE | 0.232119 | 0.060923 |
| RL | 0.212051 | 0.030088 |
| CV | 0.534779 | 0.031803 |

These are example-configuration execution checks, not manuscript-table reproduction. No hyperparameter search was run. CUDA was unavailable and remains untested. The configured 500-step budget can end before the 1e-5 proximal tolerance; the raw report preserves this fact. Increasing the budget is required before claiming fully converged benchmark performance.

Raw per-fold results are generated under `outputs/` (gitignored). No dataset is copied into the release.


### Flags extended-budget follow-up

The initial 500-step check above did not meet the inner tolerance. A follow-up used `configs/validation_flags.json` (2000 maximum inner iterations, otherwise identical fixed settings). This is a convergence follow-up, not hyperparameter tuning.

Final inner solves meeting 1e-5 tolerance: **5/5**. Outer iteration count remains a fixed budget; no claim of global joint convergence is made.

| Metric | Mean | Sample SD |
|---|---:|---:|
| AP | 0.815728 | 0.027980 |
| HL | 0.258531 | 0.029899 |
| OE | 0.232119 | 0.060923 |
| RL | 0.212051 | 0.030088 |
| CV | 0.534779 | 0.031803 |

| Fold | Final inner iterations | Proximal residual |
|---|---:|---:|
| 1 | 695 | 9.9500916e-06 |
| 2 | 642 | 9.9604281e-06 |
| 3 | 655 | 9.9832683e-06 |
| 4 | 632 | 9.9806724e-06 |
| 5 | 675 | 9.9612797e-06 |
