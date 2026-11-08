# Architecture: data → features → model → loss → rollout

Paths are relative to the repo root `fibonacci_neural_sde/`. Line numbers were checked on `main` (commit a2d27cf). Re-check them before quoting, because they drift.

## 1. Data

| Item | Where | Notes |
|---|---|---|
| Raw prices | `Data/data-20250505/security_data.txt` | SEB AM daily OHLCV, tab-separated. The model uses `ClAdjLoc` (adjusted close, local currency). |
| Loading / cache | `amgm/data/loading.py`, `amgm/data/loaders.py` | Builds `cache.pkl`, about 383 MB. `USStocksLoader` keeps only IssueId/Date/ClAdjLoc. |
| Dataset | `amgm/data/neural_SDE.py` `NeuralSDEDataset` | Samples `num_iids` stocks with `random.Random(rng_seed)`. |
| Windows | `neural_SDE.py` `make_sample` (135-165) | Stride 1, `lookback_window=252`, target = next day's price. `max_windows_per_issue=1000` keeps the earliest windows. |
| Normalisation | `neural_SDE.py` `_min_max_normalize` (177-184) | Per-window min-max to [0,1]. The target uses the same min/range, which is clamped to ≥1e-8. `sample_min` and `sample_range` are stored on the sample but **not fed to the model**. |
| Fib features | `neural_SDE.py` `calculate_features` (167-174), `amgm/utils/common.py` `calculate_fibLevels` (9-34) | Levels are `min + r·delta`, r ∈ {0,.236,.382,.5,.618,.786,1}. Features are `(x_t − L_k)/delta`, which **reduces algebraically to `pos_in_range − r_k` whatever the scaling**. The 7 features are therefore one number that the history already contains (known-issues G1). |
| Split | `experiments/neural_SDE/train_neural_SDE.py` (43-50) | `random_split` 80/20 over overlapping windows. There is no test set. |
| Config | `experiments/neural_SDE/trainer_cfg/neural_SDE/US_Stocks.py` | 100 stocks, 2014-12-31 to 2020-12-31, `rng_seed=None`, batch 256, 20 epochs, Adam lr 1e-3, wd 1e-5. `synthetic_ou.py` is an OU recovery sanity task for the MLP model. |

## 2. Models (`amgm/models/mlp.py`)

- **`NeuralSDEMoE`** (167-267) is the main model.
  - `encoder`: an MLP over `x_hist` (252 → 32 → 16, ReLU), giving `Z_t`.
  - `h = cat[Z_t, f_t]` (23-d, line 238). This one vector is shared by the gate and all six expert heads.
  - `GateHead` (138-165): SwiGLU(64), then LayerNorm, a residual SiLU, LayerNorm, and a Linear to 3 logits. Then `pi = softmax(logits / gate_temperature)` (241).
  - Experts (`ExpertHead` 125-136, 32 hidden), one μ head and one σ head each:
    - **Bounce** μ = `sign(nearest signed fib distance) · softplus(·)` (213-221, 249). It pushes *away* from the nearest level, like a support or resistance bounce.
    - **Break** μ = `sign(x_t − x_{t−1}) · softplus(·)` (227-229, 250). It follows the latest momentum.
    - **Hover** μ = `0.05 · tanh(·)` (251), σ = `0.001 + 0.02·sigmoid(·)` (261).
    - Bounce and Break σ = `softplus(·) + 1e-3`.
  - Mixing: `mu = Σ π_k μ_k` (254), `sigma = sqrt(Σ π_k σ_k²)` (264-265). The result is **one Gaussian with mixed moments**. The between-expert drift dispersion `Σπ_k(μ_k − μ̄)²` is not included.
- **`NeuralSDEMLP`** (95-122) and **`PatchNeuralSDETransformer`** (269-309) are the baselines. They return zero placeholder gates, which the diagnostics reject.
- **`MLPModel`** (5-18) is the 3-class trend classifier used by `train_classification.py`.

## 3. Training (`amgm/models/neural_SDE/runner.py`, `NeuralSDERunner`)

- NLL `_step_nll` (166-174): `0.5·[log var + (dx − μ dt)²/var]` with `var = (σ+eps)²·dt + eps`, eps=1e-6, dt=1. The constant term is dropped, so the values are negative.
- Total loss `_compute_loss` (185-212): `loss = sde_loss − gate_reward(mode, π, β) + λ·balance_loss(π)`. See `gating.md`.
- Schedulers: `EntropyBetaScheduler` (≈9-40) and `TemperatureScheduler` (≈43-72) hold for `warmup` steps, then cosine-decay over `decay_steps`. Both step in `training_step` (243+). `on_load_checkpoint` (475+) restores the temperature.
- Checkpoints: `ModelCheckpoint(monitor=trainer_cfg.get("checkpoint_monitor","val/loss"))` (`train_neural_SDE.py:162-174`). The filename suffix is the hard-coded `model_type="MoE"` (279).
- `Trainer(logger=False)` (`train_neural_SDE.py:213`): `self.log` metrics are **not** written to TensorBoard. Use `--gating-diagnostics`, which writes CSVs, or `val_metrics.json`.

## 4. Generation (`experiments/neural_SDE/generate_samples.py`)

- `_load_model_checkpoint` (35-72) loads hparams and state, but **does not call `on_load_checkpoint`, so the gate temperature stays at its constructor value**.
- `_rollout_sde` (107-150): an autoregressive Euler–Maruyama loop with dt=1. Every step recomputes min-max normalisation and fib features on the rolling window and appends its own output.
- Modes: standard (`main`, 787+), with one path per dataset window, writes `Data/Synthetic/synthetic_rollout_paths_<type>.csv`. MC (`main_mc_multi`, 562+) writes `*_MC_paths.csv` and `*_MC_valid_samples.npz`, plus ensemble plots.
- Validity: `amgm/utils/common.py` `sanity_check_path` (771-793). A path is rejected if a price is ≤0, max/S0 > 4, min/S0 < 0.05, or any |daily return| > 100%. The comment on line 784 says "95%", but the code uses 5%.
- Loader for outputs: `experiments/neural_SDE/read_valid_samples.py` `load_valid_samples_npz`.

## 5. Downstream and evaluation scripts

- `experiments/neural_SDE/train_classification.py`: TSTR (train on synthetic, test on real) trend classifier. Labels come from `amgm/data/linear_trend.py` (258-386): a 21-day linear fit, a 5-day horizon, a ±1.96σ band, 3 classes. Model type and synthetic share are hard-coded at lines 397-398.
- `experiments/neural_SDE/fibonacci_state_detection.py` `detect_fibonacci_interactions` (74+): a rule FSM with tolerance .02, min dwell 3, confirmation 10, max dwell 20. It emits hover, breakout, pullback and timeout.
- `experiments/neural_SDE/Statistical_comparison.ipynb`: event histograms with JS distance, return/vol/MDD, and a Wilcoxon test against a Vasicek with hard-coded parameters. It must run from `experiments/neural_SDE/`.
- `experiments/neural_SDE/gate_collapse_benchmark.ipynb`: trains the 5 regulariser variants (cell 4 overrides) into `workspace/neural_SDE/gating_diagnostics_benchmark/<stamp>/<variant>/`.

## 6. Signal-to-noise fact worth remembering

In every benchmark checkpoint (`val_metrics.json`, normalised units), `val/drift_abs` is 0.0004–0.0016 and `val/diffusion` is ≈0.045–0.048, so drift is about 1–3% of the per-step noise. `val/rmse_next` is 0.0499 for all variants: no variant forecasts better than the next-step noise floor. The NLL is therefore dominated by σ. Most of the gradient the gate receives is about *which σ to pick*, not which drift direction, so semantic Bounce-vs-Break routing gets a weak learning signal.
