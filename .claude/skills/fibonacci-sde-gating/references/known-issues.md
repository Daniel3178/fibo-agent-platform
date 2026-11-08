# Known issues and discrepancies

Each issue below was checked against `main` (a2d27cf) on 2026-10-27. Paths are relative to `fibonacci_neural_sde/`. These are recorded rather than fixed. Fix them as separate `fix:` commits on their own branch, following the `git-conventions` skill.

## Affects gating results directly

| # | Issue | Where | Consequence |
|---|---|---|---|
| G1 | **Fib features are degenerate.** `f_k = (x_t − L_k)/Δ` with `L_k = min + r_kΔ` reduces to `(x_t − min)/Δ − r_k`, the position in the range minus a constant. This holds **whatever the scaling**: computing the levels before the min-max step would give the same numbers. | `amgm/data/neural_SDE.py:167-174`; `amgm/utils/common.py:9-34`; mirrored in `gating_diagnostics/statistics.py` `window_features` and in `generate_samples._compute_features` | The 7 features are one number that the history already holds (`x_hist[:, -1]`). Scale information (window range relative to price, volatility) is discarded, although `sample_min`/`sample_range` are stored. The fix is a different normaliser or anchor (ATR / realised vol, swing or multi-window anchors), not rescaling. This is a root cause of non-semantic routing. |
| G2 | The gate and the experts share one input `h`; the gate has no velocity or volatility features | `amgm/models/mlp.py:235-241` | The routing signal is mixed in with a 16-d history code. See `classifier-gating.md`. |
| G3 | Moment-mixing objective: μ=Σπμ_k, σ²=Σπσ_k², a single Gaussian NLL | `mlp.py:254, 264-265`; `runner.py:166-174` | No competitive pressure for specialisation. π and expert magnitudes are not identifiable. σ² leaves out the between-expert drift dispersion. |
| G4 | The `entropy` and `adaptive_entropy` rewards are maximised by uniform per-sample gates | `gate_regularizers.py:39-54` | They directly produce the uniform-soft collapse. |
| G5 | `balance_loss` uses the soft mean p̄ | `gate_regularizers.py:84-86` | Uniform gates satisfy it, so it cannot prevent uniform-soft collapse. |
| G6 | **`val/loss` = loss_sde − R + λ·bal**, and R depends on the mode and on β | `runner.py:212`; `US_Stocks.py:62` (`checkpoint_monitor="val/loss"`) | It cannot be compared across variants. Early epochs with high β are favoured during checkpoint selection. Use `val/loss_sde` (the benchmark notebook already does). |
| G7 | **The MC generator ignores the trained temperature.** `_load_model_checkpoint` never calls `on_load_checkpoint`. | `experiments/neural_SDE/generate_samples.py:35-72` | `Data/Synthetic/{temperature,adaptive_entropy_temperature,regional_variance}_*` were sampled at T=1.0 instead of 0.5. The Exp 2 and Exp 4 loaders (`gating_diagnostics/analysis.py`) do restore T. |
| G8 | Single seed (1) for every benchmark variant | `gate_collapse_benchmark.ipynb` cell 3 | A `loss_sde` gap of about 0.01 between variants is within unknown noise. |
| G9 | `mutual_info` mode is implemented but never benchmarked or tested | `gate_regularizers.py:61-62` | It is a cheap next experiment. |
| G10 | The `compare.summary_table` "Winner Share" column shows only `p_bounce_winner_share` | `experiments/neural_SDE/gating_diagnostics/compare.py` (`summary_table`) | It misleads anyone reading the notebook table. |
| G12 | Bounce and Break drift signs agree whenever price moves away from the nearest level | `mlp.py:220` (`sign(x_t − L*)`), `mlp.py:228` (`sign(x_t − x_{t−1})`) | In about half of all states the two experts are interchangeable, so they can only be told apart while price approaches a level. |
| G13 | In Exp 2 the FSM runs on future prices scaled by the *origin* window, while the gate's inputs use *rolling-window* min-max | `gating_diagnostics/analysis.py:133-143` vs `statistics.window_features` | After a new high, a breakout above level 1.0 looks to the model like "sitting at 1.0". This deflates measured alignment. |
| G14 | Breakout and pullback labels are emitted at `exit + 9` (10 confirmation steps); hover at the dwell-3 step is causal | `fibonacci_state_detection.py:166, 186, 219-229, 274, 322`; gate read at the exit day in `analysis.py:83` | Labels are about 3% of windows (794 Bounce / 652 Hover / 198 Break in a 55k-window run). Any supervised gate is *forecasting* the outcome. |
| G11 | The benchmark notebook's narrative says the baseline "hogs ~1.0", but the saved numbers show near-uniform gates | `gate_collapse_benchmark.ipynb` cells 7 and 9 | Trust the CSVs and `gate_health.py`, not the prose. |

## Data, splitting and evaluation validity

| # | Issue | Where | Consequence |
|---|---|---|---|
| D1 | Stride-1 windows that overlap by 251 of 252 days, with a random 80/20 split and no test set | `train_neural_SDE.py:43-50` | Validation is effectively in-sample. Use time-blocked splits with an embargo before claiming generalisation. |
| D2 | Exp 2 evaluates 2018-2020, which overlaps the training period (2014-12-31 to 2020-12-31; the earliest 1000 windows per stock reach about the end of 2019) | `gating_diagnostics/analysis.py` `segment_origins` | Gate–event alignment is measured mostly on seen data. |
| D3 | `rng_seed=None`, so each run trains on a fresh, unrecorded set of 100 stocks | `trainer_cfg/neural_SDE/US_Stocks.py:15` | It is not reproducible, and generation may sample different stocks. `--gating-diagnostics` forces seed 1. |
| D4 | The window range is clamped to ≥1e-8; near-flat windows blow up the scaled targets | `neural_SDE.py:181` | NLL spikes. Consider filtering windows by minimum relative range. |
| D4b | Three raw rows have `ClAdjLoc = 0`: `01343103` on 2017-07-26, and `03661301` on 2015-08-27 and 2016-03-21 | `Data/data-20250505/security_data.txt` | Every window containing them gets a −100% spike. Nothing filters them out. |
| D5 | Paper vs code: the paper uses 50 stocks over 2014-2017, the config uses 100 stocks over 2014-2020; the paper says Vasicek is MLE-calibrated, the notebook hard-codes κ=0.015, μ=0.5, σ=0.015 | `Fibonacci_Neural_SDE.pdf` §3.1; `Statistical_comparison.ipynb` cell 6 | The baseline comparison is not fair as it stands. |
| D6 | The saved Wilcoxon results (adaptive_entropy_temperature variant) show the Neural SDE with *larger* error than Vasicek on mean return, variance and MDD (p ≥ 0.93) | `Statistical_comparison.ipynb` cell 26 | This contradicts the paper's claim for that variant. Re-check per variant and over multiple seeds. |
| D7 | The `sanity_check_path` comment says "95%"; the code uses min/S0 < 0.05 | `amgm/utils/common.py:784` | Documentation only. |

## Tooling

| # | Issue | Where | Consequence |
|---|---|---|---|
| T1 | `tests/test_gate_regularizers.py` imports functions that don't exist (`entropy_regularization`, …) | `tests/test_gate_regularizers.py:3-8` | A bare `pytest` fails at collection. Run the other files explicitly. |
| T2 | `Trainer(logger=False)` | `train_neural_SDE.py:213` | Every `self.log` metric is dropped, and `visualize_experts.py` (which reads TensorBoard) finds nothing new. |
| T3 | `--benchmark` / `--benchmark-dir` are parsed but never used | `train_neural_SDE.py:292-301` | The benchmark only runs from the notebook. |
| T4 | `model_type="MoE"` is hard-coded | `train_neural_SDE.py:279` | MLP and PatchTST checkpoints get the `_MoE` suffix, and `generate_samples.py` names its outputs from that suffix. |
| T5 | Gate and regulariser hyperparameters have no CLI flags | `train_neural_SDE.py:282-306` | Edit the config module or use notebook overrides. |
| T6 | `.venv` was created at the old path `…/Repositories/fibonacci_neural_sde/.venv` | `.venv/pyvenv.cfg` | Activation scripts may point to the old location. `.venv/bin/python` still works. |
| T7 | `Data/data-20250505/cache.pkl` (≈383 MB) is staged in git | `git status` | Unstage it before committing. |
