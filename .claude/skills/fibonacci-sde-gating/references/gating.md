# Gating: design, regularisers, diagnostics, current state

Paths are relative to `fibonacci_neural_sde/`.

## 1. Gate design (`amgm/models/mlp.py`)

- Input: the shared `h = [Z_t (16-d history encoding), f_t (7 fib distances)]` (line 238). There are no gate-specific inputs. `f_t` reduces to `pos_in_range − r_k` whatever the scaling (known-issues G1), so in practice the gate sees "history encoding + current position in the range". The features are not drowned by their dimensionality (7 of 23 dims); they simply carry no extra information. Velocity, volatility and window range relative to price are absent.
- Network: `GateHead` (138-165), SwiGLU(64) → LN → residual → LN → Linear(3).
- Output: `pi = softmax(logits / T)` (241). Routing is dense and soft: no top-k, no noise, no straight-through. K=3 is fixed: index 0 Bounce, 1 Break, 2 Hover.
- Use: `μ = Σπ_kμ_k`, `σ² = Σπ_kσ_k²`, so the gate is a moment-mixing weight, not a mixture responsibility.

**Identifiability caveat.** The NLL only sees the mixed μ and σ. For drift, `π_k · softplus_k` products can trade off freely: a small π with a large expert magnitude is equivalent to a large π with a small magnitude. Nothing in the objective forces π to carry meaning. Hover's bounded σ ∈ [0.001, 0.021] is the only strongly distinctive expert, which is why the gate mostly behaves as a *volatility selector*. Formally, with L = ½log σ² + ½r²/σ², r = dx − μ, dt = 1:

  ∂L/∂π_k = ½(σ_k²/σ²)(1 − r²/σ²) − r·μ_k/σ²

Because |μ_k| ≪ σ, the variance term dominates the gate gradient. Bounce and Break drift signs also agree whenever price moves away from the nearest level (G12).

## 2. Loss and regularisers

`loss = L_SDE − R(π; β) + λ·L_bal` (`amgm/models/neural_SDE/runner.py:212`). The functions are in `amgm/models/neural_SDE/gate_regularizers.py`. Notation: p̄_k = mean over the batch of π_k, K=3, log eps 1e-8.

| `gate_reg_mode` | Function (line) | R(π; β) | Optimum |
|---|---|---|---|
| `entropy` | `entropy_reward` (39) | β · mean_b H(π_b) | uniform π for every sample, i.e. uninformative |
| `adaptive_entropy` | `adaptive_entropy_reward` (43) | mean_b Σ_k β_k(−π_k log π_k), with β_k = β / clamp(p̄_k.detach()·K, clip, 1) | still pushes towards uniform, harder on under-used experts |
| `regional_variance` | `regional_variance_reward` (57) | β · mean_k Var_b(π_k) (biased) | balanced one-hot routing, Var = 2/9 ≈ 0.222 |
| `mutual_info` | `mutual_info_reward` (61) | β · [H(p̄) − mean_b H(π_b)] | balanced and confident; **never benchmarked** |
| balance | `balance_loss` (84) | L_bal = Σ_k (p̄_k − 1/K)² | balanced soft mean, which uniform per-sample gates also satisfy |

Dispatch goes through `gate_reward(mode, pi, beta, **options)` (73). To add a mode, add a function and register it in `GATE_REGULARIZERS`.

Schedules (`runner.py`): `EntropyBetaScheduler` holds β = `entropy_beta` for `entropy_beta_warmup_steps`, then cosine-decays to `entropy_beta_min` over `entropy_beta_decay_steps`. `TemperatureScheduler` does the same for T (`gate_temperature_max` → `_min`). A scheduler with `decay_steps=0` is disabled. β scales every mode.

Hyperparameters live in the trainer config dict (`trainer_cfg/neural_SDE/US_Stocks.py:51-63`) or in notebook overrides. **There are no CLI flags for them.**

## 3. Logged metrics

`runner.py` logs `train|val/` `loss`, `loss_sde`, `loss_gate_reg_term`, `loss_entropy_term`, `loss_balance_term`, `mean_pi_var`, `pi_min_mean`, `pi_mean_e{k}`, `pi_var_e{k}`, `drift_abs`, `diffusion`, `entropy_beta`, `gate_temperature`. Because `logger=False`, they only survive through the diagnostics callback and `val_metrics.json`.

**Compare variants on `val/loss_sde`.** `val/loss` subtracts R, which depends on the mode and on β.

## 4. Diagnostics (`experiments/neural_SDE/gating_diagnostics/`)

| Experiment | Entry point | Main outputs (in the run dir) |
|---|---|---|
| 1. Training history | `train_neural_SDE.py --trainer_cfg US_Stocks --gating-diagnostics [--diagnostics-dir D]`. Seed is forced to 1. Callback: `collection.py` `GatingDiagnostics`. | `epoch_metrics.csv` (epoch, split, n, entropy, confidence, p_{bounce,break,hover}, `_variance`, `_winner_share`), `training_batches.csv`, `assignment_changes.csv`, `validation_epoch_XXX.npz`, `split_membership.csv`, `manifest.json`, `checkpoints/` |
| 2. Real trajectories | `python -m experiments.neural_SDE.analyze_gating --training-run RUN` (or `--checkpoint CKPT --issue-ids-file F --output-dir D`; `--render-only`). Code: `analysis.py`. | `real_gates.csv` (segment, step, date, price, distance, movement, nearest_level, p_*, distance_0..6), `events.csv` (event ∈ Bounce/Break/Hover, entry/exit indices, p_* at gate time and entry, change), `event_summary.csv`, `distance_bins.csv`, `movement_bins.csv`, `segments.csv`, `real_windows.npz`, `01..08_gating.png`, `report.html` |
| 4. Temporal consistency | `python -m experiments.neural_SDE.analyze_temporal_gating --training-run RUN --paths-per-origin 100 --seed 1` (or `--render-only --output-dir`). Code: `temporal.py`, `temporal_statistics.py`. | Switches per 100 steps, dwell runs, 3×3 transition matrices, real vs synthetic paired stock-cluster bootstrap CIs, plots and a report. Default output is `Results/gating_diagnostics/temporal_seed<seed>_<stamp>/`. |

Metric definitions are in `statistics.py`:
- `GateAccumulator`: entropy = H/log 3 (normalised to [0,1]); confidence = mean max π; winner share counts ties split.
- `assignment_change`: the winner-switch fraction between epochs, excluding windows whose top-2 margin is < 0.05.

Exp 2 details:
- Segments are non-overlapping, 100 steps, 2018-01-01 to 2020-12-31 (`analysis.py` `segment_origins`). **This overlaps the training period.**
- Event mapping: pullback → Bounce, breakout → Break, hover → Hover (`analysis.py:19`).
- The gate is read at exit for Bounce/Break and at hover emission for Hover. FSM labels need up to 10 days of confirmation, so the labels use hindsight.

Exp 3 does not exist on `main`.

There is also `compare.py`, used by the benchmark notebook. Its `summary_table` "Winner Share" column is Bounce only.

Quick health summary of any run or benchmark folder:

```bash
.venv/bin/python -I <skill>/scripts/gate_health.py workspace/neural_SDE/gating_diagnostics_benchmark/<stamp>
```

## 5. Collapse taxonomy

| Pattern | Signature | gate_health flag |
|---|---|---|
| Hard collapse | One expert wins ≥95%, e.g. Hover π≈0.96 in the old `gating_diagnostics/baseline_20260930*` runs | `hard-collapse` |
| Starved expert | One expert never wins (<2%), e.g. baseline Bounce at 0.000 | `starved-expert` |
| Uniform-soft (uninformative mixing) | Normalised entropy ≥0.85, confidence <0.6, per-expert variance ≈0.01 | `uniform-soft` |
| Non-semantic specialisation | The gate is sharp and balanced, but its mean on Bounce, Break and Hover events barely differs (contrast <0.05) | `non-semantic` |

## 6. Current state (benchmark `20261004_185041`, seed 1, 20 epochs)

Reproduced with `gate_health.py`. Validation is the last epoch; "real" means Exp 2 on 2018-20.

| Variant | val entropy | val win B/Br/H | val loss_sde | real win B/Br/H | event contrast | MI(event; winner)/H(event) | Verdict |
|---|---|---|---|---|---|---|---|
| baseline (entropy) | 0.925 | .00/.46/.54 | −2.399 | .00/.65/.35 | 0.005 | 0.001 | starved, uniform-soft, non-semantic |
| adaptive_entropy | 0.928 | .37/.11/.53 | −2.387 | .61/.09/.30 | 0.011 | 0.002 | uniform-soft, non-semantic |
| temperature 5→0.5 | 0.938 | .01/.86/.13 | −2.406 | .76/.12/.12 | 0.017 | 0.009 | starved, uniform-soft, non-semantic |
| adaptive_entropy + temperature | 0.954 | .67/.25/.08 | −2.406 | .80/.15/.06 | 0.019 | 0.009 | uniform-soft, non-semantic |
| regional_variance (β 7→1.5, λ 2, T anneal) | 0.009 | .32/.33/.35 | −2.370 | .22/.41/.37 | 0.002 | 0.006 | non-semantic |

What this shows:
0. Input dependence I(X;E)/log 3 (`gate_health` `val_mi_input_expert`) is 0.035 for baseline, 0.068 for adaptive_entropy, 0.053 for temperature, 0.044 for AE+temperature and 0.991 for regional_variance.
1. Entropy-family regularisers move *which* expert wins, but leave gates near-uniform. Winner shares on those near-uniform gates are fragile, and they flip between validation and real data.
2. `regional_variance` is the only variant that produces sharp, balanced routing. It costs about 0.03 nats of `loss_sde`, and its routing is unrelated to Fibonacci events.
3. **No variant gives semantic gating.** The gate carries about 1% or less of the information about the event label. The bottleneck is the inputs and the objective (`improving-gating.md`), not the regulariser strength.
4. Everything above comes from one seed. Treat differences of about 0.01 in `loss_sde` as noise until multi-seed results exist.
