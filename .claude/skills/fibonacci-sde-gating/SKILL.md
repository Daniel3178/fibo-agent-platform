---
name: fibonacci-sde-gating
description: Understand, analyse and improve the fibonacci_neural_sde project, a mixture-of-experts Neural SDE that generates synthetic stock paths with Bounce/Break/Hover experts around Fibonacci retracement levels. The focus is the MoE gating mechanism. Use for any question or change touching the gate, gate collapse, regularisers (entropy, adaptive entropy, temperature, regional variance, mutual info), gating diagnostics (Exp 1/2/4, analyze_gating, analyze_temporal_gating), expert specialisation, classifier or supervised gating, Fibonacci features, training (train_neural_SDE.py, runner.py), sample generation, or evaluating synthetic data against real data or Vasicek.
---

# Fibonacci Neural SDE: gating skill

**Repo:** `fibonacci_neural_sde/`, a sibling of this workspace folder. From `sebx-agent-platform/` it is `../fibonacci_neural_sde`. Every path in this skill is relative to that repo root. The Python env is `.venv/bin/python` (3.12). Run everything from the repo root.

**Skill files:** `.claude/skills/fibonacci-sde-gating/` in `sebx-agent-platform/`. This README calls it `<skill>`.

## 1. What problem this is

This is **conditional generative modelling of daily stock prices**: a path-dependent Neural SDE `X_{t+1} = X_t + μ + σε` (dt = 1, normalised units), trained by **one-step Euler–Maruyama Gaussian maximum likelihood**.
- **Inputs:** a 252-day price window, min-max scaled, plus 7 distances to Fibonacci retracement levels.
- **μ and σ:** produced by a **dense soft mixture of 3 constrained experts**:
  - **Bounce:** pushes away from the nearest level.
  - **Break:** follows the latest move.
  - **Hover:** small drift, low σ.
- **Gate:** a softmax over these experts. Experts are combined by moment mixing: `μ = Σπμ_k`, `σ² = Σπσ_k²`.
- **Generation:** autoregressive Monte Carlo rollouts.
- **Evaluation:**
  - event and distribution fidelity against real data and a Vasicek baseline;
  - TSTR (train on synthetic, test on real) with a trend classifier;
  - gating diagnostics.

Statistically, it is a heteroscedastic Gaussian AR model with a mixture-of-experts head. The central open problem is that **the gate is not semantic**:
- Entropy-style regularisers give near-uniform gates, and Bounce gets starved.
- `regional_variance` gives sharp, balanced gates that are unrelated to Fibonacci events.
- In every benchmark variant, MI(event; gate winner) ≤ 0.01.

## 2. Repo map (read in this order to understand it)

1. `amgm/models/mlp.py`: `NeuralSDEMoE` (167-267), `GateHead` (138-165), `ExpertHead`.
2. `amgm/models/neural_SDE/runner.py`: NLL (166-174), loss assembly (185-212), β and temperature schedulers, logging.
3. `amgm/models/neural_SDE/gate_regularizers.py`: `gate_reward` modes and `balance_loss`.
4. `amgm/data/neural_SDE.py`: windows, min-max scaling, Fibonacci features. `amgm/utils/common.py` has `calculate_fibLevels` and `sanity_check_path`.
5. `experiments/neural_SDE/trainer_cfg/neural_SDE/US_Stocks.py`: all hyperparameters. There are no CLI flags for gate settings.
6. `experiments/neural_SDE/train_neural_SDE.py`, `generate_samples.py`, `fibonacci_state_detection.py` (the event FSM).
7. `experiments/neural_SDE/gating_diagnostics/` (collection, statistics, analysis, temporal) and the notebooks `gate_collapse_benchmark.ipynb` and `Statistical_comparison.ipynb`.

The details are in `references/architecture.md` and `references/gating.md`.

## 3. Commands

```bash
# Train (checkpoints → workspace/neural_SDE/logs/train_neural_SDE/US_Stocks/checkpoints/)
.venv/bin/python experiments/neural_SDE/train_neural_SDE.py --trainer_cfg US_Stocks
# Train with Exp 1 diagnostics (seed forced to 1) → workspace/neural_SDE/gating_diagnostics/baseline_<stamp>/
.venv/bin/python experiments/neural_SDE/train_neural_SDE.py --trainer_cfg US_Stocks --gating-diagnostics [--diagnostics-dir DIR]
# Exp 2: real-trajectory gate analysis
.venv/bin/python -m experiments.neural_SDE.analyze_gating --training-run RUN_DIR
# Exp 4: temporal consistency, real vs synthetic
.venv/bin/python -m experiments.neural_SDE.analyze_temporal_gating --training-run RUN_DIR --paths-per-origin 100 --seed 1
# Gate health summary: one run, or a benchmark folder of variants
.venv/bin/python -I <skill>/scripts/gate_health.py workspace/neural_SDE/gating_diagnostics_benchmark/20261004_185041
# Synthetic samples (MC mode)
.venv/bin/python experiments/neural_SDE/generate_samples.py --trainer_cfg US_Stocks --checkpoint-name <name> --mode mc --batch-size 1000
# Tests: name the files explicitly, because tests/test_gate_regularizers.py is broken and a bare `pytest` fails at collection
.venv/bin/python -m pytest tests/test_gating_diagnostics.py tests/test_temporal_gating.py -q
```

Benchmark variants are trained from `gate_collapse_benchmark.ipynb`; the overrides are in cell 4. `Statistical_comparison.ipynb` must be run from `experiments/neural_SDE/`.

## 4. Workflows

### Understand or explain something
Read the relevant section of `references/architecture.md` or `references/gating.md`, then open the cited lines to confirm them, because line numbers drift. Prefer the CSVs and `gate_health.py` output over the narrative in the notebooks (G11).

### Diagnose a gate (any run or checkpoint)
1. Make sure Exp 1 and Exp 2 outputs exist: train with `--gating-diagnostics`, or run `analyze_gating --checkpoint ...`.
2. Run `gate_health.py`. Read its verdict using the taxonomy in `gating.md` §5:

   | Verdict | Meaning |
   |---|---|
   | `hard-collapse` | one expert wins nearly every window |
   | `starved-expert` | one expert almost never wins |
   | `uniform-soft` | gates are near-uniform and barely depend on the input |
   | `non-semantic` | gates are sharp or balanced but don't match Fibonacci events |
   | `healthy` | none of the above |

   Also check two columns:
   - `val_mi_input_expert`: ≈0 means the gate ignores its input.
   - `real_sign_agree`: the share of windows where Bounce and Break push the same way.
3. Go deeper with `references/analysis-protocol.md`: leakage audit, stability across seeds, alignment probes, expert knockout and oracle routing, regime slicing.
4. Run Exp 4 if temporal behaviour matters.

### Improve the gate
1. Do the prerequisites in `references/improving-gating.md` §0 first:
   - checkpoint on `val/loss_sde`;
   - fixed seeds and stock list, ≥3 seeds;
   - a time-based split.
2. Pick items in the recommended order:
   - informative gate inputs plus disambiguated Bounce/Break;
   - then a mixture likelihood;
   - then annealed weak supervision.

   The classifier option is analysed in `references/classifier-gating.md`. Run its **probe test** before building anything.
3. Change one thing at a time. Add new regulariser modes to `GATE_REGULARIZERS`. If you change features, mirror the change in `generate_samples._compute_features` and `gating_diagnostics/statistics.window_features`.
4. Compare variants on **`val/loss_sde`, never `val/loss`**, plus `gate_health` metrics, all over multiple seeds (`analysis-protocol.md` step 7).
5. Commit following the `git-conventions` skill: a feature branch, Conventional Commits, ≤200 lines per commit. Never commit `Data/data-20250505/cache.pkl`.

### Evaluate synthetic data
Use `references/evaluation.md`: stylized facts, two-sample tests, fair baselines (GBM, GJR-GARCH-t, a properly calibrated OU), chronological splits, the invalid-path rate, and a Fibonacci-vs-random-levels ablation.

## 5. Pitfalls checklist

The full list, with file:line references, is in `references/known-issues.md`.

- **Fibonacci features:** they equal `pos_in_range − r_k` whatever the scaling (G1). The gate gets no level information beyond `x_t`.
- **Gate gradient:** it is dominated by σ (|μ| is about 1–3% of σ). The gate acts as a volatility selector.
- **Bounce vs Break:** their drift signs agree on 45.7% of windows (G12).
- **Loss comparability:** `val/loss` includes −R, so it can't be compared across modes (G6).
- **Sampling temperature:** `generate_samples` ignores the trained temperature (G7).
- **Leakage:** validation is in-sample (random split of overlapping windows), and Exp 2 (2018-20) overlaps training (D1, D2).
- **Single seed:** every benchmark result comes from seed 1 (G8).
- **FSM labels:** they need 9 days of hindsight, cover only about 3% of windows, and are scaled differently from the gate's inputs (G13, G14).
- **No TensorBoard:** `logger=False`, so `self.log` metrics are lost. Rely on the diagnostics CSVs and `val_metrics.json`.
- **Paper vs code:** they disagree on data range, number of stocks and Vasicek calibration. The saved Wilcoxon results don't support the paper's claim (D5, D6).

## 6. References

| File | Use it for |
|---|---|
| `references/architecture.md` | data → features → model → loss → rollout, with file:line map |
| `references/gating.md` | gate design, regulariser formulas, diagnostics outputs, collapse taxonomy, current benchmark numbers |
| `references/analysis-protocol.md` | step-by-step gate analysis with formulas, source files and how to interpret each result |
| `references/improving-gating.md` | ranked, research-backed changes, with where to implement them and how to evaluate |
| `references/classifier-gating.md` | whether a dedicated classifier or supervised gate helps: verdict, designs A–D, probe test |
| `references/evaluation.md` | evaluating the generator, baselines and splits |
| `references/known-issues.md` | verified bugs and discrepancies (G*, D*, T*) |
| `references/literature.md` | citations behind all of the above |
