# Classifier-driven gating: would a dedicated classifier help?

The hypothesis under test: the gating decision should depend on a few features, the ~7 Fibonacci distances, but they get drowned among the other inputs. A dedicated classifier could route better.

Paths are relative to `fibonacci_neural_sde/`. Citations are in `literature.md`.

## 1. Verdict

**A classifier can help, but the reason differs from the hypothesis.**

1. **The Fibonacci features are not drowned. They are empty.**
   - `(x_t − L_k)/Δ` with `L_k = min + r_kΔ` equals `pos_in_range − r_k` whatever the scaling (known-issues G1). The 7 features are one number that `x_hist[:, -1]` already contains.
   - They are 7 of the 23 gate-input dimensions, so they aren't outnumbered.
   - The real "drowning" is upstream. The regime cues (approach side, velocity, days in zone, volatility relative to range) must be pulled out of 252 raw inputs by an encoder trained only through μ and σ.
2. **The likelihood gives the gate nothing to identify regimes with.**
   - Moment mixing plus |μ| ≪ σ means the gate mostly selects σ (`gating.md` §1).
   - Unsupervised regularisers can make gates sharp but not semantic. In the benchmark, contrast is ≤0.02 and MI(event; winner) ≤0.01.
   - Semantics must come from outside: labels, or gate inputs restricted to level and regime features.
3. **The labels are sparse and partly use hindsight** (G14).

   | FSM label | Emitted at | Hindsight |
   |---|---|---|
   | hover | the step where dwell ≥ 3 in the zone | none (causal) |
   | pullback (→ Bounce) / breakout (→ Break) | `exit + 9` (10 confirmation steps) | 9 days |
   | timeout | consolidation + 20 | none; dropped in Exp 2 |

   - Exp 2 reads the gate at the **exit day** for Bounce and Break (`analysis.py:83`). The side price left on is already observable there; only the 9-day persistence is hindsight. So a causal classifier can learn much of the label, provided it gets an approach-side feature.
   - Counts in one 55k-window run: 794 Bounce, 652 Hover, 198 Break. That is about 3% of windows labelled, with imbalance pullback > hover ≫ breakout.

**Conditions for success:**
- (a) The gate or classifier gets causal features that can actually predict the labels.
- (b) Supervision is soft and annealed, or a KL prior, so the NLL can override it where an expert can't fit. For example, Hover's σ is capped at ≤0.021 (`mlp.py:261`) and can't model volatile "hover" windows.
- Expect some `val/loss_sde` cost. The question is how much.

## 2. Designs

### (A) Separate gate on a small causal feature set, no labels
- `π = softmax(G(g_t)/T)` with the `g_t` from `improving-gating.md` item 2. G is LayerNorm → 2×32 SiLU. Experts keep `h`.
- Loss unchanged; pair it with `mutual_info` or `regional_variance`.
- **Pros:** cheap and causal, and it directly tests "drowning".
- **Cons:** it may route purely on volatility, which is sharp but not Fibonacci-semantic.
- **Files:**
  - `amgm/data/neural_SDE.py` (`NeuralSDESample` 16-24; compute from raw windows before line 112, or invert via `sample_min`/`sample_range`);
  - `mlp.py` (`__init__` 191, `forward` 203-241);
  - `runner.py` (163-183);
  - mirror in `generate_samples._compute_features`, `gating_diagnostics/statistics.window_features` (65-78) and `analysis.py:135`.

### (B) Pre-trained event classifier (two-stage)
- **Stage 1:** `C(g_t) → q_t` over {Bounce, Break, Hover, None}, using dense labels from the FSM `state_trace` (`return_state_trace=True`):
  - Hover on in-zone CONSOLIDATING steps;
  - the episode outcome from entry to exit;
  - None elsewhere.
  - Freeze the levels at `entry_index`. Use weighted CE or focal loss with weights ∝ freq^−0.5.
- **Stage 2:**
  - **B1 (frozen):** `π = q[:3]/Σq[:3]`.
  - **B2 (residual):** `π = q_None·π_learned + Σ_k q_k·e_k`, so the learned gate acts only where the classifier abstains.
- **Infrastructure:** `MLPModel(input_length=|g|, hidden_sizes=(32,16), output_size=4)` (`mlp.py:5-18`), with a Lightning loop modelled on `train_classification.py`.
- **Pros:** guaranteed, interpretable semantics. Precedents: hard MoE, StableMoE distil-then-freeze, and Task-MoE / DEMix routing by labels.
- **Cons:** it captures only the causally predictable part of the labels. Experts are forced onto data they can't fit. Label noise from the 0.02 tolerance passes straight through.

### (C) Joint training with an annealed auxiliary loss on gate logits
- `L = L_SDE − R + λL_bal + α(s)·(1/|M|)·Σ_{i∈M} w_{y_i}·KL(q̃_i ‖ π_i)`
  - M = labelled windows; q̃ = smoothed one-hot (ε = 0.1) or the soft q from (B).
  - α warms up at 1.0 for 200 steps, then cosine-decays to 0.05 over about 1000 steps (copy `EntropyBetaScheduler`).
  - Compute it on pre-temperature logits.
- **Files:**
  - `route_target` and `route_mask` in the dataset (run the FSM per issue over the full series in `NeuralSDEDataset.__init__`, around lines 88-103);
  - `supervised_route_loss` in `gate_regularizers.py`;
  - the term at `runner.py:212`;
  - log `train|val/loss_route`.
- **Pros:** the NLL can override the labels, and α can be swept. Posterior regularisation argues for a prior-style constraint over hard targets.
- **Cons:** only about 3% of windows carry signal. If α is too low the gate drifts back to volatility routing.

### (D) Recommended: A, then C with a B prior, on a true mixture likelihood
1. Restrict the gate to `g_t` (A).
2. Switch to the mixture NLL (`improving-gating.md` item 1). Responsibilities become identifiable, which is the Weigend et al. 1995 time-series-regime setting.
3. Add the (C) term as an annealed KL towards the soft classifier prior (B). Better still, apply it to the posterior h.
4. Keep a small-β `mutual_info`.

## 3. Experiment protocol

0. **Probe test.** This decides the hypothesis cheaply, with no SDE retraining.
   - Using the existing checkpoint and Exp 2 outputs, fit logistic regression and a small MLP predicting the event label at `gate_index` from five feature sets:
     - (i) `f_t` (`distance_0..6` in `real_gates.csv`);
     - (ii) `Z_t` (recompute from the checkpoint encoder);
     - (iii) `h`;
     - (iv) `g_t`;
     - (v) `g_t + h`.
   - Use GroupKFold by `issue_id` with a 10-day embargo. Report macro-F1, per-class AUROC and Brier score against the class prior.
   - Read the result:
     - **(iv) ≫ (iii):** the "encoding/drowning" explanation holds; go ahead with A → D.
     - **All near chance:** the labels aren't causally predictable, and supervised gating will only hurt the NLL.
     - **(iii) ≈ (iv) > prior, but gate alignment ≈ 0:** the information is present and the *objective* ignores it. Fix the objective first (mixture NLL).
1. **Fix the evaluation:**
   - make the FSM levels causal (frozen at entry) and use the same rolling-window scaling as the model (G13);
   - use a time-based split, e.g. train ≤ 2017 and validate on 2018-2020, to match the Exp 2 window.
2. **Arms:**
   - baseline (regional_variance);
   - A;
   - A + mixture NLL;
   - B1;
   - B2;
   - C with α₀ ∈ {0.1, 0.3, 1.0};
   - D.

   Use ≥3 seeds per arm with the same issue IDs.
3. **Metrics:**
   - `val/loss_sde` mean ± std, flagging any Δ larger than 1 seed-std;
   - alignment from `event_summary.csv`: the diagonal p(expert | event) and margin = p_correct − max p_other;
   - `MI(argmax π; event)/H(event)` and `event_contrast` from `gate_health.py`;
   - the same on dense FSM labels;
   - entropy and winner share;
   - rollout fidelity: FSM on generated paths, event rate per 100 days, dwell-time JS/KS, return KS, ACF(|r|).
4. **Acceptance:**
   - alignment margin improves by ≥0.15 across seeds;
   - `val/loss_sde` stays within 1 std of the baseline, or at most 1% worse;
   - rollout event-rate JS improves.
