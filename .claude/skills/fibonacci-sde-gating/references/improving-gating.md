# Improving the gate: ranked catalogue

The items are ordered by expected impact on *semantic, non-collapsed* gating relative to effort. Each one was reconciled against the code on `main` (a2d27cf). Paths are relative to `fibonacci_neural_sde/`. Citations are in `literature.md`.

**Diagnosis behind the ranking** (see `gating.md` §1 and `known-issues.md` G1–G5, G12):
- The gate gets no input that isn't already in the history (G1).
- The moment-mixing objective gives no competitive pressure, and its gate gradient is dominated by σ.
- Bounce and Break drifts agree on 45.7% of real windows (G12).
- The entropy rewards are maximised by uniform gates.

Regularisers alone cannot fix this: `regional_variance` reaches I(X;E) ≈ 0.99 and is still non-semantic.

## 0. Prerequisites (do these before judging any gate change)

| Change | Where | Why |
|---|---|---|
| `checkpoint_monitor="val/loss_sde"` | `trainer_cfg/neural_SDE/US_Stocks.py:62` | `val/loss` includes −R (G6). |
| Fixed `rng_seed` and a fixed issue list, with ≥3 seeds (≥6 for a Wilcoxon over seeds) | `US_Stocks.py:15`; `--issue-ids-file` for Exp 2 | Seed variance is unknown (G8). |
| Time-based split with an embargo of ≥252 days; evaluate Exp 2 out of sample | `train_neural_SDE.py:43-50`; `analysis.py` `segment_origins` | Current gate statistics are in-sample (D1, D2). |
| Add a planted-regime synthetic dataset: Markov states with known drift sign and σ | next to `SyntheticNeuralSDEDataset` (`amgm/data/neural_SDE.py:202`) | This is the only ground truth for "does the gate recover regimes?". |
| Log mean σ_k and μ_k per expert | `runner.py` logging block | Shows whether experts actually differ. |

## 1. Mixture likelihood instead of moment mixing (main objective fix)

- **What:** score a real mixture density, the Jacobs/Jordan competitive objective and the MDN form:

  `L_SDE = −mean_b logsumexp_k [ log π_k − ½log(2πσ_k²dt) − (dx − μ_k dt)² / (2σ_k²dt) ]`

  with `log π = log_softmax(logits / T)`.
- **Why:**
  - The gate-logit gradient becomes `π_k − h_k`, where `h_k ∝ π_k N_k` is the posterior responsibility. Each expert's gradient is weighted by `h_k`, so experts specialise on the points they explain best.
  - Responsibilities become identifiable.
  - The between-expert drift dispersion is now inside the model.
- **Where:**
  - `NeuralSDEMoE.forward` (`mlp.py:203-267`): also return the stacked `mu_k (B,3)`, `sigma_k (B,3)` and the logits. Keep `μ = Σπμ_k` for `x_tp1_pred`.
  - `runner.py` `_step_nll` (166-174) and `_compute_loss` (185-212).
  - `generate_samples._rollout_sde` (107-150): sample k ~ π, then dx ~ N(μ_k, σ_k²). Do *not* keep using the mixed moments.
  - The MLP and PatchTST variants keep the old loss.
- **Starting point:** β=0 (entropy reward off), λ_bal=0.1 or a Switch loss (item 5), T=1.
- **Failure modes and checks:**
  - The split follows volatility rather than event type. Weigend et al. 1995 saw exactly this. Compare against a vol-only gate ablation.
  - A wide expert absorbs outliers.
  - σ_k collapses to 1e-3. Keep the floor or clamp log σ.
- **Evaluate:** `val/loss_sde` is a proper NLL under both objectives, so it is directly comparable. Measure alignment of both π (prior) and h (posterior) with events.

## 2. Informative, gate-specific inputs (feature fix)

- **What:** give the gate `g_t`, roughly 10–14 causal features computed on raw closes (or recovered via `sample_min` / `sample_range`):
  - `d_k = (P_t − L_k)/(σ20_t·P_t)`, clipped to ±5: level distances in volatility units;
  - |nearest distance|, plus an approach indicator `−sign(x_t − L*)·sign(v_t)`;
  - velocity `(P_t − P_{t−k})/(σ20·P)` for k = 1, 5, 10;
  - `log(σ20/σ60)` and range/vol `Δ/(σ20·P·√252)`;
  - days in the ±0.02 zone, and the approach side (a causal version of the FSM state).

  The gate becomes `π = softmax(G(g_t)/T)`, where G is a LayerNorm followed by a 2×32 SiLU MLP. Experts keep `h`. Optionally give the gate `Z_t.detach()` with 50% group dropout, or FiLM `γ(g)·Z + β(g)`.
- **Why:** G1. Regime variables help when they are restricted to the gate (RG-ResMoE 2026). MoE theory needs separable clusters in the router's input space (Chen et al. 2022).
- **Where:**
  - `NeuralSDESample` and the dataset (`amgm/data/neural_SDE.py:16-24, 112-115, 167-174`);
  - `NeuralSDEMoE.__init__` / `forward` (`mlp.py:191, 203-241`);
  - `runner._prepare_batch` (176-183);
  - **mirror the change** in `generate_samples._compute_features` and in `gating_diagnostics/statistics.window_features` and `analysis.py`, or the diagnostics will be wrong.
- **Failure mode:** with no labels the gate routes on volatility only. Check with a vol-only ablation.

## 3. Disambiguate the Bounce and Break experts

- **What:** define Break as continuation *through* the nearest level, `μ_break = −sign(x_t − L*)·softplus(·)`, the exact opposite of Bounce. Alternatively, let both act only in approach states.
- **Why:** today their signs agree on 45.7% of windows (G12, `gate_health.py` `real_sign_agree`).
- **Where:** `mlp.py:220-229, 249-250`.

## 4. Weak supervision from FSM labels, annealed (see `classifier-gating.md`)

- **What:** `+ α(s)·mean_{i∈M} w_{y_i}·KL(q̃_i ‖ π_i)` over the labelled windows M, with label smoothing ε = 0.1 or a soft classifier prior q.
  - α: warm up at 0.5–1.0, then cosine-decay to 0.05. Reuse the `EntropyBetaScheduler` pattern.
  - Use class weights ∝ freq^−0.5.
  - With item 1 in place, apply the KL to the posterior h rather than to π. That is posterior regularisation, and it leaks less.
- **Where:** a new `supervised_route_loss` in `gate_regularizers.py`; the term goes at `runner.py:212`; labels come from running the FSM per series in the dataset.
- **Caveat:** breakout and pullback labels need 9 days of hindsight (G14), and only about 3% of windows are labelled. Run the probe test in `classifier-gating.md` §3 step 0 first.

## 5. Replace uniform-entropy rewards with balance-only terms

- **What:** drop the `entropy` and `adaptive_entropy` rewards, whose optimum is the uniform gate. Use instead:
  - the Switch load loss: `α·K·Σ_k f_k·P_k`, with α = 1e-2, `f_k` = argmax fraction (no gradient) and `P_k` = mean π_k;
  - optionally the ST-MoE router z-loss: `1e-3·mean_b (logsumexp logits_b)²`, which stabilises logits at low T.
- **Where:** new functions in `gate_regularizers.py`, registered in `GATE_REGULARIZERS` or alongside `balance_loss` (84).
- **Note:** balancing prevents *starvation*; it does not create *specialisation*. Pair it with item 1, 2 or 4.

## 6. Run the existing `mutual_info` mode

- **What:** `gate_reg_mode="mutual_info"` with β 0.05→0.01. It needs no code change.
- **Expectation:** under moment mixing it should behave like `regional_variance`, giving confident but arbitrary gates. It is more useful combined with items 1 and 2.

## 7. Sparse gate: entmax-1.5

- **What:** replace the softmax at `mlp.py:241` with entmax-1.5, which gives exact zeros (the Tsallis-entropy argmax). Warm up with softmax for about 1 epoch to avoid dead experts.
- **Prefer it over Gumbel / straight-through at K=3.** With 3 experts you can marginalise exactly, and the mixture NLL already does so.
- **Learnable temperature** cannot be identified separately from the logit scale (`out_proj` can rescale), so it adds little. Annealing only acts as an optimisation prior, which matches the benchmark: winner shares changed but `loss_sde` did not.

## 8. Temporal smoothness and regime persistence

- **Cheap:** `λ_tv·mean|π_t − π_{t−1}|` over adjacent windows of the same stock, with λ_tv ≈ 0.1. This needs a sampler that batches contiguous runs, because `random_split` plus shuffle breaks adjacency.
- **Full:** a Hamilton/IOHMM-style filtered gate, `π_t ∝ (Aᵀ·filtered_{t−1}) ⊙ g(x_t)`. This is a substantial rewrite.
- **Compare against** Exp 4 statistics: switches per 100 steps and dwell lengths, real vs synthetic.

## 9. Gate-related generation fixes

- Restore T when sampling: call `model.on_load_checkpoint(checkpoint)` in `generate_samples._load_model_checkpoint` (G7).
- Consider log-return targets and bounded σ in return units. This removes the σ·x_range feedback loop (`evaluation.md` §1). It changes what σ means, so rerun the gating diagnostics afterwards.

## Recommended sequence

1. Prerequisites (item 0).
2. Probe test (`classifier-gating.md` §3 step 0). It decides whether supervision can help at all.
3. Items 2 + 3, then add item 1. Benchmark: baseline vs 2+3 vs 1+2+3, each with ≥3 seeds.
4. If alignment is still low but the probe says the labels are predictable, add item 4.
5. Then items 5, 6, 7 and 8 as refinements.

Change one thing at a time and run `gate_health.py` and the protocol in `analysis-protocol.md` after each.
