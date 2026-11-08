# Gating analysis protocol

This protocol extends Exp 1, 2 and 4 (`gating.md` §4). Paths are relative to `fibonacci_neural_sde/`. Every source file listed below is already written by the existing diagnostics, unless the step says "new script". Citations are in `literature.md`.

Step 1, plus the event metrics in step 3, are automated by `scripts/gate_health.py`. Run it first.

## Step 0: Leakage audit (before trusting any number)

- **Tag each window.** `split_membership.csv` (columns `dataset_index, split, issue_id, date`) records which windows were train or val. Join `real_gates.csv` to it on (issue_id, date) to tag each real window as train, val or unseen, and report every metric per tag. A large gap between tags means memorised routing.
- **Retrain for out-of-sample tests:**
  - Retrain with `end_date="2017-12-31"` and evaluate from 2018-06 onwards. The embargo must be at least 100 trading days, because of the 100-step segments plus 10-day confirmation.
  - Evaluate on stocks never used in training. With `rng_seed=None` the stock set is random, so pass the issue IDs explicitly.

## Step 1: Routing health per epoch (`epoch_metrics.csv`)

| Metric | Formula | Healthy | Collapsed |
|---|---|---|---|
| Marginal balance | H(p̄)/log 3, from the `p_*` columns | ≈1 | ≪1 (expert collapse) |
| Input dependence | I(X;E)/log 3 = H(p̄)/log 3 − `entropy` (`gate_health` `val_mi_input_expert`) | clearly > 0 | ≈0 (uniform-soft; the baseline is 0.035) |
| Importance CV² / load CV² | CV² over `p_*` / over `p_*_winner_share` | small | large |
| Effective experts | exp H(p̄) | ≈3 | →1 |

## Step 2: Routing stability

- **Within a run.** From `validation_epoch_XXX.npz` (keys `probabilities`, `dataset_index`, `sample_key`), compute the share of windows whose argmax at epoch e equals the final-epoch argmax (StableMoE-style curve).
- **Across seeds.**
  - Run `analyze_gating --checkpoint ... --issue-ids-file F` per seed, with the same F every time.
  - Align the `real_gates.csv` files on (issue_id, date).
  - Compute Cohen's κ on argmax, the mean JS divergence, and the correlation of log π_k. Experts are pinned by their structural constraints, so no permutation matching is needed.
- **Read it:**
  - κ > 0.6: reproducible routing.
  - Sharp gates with κ ≈ 0: arbitrary routing (non-semantic).

## Step 3: Semantic alignment (`events.csv`, `real_gates.csv`)

- **Soft MI.**
  - Let p(e|y) be the mean π over the events labelled y. Then I(E;Y) = H(Σ_y p(y)·p(e|y)) − Σ_y p(y)·H(p(e|y)); normalise by min(H(Y), log 3).
  - Null: permute labels *within stock* 1000×. CI: the existing stock-cluster bootstrap.
- **Contrast.** A = mean_y [π_{k(y)} | y − π_{k(y)} | ¬y] (`gate_health` `event_contrast`; currently ≤0.02).
- **Probes.** Multinomial logistic regression of the event label on log π (gate probe) and on the inputs (input probe: `distance_0..6`, `movement` from `real_gates.csv`, joined on `segment_id` with step = `gate_index`). Use GroupKFold by `issue_id`, plus a shuffled-label control.
- **Read the probes:**

  | Outcome | Meaning |
  |---|---|
  | gate probe ≈ input probe > prior | semantic gating |
  | input probe > gate probe ≈ prior | the information is there, but the **gate ignores it** (non-semantic) |
  | both ≈ prior | the events aren't predictable ahead of time (hindsight labels, G14); low alignment is expected, not a failure |

- **Watch for scaling mismatch.** The FSM uses origin-window scaling, while the gate uses rolling-window scaling (G13). Fix this before over-interpreting the probes.

## Step 4: Expert redundancy (`real_gates.csv`)

- Bounce's drift sign is sign(`distance`), and Break's is sign(`movement`) (`mlp.py:211-229`, mirrored in `statistics.py:74-77`).
- `gate_health` `real_sign_agree` gives the fraction of windows where these signs agree. It is **0.457** in every benchmark variant. On those windows the two experts differ only in magnitude, so they are weakly identifiable.
- Optional, as a new script: compute CKA or correlation between the expert heads' outputs on the same windows.

## Step 5: Knockout, oracle routing and gate-input ablation (new script)

- **Setup:** load with `gating_diagnostics/analysis.py` `load_moe_checkpoint` (restores T) and build features with `statistics.window_features`. Recompute the per-expert μ_k and σ_k from the heads (`mlp.py:249-265`), and score with the same formula as `runner._step_nll`.
- **Knockout:** set π_k = 0 and renormalise. Measure ΔNLL_k overall, per event label and per regime (step 6).
- **Oracle:** compute the NLL of each expert alone. Report regret = NLL_gate − min_k NLL_k, and how often argmax π equals the best expert (compare against 1/3).
- **Gate-only permutation:** permute `f_t` or `Z_t` *only in the gate input* and measure the JS change in π and ΔNLL.
- **Read it:**
  - **Healthy:** each knockout hurts most on its own event type, regret is small, and argmax π hits the best expert well above 1/3 of the time.
  - **Redundant experts:** ΔNLL ≈ 0 for every k.
  - **f_t ignored:** permuting `f_t` barely moves π. Expect this given G1.
- **Optional:** ALE plots of the gate logits against distance and movement (more robust than PDP for correlated inputs).

## Step 6: Regime slicing

- From `real_gates.csv` plus earlier prices (`analysis.py` `load_prices`), compute:
  - 20-day realised volatility, in terciles;
  - the 60-day trend sign;
  - an equal-weight cross-sectional "market" return;
  - a COVID window, 2020-02-19 to 2020-04-30.
- Recompute steps 1 and 3 within each slice.
- Sector isn't in the loaded columns. It may be available from `constituents.txt` (GICS), but that is unverified.
- **Read it:** a regime-aware gate shows Hover rising in low volatility and Break rising in the crash. If gate mass is the same across slices, the gate isn't regime-aware.

## Step 7: Comparing variants statistically

- **Seeds:**
  - Use ≥6 seeds per variant, and 10 if a pilot power analysis says so. With n = 5, a two-sided Wilcoxon test can't reach p < 0.05; its minimum p is 0.0625.
  - Pair variants by seed. One seed fixes stocks, split and initialisation.
- **Report:**
  - mean ± t-CI;
  - the paired difference with Cohen's d_z;
  - a hierarchical bootstrap (seeds, then stocks). With about 3 runs, prefer IQM with a stratified bootstrap.
- **Dependence:** stocks share calendar dates, so the stock-cluster bootstrap understates crisis variance. Add a stationary block bootstrap over dates, or a two-way stock × time-block bootstrap.
- **Multiple testing:** pre-register a primary set (ΔNLL, I(E;Y), I(X;E)) with a Holm correction. Use BH for exploratory grids such as bins and regimes.
