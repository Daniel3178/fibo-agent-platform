# Evaluating the generator (beyond gating)

Paths are relative to `fibonacci_neural_sde/`. Citations marked † were not re-checked by the researcher.

## 1. What exists today

| Piece | Where | What it does |
|---|---|---|
| Rollout | `experiments/neural_SDE/generate_samples.py` `_rollout_sde` (107-150) | Autoregressive Euler–Maruyama (EM) in normalised units. **Each step re-normalises the window**, and `x_next_original = x_next_norm·x_range + x_min`. The noise in price units is therefore σ·x_range, so a shock widens the range and raises later noise (a feedback loop). Prices are additive with no positivity constraint. |
| Validity | `amgm/utils/common.py` `sanity_check_path` (771-793) | Standard mode **drops** invalid paths (`generate_samples.py:817-819`); MC mode keeps them with `is_valid`. Dropping them is a selection bias that favours the model. Report the invalid fraction as a metric. |
| Distribution / event fidelity | `experiments/neural_SDE/Statistical_comparison.ipynb` | FSM event histograms with a JS distance; mean return, variance and max drawdown (MDD) per issue; a one-sided paired Wilcoxon test against Vasicek. It runs from `experiments/neural_SDE/` and the variant is chosen in cell 2. |
| TSTR | `experiments/neural_SDE/train_classification.py` | Trains the trend classifier on synthetic data and validates it on real data. Model type and synthetic share are hard-coded at lines 397-398. |
| Temporal gate realism | `analyze_temporal_gating` (Exp 4) | Real vs synthetic gate switching and dwell times (see `gating.md`). |

Saved result to know about: for `adaptive_entropy_temperature`, Wilcoxon cell 26 (49 issues) gives these mean absolute errors:

| Metric | Neural SDE | Vasicek | p |
|---|---|---|---|
| Mean return | 0.0026 | 0.0018 | 0.94 |
| Variance | 0.0026 | 0.0004 | 1.0 |
| MDD | 0.158 | 0.105 | 0.998 |

Return variance is about 6× over-dispersed, which fits the σ·x_range feedback loop.

## 2. What the problem really is

The model is a **heteroscedastic Gaussian AR(1)-style transition model fitted by one-step maximum likelihood**. It is not a continuous-time SDE fit, because dt=1 in normalised units. One-step NLL matches one-step conditionals only, and cannot penalise wrong multi-step path laws (exposure bias).

## 3. Recommended evaluation battery

Compute each item on held-out **test dates** (see §4), comparing real continuations with synthetic rollouts from the same origins.

1. **Stylized facts** (Cont 2001, Quant. Finance 1(2):223-236†, https://doi.org/10.1080/713665670):
   - heavy tails: kurtosis and Hill tail index;
   - ACF(r) ≈ 0 at lags 1-50;
   - slowly decaying ACF(|r|) and ACF(r²), i.e. volatility clustering;
   - leverage effect: corr(r_t, r²_{t+k}) < 0;
   - aggregational Gaussianity;
   - gain/loss asymmetry.

   The Quant GAN metric suite gives ACF scores, a leverage score and EMD by lag (Wiese et al. 2020, Quant. Finance 20(9), https://arxiv.org/abs/1907.06673).
2. **Two-sample tests:**
   - KS and Anderson–Darling on pooled returns;
   - kernel MMD (Gretton et al. 2012, JMLR†);
   - signature MMD on paths (Chevyrev & Oberhauser 2022†; for market generators, Buehler et al. 2020, https://arxiv.org/abs/2006.14498).
3. **Discriminative and predictive scores** (TimeGAN, Yoon et al. 2019†) and **TSTR/TRTS** (Esteban et al. 2017, https://arxiv.org/abs/1706.02633†).
4. **Event fidelity:** keep the FSM JS distances, but compute them on test dates and on *all* paths, not only valid ones.
5. **Invalid-path fraction** from `sanity_check_path`, reported per variant.
6. **Multi-seed:** run at least 5 seeds with a fixed issue universe (`rng_seed` set) and report mean ± std or CI. Apply a Holm correction when testing against several baselines.

## 4. Splitting

- Today the split is a random 80/20 over stride-1 windows (`train_neural_SDE.py:43-50`), which is leaky.
- Use a chronological split instead: train up to T1, an embargo of at least 252 days (one lookback window), validation, another embargo, then test after T2. Split on the sample's `test_dates`. This is purged/embargoed CV (López de Prado 2018, *Advances in Financial ML*, ch. 7).
- Point `generate_samples.py` `_build_dataset` (line 75) at test dates only, so rollouts start out of sample.

## 5. Fair baselines

| Baseline | Calibration | Notes |
|---|---|---|
| GBM | μ̂ = mean(log r) + σ̂²/2, σ̂ = std(log r) on the same 252-day window | the minimal fair null for daily equities |
| GJR-GARCH(1,1) with Student-t | MLE via the `arch` package | the standard daily-equity benchmark; leverage models beat GARCH(1,1) on equities (Hansen & Lunde 2005, J. Appl. Econometrics 20(7), https://doi.org/10.1002/jae.800) |
| OU/Vasicek | exact AR(1): X_{t+1} = a + bX_t + ε. OLS is the conditional MLE: κ = −ln b/Δ, θ = a/(1−b), σ² = Var(ε)·2κ/(1−b²) (Brigo et al., https://arxiv.org/abs/0812.4210) | The notebook hard-codes κ, μ, σ. Mean-reverting to mid-range already builds in a "range" prior, so this baseline is weak for stocks. |
| Heston | needs filtering-based likelihood or options | optional |

## 6. Is Fibonacci itself doing anything?

- Support and resistance levels have some empirical support:
  - Osler 2000, FRBNY EPR 6(2), https://www.newyorkfed.org/research/epr/00v06n2/0007osle.html;
  - Osler 2003, J. Finance† (order clustering at round numbers);
  - Lo, Mamaysky & Wang 2000, J. Finance†;
  - Chung & Bellotti 2021, https://arxiv.org/abs/2101.07410.
- **The Fibonacci ratios themselves don't.** Tsinaslanidis, Guijarro & Voukelatos 2022 (Expert Syst. Appl. 187) found no difference between Fibonacci and non-Fibonacci zones; this is based on the abstract only.
- **Ablation to run:** swap the ratios in `common.calculate_fibLevels` (9-34) for uniform levels (0.2/0.4/0.6/0.8) or random ones. Compare test NLL, event JS distances and gate–event alignment. Fix G1 in `known-issues.md` first, because otherwise the features are degenerate whatever the ratios are.
