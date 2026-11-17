# Deck brief: gate-input features for the Fibonacci MoE Neural SDE

> **How to use this file.** Open PowerPoint and start Claude. Attach this file and the 8 PNGs in `figures/`, then say: *"Build the deck described in presentation.md. Put the speaker notes in each slide's notes pane."* The rest of this file is written for the Claude agent.

---

## 0. Instructions for the agent

**What you are building.** A 13-slide main deck plus 6 backup slides. One presenter (Daniel) talks for **13–14 minutes**. The topic is one change to a neural network, *giving the gating network its own input features*, and what the experiments showed. The audience is a university project course: the supervisor, other students, and an opposing group. They saw a half-time talk about the same model, so they know the basic setup. They are comfortable with ML but are not specialists in mixture-of-experts or finance.

**Goal.** Results that are **easy to take in**. Every slide has one message, and that message is the slide title (a full sentence, at most ~12 words). Someone who reads only the titles should get the whole story.

**Hard rules**
1. **Use only the numbers in this file.** Section 3 is the fact sheet. Do not round them differently, do not compute new ones, and do not invent any.
2. **Speaker notes** go in the notes pane of each slide, word for word as given in section 1. They are the presenter's script. You may fix typography but must not shorten them.
3. **Body text**: at most ~25 words per slide outside tables, figures and big-number callouts. No paragraphs on slides. The notes carry the explanation.
4. **Figures**: insert the PNG named in the spec. Never redraw or alter a provided figure. Crop only where the spec says so. Keep the aspect ratio.
5. **Native visuals**: where the spec says *native chart*, *native table* or *native shapes*, build them with PowerPoint objects from the inline data so they stay editable.
6. Put **"Backup"** in the section name and title prefix of the backup slides, after a divider slide titled "Backup".

**Visual style**
- 16:9, white background, plenty of white space, one sans-serif font family (e.g. Aptos or Calibri). Titles 28–32 pt, body 18–20 pt, notes on tables 14 pt, never below 12 pt.
- **Expert colours, used everywhere and matching the figures:** Bounce `#0072B2` (blue), Break `#D55E00` (orange), Hover `#009E73` (green). Highlight colour for "our result": the same blue `#0072B2`. Neutral grey `#9E9E9E` for baselines and controls.
- Use a "big number" callout style (number 40–54 pt, a one-line label under it) for headline numbers.
- Use status chips on slide 13: green "Fixed" and amber "Open".
- Slide numbers on all slides except the title. A small footer on content slides: "Gate-input features · Experiment B".

**Vocabulary (use these terms consistently, see the glossary in section 2)**
- "the gate" (not "router"), "expert", "Bounce / Break / Hover".
- Variant names on slides: **Baseline**, **Baseline + features**, **RV** (regional variance), **RV + features**, **RV + h + features**, **RV vol-only (placebo)**.

**Timing (target 13:35)**

| # | Slide | Time |
|---|---|---|
| 1 | Title | 0:20 |
| 2 | The claim under test | 0:50 |
| 3 | Half-time recap | 1:00 |
| 4 | Diagnosis | 1:00 |
| 5 | The change | 1:15 |
| 6 | The 14 features | 1:20 |
| 7 | What the gate sees | 1:00 |
| 8 | Experiment design | 0:50 |
| 9 | How we score "meaningful" | 1:00 |
| 10 | Result 1 | 1:15 |
| 11 | Result 2 | 1:20 |
| 12 | Result 3 | 1:15 |
| 13 | Where we stand and next steps | 1:15 |
| — | Q&A | — |

If the presenter needs a 10-minute version: hide slides 3 and 7 and shorten slide 8 to one sentence (saves ~2:30).

---

## 1. Slide-by-slide specification

### Slide 1: Title

- **Title:** Giving the gate its own eyes
- **Subtitle:** Gate-input features for the Fibonacci mixture-of-experts Neural SDE
- **Small line:** Daniel Ibrahimi · Project group: Ammar Alzeno, Elias Gaghlasian, Daniel Ibrahimi, Adam Chachoud
- **Visual:** none, or a thin band of three colour blocks (blue, orange, green) as a subtle nod to the three experts.

**Speaker notes**
> Hi everyone. At half-time we showed that the gate in our Fibonacci Neural SDE doesn't do what the paper claims. In this part I'll show the one architectural change we made since then: we gave the gate its own input features. Then I'll go through what that changed, what it didn't change, and what we think the real bottleneck is.

---

### Slide 2: The claim under test

- **Title:** The paper's claim: the gate should say "Bounce" when the price bounces
- **Visual: native shapes**, a left-to-right flow:
  1. Box "Last 252 days of prices" →
  2. Box "Gate" → three small output bars labelled π Bounce / π Break / π Hover (in expert colours) →
  3. Three expert boxes, stacked, in expert colours:
     - **Bounce**: "pushes away from the nearest Fibonacci level"
     - **Break**: "follows the last move"
     - **Hover**: "tiny drift, low volatility"
  4. → Box "Blend: μ = Σ πₖ μₖ, σ² = Σ πₖ σₖ²" → "Tomorrow's price move"
  - Above the gate, a speech-bubble callout: *"π_Bounce = 0.9 should mean: this is a bounce"*
- **Body text:** none beyond the diagram labels.

**Speaker notes**
> Quick reminder of the model. It looks at the last year of prices and generates the next day's move. Three experts each propose a drift and a volatility. Bounce pushes the price away from the nearest Fibonacci level, Break follows the last move, and Hover barely moves. A gate network outputs three weights that sum to one, and the experts' outputs are blended with those weights. The paper presents this gate as interpretable: if Bounce gets weight 0.9, the model is saying "this is a bounce". That is the claim we're testing. We call a gate that behaves like this *semantic*.

---

### Slide 3: Half-time recap

- **Title:** Half-time recap: regularisers change the gate's shape, not its meaning
- **Visual:** `figures/01_expA_winner_shares.png`, left ~60% of the slide.
- **Right side:** three short rows with icons or ticks:
  - ✗ Entropy family: near-uniform gate (entropy > 0.9)
  - ✓ Regional variance: confident and balanced (entropy 0.009, winners .36 / .33 / .31)
  - ✗ All five: event_contrast ≤ 0.019
- **Bottom callout (one line):** "The gate became sharp, but not about bounces and breaks."

**Speaker notes**
> At half-time we tried to fix the gate by changing the regulariser. The chart shows which expert wins on validation windows for each of the five variants. The paper's entropy baseline never lets Bounce win at all. The other entropy-based variants spread weight roughly a third each everywhere, so the "winner" is decided by tiny margins and flips between datasets. Regional variance, the regulariser we proposed, fixed that: the gate became confident and used all three experts equally. But none of the five variants lined up with real bounce, break or hover events. The best score on our event metric was 0.019, which is essentially zero. So the regulariser controls *how sharp* the gate is, not *what it's about*. That pointed us to the gate's inputs.

---

### Slide 4: Diagnosis

- **Title:** Diagnosis: the gate's seven Fibonacci inputs are one number repeated
- **Visual: native shapes.**
  - Top: the equation, large and centred:
    **f<sub>k</sub> = (x<sub>t</sub> − L<sub>k</sub>) / range = (position in yearly range) − r<sub>k</sub>**
  - Middle: a vertical 0-to-1 price ruler with the 7 Fibonacci levels marked (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1) and a dot for today's price. Seven thin arrows from the dot to each level, all labelled with the same colour, and the label: "move the dot → all 7 arrows change by the same amount".
  - Bottom, small grey text: "The encoder already sees the position. The gate learns nothing new, and nothing about speed or volatility."
- **Small side box, titled "Two more causes (in the objective)":** "Drift is only 1–3% of volatility" · "Bounce and Break push the same way on 45.7% of windows"

**Speaker notes**
> Why can't the gate be semantic? Look at what it reads. Besides the encoder output, it gets seven Fibonacci features: the distance from today's price to each level, divided by the range. If you substitute the definition of the levels, every one of them collapses to the same number, the price's position in its yearly range, minus a constant. And the encoder already sees that position. So the gate gets nothing it didn't already have, and in particular nothing about how fast the price is moving or how volatile it is, which is exactly what separates a bounce from a breakout. There are two more causes in the training objective, shown on the right, which I'll come back to at the end. This slide is the one we could fix directly.

---

### Slide 5: The change

- **Title:** The change: the gate gets its own feature vector, the experts don't
- **Visual: native shapes**, recreating this diagram (left to right):

```
                 ┌────────────┐     ┌──────────────────┐
x_hist  ───────▶ │  Encoder   │───▶ │ h = [Z_t ‖ f_t]  │──────────────▶  Experts (always read h)  ─┐
(252 prices)     └────────────┘     │   23 numbers     │                                            │
   │                                └──────────────────┘                                            ▼
   │                                        ┆ mode "h" (original) / "h+features"            ┌───────────────┐
   │                                        ▼                                               │ μ = Σ πₖ μₖ    │
   │             ┌───────────────────┐   mode "features"   ┌────────────────────┐           │ σ² = Σ πₖ σₖ²  │
   └───────────▶ │ g_t: gate features│ ──────────────────▶ │ Gate  π = softmax  │ ────────▶ └───────────────┘
                 │ 14 numbers        │                     └────────────────────┘
                 └───────────────────┘
```

  - Fill the **g_t** box and the **Gate** box in light blue (`#0072B2` at ~12% opacity) to mark what is new.
  - Draw the `h → Gate` connector **dashed grey** with the label "h (original) · h + features".
  - Draw the `g_t → Gate` connector **solid blue** with the label "features".
- **Below the diagram, three small labelled chips:** `h`: original · `features`: gate reads only g_t · `h + features`: gate reads both (37 numbers)

**Speaker notes**
> So the change is in the architecture. We compute a separate feature vector g, only for the gate, from the same price window. The experts are untouched: they still read the encoder state plus the Fibonacci features. The gate now has three possible input modes: the original h, only the new features, or both concatenated. This follows a principle that goes back to Jordan and Jacobs in 1994: a gate can only route well if the regimes are separable in the gate's *own* input space, and the gate's inputs don't have to be the experts' inputs. Keeping the experts fixed matters for the experiment: any change in behaviour comes from what the gate sees, not from the experts.

---

### Slide 6: The 14 features

- **Title:** 14 causal features, measured in "typical days" instead of raw distance
- **Visual: native table**, 6 rows, with a small icon per row if available:

| Group | Feature(s) | What it tells the gate | Example value |
|---|---|---|---|
| Distance (7) | dist_vol₀ … dist_vol₆ = (x − Lₖ) / σ₂₀ | How many typical days away each level is | +2.5 (50% level), −3.4 (61.8%) |
| Approach (1) | approach | +1 heading toward the nearest level, −1 away | −1 |
| Velocity (3) | vel₁, vel₅, vel₁₀ | Momentum compared with noise (±1 = noise, ≥3 = strong) | 0.50 / 1.12 / 1.26 |
| Volatility trend (1) | log_vol_ratio = log(σ₂₀/σ₆₀) | Calming (<0) or heating up (>0) | −0.22 |
| Volatility level (1) | log_range_vol | Quiet trending year vs noisy year | +1.15 |
| Dwell (1) | zone_days | How long the price has sat at a level | 0 |

- **Footer row (grey, smaller):** "Placebo: **vol-only** = only the 2 volatility features, blind to the levels."
- **Small note under the title:** "σ₂₀ = typical daily move over the last 20 days. All features use only past prices and are clipped to ±5."

**Speaker notes**
> Here are the features. The key idea is a second ruler: the typical daily move over the last 20 days, sigma-20. We divide distances by it, so a level 0.05 away is one day away for a choppy stock but five days away for a calm one. That's exactly the scale information the old features lacked. On top of the seven distances we added: are we heading toward the nearest level or away; how fast are we moving compared with noise, over 1, 5 and 10 days; is volatility rising or falling; is this a noisy or a trending year; and how many days have we sat at the level. Everything is causal, so no future prices are used. The right column is a running example: a price drifting up away from the 50% level at a normal pace. Finally there's a placebo, vol-only, with just the two volatility features. Keep it in mind, it matters later.

---

### Slide 7: What the gate sees

- **Title:** What the new gate input looks like on one real stock
- **Visual:** `figures/02_gate_input_example.png`. This is a tall figure: place it on the right ~55% of the slide at full slide height.
- **Left side:** five short labels aligned with the five panels from top to bottom, connected by thin leader lines:
  1. Price (100 trading days from 2018-01-02)
  2. Distance to each level (red = above, blue = below)
  3. Velocity and approach
  4. Volatility and dwell
  5. Resulting gate weights (RV + features, seed 1)

**Speaker notes**
> This is what it looks like in practice, on one real stock over a hundred trading days in 2018. At the top is the price. Below it, the seven distance features as a heat map: red means the price is above that level, blue means below, so you can see the price sliding down through the levels. Then momentum and approach, then the two volatility features and the dwell time. At the bottom are the gate weights that the regional-variance-plus-features model produces from these inputs. Notice that the gate switches sharply between experts. The question for the rest of the talk is whether those switches mean anything.

---

### Slide 8: Experiment design

- **Title:** Experiment B: six variants, three seeds, one change per comparison
- **Visual: native shapes**, a pairing diagram with arrows labelled "one change":
  - Row 1: **Baseline** (entropy, gate reads h) → **Baseline + features**
  - Row 2: **RV** (gate reads h) → **RV + features**
  - **RV** → **RV + h + features**
  - **RV** → **RV vol-only (placebo)**, with a hatched grey box
- **Right-side fact chips (3):** "18 runs (6 × seeds 1, 2, 3)" · "Same 100 US stocks for every seed" · "Compared within each seed (seed noise ≈ 0.07–0.09 in fit)"

**Speaker notes**
> To test it we trained six variants with three random seeds each, so eighteen runs, on the same hundred stocks. Every arrow on the slide is a paired comparison where exactly one thing changes, within the same seed. That matters because the fit varies by 0.07 to 0.09 between seeds, which is bigger than most of the effects we're looking for. Pairing removes most of that noise. The four regional-variance variants differ only in what the gate reads: h, the features, both, or just volatility.

---

### Slide 9: How we score "meaningful"

- **Title:** How we score "meaningful": does each expert get more weight on its own events?
- **Left: short method strip (native shapes, 3 steps):** "Real 2018–2020 paths" → "Rule-based detector labels 1,644 events (794 Bounce · 198 Break · 652 Hover)" → "Read the gate on each event day"
- **Right: native table** (mean gate weight per event type; RV + features, seed 1). Highlight the diagonal cells in their expert colour:

| Event ↓ / Expert → | Bounce | Break | Hover |
|---|---|---|---|
| Bounce events | **0.464** | 0.389 | 0.147 |
| Break events | 0.364 | **0.398** | 0.238 |
| Hover events | 0.431 | 0.297 | **0.273** |
| Own − other | +0.048 | +0.051 | +0.107 |

- **Big-number callout under the table:** **event_contrast = +0.069** · label: "mean of the bottom row"
- **Scale strip (thin horizontal bar):** −1 "names wrong" · 0 "gate ignores events" · +1 "perfect". Mark the threshold **0.05** = "healthy" cut-off.

**Speaker notes**
> Entropy and winner shares only tell you the gate's shape. To check meaning we need labels from outside the model. A rule-based detector finds bounces, breakouts and hovers on real 2018 to 2020 paths: 1,644 events. For each event we read the gate's weights on the day it should react. Then we ask: does each expert get more weight on its own events than on the others? For Bounce here: 0.464 on bounce events, minus about 0.41 on the rest, gives +0.048. Average that over the three experts and you get the event contrast, here +0.069. Zero means the gate ignores the event type, and negative means the names are wrong. Expect small numbers: bounce and break labels need ten days of hindsight that a causal gate can't see, so we compare variants with each other, not with one. We call a gate healthy above 0.05.

---

### Slide 10: Result 1

- **Title:** Result 1: features alone improve the fit but turn the gate into a volatility dial
- **Visual:** `figures/03_same_path_all_variants.png`, **cropped to the top row only** (price, Baseline, Baseline + features). Add a thin blue outline around the "Baseline + features" panel.
- **Three big-number callouts below the figure:**
  - **−0.051 nats**: better fit on all 3 seeds
  - **100%**: of real windows won by Hover, every seed
  - **−0.47 / −0.31 / −0.28**: correlation of Hover weight with volatility trend (baseline: −0.08 to −0.13)
- **One-line takeaway at the bottom:** "Better variance prediction, not regime recognition."

**Speaker notes**
> First result: give the original baseline the features, and the fit improves on every seed, by about 0.05 nats. That sounds like good news. But look at the gate on the right. Hover sits at 0.8 to 0.9 the whole time and the weights just slide up and down. On real data Hover wins one hundred percent of windows on all three seeds. What's happening is that Hover has a small, capped volatility and the other experts have larger ones. So by sliding weight between them, the gate dials the blended volatility up and down, and the new volatility features tell it exactly how much. You can see it in the correlation: the Hover weight falls when volatility rises, at −0.47 on seed 1, versus about −0.1 for the original gate. So the fit gain is real, but it comes from predicting the size of the next move, not from recognising bounces. This is the behaviour Weigend and colleagues reported for gated experts back in 1995.

---

### Slide 11: Result 2

- **Title:** Result 2: features + regional variance give the same split on every seed and the best alignment
- **Left (~60%):** `figures/04_event_contrast_by_variant.png`
- **Right (~40%): native column chart**, titled "Spread of winner shares across seeds (lower = reproducible)":

| Variant | Mean sd of winner shares |
|---|---|
| RV | 0.171 |
| RV + features | 0.008 |

  Colour RV grey and RV + features blue. Add data labels.
- **Under the chart, three tick bullets:**
  - Winner shares .34 / .40 / .26 on **every** seed
  - Contrast up on **all 3** seeds (0.000 → 0.052)
  - **2 of 3** seeds healthy, the only ones apart from the placebo
- **Small grey footnote:** "RV + h + features: 0.018 ± 0.039, non-semantic on all seeds. Mixed with h, the features get ignored."

**Speaker notes**
> Second result, and the most positive one. Combine the features with regional variance. Left chart: event contrast per variant, with each seed as a marker and the dashed line at the 0.05 healthy threshold. RV plus features is the only variant whose mean is above the line and whose three seeds are all positive. Contrast goes from 0.000 for plain regional variance to 0.052, up on every seed. Two of three seeds get a healthy verdict. The right chart is, I think, the clearest win: reproducibility. With the original input, regional variance finds a different split on every seed. With the features it finds the same split every time: 34, 40 and 26 percent, varying by one percentage point at most. One more detail: if you give the gate both h and the features, it routes on h and the effect disappears. The features have to be the gate's only input.

---

### Slide 12: Result 3

- **Title:** Result 3: but the split is by price band, and volatility alone does as well
- **Top half, "What decides the winner?" (native bar chart, horizontal, 2 bars):** balanced accuracy of a small decision tree predicting the RV + features gate's winner:
  - From position in the yearly range alone: **0.83–0.97**
  - From the 2 volatility features alone: **0.41–0.50**
  - Mark chance at 0.33 with a dashed line. Show ranges as bars to the midpoint with the range as the data label: midpoints 0.90 and 0.455, labelled "0.83–0.97" and "0.41–0.50".
- **Bottom half, native table "Full features vs. placebo":**

| | RV + features (14) | RV vol-only (2) |
|---|---|---|
| event_contrast | 0.052 ± 0.017 | 0.048 ± 0.101 |
| event_mi | 0.009 ± 0.009 | **0.027 ± 0.005** |
| Verdicts | healthy, healthy, non-semantic | healthy, healthy, non-semantic |

- **Right-hand callout (blue border):** "A price band is not an event. A price in the middle band can bounce, break or hover."

**Speaker notes**
> But what is that reproducible split made of? We trained a small decision tree to predict which expert wins from the gate's inputs. The position in the yearly range alone predicts the winner with 83 to 97 percent balanced accuracy. So the gate is cutting the price range into horizontal bands: low, middle, high. That's stable, but a band isn't an event. Then the placebo. The vol-only gate, which can't see a single Fibonacci level, matches the full feature set on event contrast and actually beats it on event mutual information on all three seeds. So whatever event information the gate carries is no more than volatility alone provides. Its two healthy seeds are luck in naming: on seed 3 it gives the noisy windows to Break instead of Bounce and its contrast goes to −0.067. The level, approach, velocity and dwell features help stability, but they don't add event meaning.

---

### Slide 13: Where we stand and next steps

- **Title:** Better inputs were necessary, not sufficient. Next: change the objective
- **Left: native table with status chips:**

| Problem | Status |
|---|---|
| Hard collapse / starved expert | 🟢 Fixed: RV + features holds on every seed |
| Uniform-soft gate | 🟢 Fixed: regional variance |
| Split depends on the seed | 🟢 Fixed: gate reads features only |
| Gate is not Fibonacci-semantic | 🟠 Open |
| Expert names not pinned to behaviours | 🟠 Open |

  (Use coloured chips instead of emoji if available.)
- **Right: "Next steps", three numbered cards:**
  1. **True mixture likelihood**: −log Σ πₖ 𝒩(Δx | μₖ, σₖ²), so experts compete for each data point
  2. **Make Break the mirror of Bounce**: continuation *through* the nearest level, so the two never agree
  3. **Keep the vol-only placebo and use 6 seeds**: pass = features beat vol-only on both metrics, every seed
- **Bottom banner, one line:** "Under moment mixing, the loss gives the gate no reason to separate Bounce from Break."

**Speaker notes**
> To sum up. With regional variance plus the new features, collapse, starvation, the uniform gate and seed dependence are all fixed. The gate is now stable and reproducible. But it is still not about Fibonacci events, and the expert names still aren't tied to behaviour. So the new inputs were a necessary change but not a sufficient one. The bottleneck has moved to the training objective. Because the experts are blended into one Gaussian before the loss, the loss can't tell "trust this expert a little" from "trust it a lot but let it whisper", and the gate's gradient is mostly about volatility. Next we want a true mixture likelihood, where experts compete for each data point. We also want Break redefined as the exact opposite of Bounce, and we'll keep the placebo as the bar to beat, with six seeds so the tests have power. Thanks, happy to take questions.

---

## Backup slides (after a divider slide titled "Backup")

### B1: Event matrices for every variant
- **Title:** Backup: mean gate weight per event type, all variants (seed 1)
- **Visual:** `figures/B1_event_matrix_grid.png`, full width.
- **Notes:**
> Rows are detected event types and columns are experts. A semantic gate would show a bright diagonal. Baseline + features is Hover-dominated on every row, about 0.85 to 0.88. The other gates give almost the same row on every event type. RV + features is the only one with a visible lean along the diagonal, at contrast +0.069, and even that is small.

### B2: Gate weight vs. distance to the nearest level
- **Title:** Backup: the RV + features gate splits by which level is nearest
- **Visual:** `figures/B2_rvfeat_gate_vs_distance.png`
- **Notes:**
> The bottom panels split the windows by nearest level. Near the 50% and 61.8% levels Bounce gets almost all the weight. Near 0% it's Hover, and near 100% it's Break. The weight depends on *which* level is nearest, which is price position, more than on what the price is doing at that level. This is the price-band split from slide 12.

### B3a: Concrete level interactions, RV (gate reads h)
- **Title:** Backup: one Bounce, one Break and one Hover, as the plain RV gate sees them
- **Visual:** `figures/B3a_rv_event_examples.png`
- **Small label, top right:** "Gate reads h (original input). Compare with the next slide."
- **Notes:**
> These are the same three real events I'll show on the next slide, but with plain regional variance, where the gate reads the original input h. The events were picked by duration, not by gate quality. The gate holds long, nearly constant blocks. On the bounce, it's on Break when the price leaves the 50% level, and it switches to Hover about two weeks later as the price falls to the bottom of its range. On the breakout, Bounce leads for almost the whole hundred days, including the move through the top level. The hover example is the same stock and path as the bounce, so its gate trace is the same. The gate is confident, but the expert it picks doesn't match the event.

### B3b: Concrete level interactions, RV + features
- **Title:** Backup: the same three events, as the RV + features gate sees them
- **Visual:** `figures/B3b_rvfeat_event_examples.png`
- **Small label, top right:** "Gate reads the 14 features."
- **Notes:**
> The same three events, now with the gate reading the new features. The routing changes a lot: it switches much more often, and in the bounce window it moves into a mixed Break and Hover phase instead of one long block. But it switches far more often than events happen, so one switch near an event is not evidence that the gate recognised it. That is why we rely on the aggregate event metrics, not on single examples.

### B4: Paired differences per seed
- **Title:** Backup: paired differences, variant minus reference, same seed
- **Visual: native table**

| Pair | Metric | Seed 1 | Seed 2 | Seed 3 | Mean ± sd | Same sign on all 3? |
|---|---|---|---|---|---|---|
| Baseline + feat − Baseline | fit (val_loss_sde) | −0.057 | −0.035 | −0.061 | −0.051 ± 0.014 | Yes, better (p ≈ 0.02) |
| RV + feat − RV | event_contrast | +0.057 | +0.094 | +0.004 | +0.051 ± 0.045 | Yes (p ≈ 0.19) |
| RV + feat − RV | fit (val_loss_sde) | −0.063 | +0.036 | −0.041 | −0.023 ± 0.052 | No |
| vol-only − RV | event_mi | +0.016 | +0.028 | +0.027 | +0.024 ± 0.007 | Yes (p ≈ 0.03) |
| RV + h + feat − RV | event_contrast | −0.039 | +0.080 | +0.011 | +0.018 ± 0.060 | No |

- **Footnote:** "Paired t-tests, 2 degrees of freedom, on rounded values. Indications only."
- **Notes:**
> Lower fit is better. The only effects with the same sign on all three seeds are the fit gain from features on the baseline, the contrast gain from features under regional variance, and the event-MI gain of the vol-only placebo.

### B5: Limitations
- **Title:** Backup: limitations
- **Bullets:**
  - 3 seeds: 3 of 3 in the same direction gives a two-sided sign-test p of only 0.25
  - "Healthy" uses a heuristic 0.05 threshold. RV + features scores 0.069, 0.053 and 0.034
  - Validation is effectively in-sample: overlapping windows, random split, and the 2018–2020 evaluation overlaps training
  - Labels are sparse (~3% of windows) and Bounce/Break need 10 days of hindsight
  - The vol-only run on seed 2 looks like a failed run (fit 0.48 worse than its reference)
  - Generation quality was not re-assessed for these variants
- **Notes:**
> These results rank variants against each other. They are not out-of-sample evidence. With six seeds, six out of six in the same direction would give p ≈ 0.03.

---

## 2. Glossary (for the agent; do not invent other definitions)

| Term | Meaning |
|---|---|
| Gate | Small network that outputs weights π = (π_Bounce, π_Break, π_Hover), positive and summing to 1 |
| Moment mixing | Blending the experts' drift and variance with the gate weights into one Gaussian: μ = Σ πₖ μₖ, σ² = Σ πₖ σₖ² |
| h | The original gate and expert input: 16-number encoder summary plus the 7 Fibonacci distances (23 numbers) |
| g_t / price_v1 | The 14 new gate-only features |
| vol-only | Placebo gate input: only log_vol_ratio and log_range_vol |
| σ₂₀ | Standard deviation of daily price changes over the last 20 days ("a typical day") |
| RV (regional variance) | Regulariser that rewards the gate for giving different windows different weights (confident and balanced routing) |
| Entropy (gate) | 1 = uniform weights, 0 = fully confident |
| Winner share | Fraction of windows where an expert has the largest weight |
| event_contrast | Mean over experts of (weight on own events − weight on other events). 0 = ignores events, < 0 = names wrong, ≥ 0.05 = "healthy" |
| event_mi | Share of the event label's uncertainty explained by which expert wins. Ignores expert names |
| Verdicts | hard-collapse, starved, uniform-soft, non-semantic, healthy (fixed rules of thumb) |
| Seed | Random initialisation. 3 seeds per variant |

---

## 3. Fact sheet (the only numbers allowed in the deck)

**Setup**
- 100 US stocks, daily prices 2014-12-31 to 2020-12-31. 252-day windows. 20 epochs.
- Event evaluation on real 2018–2020 paths: 1,644 events = 794 Bounce (48%), 198 Break (12%), 652 Hover (40%). About 3% of windows are labelled.
- Bounce/Break labels need 10 days of hindsight. Zone tolerance 0.02.
- Seed-to-seed spread in fit: 0.07–0.09.

**Diagnosis**
- All 7 Fibonacci features = position in range − rₖ (rₖ ∈ {0, 0.236, 0.382, 0.5, 0.618, 0.786, 1}).
- Daily drift is 1–3% of volatility (|μ| 0.0004–0.0016 vs σ ≈ 0.047).
- Bounce and Break drift signs agree on 45.7% of real windows.

**Experiment A (half-time, seed 1, gate reads h)**

| Variant | Val entropy | Val winners B / Br / H | Event contrast |
|---|---|---|---|
| Entropy (baseline) | 0.906 | .00 / .35 / .65 | 0.010 |
| Adaptive entropy | 0.935 | .36 / .15 / .49 | 0.012 |
| Temperature | 0.934 | .01 / .83 / .16 | 0.018 |
| Adaptive ent. + temp. | 0.948 | .62 / .31 / .07 | 0.019 |
| Regional variance | 0.009 (confidence 0.997) | .36 / .33 / .31 | 0.012 |

**Experiment B (mean ± sd over seeds 1–3; winners on real paths)**

| Variant | Fit (val_loss_sde, lower = better) | Entropy | event_contrast | event_mi | Real winners B / Br / H | sd of winners across seeds |
|---|---|---|---|---|---|---|
| Baseline | −2.470 ± 0.087 | 0.882 | 0.008 ± 0.002 | 0.003 ± 0.003 | .12 / .38 / .50 | .21 / .42 / .44 |
| Baseline + features | −2.521 ± 0.094 | 0.607 | 0.016 ± 0.006 | 0.001 ± 0.001 | .00 / .00 / 1.00 | 0 / 0 / 0 |
| RV | −2.478 ± 0.072 | 0.012 | 0.000 ± 0.037 | 0.003 ± 0.001 | .33 / .41 / .26 | .09 / .20 / .23 |
| RV + features | −2.500 ± 0.082 | 0.021 | 0.052 ± 0.017 | 0.009 ± 0.009 | .34 / .40 / .26 | .01 / .01 / .00 |
| RV + h + features | −2.464 ± 0.064 | 0.025 | 0.018 ± 0.039 | 0.003 ± 0.001 | .36 / .33 / .31 | .04 / .07 / .09 |
| RV vol-only (placebo) | −2.308 ± 0.288 | 0.060 | 0.048 ± 0.101 | 0.027 ± 0.005 | .37 / .36 / .27 | .04 / .12 / .12 |

- Mean sd of winner shares across seeds: Baseline 0.359 · Baseline + features 0.000 (always Hover) · RV 0.171 · RV + features 0.008 · RV + h + features 0.066 · vol-only 0.091.
- RV + features event_contrast per seed: +0.069, +0.053, +0.034. Healthy on seeds 1 and 2.
- Verdicts across all 18 runs: 4 healthy (2 RV + features, 2 vol-only), 7 non-semantic only, 7 starved, collapsed or uniform-soft.
- Correlation of the Hover weight with log_vol_ratio (Baseline + features): −0.47, −0.31, −0.28 (seeds 1–3). Original baseline: −0.08 to −0.13.
- Decision tree predicting the RV + features winner (balanced accuracy, chance 0.33): position in range alone 0.83–0.97; 2 volatility features alone 0.41–0.50; the 12 non-volatility features 0.91–0.98.
- Classifier predicting the event label directly: 0.370 from all 14 features, 0.367 from vol-only, 0.333 chance.
- vol-only seed 3: event_contrast −0.067, event_mi 0.029 (separates events but gives them the wrong names).
