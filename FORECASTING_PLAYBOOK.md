# A Playbook for Time-Series Forecasting

A domain-agnostic, step-by-step methodology for building and *honestly evaluating* forecasting
models on time-series data — one or more target series observed on a regular cadence, with
optional exogenous companion series. It distills a full worked exercise into a repeatable
recipe you can drop onto a new dataset in any domain (environmental sensing, energy demand,
traffic, finance, demand planning, IoT telemetry, epidemiology, …).

The throughline: **decide how you will judge success before you model, measure the noise floor
of that judgement, and never let a difference smaller than the noise become a "finding."** Most
of the value in a forecasting project is in the evaluation protocol, not the model — so that
half of this guide is deliberately the most detailed, and you should resist the temptation to
skip it for the model zoo.

---

## How to read this document

The phases are ordered as you should execute them. Each states its **goal**, the **decisions**
it produces, and — where relevant — **figure specs** written to be *information-dense and
multi-dimensional* (every plot should encode ≥3 variables and drive a specific decision; a plot
that shows one number is a table).

> **Principle boxes** like this flag the load-bearing ideas. If you skim, read these.

> **Principle — Two modes of work.** Keep *exploratory* work (searching, tuning, hunting for
> signal) and *confirmatory* work (the held-out number you report) physically separate, in both
> code and write-up. Conflating them is the most common way forecasting results are silently
> inflated.

---

## Phase 0 — Frame the problem and the evaluation *first*

**Goal:** a short contract you commit to before touching a model. Six decisions:

1. **Target & cadence.** What variable `y`, at what sampling interval Δt? Is the grid regular,
   or does it need snapping/resampling/aggregation? Is it continuous, count, intermittent
   (many zeros), or bounded?
2. **Forecast setup.** "Given observations up to *t*, predict `y` at *t+h*." Index every target
   at its **forecast origin** *t*, not the target time *t+h* — this matches production use and
   makes leakage easy to reason about.
3. **Horizons & forecast strategy.** Forecast at several lead times, and decide *how* you
   produce a multi-step forecast (this is a real architectural choice, not an afterthought):
   - **Recursive** — one one-step model fed its own predictions. Compact; compounds error.
   - **Direct** — a separate model per horizon. No compounding; ignores cross-horizon
     structure; more models to train.
   - **Joint / multi-output** — one model emits the whole horizon vector at once (most DL
     forecasters; sequence-to-sequence).
   - **DirRec / hybrids** — direct models that also consume earlier-horizon predictions.
4. **Point vs probabilistic.** Do you need a number, or a distribution/interval? Many real
   decisions (inventory, capacity, risk) need quantiles. Decide now — it changes models *and*
   metrics (Phases 8, 10).
5. **One series or many (local vs global).** A single target → a *local* model. Many related
   series (stores, sensors, SKUs) → a *global* model trained across all of them usually wins
   (Phase 8). Even with one target, exogenous companions make this multivariate.
6. **Metric, baselines, and the held-out period** (Phases 3–5). Decide these *now*; they govern
   everything downstream. Reserve a final, never-touched slice and write down its boundaries.

> **Principle — The horizon is a first-class axis.** Treat "(strategy, model, feature set,
> hyperparameters)" as a function *of horizon*. Most real datasets reward a different recipe at
> *t+1* than at *t+100*.

---

## Phase 1 — Acquisition, provenance, and a reproducible pipeline

**Goal:** a clean, gap-aware, timezone-correct dataset you can regenerate from scratch, plus a
written record of how the data behaves.

- **One reproducible ETL path.** Raw → cleaned, schema-unified, single canonical table with a
  gap-free time index. Script it; no manual steps.
- **Sentinels, units, ranges.** Map domain missing-value sentinels (`-99.9`, `9999`, `""`) to
  `NaN`; verify units and physical ranges; confirm the timezone and whether daylight-saving
  shifts exist (prefer a fixed offset if the source is fixed-offset).
- **Provenance & silent drift.** Re-download a slice you already have and diff it. Sources
  *revise history* without notice — values, not just new rows. If they do, document the
  signature (how much changed, where, which fields, whether it's one-sided) and **pin a baseline
  metric with a test** so a future re-pull is caught instead of silently shifting your headline.
- **Point-in-time discipline (flagged here, enforced in Phase 6).** Record, per input, *when a
  value actually became available* vs the timestamp it describes. Revised/late-arriving data is
  the deepest leakage trap (see 6.5).

> **Principle — Absolute metrics aren't portable across data vintages.** If the data can be
> revised (or you switch sources), absolute error is not comparable across snapshots — but
> *skill versus a baseline* (Phase 3) usually is. Lead with skill.

---

## Phase 2 — Exploratory data analysis

**Goal:** understand structure, scale, gaps, trend, multi-scale seasonality, autocorrelation,
stationarity, and which companions carry information — converting each into a concrete
downstream decision (lag grid, transform, differencing, baseline form, candidate features).

Build the following. Each is specified to be multi-dimensional.

### Fig 2.1 — Coverage / availability matrix
**Encoding:** x = time (binned), y = each series/feature, colour = fraction present (non-NaN);
rows ordered by first-valid timestamp. **Reads:** the joint (source × time × completeness) trade
space — when each source switched on, the common overlap window, which features are too sparse.
**Drives:** the length-vs-breadth call (long history × few sources vs short × many) and the
sparse-column drop threshold.

### Fig 2.2 — Target distribution, raw vs transformed
**Encoding:** overlaid histograms/violins of `y` raw, log, and Box-Cox/Yeo-Johnson; annotate
skew, tail mass, zero-fraction; inset empirical CDF. **Reads:** heavy tails, zero-inflation /
intermittency, multimodality, the cost of a Gaussian assumption. **Drives:** target transform,
robust vs standard scaling, and whether to model quantiles or a count/intermittent likelihood.

### Fig 2.3 — Decomposition & stationarity
**Encoding:** an **STL (seasonal-trend-Loess) decomposition** — stacked panels of observed,
trend, one or more seasonal components, and remainder — beside a **rolling mean/variance** strip
and the verdicts of **unit-root tests (ADF, KPSS)**. **Reads:** how much variance is trend vs
seasonal vs irregular, whether the level/variance drift (non-stationarity), and the strength of
each seasonal cycle. **Drives:** whether to difference / detrend, which seasonalities to encode,
and whether variance-stabilising transforms are needed. (Add a **periodogram / spectral density**
panel to *discover* unknown cycle lengths rather than assuming them.)

### Fig 2.4 — Autocorrelation across lead time
**Encoding:** ACF and PACF vs lag, **multiple series overlaid** (target + each companion), with
markers at the lags crossing key thresholds; optionally overlay "persistence error vs horizon"
derived from the ACF. **Reads:** how fast the process decorrelates → how hard each horizon is and
how strong persistence will be where. **Drives:** the lag/rolling grid (place lags at the
autocorrelation knees, not round numbers) and the expected persistence↔seasonal-naive crossover.

### Fig 2.5 — Multi-scale seasonality calendar
**Encoding:** 2-D heatmap of mean `y` over two cyclical axes (e.g. month × hour-of-day), colour
= mean, with a second panel for variance and marginal profiles. **Reads:** diurnal, weekly, and
annual structure *and their interaction* in one frame (a 1-D seasonal curve hides the
interaction). **Drives:** calendar features and the form of the seasonal baseline (condition on
the cycle that actually moves the mean).

### Fig 2.6 — Cross-source lead–lag structure
**Encoding:** matrix of pairwise correlations at best lead/lag (colour = correlation, annotation
= lag at peak cross-correlation), or small-multiple lagged cross-correlation curves. **Reads:**
which companions *lead* the target and by how much, and which are mutually redundant.
**Drives:** the candidate exogenous set and whether any series is a genuine leading indicator vs
a co-mover. (Caution: at an aggregate-energy level, co-located series often move *together*
[lag 0] even when physics implies a delay — verify the lead empirically before engineering one.)

### Fig 2.7 — Feature × horizon predictive screen
**Encoding:** heatmap, y = candidate feature (raw + engineered), x = horizon, colour =
predictive strength of *feature(t)* vs *target(t+h)* (rank/|corr| **or mutual information** so
non-linear links show); diverging map if signed. **Reads:** the (feature × horizon × strength)
cube — which inputs matter at which lead time, and how signal decays with horizon. **Drives:** an
informed first shortlist and the expectation that importance is horizon-dependent.

> **Principle — Every EDA figure ends in a decision.** If you can't name the modelling choice a
> plot changes, cut it.

---

## Phase 3 — Baselines before models

**Goal:** the references that turn an absolute error into a *skill* statement, and that double as
genuinely strong competitors. Build these before any ML; a model that can't beat them isn't one.

- **Persistence / naïve:** `ŷ(t+h) = y(t)`. Strong at short horizons for autocorrelated series.
- **Seasonal-naïve:** `ŷ(t+h) = y(t+h−m)` for period *m*. The baseline to beat for strongly
  seasonal data, and often surprisingly hard.
- **Seasonal mean / climatology:** the cycle-conditioned historical mean (e.g. mean by
  hour-of-day or week-of-year of the *target* time). Horizon-independent; a "regress to the
  seasonal mean" floor.
- **Drift / random walk + trend**, and **Theta** (a decomposition method that won M3 and remains
  a top statistical baseline) — cheap, and frequently competitive with ML at long horizons.

Expect a **crossover**: persistence/seasonal-naïve win short, seasonal-mean/Theta win long; the
"better baseline" is their lower envelope. Locating the crossover tells you *what kind of skill
is even available* at each horizon.

> **Skill & scale-free reference.** Report **skill** = `1 − error(model)/error(baseline)`
> (positive = beats baseline), and prefer a **scaled** error like **MASE** (mean absolute error
> divided by the in-sample one-step naïve MAE) so results are comparable across series and
> scales. Skill and MASE are portable across data vintages where raw RMSE is not.

---

## Phase 4 — The evaluation protocol (test selection) — the crux

**Goal:** a selection-and-reporting scheme that doesn't lie to you. Most of the rigor lives here.

### 4.1 Split chronologically, never shuffle
Random K-fold leaks the future into the past through autocorrelation. Split by time, always.

### 4.2 Three roles, kept separate
- **Train** — fit parameters.
- **Validation** — choose hyperparameters, features, architectures.
- **Test** — touched *once*, to report.

If you tune on the test set, your headline is optimistically biased by exactly the amount of
searching you did.

### 4.3 Rolling-origin (walk-forward) backtesting — for *selection*
A single train/validation split is **one noisy draw**. Instead, validate on several consecutive
held-out blocks with an expanding (or sliding) training window:

```
fold 1:  train [........]      val [—]
fold 2:  train [..........]    val  [—]
fold 3:  train [............]  val   [—]   …
```

Select on the **mean across folds**; the **fold-to-fold spread is your real uncertainty**. This
de-biases selection and exposes regime sensitivity (a calm period vs a volatile one can differ
by more than every model gap combined).

These are **validation** folds — you iterate on them, so by definition they are not the test
set. Keep the Phase-4.2 **test** — the final, once-touched held-out (or the pre-committed blind
period below) — untouched until the recipe is frozen, then score it *once* (optionally as its
own rolling-origin pass you don't tune against). (Naming caveat: scikit-learn's CV API labels
*every* held-out fold `test` [`TimeSeriesSplit` → `(train_index, test_index)`]; in the
train/validation/test deployment vocabulary used here, folds you select on are **validation**.)

### 4.4 Embargo the horizon (purge leakage at the seam)
Because the target looks `h` steps ahead, the last `h` training origins "see" into the held-out
block. Drop a **gap of `h` steps** between train and held-out in every fold (and at the
train/test seam too). Equivalent to a purge/embargo in financial backtesting.

### 4.5 A pinned comparison window
For head-to-head comparisons, fix the held-out boundaries so that changing a model/feature
changes *only the thing under comparison, not the rows*. (When ablating sources, also clip to the
common overlap so every variant trains on identical rows.)

> **Principle — One window does triple duty by default; stop it.** Hyperparameter tuning,
> feature/source selection, and final reporting collapsing onto the same rows is the core failure
> mode. Each selection layer on shared data biases the reported number downward. Select on
> validation/folds; report on the held-out window once.

> **Principle — Pre-commit a true blind set.** Reserve a genuinely future slice (e.g. the next
> period as it arrives). Pre-register the candidates. Score by *re-fitting the same pipeline* on
> the same training data — not by loading a serialized model — and enforce the feature schema so
> drift surfaces as an error, not a silent wrong prediction.

---

## Phase 5 — Understand the noise floor of your KPI

**Goal:** know the uncertainty of your headline metric *before* comparing models, so you can tell
a finding from a coin flip. Give this its own section; it changes how you read everything else.

### 5.1 Absolute uncertainty — block bootstrap
Resample the held-out residuals to put a confidence interval on the metric. Because residuals are
**autocorrelated**, resample *contiguous blocks* (moving-block bootstrap) longer than the
dominant autocorrelation scale — an i.i.d. bootstrap badly understates the spread. Report the
metric's standard error / 95% CI. (Common outcome: the CI half-width *exceeds* the gaps between
your top models — which is the point.)

### 5.2 Model-vs-model — *paired* test
Two models scored on the **same rows** have correlated errors (they see the same hard periods),
so the *difference* is far better resolved than either absolute metric. Bootstrap the **paired**
per-origin error difference (same resampled blocks indexing both models), or use a
**Diebold–Mariano** test. A difference is real only if its paired CI excludes zero.

### 5.3 Stochastic models need seed variance
For models with random initialisation/optimisation (neural nets, some boosters), a single seed's
score has run-to-run variance that can exceed the hyperparameter effect you're ranking. Average
over several seeds and report the spread.

> **Principle — Sub-noise gaps are not results.** Absolute CIs for two models can overlap heavily
> while their *paired* difference is decisively non-zero — or vanishingly small. Judge every
> model/feature change by the paired interval, not the point estimate. A per-horizon "winner"
> chosen by a tiny gap on one window is usually selecting noise; report which contenders *tie*.

---

## Phase 6 — Feature engineering (leakage-safe)

**Goal:** a design matrix that exposes structure without smuggling in the future — including the
*operational* future.

- **Past-only transforms.** Lags, rolling stats (mean/std/min/max/quantile), momentum/deltas,
  expanding stats. **Shift any window feature so it ends at `t−1`, not `t`** — a rolling stat
  including the current (or future) step is leakage. Enforce with a test. Place lags by
  autocorrelation (Fig 2.4), not round numbers.
- **Differencing / detrending** where Phase 2.3 flagged non-stationarity, so the model sees a
  stable target; remember to invert the transform when scoring.
- **Circular encoding.** Angular variables (direction, hour, day-of-year, phase) → `(sin, cos)`
  so 359°→1° is 2° apart, not 358°.
- **Calendar / event features.** Hour, day-of-week, day-of-year, holidays, promotions, regime
  flags — as the seasonality and known events demand.
- **Residual / boosted-on-baseline targets.** Predict `y(t+h) − baseline(t)` instead of the
  level. Letting persistence/seasonal-naïve carry the level and a model learn only the *delta* is
  often a meaningfully easier problem and makes skill native. (Strong at short horizons; can hurt
  at long ones once the residual is mostly noise — verify per horizon.)
- **Two representations.** Engineered lag/rolling matrices suit linear/tree models; raw windowed
  channels suit sequence/DL models. Build whichever the model consumes.

### 6.5 Point-in-time correctness (the leakage most people miss)
Look-ahead in *features* (above) is the obvious leak; the subtle one is **feature availability
lag** — using a value the model could not actually have had at prediction time. Two traps:
- **Revised/restated data** (economic series, settled metrics, late-arriving telemetry): train on
  the *vintage available at origin t*, not the final revised value.
- **Publication/ingestion latency**: if an input arrives with a delay in production, lag it by
  that delay in training too.
Build features against a **point-in-time (as-of) view** of every source. This is the dominant
real-world leakage source in finance and operations and silently inflates backtests.

> **Preprocessing, fit on train only.** Drop columns whose *train* missing-fraction exceeds a
> threshold (a near-empty column mean-imputes to a near-constant that quietly corrodes gradient
> models). Impute, then scale — **robust (median/IQR) scaling for heavy-tailed data** so spikes
> don't dominate; pass through bounded encodings. Persist the fitted transformer and **enforce
> its schema at inference** (missing required column → error; unexpected column → drop).

---

## Phase 7 — Feature importance & selection

**Goal:** the smallest input set that pays its way, *per (horizon, model family)* — chosen with
the right tool for the granularity of the decision. Climb this ladder; don't default to the most
expensive rung.

1. **Filter (model-free).** Univariate screens — correlation, **mutual information**, the Fig 2.7
   feature×horizon panel. Cheap, ignore interactions; use to shortlist, not to decide.
2. **Embedded (selection for free).** **L1/elastic-net** zeroes coefficients along a
   regularisation path; **tree/GBM importances** rank inputs during fitting. Your default when
   you have many individual features — the model selects as it learns.
3. **Model-agnostic attribution.** **Permutation importance** (shuffle one input, measure the
   performance drop — no refit) and **SHAP** (per-prediction attribution + interaction structure).
   The modern defaults for "which inputs matter and how," and they expose interactions filters
   miss.
4. **Wrapper search.** **Forward selection / backward elimination / RFE** — systematically add or
   remove inputs, refitting and scoring each candidate set. Powerful but O(n) refits and noisy;
   reserve for a shortlist.

### Grouped ablation — the special case for *data sources*
When the unit of decision is a whole **source/sensor/feature-family** (each with real acquisition
cost), do a two-sided group ablation — the most decision-relevant flavour of wrapper search:
- **Add-one:** from the minimal model, add one source; record the change (marginal value *in
  isolation*).
- **Drop-one (leave-one-group-out / LOCO):** from the full model, remove one source (marginal
  value *given the rest*).
Their **disagreement is the signal**: add-one ≫ drop-one ⇒ redundant with others; drop-one ≫
add-one ⇒ complementary only in context. Keep a source if add-one helps *or* drop-one hurts by
more than a threshold expressed **relative to the Phase-5 noise floor** (never on a sub-noise
gain). Run every cell on identical rows (fixed overlap window) so only the columns change.

> **Principle — Selection is conditional, and "more" isn't monotone.** The right set differs by
> model family and horizon (linear wants a small low-variance subspace; trees extract value linear
> models discard; sequence models exploit temporal/phase structure). And adding inputs is *not*
> reliably helpful: under weak regularisation or little data, extra columns are noise the model
> must absorb and can push the "everything in" ceiling *below* a parsimonious model — while with
> ample data or strong regularisation they're often neutral. Don't assume a direction; measure it
> with a rolling-origin paired test.

---

## Phase 8 — Model architectures

**Goal:** the right family per horizon, chosen on validation/folds — not a leaderboard of
one-window scores. Climb the ladder, stopping when added complexity stops paying (paired test);
**simpler families routinely win, especially with limited data or long horizons.**

1. **Classical statistical.** ARIMA/SARIMAX, exponential smoothing (ETS / Holt-Winters),
   state-space / structural models (Kalman filtering, BSTS), Theta. Often the right answer for
   univariate or low-data problems and *brutal* baselines — the M3/M4 competitions are littered
   with ML models that couldn't beat them. Native uncertainty, interpretable components.
2. **Regularised linear on engineered features** (Ridge / Lasso / Elastic-Net). Fast, strong,
   interpretable; frequently the *long-horizon* champion because shrinkage resists noise once
   signal is weak. A stiff baseline for everything above it.
3. **Gradient-boosted trees** (LightGBM / XGBoost / CatBoost / HistGBM). Non-linearity and
   interactions, native missing-value handling; the **global** GBM was the M5 winner. Often best
   at *short* horizons (rich residual signal) and worst at *long* ones (it fits the noise the
   residual has become). Prefer boosting on the baseline residual.
4. **Sequence neural nets** (RNN / GRU / LSTM / TCN) on raw windowed channels. Worth the cost when
   temporal/phase or cross-series dynamics carry signal the engineered features can't. Tune **per
   horizon** (longer horizons want longer context) and **average over seeds**.
5. **Forecasting-specific deep learning.** **N-BEATS / N-HiTS** (pure-DL, strong, partly
   interpretable), **DeepAR** (autoregressive *probabilistic* RNN), **Temporal Fusion Transformer**
   (covariates + quantiles + attention interpretability), and the transformer line
   (Informer → Autoformer → FEDformer → PatchTST → iTransformer). Shine with many series, rich
   covariates, and scale. **Honest caveat:** a plain linear model (DLinear) matches or beats many
   transformer variants on standard benchmarks — don't assume DL is the ceiling.
6. **Foundation / pretrained models.** **Chronos, TimesFM, TimeGPT, Moirai, Lag-Llama** —
   zero/few-shot forecasting from a pretrained backbone. Excellent for prototyping, cold-start,
   and many-series settings; benchmark them, don't adopt blindly.
7. **Decomposition / industry frameworks.** **Prophet** (trend + multi-seasonality + holidays,
   analyst-friendly) and STL-plus-a-model. Interpretable, robust to missing data and business
   calendars; rarely the most accurate but cheap and legible.

### Cross-cutting axes (often matter more than the specific model)
- **Multi-horizon strategy** (Phase 0.3): direct vs recursive vs joint multi-output vs DirRec.
- **Global vs local:** with many related series, one **global** model (shared parameters across
  series, series id / static covariates as features) usually beats per-series **local** models —
  more data per parameter, cross-series generalisation. This is the single biggest lever the
  classical/local mindset misses.
- **Point vs probabilistic:** quantile regression, GBM with pinball loss, DeepAR/TFT
  distributions, or **conformal prediction** (incl. adaptive/time-series variants) wrapped around
  any point model to get calibrated intervals.
- **Ensembling.** Average the surviving near-equal models (simple mean / `nanmean`, or stacking
  on validation). With several correlated-but-good models the average reliably shaves error at no
  modelling risk — the M4 winner was a hybrid ensemble — and is usually the sensible shipping
  artefact.

---

## Phase 9 — Comparison techniques

**Goal:** rank candidates honestly.

- **Pair on common rows.** Compare predictions on identical origins; intersect to where both
  produced a value.
- **Rolling-origin means with fold spread**, not single-window points.
- **Paired bootstrap / Diebold–Mariano** for every "A beats B" claim (Phase 5.2).
- **Report ties explicitly.** When several models fall inside the winner's CI, the decision turns
  on simplicity, stability, latency, and cost — not a noise-level gap. Prefer the simplest model
  in the tied set.

---

## Phase 10 — KPIs / metrics

Report a small, complementary set — never a single number. Each KPI carries its uncertainty
(Phase 5) and is stratified by regime.

**Point accuracy**
- **RMSE** — penalises large misses; matches the squared objective and spike-sensitive domains.
- **MAE** — robust, in target units; a large RMSE/MAE ratio flags heavy-tailed errors.
- **Bias** (mean signed error) — systematic over/under-prediction, often concentrated in the
  regimes you care about most (extremes, peaks).

**Scale-free / cross-series** (so errors compare across series and magnitudes)
- **MASE** — error scaled by the in-sample naïve error; the M-competition standard, robust to
  scale and to zeros.
- **sMAPE / WAPE** — percentage errors; useful but beware MAPE's blow-up near zero and its
  asymmetry — prefer **WAPE** (weighted absolute percentage) for volumes.
- **Skill vs baseline** — `1 − error/error_baseline`; the portable headline.

**Probabilistic** (if you forecast distributions/intervals — Phase 8)
- **Pinball / quantile loss** per quantile; **CRPS** for the whole predictive distribution.
- **Calibration:** interval **coverage** vs nominal and a **PIT histogram** (should be uniform);
  **Winkler / interval score** for interval sharpness-plus-coverage.

> **Principle — Diagnose *where* error lives, not just how much.** Stratify the headline metric by
> regime (calm/volatile, peak/off-peak, by season, by series) and by horizon. The worst 10% of
> conditions usually drives real-world value and is where models silently fail.

---

## Phase 11 — Headline figures

The few plots that carry the story. Keep them dense.

### Fig 11.1 — Skill / error vs horizon (the money chart)
**Encoding:** x = horizon, y = error; one line per surviving (model, feature-set) winner, stars on
the horizons each wins; the **better-baseline lower envelope** as a faint backdrop; an **error
band = fold spread** on the winner. **Reads:** which recipe to use at each lead time, how skill
decays, and how close the best model runs to the baseline (where the ceiling is). **Drives:** the
deployment recipe and where further effort is futile.

### Fig 11.2 — Contender forest plot with CIs
**Encoding:** small-multiples by horizon; y = each contender, x = mean rolling-origin metric with a
**bootstrap CI bar**; colour = {winner / tied-with-winner / significantly-worse} (colour from the
*paired* test, bar from the *absolute* bootstrap); shade the winner's CI. **Reads:** ranking,
absolute uncertainty (wide, overlapping bars), and the paired verdict (colour) at once — making
visible that single-window CIs overlap yet paired differences still separate. **Drives:** honest
winner/tie calls.

### Fig 11.3 — Residual diagnostics
**Encoding:** 2-D density (hexbin) of residual vs prediction with marginal histograms, plus a
residual-vs-key-feature panel with quantile bands; for probabilistic models, a **reliability /
PIT** panel. **Reads:** heteroscedasticity, tail bias (systematic under-prediction of extremes),
which feature ranges are mis-modelled, and whether intervals are calibrated. **Drives:** target
transform, quantile modelling, conformal calibration, or a targeted feature for the failing regime.

### Fig 11.4 — Importance / ablation (input × horizon)
**Encoding:** signed-colour heatmap(s) — for grouped sources, two panels (add-one gain, drop-one
cost); for features, a permutation-importance or mean-|SHAP| heatmap, y = input, x = horizon,
colour = effect. **Reads:** the (input × horizon × value) cube. **Drives:** the per-horizon kept
input set.

---

## Phase 12 — Results write-up: structure & order

Order the narrative so each result rests on the one before it:

1. **Objective & evaluation contract** — target, horizons, forecast strategy, point/probabilistic,
   metric, baselines, the pinned test window. State the rules before any number.
2. **Data & provenance** — sources, cadence, cleaning, drift/revision behaviour (with the pinned
   baseline test that guards it), point-in-time handling.
3. **EDA** — the decision-driving figures (Phase 2), each ending in the choice it informed.
4. **Baselines** — naïve/seasonal-naïve/seasonal-mean/Theta, the crossover, skill + MASE. Every
   later number is reported as skill against these.
5. **Evaluation protocol & noise floor** — split, rolling-origin, embargo, bootstrap noise floor.
   Put this *before* model results so the reader knows the resolution of every comparison.
6. **Feature importance & selection** — the ladder, grouped ablation where relevant, thresholded
   against noise, per (horizon, family).
7. **Model selection** — the architecture ladder + cross-cutting axes; hyperparameter/seed
   handling; ensembling. Separate the **exploratory** sweep from the **confirmatory** rolling-origin
   selection, and label them.
8. **Significance** — which differences are real (paired tests, forest plot). Lead with the
   *structural* findings (family×horizon, breadth×horizon, strategy effects) that survive, not the
   individual sub-noise winners.
9. **Blind / real-world validation** — pre-committed candidates scored on a truly unseen period.
10. **Roadmap & negative results** — what to try next *and what you tried that didn't work.*

> **Principle — Lead with the baseline, report skill, show uncertainty, every time.** "RMSE 0.45"
> is unfalsifiable; "skill +0.09 vs seasonal-mean, paired CI [+0.04, +0.13] over 5 folds" is a
> claim. Negative results (a feature/source/architecture that *didn't* help) are first-class —
> they redirect effort and stop re-treading.

---

## Phase 13 — Reproducibility & experiment logging

- **Append-only run log.** One record per experiment (name, metrics, data sources, train/val/test
  windows, feature count, hyperparameters, git SHA, timestamp) in a single committed file; read it
  back as a dataframe for every chart and table. The log *is* the results database.
- **Deterministic environment.** Pin dependencies and the lockfile; seed every stochastic step.
- **Scripts over notebooks for anything reported**, so results regenerate end-to-end.
- **Cheap re-derivation.** A `--plot-only` / cached path so figures rebuild from the log without
  re-running the expensive sweep.
- **Leverage the ecosystem.** Mature libraries encode much of this: `statsforecast` / `mlforecast`
  / `neuralforecast` (Nixtla), `sktime`, `darts`, `GluonTS`, `Prophet`. Use them for baselines,
  backtesting splitters, and reference models rather than re-implementing — but keep the
  evaluation protocol (Phases 4–5) under your own control.

---

## Appendix — One-page checklist

- [ ] Target, cadence, horizons, **forecast strategy** (direct/recursive/joint), point-vs-
      probabilistic, **local-vs-global**, metric, baselines, and held-out window — decided **before**
      modelling.
- [ ] Reproducible ETL; sentinels→NaN; timezone verified; drift/revision checked and a baseline
      pinned by a test; point-in-time view of every source.
- [ ] EDA figures, each ending in a decision: coverage, distribution, **STL+stationarity+spectral**,
      ACF/PACF, seasonality calendar, lead–lag, feature×horizon.
- [ ] Baselines: persistence, **seasonal-naïve**, seasonal-mean, drift, **Theta**; crossover located;
      skill + **MASE** defined.
- [ ] Chronological split; train/val/test separated; rolling-origin with a **horizon embargo**.
- [ ] **Noise floor measured** (block bootstrap) **before** comparing models; paired test ready.
- [ ] Leakage-safe features (window features end at `t−1`; circular encoding; residual targets);
      **point-in-time / availability-lag** respected; preprocessing fit on train only with schema
      enforcement.
- [ ] Importance/selection ladder used (filter → embedded → permutation/SHAP → wrapper); grouped
      ablation for source decisions; thresholded **against the noise floor**; per (horizon, family).
- [ ] Architecture ladder climbed from **classical** upward, stopping when the paired test stops
      paying; per-horizon tuning; seeds averaged; ensembling considered.
- [ ] Every "A beats B" backed by a **paired** test; ties reported; simplest tied model preferred.
- [ ] KPIs: point (RMSE/MAE/bias) + **scale-free (MASE/sMAPE/WAPE)** + skill + **probabilistic
      (pinball/CRPS/coverage)**, each **with uncertainty**, stratified by regime.
- [ ] Headline figures: error-vs-horizon, forest-with-CIs, residual/calibration diagnostics,
      importance/ablation.
- [ ] Write-up ordered objective→data→EDA→baselines→protocol/noise→importance→models→significance→
      blind→roadmap; exploratory and confirmatory work separated.
- [ ] A genuinely blind future period reserved and pre-committed.

> **The one rule, restated.** Decide how you'll judge success, measure the noise in that
> judgement, and treat any difference smaller than the noise as the noise it is.
