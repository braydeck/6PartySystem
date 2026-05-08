# Clustering Session Brief — CES 2024 Political Typology
**Prepared:** 2026-03-02
**Dataset:** 2024 Cooperative Election Study (CCES), Common Content
**Working directory:** `/Users/bdecker/Documents/STV/Claude/`

---

## 1. WHAT WAS BUILT

### 1.1 Final EFA Item Set — 24 Items

All 24 items entered the polychoric correlation matrix and PAF factor model. Each item is coded **LOW = conservative** in the polychoric matrix (REV_BINARY items were recoded as `3 − x`; CC24_325 recoded as `40 − raw`; CC24_303 direction verified empirically). Factor anchoring below is based on highest absolute pattern loading in the final 24-item k=5 oblimin solution (`efa_loadings_k5_final.csv`).

| Variable | Question Text (abbreviated) | Domain | Primary Factor | \|λ\| |
|----------|----------------------------|--------|----------------|-------|
| **CC24_321d** | Support increase police by 10% | Policing | **F1** | 0.734 |
| **CC24_323b** | Support increase border patrols | Immigration | **F1** | 0.705 |
| **CC24_340f** | Support deny asylum at border | Immigration | **F1** | 0.664 |
| **CC24_321e** | Oppose decrease police by 10% [rev] | Policing | **F1** | 0.653 |
| **CC24_340e** | Support renew post-9/11 surveillance programs | Civil Rights | **F1** | 0.493 |
| **CC24_421_2** | Disagree: state/local election in 2024 was fair [high=distrust] | Election Trust | **F2** | 0.901 |
| **CC24_421_1** | Disagree: U.S. elections are fair [high=distrust] | Election Trust | **F2** | 0.726 |
| **CC24_423** | Low trust: federal government [4-pt; high=less trust] | Govt Trust | **F3** | 0.663 |
| **CC24_424** | Low trust: state government [4-pt; high=less trust] | Govt Trust | **F3** | 0.476 |
| **pew_churatd** | Church attendance frequency | Religion | **F4** | 0.688 |
| **CC24_325** | Abortion weeks limit (40 − raw; high = more restrictive) | Abortion | **F4** | 0.688 |
| **CC24_340c** | Oppose require same-sex marriage recognition [rev] | Civil Rights | **F4** | 0.651 |
| **CC24_340b** | Oppose prohibit abortion service restrictions [rev] | Civil Rights | **F4** | 0.489 |
| **CC24_324b** | Support permit abortion only rape/incest/life danger | Abortion | **F4** | 0.297 |
| **CC24_440b** | Agree: racial problems are rare/isolated | Racial/Gender | **F5** | 0.616 |
| **CC24_321b** | Support easier concealed carry permits | Guns | **F5** | 0.557 |
| **CC24_341c** | Oppose allow $400k+ tax rates to rise [rev] | Tax | **F5** | 0.534 |
| **CC24_323d** | Oppose Dreamers pathway to citizenship [rev] | Immigration | **F5** | 0.540 |
| **CC24_323a** | Oppose grant legal status to working immigrants [rev] | Immigration | **F5** | 0.520 |
| **CC24_440c** | Agree: women seek to gain power over men | Racial/Gender | **F5** | 0.437 |
| **CC24_341d** | Oppose $150B infrastructure spending [rev] | Tax | **F5** | 0.365 |
| **CC24_341a** | Support extend 2017 tax cuts | Tax | diffuse | 0.260 |
| **CC24_303** | Perceived price change past year (high = inflation) | Econ | diffuse | 0.219 |
| **CC24_302** | Household income: change past year | Econ | none | 0.172 |

> **Note on CC24_341a, CC24_303, CC24_302:** These three items have weak or no clean factor loading in the 24-item solution. CC24_302 has no clean factor anchor (max \|λ\|=0.172, h²=0.153) and was marked DROP in the recommendation table. All 24 items contributed to Thomson regression factor scoring via R⁻¹, but their signal contribution is negligible.

> **Note on CC24_340e:** Unstable across k solutions (shifts between F1 and F4). Treated with caution. In the final 24-item solution it loads cleanly on F1 (\|λ\|=0.493).

> **Sign convention:** All negative loadings in `efa_loadings_k5_final.csv` reflect the oblique rotation solution *before* empirical sign-flipping of factor scores. Factor score directions were verified empirically (see §3.3).

---

### 1.2 Five Final Factor Score Names and Interpretations

| Factor Score | Label | Cramér's V (with party ID) | Direction |
|---|---|---|---|
| **FS_F1** | Enforcement Orientation | **0.346** | HIGH = enforcement-conservative |
| **FS_F2** | Election Distrust | **0.106** | HIGH = distrusts elections |
| **FS_F3** | Government Trust | **0.059** | HIGH = distrusts government |
| **FS_F4_resid** | Reproductive Rights / Religion *(F1-residualized)* | **0.183** | HIGH = socially conservative |
| **FS_F5_resid** | Values Conservatism *(F1-residualized)* | **0.283** | HIGH = values-conservative |

**Raw (pre-residualization) Cramér's V:** F4 = 0.280, F5 = 0.425. Both reduced after residualization.

**What each factor represents:**

- **F1 — Enforcement Orientation:** Attitudes toward border security (patrols, asylum denial), domestic policing (increase/decrease 10%), and post-9/11 surveillance. The most politically informative factor (V=0.346). Anchored by enforcement-side policy preferences rather than ideological identity.

- **F2 — Election Distrust:** Disbelief that U.S. elections are conducted fairly (national and state/local levels). Near-cross-partisan (V=0.106) — skepticism spans both major parties, skewing toward Republicans in 2024 but not exclusively. Genuinely orthogonal to most other factors.

- **F3 — Government Trust:** Low trust in federal and state government institutions. Near-orthogonal to party ID (V=0.059) — the "Not sure" responses (10,526 respondents) were imputed to the midpoint before factor scoring; see `govt_trust_imputed` flag. This factor is the least partisan of all five.

- **F4 / F4_resid — Reproductive Rights / Religion:** Opposition to abortion access, same-sex marriage recognition, and frequency of church attendance. **F4_resid** is this dimension *after removing the variance it shared with F1* — i.e., the portion of social conservatism that is independent of enforcement orientation. β(F4 ~ F1) = 0.2229; 12.5% of F4 variance removed.

- **F5 / F5_resid — Values Conservatism:** Racial minimization ("racial problems are rare"), gender traditionalism ("women seek power over men"), fiscal conservatism (tax cuts, oppose infrastructure), gun rights, and opposition to immigration pathways (legal status, Dreamers). **F5_resid** is this dimension after removing overlap with F1. β(F5 ~ F1) = 0.5651; 25.8% of F5 variance removed. The strong F1 overlap is primarily via the immigration sub-items (CC24_323a, CC24_323d).

---

### 1.3 Residualization Decisions

**Both F4 and F5 were residualized on F1** (weighted OLS; each independently regressed on F1 using `commonpostweight`).

**Why:** The three conservative partisan factors formed a near-collinear bloc in the raw factor score space:

| Pair | Raw correlation | After residualization |
|------|----------------|----------------------|
| F1 × F4 | +0.354 | **0.000** (by construction) |
| F1 × F5 | +0.508 | **0.000** (by construction) |
| F4 × F5 | +0.486 | **0.381** (reduced but non-zero) |

Condition number: **5.50 → 3.19**. Without residualization, the clustering algorithm would treat F1, F4, and F5 as three nearly parallel axes, effectively triple-weighting conservative ideology relative to the cross-partisan dimensions. After residualization, each factor contributes more independent information.

The residualized F4_resid and F5_resid represent:
- **F4_resid:** "What does your position on reproductive rights / religiosity add, *beyond* what your enforcement attitudes already predict?"
- **F5_resid:** "What do your racial and cultural values add, *beyond* your enforcement orientation?"

These are substantively meaningful orthogonal decompositions, not just a technical correction.

---

### 1.4 Sample Size and Weights

| Metric | Value |
|--------|-------|
| Original CCES 2024 N | 60,000 |
| Post-survey completers (tookpost=Yes) | 49,432 |
| After listwise deletion (pre-govt trust fix) | ~35,971 |
| **Final N after midpoint imputation** | **45,707** |
| Weight variable | `commonpostweight` |
| Weighted N (approximate) | 44,834 |

---

### 1.5 The `govt_trust_imputed` Flag

**Variable name:** `govt_trust_imputed` (integer, 0/1)
**Definition:** 1 if the respondent answered "Not sure" (value = 8) on *either* CC24_423 (federal govt trust) *or* CC24_424 (state govt trust) in the raw data.
**N flagged:** 10,526 respondents (≈23% of the final analytic sample).
**Created:** Before any recoding, so it reflects original raw responses.
**Purpose:** Allows downstream validity checking — e.g., do "not sure" respondents cluster differently? Are they disproportionately excluded from certain clusters? Do their cluster assignments show unusually low confidence?

The "Not sure" group showed a U-shaped ideology distribution (higher rates among very liberals and very conservatives than moderates), suggesting genuine ambivalence rather than a missing-at-random mechanism. After recoding value 8 → 2 (midpoint of the 1–3 scale), N recovered from ~35,971 to 45,707.

---

## 2. WHAT WAS DROPPED AND WHY

### 2.1 Items Dropped Before EFA (Steps 1–4 Collinearity Screening)

**Partisan proxies — too collinear with party ID to distinguish typologies:**

| Variable | Description | PID R² | Reason |
|----------|-------------|--------|--------|
| CC24_330a | Ideology self-placement (7-pt liberal–conservative) | 0.568 | Direct partisan proxy; would dominate Factor 1 |
| CC24_301 | National economy retrospective perception | 0.338 | Partisan perceptual filter; reflects partisan identity as much as economic reality |
| CC24_312a | Biden approval rating | 0.568 | Presidential approval = partisan proxy; r=0.977 with CC24_312i |
| CC24_312i | Harris approval rating | 0.630 | Same as above; nearly collinear with CC24_312a |
| CC24_312d–h | Other presidential/VP approval items | high | Direction-inconsistent across respondent partisanship; structurally bimodal |

**Compositional data artifact — check-all-that-apply structure:**

| Variable(s) | Description | Reason |
|-------------|-------------|--------|
| CC24_420_1 through CC24_420_7 | Most important issue selections | Check-all-that-apply creates compositional data that violates polychoric EFA assumptions; individual item endorsement confounds item difficulty with position |

**Ceiling/floor effects — insufficient variance for factor analysis:**

| Variable | Description | Marginal | Reason |
|----------|-------------|---------|--------|
| CC24_321c | Support universal background checks | 93% support | Near-universal; no meaningful variance to factor |
| CC24_324c | Oppose making abortion illegal in all cases | 89% oppose | Near-universal; ceiling effect |

**Redundant — polychoric r > 0.70 with retained item:**

| Variable | Redundant with | r | Reason |
|----------|---------------|---|--------|
| pew_religimp | pew_churatd | >0.70 | Church attendance retained as more behavioral measure |
| pew_bornagain | pew_churatd | >0.70 | Born-again status redundant with attendance |
| CC24_341b | CC24_341c or CC24_341a | >0.70 | Tax item cluster redundancy |
| CC24_324d | CC24_325 | >0.70 | Abortion item cluster redundancy |
| CC24_440a | CC24_440b | >0.70 | Racial attitudes cluster redundancy |
| CC24_440d | CC24_440c | >0.70 | Gender attitudes cluster redundancy |

**User decision — substantive scope exclusion:**

| Variable | Description | Reason |
|----------|-------------|--------|
| CC24_441a | Racial resentment item | User decision: excluded from typology construction |
| CC24_441b | Racial resentment item | User decision: excluded from typology construction |
| CC24_309d_8 | — | High missingness / user decision |
| CC24_312b | — | High missingness / user decision |
| CC24_312c | — | High missingness / user decision |
| pew_prayer | Prayer frequency | High missingness / user decision |

---

### 2.2 Item Dropped After EFA Entry (25 → 24 Items)

| Variable | Description | Reason |
|----------|-------------|--------|
| CC24_340a | Oppose prohibit contraceptive restrictions [rev] | **Near-Heywood case**: pattern loading = −0.947 to −1.003 across k solutions. Indicates near-perfect communality driven by a single factor — likely a ceiling effect. Empirical distribution confirmed near-universal agreement, making this item uninformative for typology differentiation. Dropping it improved solution stability. |

---

### 2.3 Item Retained in EFA But Excluded From Clustering Interpretation

| Variable | Description | Issue |
|----------|-------------|-------|
| CC24_302 | Household income: change past year | h²=0.153, max \|λ\|=0.172 — no clean factor loading. Contributes negligible signal to any factor. Marked DROP in recommendation table. Included in Thomson scoring computation but effectively inert. |

---

## 3. KEY METHODOLOGICAL DECISIONS AND REASONING

### 3.1 Why Polychoric Correlation Rather Than Pearson

All 24 items are ordered categorical variables (2–4 response categories). Pearson correlation assumes continuous, normally distributed variables and is biased downward for ordinal data — the degree of bias increases as the number of categories decreases and as item skewness increases. Polychoric correlation instead estimates the latent bivariate normal correlation underlying each pair of ordinal items, which is the appropriate input for factor analysis of ordinal political attitude data. This is the methodological standard in political science survey research (e.g., Ansolabehere et al., 2008; DeVellis, 2017).

The polychoric matrix was computed once using the `commonpostweight`-weighted sample (listwise N≈45,143 at time of computation) and saved to `polychoric_matrix.csv`. **Do not recompute.**

---

### 3.2 Why Oblimin Rotation Rather Than Varimax

Varimax (orthogonal rotation) imposes the constraint that all factors are uncorrelated. This is theoretically untenable for political attitudes: conservative respondents tend to hold consistently conservative positions across enforcement, reproduction, and values domains. Forcing orthogonality would artificially spread correlated variance across multiple factors, distorting the solution.

Oblimin (oblique rotation) allows factors to be correlated, producing a more interpretable simple structure while preserving the correlated architecture of political ideology. The resulting phi matrix confirmed the expected pattern: three conservative partisan factors are moderately correlated (F1×F4=+0.354, F1×F5=+0.508, F4×F5=+0.486), while cross-partisan factors (F2, F3) are near-orthogonal to all others.

---

### 3.3 Why k=5 Over k=4 Despite Parallel Analysis Supporting k=4

Parallel analysis (100 simulated datasets, eigenvalue comparison) strictly supported k=4. The 5th factor's empirical eigenvalue (1.026) fell marginally below the parallel analysis threshold (1.028, Δ = −0.002). This is an arbitrarily close miss.

k=5 was adopted for two substantive reasons:

1. **Interpretive distinctiveness:** The k=4 solution collapsed what k=5 separates into F4 (religious/reproductive social conservatism) and F5 (racial/fiscal/cultural values conservatism) into a single undifferentiated "conservative values" bloc. The distinction between *institutional religious and reproductive conservatism* (F4: church attendance, abortion restrictions, same-sex marriage) and *cultural-racial-fiscal conservatism* (F5: racial minimization, gender traditionalism, gun rights, tax opposition) is substantively important for political typology work and well-documented in the political science literature (e.g., Egan, 2020).

2. **Marginal cost is low:** A parallel analysis miss of Δ=−0.002 is trivially small, and the F5 communalities and loadings were robust. The 5-factor solution explained 43.0% of common variance vs. ~38% for k=4.

**Sign convention note:** In oblique rotation, pattern matrix loading signs do **not** reliably indicate factor score direction. The Thomson regression scoring matrix B = R⁻¹S depends on the full polychoric correlation structure and can produce factor scores where the "conservative" direction is opposite to what the loading signs suggest. All three partisan factors (F1, F4, F5) were empirically verified:
- For each, weighted mean factor scores were computed for Democrats (pid3=1) and Republicans (pid3=2)
- F1 and F4 required sign flips (Rep mean < Dem mean in raw output)
- F5 was already correctly oriented (Rep mean > Dem mean)

---

### 3.4 Why Midpoint Recoding for Government Trust "Not Sure" Responses

CC24_423 and CC24_424 include a response option of 8 ("Not sure"), which is not an ordered position on the 1–3 trust scale. 10,526 respondents (≈23%) answered "Not sure" on at least one of the two items.

Listwise deletion would have excluded these respondents, reducing N from ~45,700 to ~35,971 — a 21% loss. The "Not sure" group is not missing at random: it shows a **U-shaped ideology pattern** (disproportionately prevalent among very liberals and very conservatives vs. moderates), and by party: Democrats 9.7%, Independents 17.9%, very conservatives 22.8%. Listwise deletion would have over-represented partisan moderates.

Midpoint recoding (value 8 → 2 on the 1–3 scale) treats "Not sure" as neutral trust — a defensible substantive interpretation. The `govt_trust_imputed` flag preserves the ability to check whether this group's cluster assignments differ meaningfully from the non-imputed respondents.

---

### 3.5 Why Option C Residualization Over Options A and B

Three clustering input options were evaluated:

**Option A — All 5 raw factor scores as-is:**
Condition number = 5.50; F1×F5 = +0.508, F1×F4 = +0.354, F4×F5 = +0.486. The three conservative factors form a near-collinear bloc. A clustering algorithm operating on these inputs would effectively triple-weight general conservative ideology. The resulting clusters would likely differentiate conservatives from liberals at the expense of within-conservative differentiation.

**Option B — F2, F3, F4 only (drop F1 and F5):**
Would discard F1 (V=0.346, the single most politically informative factor) and F5 (V=0.280 raw). This loses the enforcement/immigration dimension entirely and half the values-conservatism signal. Not justified by any principled criterion.

**Option C — Residualize F4 and F5 on F1:**
- Eliminates F1→F4 and F1→F5 collinearity by construction (residuals are orthogonal to F1)
- Retains all five substantive dimensions
- Condition number drops to 3.19
- The residualized factors represent analytically distinct variance: "social conservatism independent of enforcement orientation" and "cultural/values conservatism independent of enforcement orientation"
- Cramér's V for residualized factors: F4_resid=0.183, F5_resid=0.283 — still meaningful partisan signal, just the independent portion

**F2 and F3 were not residualized** because they are already near-orthogonal to F1 (r=0.048 and r=0.054 respectively) and to each other (r=0.181). No correction needed.

---

### 3.6 Why DPGMM Over Fixed-k Clustering

Fixed-k methods (k-means, fixed GMM) require specifying the number of clusters in advance. For an empirical political typology of the U.S. electorate, there is no strong theoretical prior for the exact number of voter segments, and choosing k by grid search introduces researcher degrees of freedom.

**Dirichlet Process Gaussian Mixture Model (DPGMM)** treats the number of mixture components as a random variable estimated from data:

1. **Data-driven k selection:** The Dirichlet Process prior penalizes unnecessary components. With an upper bound of n_components=10, the model assigns negligible weight to components unsupported by the data. Effective cluster count is determined empirically (components with weight > 0.01).

2. **Full covariance matrices:** `covariance_type='full'` allows each cluster to have its own shape, orientation, and scale in the 5-dimensional input space. Political opinion clusters are not spherical (k-means assumption) or identically shaped (tied covariance assumption). Full covariance is essential for capturing, e.g., a cluster that is tight on F1 but diffuse on F3.

3. **Soft assignments:** Respondents receive a probability vector over all components, not a hard binary assignment. This is valuable for identifying "mixed" respondents who do not fall cleanly into one segment.

4. **Regularization:** The Dirichlet Process prior with `weight_concentration_prior_type='dirichlet_process'` provides built-in regularization against overfitting. Components compete for respondents; weak components collapse.

5. **n_init=5:** Multiple random initializations guard against the EM algorithm converging to a local optimum.

**Survey weight application:** `sample_weight=commonpostweight` ensures that cluster memberships reflect the population distribution, not the biased raw sample composition.

---

## 4. FILES AND PATHS

### 4.1 Primary Analysis Files

| File | Description |
|------|-------------|
| `/Users/bdecker/Documents/STV/Claude/efa_factor_scores.csv` | **Primary output.** Factor scores, residualized scores, weights, flags. All clustering work starts here. |
| `/Users/bdecker/Documents/STV/Claude/polychoric_matrix.csv` | 25×25 polychoric correlation matrix. **DO NOT RECOMPUTE.** |
| `/Users/bdecker/Documents/STV/Claude/efa_update.py` | Full pipeline: load raw data → recode → Thomson scoring → sign flip → residualization → save |
| `/Users/bdecker/Documents/STV/Claude/efa_loadings_k5_final.csv` | 24-item k=5 oblimin pattern matrix (pre-sign-flip, as output by factor_analyzer) |
| `/Users/bdecker/Documents/STV/Claude/efa_phi_k5_final.csv` | Inter-factor correlation matrix phi (pre-sign-flip) |
| `/Users/bdecker/Documents/STV/Claude/efa_cramersv_actual.csv` | Cramér's V computed from actual weighted factor score quartiles vs. pid3 |
| `/Users/bdecker/Documents/STV/Claude/efa_recommendation_table.csv` | Per-item KEEP/REVIEW/DROP table (from 25-item solution; CC24_340a still present) |
| `/Users/bdecker/Documents/STV/Claude/efa_checkpoint_summary.txt` | Steps 1–4 summary: collinearity screening, dropped items, polychoric notes |
| `/Users/bdecker/Documents/STV/Claude/efa_narrative_interpretation.md` | Full narrative description of k=5 factor solution |
| `/Users/bdecker/Documents/STV/Claude/efa_variable_list.csv` | Full variable list with PID R² values and keep/drop status |
| `/Users/bdecker/Documents/STV/Claude/efa_variance_summary.csv` | Variance explained by factor and k solution |
| `/Users/bdecker/Documents/STV/Claude/efa_parallel_analysis.csv` | Parallel analysis eigenvalue comparison (empirical vs. simulated) |

### 4.2 Raw Data

| File | Description |
|------|-------------|
| `/Users/bdecker/Documents/STV/2024 CES Base/CCES24_Common_OUTPUT_vv_topost_final.dta` | Original CCES 2024 Stata file (read-only) |

### 4.3 Columns in `efa_factor_scores.csv`

| Column | Type | Description |
|--------|------|-------------|
| `pid3` | int | Party ID 3-category (1=Democrat, 2=Republican, 3=Independent/Other) |
| `ideo5` | int | Ideology 5-category (1=Very Liberal … 5=Very Conservative) |
| `inputstate` | int | State FIPS code |
| `commonpostweight` | float | Survey weight (use for all weighted analyses) |
| `govt_trust_imputed` | int (0/1) | 1 = respondent had "Not sure" (value 8) on CC24_423 or CC24_424 |
| `FS_F1` | float | Enforcement Orientation score (HIGH = enforcement-conservative) |
| `FS_F2` | float | Election Distrust score (HIGH = distrusts elections) |
| `FS_F3` | float | Government Trust score (HIGH = distrusts government) |
| `FS_F4` | float | Repro Rights/Religion score, **raw** (before residualization) |
| `FS_F5` | float | Values Conservatism score, **raw** (before residualization) |
| `FS_F4_resid` | float | F4 residualized on F1 (use this for clustering) |
| `FS_F5_resid` | float | F5 residualized on F1 (use this for clustering) |

**N = 45,707 rows. No NaN values in any column.**

---

## 5. NEXT SESSION INSTRUCTIONS

### 5.1 Setup

```python
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

DATA_DIR = "/Users/bdecker/Documents/STV/Claude"

# Load factor scores
fs = pd.read_csv(f"{DATA_DIR}/efa_factor_scores.csv")

# Clustering input columns
CLUSTER_COLS = ["FS_F1", "FS_F2", "FS_F3", "FS_F4_resid", "FS_F5_resid"]

X = fs[CLUSTER_COLS].values
w = fs["commonpostweight"].values
```

### 5.2 Run DPGMM

```python
dpgmm = BayesianGaussianMixture(
    n_components=10,                              # upper bound; effective k will be lower
    covariance_type='full',                       # each cluster has its own shape
    weight_concentration_prior_type='dirichlet_process',
    n_init=5,                                     # guard against local optima
    random_state=42,
    max_iter=500,
)

dpgmm.fit(X, sample_weight=w)
```

### 5.3 Report Required After Fitting

1. **Effective cluster count:** components where `dpgmm.weights_[k] > 0.01`
2. **Cluster sizes:** weighted N per cluster (apply `commonpostweight`)
3. **Weighted factor score means per cluster** for all 5 inputs: `FS_F1, FS_F2, FS_F3, FS_F4_resid, FS_F5_resid`
4. **Hard cluster assignments:** `cluster = dpgmm.predict(X)`
5. **Soft assignment matrix:** `probs = dpgmm.predict_proba(X)` — report mean max-probability per cluster (confidence indicator)

> **Do NOT name clusters yet.** Report numbers and factor score profiles only. Naming comes after reviewing the profile table.

> **Do NOT use fixed k.** The DPGMM determines effective k from the data.

### 5.4 Save Cluster Assignments

```python
save_cols = ["pid3", "ideo5", "inputstate", "commonpostweight",
             "govt_trust_imputed"] + CLUSTER_COLS

out = fs[save_cols].copy()
out["cluster"] = dpgmm.predict(X)

# Optionally save soft assignment probabilities
probs = dpgmm.predict_proba(X)
n_eff = (dpgmm.weights_ > 0.01).sum()
for k in range(n_eff):
    out[f"prob_cluster_{k}"] = probs[:, k]

out.to_csv(f"{DATA_DIR}/typology_cluster_assignments.csv", index=False)
```

### 5.5 Validation Cross-Tabs to Run After Clustering

For each cluster, compute weighted marginals on:
- `pid3` (party ID — expected strong differentiation on F1/F4/F5 clusters)
- `ideo5` (ideology — expected but should not be the only differentiator)
- `govt_trust_imputed` (check whether "not sure" respondents concentrate in specific clusters)
- `FS_F2` mean (election distrust — should cut across partisan clusters)
- `FS_F3` mean (govt trust — should be near-independent of partisan clusters)

### 5.6 Key Cautions for Next Session

- **Do not re-run EFA or recompute factor scores.** All factor scores are finalized in `efa_factor_scores.csv`.
- **Do not recompute polychoric_matrix.csv.**
- **Always apply `commonpostweight`** — unweighted analyses will over-represent certain demographic groups in the raw sample.
- **`FS_F4` and `FS_F5` are raw (un-residualized) — use `FS_F4_resid` and `FS_F5_resid` for clustering.** Raw versions are retained in the file for post-hoc interpretation only.
- **F3 (govt trust) has V=0.059** — it contributes meaningful variance to clustering but should not be over-interpreted as a partisan differentiator.
- **CC24_340e (surveillance)** loads on F1 but was flagged as unstable across solutions. If surveillance-related clustering appears unexpected, this item may be worth examining separately.
- **`govt_trust_imputed` respondents (N=10,526)** are included in clustering. Check whether they concentrate in specific clusters in validation.

---

*End of brief. All files at `/Users/bdecker/Documents/STV/Claude/`. Raw data at `/Users/bdecker/Documents/STV/2024 CES Base/`.*
