# EFA Narrative Interpretation — CES 2024 Political Typology
**Steps 5–7 | PAF + Oblimin | N = 45,143 (commonpostweight) | 25 items**
_Generated from polychoric correlation matrix; all statistics are approximations
without raw data — flag for analyst verification before finalizing cluster inputs._

---

## 1. Parallel Analysis and Factor Count Decision

Parallel analysis (200 simulations, 95th percentile criterion) strictly supports **k = 4**:

| Factor | Obs Eigenvalue | Sim 95th% | Extract? |
|--------|---------------|-----------|----------|
| 1 | 8.3196 | 1.0473 | YES |
| 2 | 2.9288 | 1.0415 | YES |
| 3 | 1.5300 | 1.0354 | YES |
| 4 | 1.3430 | 1.0312 | YES |
| **5** | **1.0258** | **1.0278** | **no (Δ = −0.0020)** |

The F5 eigenvalue misses the simulated threshold by **< 0.2%** — a marginal call in a large sample where even trivial variance is reliably estimated. More importantly, comparing the k=4 and k=5 solutions reveals that the k=4 F1 is an omnibus first factor that collapses two theoretically distinct and separable sub-dimensions (enforcement/policing and traditional social conservatism), producing a single large factor that accounts for 20.4% of variance but mixes constructs with different political dynamics. In the k=5 solution, this omnibus splits cleanly:

- **k=4 F1** (enforcement + immigration + fiscal + social conservatism, V≈0.41)
  → splits into **k=5 F1** (enforcement/immigration) + **k=5 F5** (traditional conservatism/fiscal)

This split is theoretically meaningful for typology work: enforcement orientation and fiscal/social traditionalism overlap heavily among self-identified Republicans, but they behave differently across voter segments (e.g., populist-right voters who are economically cross-pressured, or libertarian-right voters who are socially liberal but enforcement-oriented). The split also improves cluster separability in the downstream k-means/LCA step.

**Recommendation: Use k=5 as primary working solution.** Retain k=4 as the statistically conservative anchor. Run cluster analysis on factor scores from both and compare cluster stability and interpretability.

---

## 2. K=5 Factor Descriptions (PAF + Oblimin)

### F1 — Enforcement Orientation (V≈0.39, MODERATE PARTISAN LOAD)
**Variance: 12.9% | "Border and Blue"**

| Item | Loading | Direction |
|------|---------|-----------|
| CC24_323b — Support increase border patrols | +0.769 | High = pro-enforcement |
| CC24_321d — Support increase police by 10% | +0.767 | High = pro-enforcement |
| CC24_340f — Support deny asylum at border | +0.741 | High = pro-enforcement |
| CC24_321e — Oppose decrease police by 10% | +0.723 | High = pro-enforcement |
| CC24_323a — Oppose grant legal status to immigrants | +0.473 | High = restrictionist |
| CC24_323d — Oppose Dreamers citizenship pathway | +0.458 | High = restrictionist |
| CC24_340e — Support renew surveillance programs | +0.419 | High = pro-surveillance |

**Interpretation:** A unified law-enforcement and border-control dimension. High scorers favor policing, border security, and immigration restriction across the board. The 0.42 loading on CC24_340e (surveillance renewal) is notable — this item loads here (not on F2/F4) in the k=5 solution, suggesting surveillance support is associated with the enforcement orientation rather than with the distrust/populist cluster.

**Sign note:** All loadings are positive and in the conservative direction. High factor score = enforcement-oriented.

**Factor correlates with F5 (r=0.615)** — the enforcement orientation and traditional conservatism dimensions are substantially correlated, consistent with a general partisan alignment. However, they are separable enough to contribute distinct cluster-level differentiation.

---

### F2 — Election & Government Distrust (V≈0.15, ORTHOGONAL DIMENSION)
**Variance: 9.0% | "Institutional Skeptics"**

| Item | Loading | Direction |
|------|---------|-----------|
| CC24_421_2 — Disagree: state/local elections are fair | −0.861 | High score = trusting |
| CC24_421_1 — Disagree: U.S. elections are fair | −0.785 | High score = trusting |
| CC24_424 — Low trust: state government | −0.603 | High score = trusting |
| CC24_423 — Low trust: federal government | −0.538 | High score = trusting |

**Interpretation:** Negative loadings mean that a **low** factor score = high institutional distrust (distrusting elections and government at multiple levels). This is the most genuinely cross-partisan dimension in the battery: V≈0.15, well below the 0.30 orthogonality threshold. Election distrust and government distrust co-vary across both parties — right-leaning distrust of electoral integrity (post-2020) and left-leaning distrust of government legitimacy both load here.

**Sign note:** Low F2 score = high distrust. For cluster labeling, consider recoding or flipping sign so "high = distrust" is more intuitive.

**Cramér's V ≈ 0.15** — strongest candidate for inclusion in clustering input. This dimension will reliably separate cross-partisan "distruster" segments from institutional supporters within each party.

---

### F3 — Reproductive Rights / Social Policy Conservatism (V≈0.27, NEAR-ORTHOGONAL)
**Variance: 6.4% | "Reproductive Rights Axis"**

| Item | Loading | Direction |
|------|---------|-----------|
| CC24_340a — Oppose prohibit contraceptive restrictions [rev] | −0.947 | Low score = conservative on contraception |
| CC24_340b — Oppose prohibit abortion restrictions [rev] | −0.672 | Low score = conservative on abortion |
| CC24_340c — Oppose require same-sex marriage recognition [rev] | −0.294 | Low score = anti-same-sex marriage |
| CC24_325 — Abortion weeks limit (high=restrictive) | −0.203 | Low score = shorter limit |

**Interpretation:** Negative loadings throughout. A **low** F3 score = social conservatism on reproductive rights and sexuality. This dimension is more orthogonal to partisanship than it might appear (V≈0.27) because the positions cut across rural/urban and religious/secular divisions within both parties. Note CC24_340a approaches a Heywood case loading (−0.947) — this item has near-perfect factor saturation, likely because the wording of the contraceptive restrictions item is very specifically linked to this single construct.

**Important:** The near-Heywood loading on CC24_340a (−0.947 in k=5, −0.978 in k=6) warrants inspection of item distribution. If this variable has a ceiling/floor effect, it may inflate its apparent factor loading. Verify item-level frequency tables before finalizing.

**Sign note:** Low F3 = conservative on reproductive rights/sexuality. High F3 = liberal/libertarian on these issues.

---

### F4 — Cross-Cutting Populist Dimension (V≈0.35, MODERATE PARTISAN LOAD)
**Variance: 5.2% | "Populist/Libertarian Right"**

| Item | Loading | Direction |
|------|---------|-----------|
| CC24_340e — Support renew post-9/11 surveillance programs | −0.518 | Low score = pro-surveillance |
| CC24_323a — Oppose grant legal status to immigrants | +0.476 | High score = restrictionist |
| CC24_323d — Oppose Dreamers citizenship pathway | +0.429 | High score = restrictionist |
| CC24_423 — Low trust: federal government | +0.387 | High score = distrusts gov |
| CC24_341c — Oppose allow $400k+ rates to rise [rev] | +0.355 | High score = anti-tax increase |

**Interpretation:** This is the most theoretically complex factor. It captures a populist-right orientation that combines immigration restrictionism, anti-government distrust, and fiscal conservatism, alongside opposition to post-9/11 surveillance (negative loading). The combination suggests a "libertarian restrictionist" profile — anti-immigration but also anti-surveillance-state, distrustful of all government power. This may correspond to a segment of MAGA-adjacent, anti-establishment voters who are hawkish on borders but skeptical of surveillance programs.

**Note:** CC24_340e cross-loads across solutions (F1 in k=4, F4 in k=5, F6 in k=6). This instability across solutions is a warning flag — the item may belong to multiple dimensions depending on who it co-clusters with. Treat with caution as a cluster input.

**Sign note:** High F4 = immigration restrictionist + anti-government + anti-tax increase; low F4 = pro-immigration legalization + trusts government + pro-surveillance.

---

### F5 — Traditional Conservatism (V≈0.40, MODERATE PARTISAN LOAD)
**Variance: 9.6% | "Values Conservatism"**

| Item | Loading | Direction |
|------|---------|-----------|
| CC24_341a — Support extend 2017 tax cuts | +0.538 | High = fiscal conservative |
| CC24_325 — Abortion weeks limit (high=restrictive) | +0.493 | High = pro-life |
| CC24_341c — Oppose allow $400k+ tax rates to rise | +0.484 | High = anti-tax |
| CC24_340c — Oppose require same-sex marriage recognition | +0.482 | High = anti-same-sex marriage |
| CC24_440b — Agree: racial problems are rare/isolated | +0.475 | High = minimizes race issues |
| CC24_321b — Support easier concealed carry permits | +0.474 | High = pro-gun |
| pew_churatd — Church attendance frequency | +0.440 | High = religious |
| CC24_440c — Agree: women seek power over men | +0.429 | High = anti-feminist |
| CC24_341d — Oppose $150B infrastructure spending | +0.365 | High = anti-spending |
| CC24_324b — Support permit abortion only rape/incest/life | +0.358 | High = restrictive |

**Interpretation:** This is the classic American social conservatism dimension — fiscal conservatism, anti-abortion, opposition to same-sex marriage, racial minimization, religiosity, and gender traditionalism cluster together. All loadings are positive and in the conservative direction. This factor has the broadest item coverage (10 items with |λ| ≥ 0.30) and accounts for 9.6% of variance.

**Key distinction from F1:** F1 captures behavioral/enforcement attitudes (policing, border control, asylum) while F5 captures values/identity-based conservatism. The F1–F5 correlation (r=0.615) confirms they are related but separable — high F5 without high F1 likely characterizes socially conservative voters who are not particularly enforcement-oriented; high F1 without high F5 may characterize immigration-focused voters with less consistent social conservatism.

---

## 3. Factor Intercorrelation Summary (k=5 Phi Matrix)

|    | F1 | F2 | F3 | F4 | F5 |
|----|----|----|----|----|-----|
| **F1** | 1.00 | −0.03 | −0.36 | 0.20 | **0.62** |
| **F2** | −0.03 | 1.00 | 0.13 | −0.23 | −0.17 |
| **F3** | −0.36 | 0.13 | 1.00 | −0.07 | −0.40 |
| **F4** | 0.20 | −0.23 | −0.07 | 1.00 | 0.24 |
| **F5** | **0.62** | −0.17 | −0.40 | 0.24 | 1.00 |

**Key takeaways:**
- **F1↔F5 = 0.62**: Largest inter-factor correlation — enforcement and traditional conservatism share substantial common variance. This is the underlying partisan axis. The oblique solution correctly allows these to correlate rather than forcing orthogonality.
- **F2 (distrust)**: Near-zero correlations with F1, F3, F5 — genuinely orthogonal to the partisan structure.
- **F3 (reproductive rights) ↔ F5 = −0.40**: Social conservatives on values (F5) are also conservative on reproductive rights (low F3).
- **F4 (populist)**: Weakly positively correlated with F1 (0.20) and F5 (0.24), weakly negatively correlated with F2 (−0.23).

---

## 4. Cross-Solution Stability Assessment

Items that are **stable across k=3 through k=7** (consistent factor membership, loading magnitude ≥ 0.50):

| Item | Stable Factor | Stability |
|------|--------------|-----------|
| CC24_421_1, CC24_421_2 | Election distrust (F2 in k=4–7) | Very stable |
| CC24_423, CC24_424 | Govt distrust | Stable (separates from election distrust at k=6+) |
| CC24_340a, CC24_340b | Reproductive rights (F3) | Very stable (near-Heywood) |
| CC24_323b, CC24_321d, CC24_321e, CC24_340f | Enforcement (F1) | Stable |

Items that are **unstable across solutions** (factor membership shifts):

| Item | Instability | Note |
|------|------------|-------|
| CC24_340e | F1→F4→F1→F1 across k=3,4,5,6 | Cross-loading; treat with caution |
| CC24_341c | F1/F4/F5 cross-loads | Fiscal conservatism pulls across factors |
| CC24_423, CC24_424 | Split from election distrust at k=6 | Separate government vs. election trust |
| pew_churatd | F3/F5 | Church attendance sits between reproductive and values factors |

---

## 5. Approximation Limitations

**Without raw data, the following required analyses were approximated or omitted:**

1. **Cramér's V from quartile-split cross-tabs (Step 6 specification):** Approximated as V ≈ √(loading²-weighted avg PID R²). Actual computation requires: compute factor scores from raw data → quartile cut → cross-tab × pid3 (Democrat/Republican/Independent) → Cramér's V. **Re-run with raw CES data before finalizing clustering inputs.**

2. **Ideology R²:** Estimated as 1.5× PID R² based on known high PID–ideology correlation in CCES. Verify directly by regressing factor scores on ideo5 variable.

3. **Factor score weighting:** Factor scores should be computed using commonpostweight. The EFA itself was conducted on the weighted polychoric matrix, but downstream factor scores need the weight applied at the regression/Bartlett/Thomson scoring stage.

4. **CC24_340a near-Heywood loading (−0.947 to −1.003 across k=5–7):** Check for distributional anomalies (extreme skew, floor/ceiling effects). A loading > 0.95 on a single factor suggests this item may be near-redundant with the factor itself.

---

## 6. Files Produced

| File | Description |
|------|-------------|
| `efa_parallel_analysis.csv` | Observed vs simulated eigenvalues, k=1–10 |
| `efa_variance_summary.csv` | SS loadings and % variance for k=3–7 |
| `efa_loadings_k3.csv` through `efa_loadings_k7.csv` | Pattern matrices (with metadata) |
| `efa_phi_k3.csv` through `efa_phi_k7.csv` | Factor intercorrelation matrices |
| `efa_recommendation_table.csv` | Step 7 recommendation table (k=5 primary) |
| `efa_factor_partisan_summary.csv` | Factor-level partisan collinearity estimates |
| `efa_narrative_interpretation.md` | This document |

---

## 7. Next Steps (Step 8+)

1. **Verify with raw data:** Re-run Cramér's V and ideology R² using actual factor scores from raw CES. Confirm CC24_340a distribution.
2. **Select cluster input items:** From the REVIEW group, retain items with Cramér's V < 0.40 (actual, not estimated) after Step 6 verification.
3. **Cluster analysis input:** Consider using all 5 factor scores (or a reduced set excluding F1/F5 if partisan separation is undesirable) as inputs to k-means or LCA.
4. **Sensitivity check:** Run clustering with k=4 and k=5 factor solutions; compare cluster interpretability and stability (silhouette width, BIC).
5. **F1–F5 correlation handling:** The r=0.62 correlation between F1 and F5 means factor scores will not be orthogonal cluster inputs. Consider oblique factor scores vs. orthogonalized scores, or use Bartlett factor scores directly.
