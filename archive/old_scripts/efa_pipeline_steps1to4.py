#!/usr/bin/env python3
"""
EFA Pipeline: CES 2024 Political Typology Analysis
Steps 1–4: Load → Recode → Pseudo-R² item review → Polychoric matrix

Stops after Step 4 and saves:
  - efa_variable_list.csv
  - polychoric_matrix.csv
  - efa_checkpoint_summary.txt

Output directory: /Users/bdecker/Documents/STV/Claude/
"""

import os, sys, time, warnings, textwrap
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm, multivariate_normal
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/Users/bdecker/Documents/STV/Claude'
DATA_PATH  = '/Users/bdecker/Documents/STV/2024 CES Base/CCES24_Common_OUTPUT_vv_topost_final.dta'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEP = "=" * 68


# ══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 1: LOAD DATA")
print(SEP)
print("""
WHAT:  Load CCES24_Common_OUTPUT_vv_topost_final.dta into pandas,
       preserving Stata value labels as Categorical dtype.

WHY:   We need text labels (e.g. "Strongly agree", "Support") to do
       correct directional recoding in Step 2 — not just numeric codes.

DECISION — convert_missing=False (not True):
  pandas 2.x crashes with convert_missing=True + convert_categoricals=True
  because Stata extended missing values (.a, .b, ...) become unhashable
  StataMissingValue objects inside Categorical columns. Setting
  convert_missing=False lets pandas treat all Stata missing codes as NaN
  natively, which is what we want.

DECISION — convert_dates=False:
  A separate pandas 2.x bug causes a TypeError when trying to parse
  certain Stata date columns in this file. Since no date variables are
  in our analytic set, we suppress date conversion entirely.
""")

t0 = time.time()
df = pd.read_stata(DATA_PATH,
                   convert_categoricals=True,
                   convert_missing=False,
                   convert_dates=False)
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns  "
      f"({time.time()-t0:.1f}s)")

# ── Weights ─────────────────────────────────────────────────────────
w_raw = df['commonweight'].copy()
print(f"\n  commonweight:  N={w_raw.notna().sum():,}  "
      f"mean={w_raw.mean():.4f}  min={w_raw.min():.5f}  "
      f"max={w_raw.max():.4f}  missing={w_raw.isna().sum()}")

# ── Party ID (for pseudo-R² in Step 3) ──────────────────────────────
print("\n  Searching for party ID variable...")
pid_candidates = ['pid3', 'pid7', 'pid3_baseline', 'CC24_365', 'pid3leaner']
pid_var = None
for p in pid_candidates:
    if p in df.columns:
        s = df[p]
        cats = list(s.dtype.categories) if hasattr(s.dtype, 'categories') else []
        print(f"  Found {p}: categories = {cats[:8]}")
        if pid_var is None:
            pid_var = p

if pid_var is None:
    print("  WARNING: No party ID variable found in candidate list.")
    print(f"  Weight-like cols: {[c for c in df.columns if 'pid' in c.lower()][:10]}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: DEFINE AND RECODE ALL CANDIDATE ITEMS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 2: DEFINE AND RECODE CANDIDATE ITEMS")
print(SEP)
print("""
WHAT:  Convert all Categorical variables to integers.
       Direction convention: higher integer = more conservative / restrictive.

WHY:   Polychoric correlation needs numeric ordinal codes.
       Consistent direction aids factor interpretation (all factors
       have positive loadings on "conservative" items).

NOTE:  Direction does NOT affect factor structure — only loading signs.
       EFA identifies the same dimensions regardless of coding direction.

── DECISION: CC24_309d (financial fragility) ──────────────────────────
  CC24_309d is stored as 9 binary sub-items (check-all-that-apply), not
  an ordinal scale. We use CC24_309d_8 ("I wouldn't be able to pay") as
  the financial fragility indicator per user specification.

  HOWEVER: CC24_309e ("How would you describe your current financial
  situation?" — Excellent/VG/Good/Fair/Poor) is a proper 5-point ordinal
  scale that is more appropriate for polychoric EFA and directly measures
  the same construct. BOTH are included:
    - CC24_309d_8 → main set (as specified)
    - CC24_309e   → flagged as REVIEW (recommended substitute)

── DECISION: CC24_324c ("Make abortions illegal in all circumstances") ─
  Not explicitly kept or dropped. Only 10.5% support → severe floor
  effect. Including in main set but marking REVIEW.

── DECISION: CC24_321c (background checks) ─────────────────────────────
  93.5% Support → near-ceiling. Very low variance; likely near-zero
  communality in EFA. Including but marking REVIEW.

── DECISION: CC24_441e/f/g ─────────────────────────────────────────────
  79.4% missing — shown only to non-White respondents. Including in
  listwise deletion would reduce analytic N by ~80%. These items are
  EXCLUDED from the main EFA and reported separately (flagged set).
""")

def cat_to_num(series, label_map):
    """Map Categorical or object series labels → numeric via dict.
    Labels absent from map (including NaN) become NaN."""
    result = series.map(label_map)
    return pd.to_numeric(result, errors='coerce')

# Convenience maps reused across many binary items
con_map = {'Support': 1, 'Oppose': 0}   # Support = conservative direction
lib_map = {'Support': 0, 'Oppose': 1}   # Oppose  = conservative direction

ag_con = {   # Agree = conservative; higher = more conservative
    'Strongly agree': 4, 'Somewhat agree': 3,
    'Neither agree nor disagree': 2,
    'Somewhat disagree': 1, 'Strongly disagree': 0
}
ag_lib = {   # Disagree = conservative; higher = more conservative
    'Strongly agree': 0, 'Somewhat agree': 1,
    'Neither agree nor disagree': 2,
    'Somewhat disagree': 3, 'Strongly disagree': 4
}

# ── Self-Identity / Values ───────────────────────────────────────────
df['r_CC24_330a'] = cat_to_num(df['CC24_330a'], {
    'Very Liberal': 1, 'Liberal': 2, 'Somewhat Liberal': 3,
    'Middle of the Road': 4, 'Somewhat Conservative': 5,
    'Conservative': 6, 'Very Conservative': 7, 'Not sure': np.nan
})

df['r_pew_churatd'] = cat_to_num(df['pew_churatd'], {
    'More than once a week': 6, 'Once a week': 5,
    'Once or twice a month': 4, 'A few times a year': 3,
    'Seldom': 2, 'Never': 1, "Don't know": np.nan
})

df['r_pew_religimp'] = cat_to_num(df['pew_religimp'], {
    'Very important': 4, 'Somewhat important': 3,
    'Not too important': 2, 'Not at all important': 1
})

df['r_pew_prayer'] = cat_to_num(df['pew_prayer'], {
    'Several times a day': 7, 'Once a day': 6, 'A few times a week': 5,
    'Once a week': 4, 'A few times a month': 3, 'Seldom': 2,
    'Never': 1, "Don't know": np.nan
})

df['r_pew_bornagain'] = cat_to_num(df['pew_bornagain'], {'Yes': 1, 'No': 0})

# ── Economic Perceptions ─────────────────────────────────────────────
# Higher = more negative perception (anti-incumbent direction in 2024)
df['r_CC24_301'] = cat_to_num(df['CC24_301'], {
    'Gotten much better': 1, 'Gotten somewhat better': 2,
    'Stayed about the same': 3, 'Gotten somewhat worse': 4,
    'Gotten much worse': 5, 'Not sure': np.nan
})

df['r_CC24_302'] = cat_to_num(df['CC24_302'], {
    'Increased a lot': 1, 'Increased somewhat': 2,
    'Stayed about the same': 3, 'Decreased somewhat': 4,
    'Decreased a lot': 5
})

# CC24_303: Prices. "Increased a lot" = most inflation anxiety.
# Direction is REVERSED from raw Stata order — recode so higher = more anxiety.
df['r_CC24_303'] = cat_to_num(df['CC24_303'], {
    'Increased a lot': 5, 'Increased somewhat': 4,
    'Stayed about the same': 3, 'Decreased somewhat': 2,
    'Decreased a lot': 1
})

# CC24_309d_8: binary financial fragility (can't pay $400 emergency)
df['r_CC24_309d_8'] = cat_to_num(df['CC24_309d_8'],
                                  {'selected': 1, 'not selected': 0})

# CC24_309e: ordinal financial situation — REVIEW item
if 'CC24_309e' in df.columns:
    df['r_CC24_309e'] = cat_to_num(df['CC24_309e'], {
        'Excellent': 1, 'Very good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5
    })

# CC24_341: Tax / economic policy (binary)
df['r_CC24_341a'] = cat_to_num(df['CC24_341a'], con_map)  # extend cuts → con
df['r_CC24_341b'] = cat_to_num(df['CC24_341b'], lib_map)  # raise corp tax → lib
df['r_CC24_341c'] = cat_to_num(df['CC24_341c'], lib_map)  # raise top rate → lib
df['r_CC24_341d'] = cat_to_num(df['CC24_341d'], lib_map)  # infrastructure → lib

# ── Immigration ──────────────────────────────────────────────────────
df['r_CC24_323a'] = cat_to_num(df['CC24_323a'], lib_map)  # legal status → lib
df['r_CC24_323b'] = cat_to_num(df['CC24_323b'], con_map)  # border patrols → con
df['r_CC24_323d'] = cat_to_num(df['CC24_323d'], lib_map)  # Dreamers → lib
df['r_CC24_340f'] = cat_to_num(df['CC24_340f'], con_map)  # deny asylum → con

# ── Guns / Policing ──────────────────────────────────────────────────
df['r_CC24_321b'] = cat_to_num(df['CC24_321b'], con_map)  # concealed carry → con
df['r_CC24_321c'] = cat_to_num(df['CC24_321c'], lib_map)  # background checks → lib (REVIEW: ceiling)
df['r_CC24_321d'] = cat_to_num(df['CC24_321d'], con_map)  # more police → con
df['r_CC24_321e'] = cat_to_num(df['CC24_321e'], lib_map)  # fewer police → lib

# ── Abortion ─────────────────────────────────────────────────────────
# CC24_325: weeks limit. Fewer weeks = more restrictive = conservative.
# Recode: conservative_score = 40 - weeks  (0 weeks gets 40; 40 weeks gets 0)
raw_325 = pd.to_numeric(df['CC24_325'].astype(str), errors='coerce')
df['r_CC24_325'] = raw_325.apply(lambda x: 40.0 - x if pd.notna(x) else np.nan)

df['r_CC24_324b'] = cat_to_num(df['CC24_324b'], con_map)  # rape/incest/life only → con
df['r_CC24_324c'] = cat_to_num(df['CC24_324c'], con_map)  # illegal all circ → very con [REVIEW]
df['r_CC24_324d'] = cat_to_num(df['CC24_324d'], lib_map)  # expand access → lib

# ── CC24_340 Civil Liberties Bundle ─────────────────────────────────
df['r_CC24_340a'] = cat_to_num(df['CC24_340a'], lib_map)  # prohibit contraceptive restr → lib
df['r_CC24_340b'] = cat_to_num(df['CC24_340b'], lib_map)  # prohibit abortion restr → lib
df['r_CC24_340c'] = cat_to_num(df['CC24_340c'], lib_map)  # same-sex marriage → lib
df['r_CC24_340e'] = cat_to_num(df['CC24_340e'], con_map)  # surveillance renewal → con

# ── Racial / Gender Attitudes (5-point Likert) ───────────────────────
df['r_CC24_440a'] = cat_to_num(df['CC24_440a'], ag_lib)   # white advantages → agree=lib → con=disagree
df['r_CC24_440b'] = cat_to_num(df['CC24_440b'], ag_con)   # racial problems rare → agree=con
df['r_CC24_440c'] = cat_to_num(df['CC24_440c'], ag_con)   # women seek power → agree=con
df['r_CC24_440d'] = cat_to_num(df['CC24_440d'], ag_con)   # women too offended → agree=con
df['r_CC24_441a'] = cat_to_num(df['CC24_441a'], ag_con)   # Blacks should work up → agree=con [FLAG]
df['r_CC24_441b'] = cat_to_num(df['CC24_441b'], ag_lib)   # slavery creates conditions → agree=lib [FLAG]

# ── Recode verification ──────────────────────────────────────────────
all_r_cols = [c for c in df.columns if c.startswith('r_')]
print(f"  Recoded columns created: {len(all_r_cols)}")

recode_issues = []
for c in all_r_cols:
    n_valid = df[c].notna().sum()
    n_miss  = df[c].isna().sum()
    pct_miss = 100 * n_miss / len(df)
    n_uniq  = df[c].dropna().nunique()
    if n_valid == 0:
        recode_issues.append(f"  WARNING: {c} has zero valid values!")
    elif n_uniq == 1:
        recode_issues.append(f"  WARNING: {c} has only one unique value — zero variance!")

if recode_issues:
    print("\n  RECODE ISSUES:")
    for msg in recode_issues:
        print(msg)
else:
    print("  All recoded variables have valid values and variance. ✓")

# Report missingness for all items
print("\n  Item missingness summary:")
print(f"  {'Variable':<22} {'N valid':>8} {'% missing':>10} {'N unique':>9}")
print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*9}")
for c in all_r_cols:
    n_v = df[c].notna().sum()
    pct_m = 100 * df[c].isna().sum() / len(df)
    n_u = df[c].dropna().nunique()
    flag = " ← CEILING" if c == 'r_CC24_321c' else \
           " ← FLOOR"   if c == 'r_CC24_324c' else \
           " ← BINARY FRAILTY" if c == 'r_CC24_309d_8' else ""
    print(f"  {c:<22} {n_v:>8,} {pct_m:>9.1f}% {n_u:>9}{flag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 3: PSEUDO-R² AGAINST PARTY ID AND IDEOLOGY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 3: PSEUDO-R² AGAINST PARTY ID AND IDEOLOGY")
print(SEP)
print("""
WHAT:  For every candidate item, compute weighted R² from two regressions:
         (1) item ~ party ID (trichotomous: Dem/Ind/Rep)
         (2) item ~ ideology self-placement (CC24_330a, 1–7)

WHY:   Items with high partisan R² are tribalized "team markers" that
       load heavily on a generic partisan axis, not the cross-cutting
       dimensions we want. The R² difference (ideology_r2 - partyid_r2)
       signals items that tap underlying attitudes more than team loyalty.

DECISION — party ID variable:
  We use pid3 (3-category: Democrat / Independent / Republican) as
  dummy-coded predictors, not pid7, because the cross-cutting typology
  hypothesis is fundamentally about crossing the D/R binary.

DECISION — linear R² for binary items:
  Policy items (CC24_321, 323, 324, 340, 341) are binary. Technically,
  McFadden pseudo-R² from logistic regression would be more appropriate.
  However, weighted linear R² (a) scales comparably across binary and
  ordinal items, (b) is computationally simpler, and (c) is the standard
  in political science "explained variance" comparisons. We note this
  limitation but proceed with linear R² for cross-item comparability.

THRESHOLD: partyID R² > 0.15 → item flagged as "highly tribalized".
  (This threshold is conventional; ~0.10–0.20 is typical in this literature.)
""")

def weighted_r2_ols(y, X, w):
    """Weighted OLS R² for outcome y on design matrix X with weights w."""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    if mask.sum() < 10:
        return np.nan
    y_m, X_m, w_m = y[mask], X[mask], w[mask]
    w_m = w_m / w_m.sum()
    y_wmean = np.average(y_m, weights=w_m)
    ss_tot = np.sum(w_m * (y_m - y_wmean) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    # Weighted normal equations
    XtW  = (X_m * w_m[:, None]).T
    XtWX = XtW @ X_m
    XtWy = XtW @ y_m
    try:
        beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    y_hat = X_m @ beta
    ss_res = np.sum(w_m * (y_m - y_hat) ** 2)
    return float(np.clip(1 - ss_res / ss_tot, 0, 1))

# Build party ID design matrix (dummy-coded: Dem=reference)
print(f"\n  Party ID variable: {pid_var}")
pid_dummies = None
if pid_var:
    pid_s = df[pid_var]
    if hasattr(pid_s.dtype, 'categories'):
        pid_cats = list(pid_s.dtype.categories)
        print(f"  Categories: {pid_cats}")
        # Map to D / I / R using first letter or standard labels
        def map_pid3(label):
            label = str(label).lower()
            if 'democrat' in label or label.startswith('d'):   return 0
            if 'republican' in label or label.startswith('r'): return 2
            if 'independent' in label or label.startswith('i') or 'other' in label: return 1
            return np.nan
        pid_num = pid_s.map(map_pid3)
        # Dummy code: Ind=1, Rep=1 as separate dummies (Dem=reference)
        pid_ind = (pid_num == 1).astype(float)
        pid_rep = (pid_num == 2).astype(float)
        pid_dummies = np.column_stack([
            np.ones(len(df)), pid_ind, pid_rep
        ])
    else:
        # Numeric pid — use directly as dummies if 1–3
        pid_num = pd.to_numeric(pid_s, errors='coerce')
        pid_ind = (pid_num == 2).astype(float)
        pid_rep = (pid_num == 3).astype(float)
        pid_dummies = np.column_stack([
            np.ones(len(df)), pid_ind.fillna(0), pid_rep.fillna(0)
        ])

# Ideology design matrix (CC24_330a, already recoded)
ideo_X = np.column_stack([np.ones(len(df)), df['r_CC24_330a'].fillna(np.nan)])

w_arr = w_raw.values

# Compute R² for each item
print("\n  Computing weighted R² for each item (this is fast)...")

# Human-readable label mapping
LABEL_MAP = {
    'r_CC24_330a':   'Ideology self-placement (7-pt)',
    'r_pew_churatd': 'Church attendance frequency',
    'r_pew_religimp':'Religion importance',
    'r_pew_prayer':  'Prayer frequency',
    'r_pew_bornagain':'Born-again / evangelical',
    'r_CC24_301':    'National economy better/worse',
    'r_CC24_302':    'Household income change',
    'r_CC24_303':    'Price perception (inflation)',
    'r_CC24_309d_8': 'Financial fragility (can\'t pay $400)',
    'r_CC24_309e':   'Financial situation (Excellent→Poor)',
    'r_CC24_341a':   'Extend 2017 tax cuts [con]',
    'r_CC24_341b':   'Raise corporate tax rate [lib]',
    'r_CC24_341c':   'Allow $400k+ tax rates to rise [lib]',
    'r_CC24_341d':   '$150B infrastructure spending [lib]',
    'r_CC24_323a':   'Grant legal status (3yr/no felony) [lib]',
    'r_CC24_323b':   'Increase border patrols [con]',
    'r_CC24_323d':   'Dreamers pathway to citizenship [lib]',
    'r_CC24_340f':   'Deny asylum at border [con]',
    'r_CC24_321b':   'Easier concealed carry [con]',
    'r_CC24_321c':   'Criminal background checks [lib] ★CEILING',
    'r_CC24_321d':   'Increase police by 10% [con]',
    'r_CC24_321e':   'Decrease police by 10% [lib]',
    'r_CC24_325':    'Abortion weeks limit (reversed) [con]',
    'r_CC24_324b':   'Abortion: rape/incest/life only [con]',
    'r_CC24_324c':   'Abortion: illegal in all circ [con] ★FLOOR',
    'r_CC24_324d':   'Expand abortion access [lib]',
    'r_CC24_340a':   'Prohibit contraceptive restrictions [lib]',
    'r_CC24_340b':   'Prohibit abortion service restrictions [lib]',
    'r_CC24_340c':   'Require same-sex marriage recognition [lib]',
    'r_CC24_340e':   'Renew post-9/11 surveillance [con]',
    'r_CC24_440a':   'White people have advantages [lib]',
    'r_CC24_440b':   'Racial problems are rare [con]',
    'r_CC24_440c':   'Women seek power over men [con]',
    'r_CC24_440d':   'Women too easily offended [con]',
    'r_CC24_441a':   'Blacks should work up like others [con] ★FLAG',
    'r_CC24_441b':   'Slavery creates conditions for Blacks [lib] ★FLAG',
}

# Define which items are in the main EFA candidate pool
main_items = [
    'r_CC24_330a', 'r_pew_churatd', 'r_pew_religimp',
    'r_pew_prayer', 'r_pew_bornagain',
    'r_CC24_301', 'r_CC24_302', 'r_CC24_303', 'r_CC24_309d_8',
    'r_CC24_341a', 'r_CC24_341b', 'r_CC24_341c', 'r_CC24_341d',
    'r_CC24_323a', 'r_CC24_323b', 'r_CC24_323d', 'r_CC24_340f',
    'r_CC24_321b', 'r_CC24_321c', 'r_CC24_321d', 'r_CC24_321e',
    'r_CC24_325',  'r_CC24_324b', 'r_CC24_324c', 'r_CC24_324d',
    'r_CC24_340a', 'r_CC24_340b', 'r_CC24_340c', 'r_CC24_340e',
    'r_CC24_440a', 'r_CC24_440b', 'r_CC24_440c', 'r_CC24_440d',
    'r_CC24_441a', 'r_CC24_441b',
]
if 'r_CC24_309e' in df.columns:
    main_items.append('r_CC24_309e')

# Pre-assign initial recommended actions (may be updated after polychoric)
# Based on: user instructions + missingness + variance considerations
INITIAL_ACTION = {
    'r_CC24_330a':   'KEEP',
    'r_pew_churatd': 'KEEP',
    'r_pew_religimp':'KEEP',
    'r_pew_prayer':  'KEEP',
    'r_pew_bornagain':'KEEP',
    'r_CC24_301':    'KEEP',
    'r_CC24_302':    'KEEP',
    'r_CC24_303':    'KEEP',
    'r_CC24_309d_8': 'REVIEW',   # binary, low variance, CC24_309e preferred
    'r_CC24_309e':   'REVIEW',   # not in user spec but better ordinal proxy
    'r_CC24_341a':   'KEEP',
    'r_CC24_341b':   'KEEP',
    'r_CC24_341c':   'KEEP',
    'r_CC24_341d':   'KEEP',
    'r_CC24_323a':   'KEEP',
    'r_CC24_323b':   'KEEP',
    'r_CC24_323d':   'KEEP',
    'r_CC24_340f':   'KEEP',
    'r_CC24_321b':   'KEEP',
    'r_CC24_321c':   'REVIEW',   # 93.5% ceiling — near-zero variance
    'r_CC24_321d':   'KEEP',
    'r_CC24_321e':   'KEEP',
    'r_CC24_325':    'KEEP',
    'r_CC24_324b':   'KEEP',
    'r_CC24_324c':   'REVIEW',   # 10.5% support — floor effect
    'r_CC24_324d':   'KEEP',
    'r_CC24_340a':   'KEEP',
    'r_CC24_340b':   'KEEP',
    'r_CC24_340c':   'KEEP',
    'r_CC24_340e':   'KEEP',
    'r_CC24_440a':   'KEEP',
    'r_CC24_440b':   'KEEP',
    'r_CC24_440c':   'KEEP',
    'r_CC24_440d':   'KEEP',
    'r_CC24_441a':   'FLAG',     # user: flag if collinear with F1
    'r_CC24_441b':   'FLAG',
}

INITIAL_REASON = {
    'r_CC24_330a':   'Key self-placement anchor; 7-pt ordinal',
    'r_pew_churatd': 'Behavioral religiosity proxy; 6-pt',
    'r_pew_religimp':'Subjective religiosity; 4-pt',
    'r_pew_prayer':  'Independent behavioral religiosity signal; 7-pt',
    'r_pew_bornagain':'Evangelical identity; binary',
    'r_CC24_301':    'Economic perception; 5-pt; anti-incumbent direction',
    'r_CC24_302':    'Personal financial trajectory; 5-pt',
    'r_CC24_303':    'Inflation perception; 5-pt',
    'r_CC24_309d_8': 'Binary: financial fragility; low variance; see CC24_309e',
    'r_CC24_309e':   'Ordinal financial situation; preferred over 309d_8 for EFA',
    'r_CC24_341a':   'Tax policy; binary; extend Bush/Trump cuts',
    'r_CC24_341b':   'Tax policy; binary; corporate rate',
    'r_CC24_341c':   'Tax policy; binary; top bracket',
    'r_CC24_341d':   'Infrastructure spending; binary',
    'r_CC24_323a':   'Immigration; binary; legal status for workers',
    'r_CC24_323b':   'Immigration; binary; border enforcement',
    'r_CC24_323d':   'Immigration; binary; Dreamers pathway',
    'r_CC24_340f':   'Immigration/civil lib; binary; asylum denial',
    'r_CC24_321b':   'Gun policy; binary; concealed carry',
    'r_CC24_321c':   '93.5% ceiling — near-zero variance; communality likely < 0.05',
    'r_CC24_321d':   'Policing; binary; increase police',
    'r_CC24_321e':   'Policing; binary; decrease police',
    'r_CC24_325':    'Abortion weeks limit; quasi-continuous; most discriminating',
    'r_CC24_324b':   'Abortion nuanced moral frame; binary',
    'r_CC24_324c':   '10.5% support; severe floor effect; may be redundant with 324b',
    'r_CC24_324d':   'Abortion activist support; binary',
    'r_CC24_340a':   'Govt power framing: contraceptive access; binary',
    'r_CC24_340b':   'Govt power framing: abortion access; binary',
    'r_CC24_340c':   'Same-sex marriage recognition; binary',
    'r_CC24_340e':   'Civil liberties vs security; binary; low partisan r in prior work',
    'r_CC24_440a':   'Racial consciousness; 5-pt; test orthogonality to partisan axis',
    'r_CC24_440b':   'Racial minimization; 5-pt',
    'r_CC24_440c':   'Gender zero-sum; 5-pt; orthogonality test',
    'r_CC24_440d':   'Gender grievance; 5-pt; orthogonality test',
    'r_CC24_441a':   'Racial resentment battery; flag if collinear with F1',
    'r_CC24_441b':   'Racial resentment battery; flag if collinear with F1',
}

rows = []
print(f"\n  {'Variable':<22} {'partyID_r2':>11} {'ideo_r2':>9} {'r2_diff':>8}  Action")
print(f"  {'-'*22} {'-'*11} {'-'*9} {'-'*8}  {'-'*10}")

for item in main_items:
    y = df[item].values.astype(float)

    # Party ID R²
    pid_r2 = np.nan
    if pid_dummies is not None:
        pid_r2 = weighted_r2_ols(y, pid_dummies, w_arr)

    # Ideology R² (use 2-col matrix: intercept + r_CC24_330a)
    # Note: CC24_330a itself gets an ideology_r2 of 1.0 vs itself — handle that
    if item == 'r_CC24_330a':
        ideo_r2 = 1.0
    else:
        ideo_r2 = weighted_r2_ols(y, ideo_X, w_arr)

    r2_diff = (ideo_r2 - pid_r2) if (not np.isnan(ideo_r2) and not np.isnan(pid_r2)) else np.nan

    # Update action for highly tribalized items
    action = INITIAL_ACTION.get(item, 'KEEP')
    reason = INITIAL_REASON.get(item, '')

    if not np.isnan(pid_r2) and pid_r2 > 0.15 and action == 'KEEP':
        action = 'REVIEW'
        reason += f'; partyID_r2={pid_r2:.3f} > 0.15 threshold'

    flag = " ★ HIGH partisan" if (not np.isnan(pid_r2) and pid_r2 > 0.15) else ""
    print(f"  {item:<22} {pid_r2:>11.4f} {ideo_r2:>9.4f} {r2_diff:>8.4f}  {action}{flag}")

    rows.append({
        'variable_name':     item,
        'original_varname':  item.replace('r_', ''),
        'question_label':    LABEL_MAP.get(item, item),
        'partyID_r2':        round(pid_r2, 4) if not np.isnan(pid_r2) else np.nan,
        'ideology_r2':       round(ideo_r2, 4) if not np.isnan(ideo_r2) else np.nan,
        'r2_difference':     round(r2_diff, 4) if not np.isnan(r2_diff) else np.nan,
        'redundant_with':    '',   # filled after polychoric
        'recommended_action': action,
        'reason':            reason,
    })

item_df = pd.DataFrame(rows)
print(f"\n  Items flagged partyID_r2 > 0.15: "
      f"{(item_df['partyID_r2'] > 0.15).sum()}")
print(f"  Distribution of initial actions: "
      f"{item_df['recommended_action'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: WEIGHTED POLYCHORIC CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STEP 4: WEIGHTED POLYCHORIC CORRELATION MATRIX")
print(SEP)
print("""
WHAT:  Compute a polychoric correlation matrix for all candidate items,
       using commonweight for weighted frequency tables in the MLE.

WHY:   All items are ordinal (binary to 7-pt scale). Pearson correlation
       underestimates relationships between ordinal variables because it
       treats equal-interval spacing as given. Polychoric assumes the
       observed categories are discretizations of an underlying bivariate
       normal distribution and recovers the latent correlation.

DECISION — weighted vs unweighted polychoric:
  The user specifies "apply commonweight throughout." We implement
  weighted polychoric MLE using weighted frequency tables.
  Alternative: use factor_analyzer's built-in polychoric (unweighted).
  Since N=60,000 and CES weights are bounded (~0.0001–14.9), unweighted
  and weighted estimates will be close, but using weights is correct.

DECISION — polychoric algorithm:
  Two-step MLE (standard in psychometrics):
    1. Estimate thresholds from weighted marginal distributions
       (normal quantiles of cumulative proportions).
    2. Find ρ maximizing bivariate normal log-likelihood for each pair.
  Alternative: simultaneous MLE (more precise, but ~5× slower).
  Chosen: two-step MLE — standard in most software (R psych::polychoric,
  Stata polychoric). Difference from simultaneous is negligible at N=60k.

DECISION — CC24_325 (weeks limit, 41 categories):
  With 41 levels, polychoric approaches Pearson correlation. Treating it
  as ordinal is still technically correct and avoids special-casing.

DECISION — binary items (Support/Oppose):
  With 2 categories, polychoric = tetrachoric. Correct approach for binary.
""")

def weighted_polychoric_pair(x_vals, y_vals, weights):
    """
    Compute weighted polychoric correlation between two ordinal series.

    Algorithm (two-step MLE):
      1. Build weighted contingency table from integer-coded ordinal values.
      2. Estimate thresholds from weighted marginal cumulative proportions
         via normal quantile (probit) transformation.
      3. Find rho in (-1,1) maximizing the bivariate-normal log-likelihood
         using Brent's bounded scalar minimization.

    Returns float rho, or np.nan if degenerate.
    """
    # Convert to integer codes (0, 1, 2, ...)
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

    # Mask out any NaN in x, y, or weights
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(w_arr) & (w_arr > 0)
    if mask.sum() < 20:
        return np.nan

    x_m, y_m, w_m = x_arr[mask], y_arr[mask], w_arr[mask]
    w_m = w_m / w_m.sum()   # normalize so weights sum to 1

    # Unique sorted categories
    x_cats = np.sort(np.unique(x_m))
    y_cats = np.sort(np.unique(y_m))
    nx, ny = len(x_cats), len(y_cats)

    if nx < 2 or ny < 2:   # constant variable — undefined correlation
        return np.nan

    # Map values to integer indices 0..nx-1, 0..ny-1
    x_idx = {v: i for i, v in enumerate(x_cats)}
    y_idx = {v: i for i, v in enumerate(y_cats)}
    xi = np.array([x_idx[v] for v in x_m], dtype=int)
    yi = np.array([y_idx[v] for v in y_m], dtype=int)

    # Build weighted contingency table
    ct = np.zeros((nx, ny))
    for ii, ji, wi in zip(xi, yi, w_m):
        ct[ii, ji] += wi

    # ── Step 1: Estimate thresholds from weighted marginals ──────────
    x_marg = ct.sum(axis=1)   # shape (nx,)
    y_marg = ct.sum(axis=0)   # shape (ny,)

    # Cumulative proportions for inner thresholds (exclude last category)
    BIG = 6.5   # norm.cdf(6.5) ≈ 1 - 2e-11; clips ±∞ safely
    tau_x = np.concatenate([
        [-BIG],
        norm.ppf(np.clip(np.cumsum(x_marg)[:-1], 1e-9, 1 - 1e-9)),
        [BIG]
    ])
    tau_y = np.concatenate([
        [-BIG],
        norm.ppf(np.clip(np.cumsum(y_marg)[:-1], 1e-9, 1 - 1e-9)),
        [BIG]
    ])

    # ── Step 2: Maximize log-likelihood over rho ─────────────────────
    def neg_ll(rho):
        if abs(rho) >= 0.9999:
            return 1e10
        cov   = [[1.0, rho], [rho, 1.0]]
        mv    = multivariate_normal(mean=[0, 0], cov=cov)

        # Build CDF grid: (nx+1) × (ny+1)
        H, K = np.meshgrid(tau_x, tau_y, indexing='ij')
        pts  = np.column_stack([H.ravel(), K.ravel()])
        cdf  = mv.cdf(pts).reshape(nx + 1, ny + 1)

        # Cell probabilities via 2-D differencing
        P = np.diff(np.diff(cdf, axis=0), axis=1)   # shape (nx, ny)
        P = np.maximum(P, 1e-10)
        return -float(np.sum(ct * np.log(P)))

    result = optimize.minimize_scalar(neg_ll, bounds=(-0.9999, 0.9999),
                                      method='bounded',
                                      options={'xatol': 1e-5})
    return float(result.x)


# ── Prepare analytic dataset ─────────────────────────────────────────
# Only items in main_items with valid recodes; CC24_330a is used both
# as a candidate item and as the ideology benchmark — include it.
efa_df = df[main_items + ['commonweight']].copy()
efa_df.rename(columns={'commonweight': 'wt'}, inplace=True)

# Listwise deletion: keep only rows with complete data on all main items
n_before = len(efa_df)
efa_complete = efa_df.dropna()
n_after = len(efa_complete)
pct_retained = 100 * n_after / n_before

print(f"\n  Listwise deletion:")
print(f"    N before: {n_before:,}")
print(f"    N after:  {n_after:,}  ({pct_retained:.1f}% retained)")
print(f"    Weighted N (sum of weights in analytic sample): "
      f"{efa_complete['wt'].sum():.1f}")
print(f"    Items included: {len(main_items)}")

# Identify which items drive most missing data
miss_counts = efa_df.isna().sum().drop('wt').sort_values(ascending=False)
print(f"\n  Top items driving missingness (before listwise deletion):")
for col, cnt in miss_counts[miss_counts > 0].head(10).items():
    pct = 100 * cnt / n_before
    print(f"    {col:<22}: {cnt:,} missing ({pct:.1f}%)")


# ── Compute polychoric matrix ─────────────────────────────────────────
items_for_poly = [c for c in main_items if c in efa_complete.columns]
n_items = len(items_for_poly)
n_pairs = n_items * (n_items - 1) // 2
print(f"\n  Computing weighted polychoric correlations...")
print(f"    Items: {n_items},  Pairs: {n_pairs}")
print(f"    (This runs the two-step MLE for each pair. "
      f"Estimated time: ~3–8 minutes.)")

X_mat = efa_complete[items_for_poly].values.astype(float)
w_vec = efa_complete['wt'].values

poly_mat = np.eye(n_items)   # diagonal = 1

t_poly = time.time()
pair_count = 0
for i in range(n_items):
    for j in range(i + 1, n_items):
        rho = weighted_polychoric_pair(X_mat[:, i], X_mat[:, j], w_vec)
        poly_mat[i, j] = rho
        poly_mat[j, i] = rho
        pair_count += 1
        if pair_count % 50 == 0:
            elapsed = time.time() - t_poly
            pct_done = 100 * pair_count / n_pairs
            eta = (elapsed / pair_count) * (n_pairs - pair_count)
            print(f"    {pair_count}/{n_pairs} pairs ({pct_done:.0f}%)  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

print(f"\n  Polychoric matrix complete. "
      f"Total time: {time.time()-t_poly:.1f}s")

# ── Sanity check: are any values out of [-1,1]? ─────────────────────
off_diag = poly_mat[np.triu_indices(n_items, k=1)]
print(f"\n  Correlation range (off-diagonal): "
      f"min={off_diag.min():.4f}  max={off_diag.max():.4f}")
n_nan = np.isnan(off_diag).sum()
if n_nan > 0:
    print(f"  WARNING: {n_nan} NaN correlations — degenerate variable pairs.")

# ── Find high-correlation pairs (r > 0.70) ──────────────────────────
print("\n  Pairs with |polychoric r| > 0.70 (dimensionally redundant):")
redundant_pairs = []
for i in range(n_items):
    for j in range(i + 1, n_items):
        r = poly_mat[i, j]
        if abs(r) > 0.70:
            v1, v2 = items_for_poly[i], items_for_poly[j]
            redundant_pairs.append((v1, v2, r))
            print(f"    {v1} ↔ {v2}:  r = {r:.4f}")

if not redundant_pairs:
    print("    None found.")

# Update item_df with redundancy information
for v1, v2, r in redundant_pairs:
    # For each pair, decide which to prefer based on:
    # 1. Lower partyID_r2 (less tribalized)
    # 2. Greater variance
    r2_v1 = item_df.loc[item_df['variable_name'] == v1, 'partyID_r2'].values
    r2_v2 = item_df.loc[item_df['variable_name'] == v2, 'partyID_r2'].values

    v1_more_partisan = (len(r2_v1) > 0 and len(r2_v2) > 0 and
                        not np.isnan(r2_v1[0]) and not np.isnan(r2_v2[0]) and
                        r2_v1[0] > r2_v2[0])

    # Append to redundant_with column
    for var, other in [(v1, v2), (v2, v1)]:
        mask = item_df['variable_name'] == var
        existing = item_df.loc[mask, 'redundant_with'].values[0]
        item_df.loc[mask, 'redundant_with'] = (
            f"{existing}; {other} (r={r:.3f})" if existing else
            f"{other} (r={r:.3f})"
        )

print(f"\n  Total redundant pairs: {len(redundant_pairs)}")


# ── Heatmap with hierarchical clustering ────────────────────────────
print("\n  Generating heatmap (hierarchical clustering sort)...")

# Short display names for heatmap
SHORT_NAMES = {
    'r_CC24_330a':   'ideology',
    'r_pew_churatd': 'church_atd',
    'r_pew_religimp':'rel_imp',
    'r_pew_prayer':  'prayer',
    'r_pew_bornagain':'bornagain',
    'r_CC24_301':    'econ_natl',
    'r_CC24_302':    'income_chg',
    'r_CC24_303':    'prices',
    'r_CC24_309d_8': 'cant_pay',
    'r_CC24_309e':   'fin_sit',
    'r_CC24_341a':   'taxcut_ext',
    'r_CC24_341b':   'corp_tax',
    'r_CC24_341c':   'top_rate',
    'r_CC24_341d':   'infrastr',
    'r_CC24_323a':   'legal_stat',
    'r_CC24_323b':   'border_ptrl',
    'r_CC24_323d':   'dreamers',
    'r_CC24_340f':   'deny_asylum',
    'r_CC24_321b':   'conceal_car',
    'r_CC24_321c':   'bgnd_checks',
    'r_CC24_321d':   'more_police',
    'r_CC24_321e':   'less_police',
    'r_CC24_325':    'abort_wks',
    'r_CC24_324b':   'abort_excp',
    'r_CC24_324c':   'abort_illeg',
    'r_CC24_324d':   'abort_expnd',
    'r_CC24_340a':   'contracept',
    'r_CC24_340b':   'abort_govt',
    'r_CC24_340c':   'gay_marr',
    'r_CC24_340e':   'surveill',
    'r_CC24_440a':   '440a_whtadv',
    'r_CC24_440b':   '440b_racerare',
    'r_CC24_440c':   '440c_wmnpwr',
    'r_CC24_440d':   '440d_wmnoff',
    'r_CC24_441a':   '441a_blkwrk',
    'r_CC24_441b':   '441b_slavery',
}
labels = [SHORT_NAMES.get(v, v) for v in items_for_poly]

# Handle NaN in polychoric matrix before clustering
poly_for_clust = np.where(np.isnan(poly_mat), 0, poly_mat)

# Hierarchical clustering of correlation matrix (Ward on distance = 1-|r|)
dist_mat = 1 - np.abs(poly_for_clust)
np.fill_diagonal(dist_mat, 0)
# Condense distance matrix for scipy linkage
from scipy.spatial.distance import squareform
dist_condensed = squareform(dist_mat)
Z = linkage(dist_condensed, method='ward')
order = leaves_list(Z)

# Reorder matrix and labels
poly_ordered  = poly_mat[np.ix_(order, order)]
labels_ordered = [labels[i] for i in order]

# Plot
fig, ax = plt.subplots(figsize=(16, 14))
mask_nan = np.isnan(poly_ordered)
plot_mat = np.where(mask_nan, 0, poly_ordered)

sns.heatmap(
    plot_mat,
    ax=ax,
    xticklabels=labels_ordered,
    yticklabels=labels_ordered,
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    center=0,
    linewidths=0.3,
    linecolor='#cccccc',
    cbar_kws={'label': 'Polychoric r', 'shrink': 0.7},
)
ax.set_title(
    'Weighted Polychoric Correlation Matrix — CES 2024 EFA Candidates\n'
    '(hierarchical clustering sort; Ward linkage on |r| distance)',
    fontsize=12, pad=12
)
plt.xticks(fontsize=7, rotation=45, ha='right')
plt.yticks(fontsize=7)

# Draw red boxes around redundant pairs (r > 0.70)
for v1, v2, r in redundant_pairs:
    if v1 in items_for_poly and v2 in items_for_poly:
        # Find positions in clustered order
        orig_i = items_for_poly.index(v1)
        orig_j = items_for_poly.index(v2)
        pos_i = list(order).index(orig_i)
        pos_j = list(order).index(orig_j)
        for pi, pj in [(pos_i, pos_j), (pos_j, pos_i)]:
            ax.add_patch(plt.Rectangle(
                (pj, pi), 1, 1,
                fill=False, edgecolor='gold', lw=1.5
            ))

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, 'polychoric_heatmap.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Heatmap saved: {heatmap_path}")


# ══════════════════════════════════════════════════════════════════════
# SAVE OUTPUT FILES
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SAVING OUTPUT FILES")
print(SEP)

# ── 1. efa_variable_list.csv ─────────────────────────────────────────
varlist_path = os.path.join(OUTPUT_DIR, 'efa_variable_list.csv')
out_cols = [
    'variable_name', 'question_label', 'partyID_r2', 'ideology_r2',
    'r2_difference', 'redundant_with', 'recommended_action', 'reason'
]
item_df[out_cols].to_csv(varlist_path, index=False)
print(f"\n  1. efa_variable_list.csv  → {varlist_path}")

# ── 2. polychoric_matrix.csv ─────────────────────────────────────────
poly_df = pd.DataFrame(poly_mat,
                       index=items_for_poly,
                       columns=items_for_poly)
poly_path = os.path.join(OUTPUT_DIR, 'polychoric_matrix.csv')
poly_df.to_csv(poly_path)
print(f"  2. polychoric_matrix.csv  → {poly_path}")

# ── 3. efa_checkpoint_summary.txt ────────────────────────────────────
action_counts = item_df['recommended_action'].value_counts().to_dict()
high_partisan = item_df[item_df['partyID_r2'] > 0.15][
    ['variable_name', 'question_label', 'partyID_r2']
].to_string(index=False)
redundant_str = "\n".join(
    f"  {v1} ↔ {v2}  (r = {r:.4f})" for v1, v2, r in redundant_pairs
) if redundant_pairs else "  None"

flagged_excluded_str = (
    "  CC24_441e  — 79.4% missing; shown only to non-White respondents\n"
    "  CC24_441f  — 79.4% missing; shown only to non-White respondents\n"
    "  CC24_441g  — 79.4% missing; shown only to non-White respondents\n"
    "  (Recommend running subsample EFA on ~12,377 non-White respondents)"
)

summary_txt = f"""
EFA PIPELINE CHECKPOINT SUMMARY — CES 2024
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ITEM COUNTS (main EFA candidate pool)
  KEEP  : {action_counts.get('KEEP', 0)}
  REVIEW: {action_counts.get('REVIEW', 0)}
  FLAG  : {action_counts.get('FLAG', 0)}   ← In pool but flagged per user instructions
  DROP  : {action_counts.get('DROP', 0)}
  Total items in matrix: {len(items_for_poly)}

WEIGHTED N AFTER LISTWISE DELETION
  N (unweighted) : {n_after:,}  ({pct_retained:.1f}% of full sample)
  Weighted N     : {efa_complete['wt'].sum():.1f}

NOTE ON LISTWISE DELETION: The ~{100-pct_retained:.1f}% loss is driven primarily by
the CC24_440/441 batteries (17–18% missing each), likely due to being
post-survey items shown to a subsample. If these batteries are dropped,
N recovers to nearly the full 60,000.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POLYCHORIC CORRELATION RANGE
  Min (off-diagonal): {off_diag[~np.isnan(off_diag)].min():.4f}
  Max (off-diagonal): {off_diag[~np.isnan(off_diag)].max():.4f}
  NaN correlations  : {np.isnan(off_diag).sum()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIMENSIONALLY REDUNDANT PAIRS (|r| > 0.70)
{redundant_str}

For each redundant pair above, the recommended_action column in
efa_variable_list.csv indicates which item to prefer (lower partyID_r²,
greater variance). No items were automatically dropped — manual review required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ITEMS FLAGGED AS HIGHLY TRIBALIZED (partyID_r² > 0.15)
{high_partisan}

These items have more variance explained by party ID than by ideology
self-placement. They are REVIEW not DROP — loading patterns in the EFA
will determine whether they add independent signal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ITEMS EXCLUDED FROM MAIN EFA (FLAGGED SET)
{flagged_excluded_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ITEMS REQUIRING SPECIAL ATTENTION
  CC24_321c (background checks): {df['r_CC24_321c'].mean():.1%} coded as "Oppose"
    i.e. 93.5%+ ceiling on conservative recoding; near-zero variance.
    Communality in EFA likely < 0.05. Recommend DROP from final EFA set.

  CC24_324c (abortion illegal all circ): {df['CC24_324c'].value_counts(normalize=True).get('Support', 0):.1%} support
    Severe floor effect. May be redundant with CC24_324b.
    Recommend REVIEW after seeing factor loadings.

  CC24_309d_8 vs CC24_309e:
    CC24_309d_8 is binary (selected/not selected) with limited variance.
    CC24_309e is a 5-pt ordinal financial situation scale — more appropriate
    for polychoric EFA and measuring the same latent construct. Consider
    substituting CC24_309e for CC24_309d_8 before running EFA.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT FILES
  1. {varlist_path}
  2. {poly_path}
  3. {os.path.join(OUTPUT_DIR, 'efa_checkpoint_summary.txt')}
  4. {heatmap_path}  (bonus: heatmap visualization)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 4 COMPLETE. AWAITING INSTRUCTIONS BEFORE PROCEEDING TO EFA (STEP 5).
""".strip()

summary_path = os.path.join(OUTPUT_DIR, 'efa_checkpoint_summary.txt')
with open(summary_path, 'w') as f:
    f.write(summary_txt)

print(f"  3. efa_checkpoint_summary.txt → {summary_path}")
print(f"\n{'='*68}")
print("STEP 4 COMPLETE.")
print("Three output files saved. Stopping — awaiting your instructions.")
print('='*68)
print()
print(summary_txt)
