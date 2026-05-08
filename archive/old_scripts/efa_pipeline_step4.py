#!/usr/bin/env python3
"""
=============================================================================
EFA PIPELINE: CES 2024 POLITICAL TYPOLOGY ANALYSIS
Steps 1–4: Load → Define → Item Diagnostics → Polychoric Correlation Matrix
=============================================================================

Goal: Identify latent attitude dimensions that cross-cut the partisan axis,
      NOT reproduce the standard left-right dimension.

Output directory: /Users/bdecker/Documents/STV/Claude/
Stops after Step 4. Awaiting instruction before proceeding to EFA.
"""

import sys
import subprocess
import os
import warnings
import textwrap

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# PACKAGE INSTALL
# ---------------------------------------------------------------------------
for pkg in ['factor_analyzer', 'statsmodels', 'pingouin']:
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                   capture_output=True)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

OUTPUT_DIR = '/Users/bdecker/Documents/STV/Claude'
DATA_PATH  = ('/Users/bdecker/Documents/STV/2024 CES Base/'
              'CCES24_Common_OUTPUT_vv_topost_final.dta')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DIVIDER = '=' * 72


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print(f'\n{DIVIDER}')
print('STEP 1: LOAD DATA')
print(DIVIDER)
print("""
WHAT: Load CES 2024 .dta with convert_categoricals=True to preserve value
      labels as Categorical dtype, enabling label-aware recoding.

WHY:  We need text labels ("Strongly agree", "Support", etc.) to correctly
      assign numeric direction (higher = more conservative/restrictive).
      Reading as raw numeric codes would require a separate lookup table.

ASSUMPTION: 'commonweight' is the correct weighting variable. Weights are
      already normalized (mean ≈ 1.0 across full sample).

PARAMETER DECISION: convert_missing=False (not True). The combination of
      convert_missing=True + convert_categoricals=True fails in pandas 2.x
      because Stata extended missing values (StataMissingValue objects) are
      unhashable and cannot be factorized into Categorical columns. With
      convert_missing=False, Stata's .a/.b/etc. extended missings are
      coerced to NaN by the Categorical factorization, which is what we want.
""")

df = pd.read_stata(DATA_PATH, convert_categoricals=True,
                   convert_missing=False, convert_dates=False)
print(f'  Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns')

w = df['commonweight']
print(f'  commonweight: N={w.notna().sum():,}, mean={w.mean():.4f}, '
      f'std={w.std():.4f}, min={w.min():.6f}, max={w.max():.4f}')
print(f'  Weight > 5: {(w > 5).sum()} rows ({100*(w > 5).mean():.1f}%)')
print(f'  Weight < 0.1: {(w < 0.1).sum()} rows ({100*(w < 0.1).mean():.1f}%)')

print("""
STEP 1 SUMMARY: 60,000 respondents, 694 columns. Weights bounded (0.0001–14.9);
mean = 1.0 confirms normalization. Extreme high weights (>5) affect ~2% of
sample — these will upweight in the polychoric computation but are within
normal CES weighting range. Proceed to variable definition and recoding.
""")


# =============================================================================
# STEP 2: DEFINE CANDIDATE ITEM SET AND RECODE
# =============================================================================
print(f'\n{DIVIDER}')
print('STEP 2: DEFINE AND RECODE CANDIDATE ITEMS')
print(DIVIDER)
print("""
WHAT: Define the 35-item EFA candidate pool. Recode all variables to a
      consistent numeric scale where higher values = more
      conservative / traditional / restrictive / anti-government.

WHY:  Polychoric correlation requires numeric ordinal codes. Consistent
      directionality makes factor loadings interpretable at a glance
      (positive loading = item moves with conservative pole of factor).

KEY DECISIONS:

  1. CC24_309d (Emergency expense capacity): The user specified "CC24_309d"
     but this is stored as multi-select binary sub-items (CC24_309d_1..9).
     Sub-item CC24_309d_8 = "I wouldn't be able to pay" is the most direct
     financial fragility measure. ALTERNATIVE: CC24_309e ("How would you
     describe your current financial situation?" Excellent→Poor) is a 5-point
     ordinal that is MORE APPROPRIATE for polychoric EFA. We use CC24_309d_8
     as the user specified CC24_309d, but flag this. Note that CC24_309d_8 is
     binary (tetrachoric case) with ~20% selecting it.

  2. CC24_324c ("Make abortion illegal in all circumstances"): Not explicitly
     kept or dropped by user. It has severe floor effect (89% Oppose). We
     INCLUDE it but flag it for review — its loading will tell us if it adds
     independent signal or is just a more extreme version of CC24_324b.

  3. CC24_321c ("Background checks"): 93% Support — near-ceiling. We INCLUDE
     but flag. With only 7% variation, it will likely show low communality.

  4. CC24_441e/f/g: 79% missing (shown only to non-White respondents ~20%
     of sample). Listwise deletion on 35 items would reduce N to ~12,000
     and introduce selection bias. EXCLUDE from main EFA. Flag separately
     for a non-White subsample analysis.

  5. Direction coding for pew variables: Religious practice items (attendance,
     prayer, importance, born-again) are proxies for religious conservatism.
     Higher values = more religious. Whether religion loads as "conservative"
     or as an independent dimension is an empirical question the EFA will answer.

  6. CC24_302 (Household income): "Decreased a lot" coded as 5 (highest).
     Direction: higher = household income fell = worse personal finances.
     Note this is NOT the same as being conservative — it measures economic
     vulnerability, which may load on a separate dimension.

  7. CC24_303 (Prices): REVERSED so higher = more perceived inflation.
     The raw Stata order has "Increased a lot" as category 1; we recode
     to 5 so that high inflation perception = high numeric value.
""")

# ---------------------------------------------------------------------------
# 2a. Helper: convert Categorical column → numeric with explicit label map
# ---------------------------------------------------------------------------
def cat_to_num(series, label_map):
    """
    Map Categorical labels to numeric values per label_map.
    Keys not in label_map (including NaN categories) → NaN.
    """
    return series.map(label_map).astype(float)

# ---------------------------------------------------------------------------
# 2b. Recode each variable
# ---------------------------------------------------------------------------
recode = {}

# --- IDEOLOGY SELF-PLACEMENT ---
recode['CC24_330a'] = cat_to_num(df['CC24_330a'], {
    'Very Liberal': 1, 'Liberal': 2, 'Somewhat Liberal': 3,
    'Middle of the Road': 4, 'Somewhat Conservative': 5,
    'Conservative': 6, 'Very Conservative': 7,
    'Not sure': np.nan  # treated as missing; not a scale point
})

# --- PEW RELIGION BATTERY ---
recode['pew_churatd'] = cat_to_num(df['pew_churatd'], {
    'Never': 1, 'Seldom': 2, 'A few times a year': 3,
    'Once or twice a month': 4, 'Once a week': 5,
    'More than once a week': 6,
    "Don't know": np.nan
})
recode['pew_religimp'] = cat_to_num(df['pew_religimp'], {
    'Not at all important': 1, 'Not too important': 2,
    'Somewhat important': 3, 'Very important': 4
})
recode['pew_prayer'] = cat_to_num(df['pew_prayer'], {
    'Never': 1, 'Seldom': 2, 'A few times a month': 3,
    'Once a week': 4, 'A few times a week': 5,
    'Once a day': 6, 'Several times a day': 7,
    "Don't know": np.nan
})
recode['pew_bornagain'] = cat_to_num(df['pew_bornagain'], {
    'No': 0, 'Yes': 1
})

# --- ECONOMIC PERCEPTIONS ---
# Higher = more negative / worse perception
recode['CC24_301'] = cat_to_num(df['CC24_301'], {
    'Gotten much better': 1, 'Gotten somewhat better': 2,
    'Stayed about the same': 3, 'Gotten somewhat worse': 4,
    'Gotten much worse': 5, 'Not sure': np.nan
})
recode['CC24_302'] = cat_to_num(df['CC24_302'], {
    'Increased a lot': 1, 'Increased somewhat': 2,
    'Stayed about the same': 3, 'Decreased somewhat': 4,
    'Decreased a lot': 5
})
# REVERSED: higher = more perceived inflation
recode['CC24_303'] = cat_to_num(df['CC24_303'], {
    'Decreased a lot': 1, 'Decreased somewhat': 2,
    'Stayed about the same': 3, 'Increased somewhat': 4,
    'Increased a lot': 5
})

# --- FINANCIAL FRAGILITY ---
# DECISION: Using CC24_309d_8 ("I wouldn't be able to pay") as specified.
# It is binary: selected=1 (can't pay = financial fragility), not selected=0.
# NOTE: CC24_309e (5-point financial situation scale) is a better ordinal
# measure for polychoric EFA. User should consider swapping this item.
recode['CC24_309d_8'] = cat_to_num(df['CC24_309d_8'], {
    'selected': 1, 'not selected': 0
})

# --- TAX / ECONOMIC POLICY (Support/Oppose) ---
# Extend 2017 tax cuts: Support = conservative → higher
recode['CC24_341a'] = cat_to_num(df['CC24_341a'], {'Support': 1, 'Oppose': 0})
# Raise corporate tax: Support = liberal → REVERSED (Oppose = conservative)
recode['CC24_341b'] = cat_to_num(df['CC24_341b'], {'Support': 0, 'Oppose': 1})
# Allow $400k+ rates to rise: Support = liberal → REVERSED
recode['CC24_341c'] = cat_to_num(df['CC24_341c'], {'Support': 0, 'Oppose': 1})
# Infrastructure spending: Support = liberal → REVERSED
recode['CC24_341d'] = cat_to_num(df['CC24_341d'], {'Support': 0, 'Oppose': 1})

# --- IMMIGRATION (CC24_323grid; dropping CC24_323c wall item) ---
# Grant legal status: Support = liberal → REVERSED
recode['CC24_323a'] = cat_to_num(df['CC24_323a'], {'Support': 0, 'Oppose': 1})
# Increase border patrols: Support = conservative → higher
recode['CC24_323b'] = cat_to_num(df['CC24_323b'], {'Support': 1, 'Oppose': 0})
# Dreamers: Support = liberal → REVERSED
recode['CC24_323d'] = cat_to_num(df['CC24_323d'], {'Support': 0, 'Oppose': 1})

# --- GUNS / POLICING (CC24_321grid; dropping CC24_321a assault rifle ban) ---
# Concealed carry easier: Support = conservative → higher
recode['CC24_321b'] = cat_to_num(df['CC24_321b'], {'Support': 1, 'Oppose': 0})
# Background checks: Support = liberal → REVERSED (FLAG: 93% Support → near-ceiling)
recode['CC24_321c'] = cat_to_num(df['CC24_321c'], {'Support': 0, 'Oppose': 1})
# Increase police 10%: Support = conservative → higher
recode['CC24_321d'] = cat_to_num(df['CC24_321d'], {'Support': 1, 'Oppose': 0})
# Decrease police 10%: Support = liberal → REVERSED
recode['CC24_321e'] = cat_to_num(df['CC24_321e'], {'Support': 0, 'Oppose': 1})

# --- ABORTION ---
# CC24_325: weeks limit (0=always illegal → 40=no limit).
# REVERSED: higher = more restrictive = more conservative
# Raw is Categorical with float labels; convert to float first
cc325_num = pd.to_numeric(df['CC24_325'].astype(str), errors='coerce')
recode['CC24_325'] = 40.0 - cc325_num  # 0→40 (most conservative), 40→0

# Permit only rape/incest/life: Support = conservative → higher
recode['CC24_324b'] = cat_to_num(df['CC24_324b'], {'Support': 1, 'Oppose': 0})
# Make illegal ALL circumstances: Support = very conservative (FLAG: 89% Oppose)
recode['CC24_324c'] = cat_to_num(df['CC24_324c'], {'Support': 1, 'Oppose': 0})
# Expand access: Support = liberal → REVERSED
recode['CC24_324d'] = cat_to_num(df['CC24_324d'], {'Support': 0, 'Oppose': 1})

# --- CC24_340 CIVIL RIGHTS / CIVIL LIBERTIES BUNDLE ---
# Prohibit contraceptive restrictions: Support = liberal → REVERSED
recode['CC24_340a'] = cat_to_num(df['CC24_340a'], {'Support': 0, 'Oppose': 1})
# Prohibit abortion service restrictions: Support = liberal → REVERSED
recode['CC24_340b'] = cat_to_num(df['CC24_340b'], {'Support': 0, 'Oppose': 1})
# Require same-sex marriage recognition: Support = liberal → REVERSED
recode['CC24_340c'] = cat_to_num(df['CC24_340c'], {'Support': 0, 'Oppose': 1})
# Renew post-9/11 surveillance (warrantless): Support = conservative/security → higher
recode['CC24_340e'] = cat_to_num(df['CC24_340e'], {'Support': 1, 'Oppose': 0})
# Deny asylum: Support = conservative/restrictive → higher
recode['CC24_340f'] = cat_to_num(df['CC24_340f'], {'Support': 1, 'Oppose': 0})

# --- RACIAL / GENDER ATTITUDES (5-point Agree–Disagree) ---
# Strongly agree=0 in raw Categorical codes; Strongly disagree=4
# We want: conservative response = higher numeric value

# CC24_440a "White advantages": Agree = liberal/progressive
#   → Strongly agree=1, Strongly disagree=5 (conservative = deny white privilege)
_440_map_lib_agree = {'Strongly agree': 1, 'Somewhat agree': 2,
                      'Neither agree nor disagree': 3,
                      'Somewhat disagree': 4, 'Strongly disagree': 5}

# CC24_440b-d, CC24_441a: Agree = conservative/resentful
#   → REVERSE: Strongly agree=5, Strongly disagree=1
_440_map_con_agree = {'Strongly agree': 5, 'Somewhat agree': 4,
                      'Neither agree nor disagree': 3,
                      'Somewhat disagree': 2, 'Strongly disagree': 1}

recode['CC24_440a'] = cat_to_num(df['CC24_440a'], _440_map_lib_agree)
recode['CC24_440b'] = cat_to_num(df['CC24_440b'], _440_map_con_agree)
recode['CC24_440c'] = cat_to_num(df['CC24_440c'], _440_map_con_agree)
recode['CC24_440d'] = cat_to_num(df['CC24_440d'], _440_map_con_agree)

# CC24_441a ("Blacks should work up without special favors"): Agree = conservative
recode['CC24_441a'] = cat_to_num(df['CC24_441a'], _440_map_con_agree)
# CC24_441b ("Slavery creates conditions"): Agree = liberal
recode['CC24_441b'] = cat_to_num(df['CC24_441b'], _440_map_lib_agree)

# --- CC24_420: MILITARY INTERVENTIONISM (all 7 items) ---
# Battery: "Would you approve of the use of U.S. military troops in order to...?"
#   (Check all that apply)
# Items:
#   CC24_420_1: Ensure the supply of oil          (r_pid7 = +0.145 — weakly Republican)
#   CC24_420_2: Destroy a terrorist camp          (r_pid7 = +0.140 — weakly Republican)
#   CC24_420_3: Intervene in genocide/civil war   (r_pid7 = -0.052 — near-zero)
#   CC24_420_4: Assist the spread of democracy    (r_pid7 = -0.037 — near-zero)
#   CC24_420_5: Protect American allies under attack (r_pid7 = +0.053 — near-zero)
#   CC24_420_6: Help the UN uphold international law (r_pid7 = -0.110 — weakly Democrat)
#   CC24_420_7: None of the above                 (r_pid7 = +0.030 — near-zero; isolationism)
#
# All partisan correlations are very weak (max |r| = 0.145), confirming these
# items are genuinely cross-partisan — exactly what the EFA is designed to surface.
#
# CODING: selected=1 for all items (approve = 1, not approve = 0).
# DIRECTION: Not reversed. Low partisan signal means direction convention is
#   largely irrelevant; EFA loadings will define the structure empirically.
#
# SPECIAL NOTE — CC24_420_7 ("None"):
#   This is a mutually exclusive "None of the above" option. Zero respondents
#   selected both item 7 and any other item. Including it will produce negative
#   polychoric correlations with items 1–6 by construction. It captures
#   pure isolationism. FLAG for potential exclusion if it distorts factor structure.
#
# MISSINGNESS: All 7 items have identical N missing = 10,568 (same post-survey
#   subsample as CC24_440/441). Listwise deletion N will not decrease further.

_mil_map = {'selected': 1, 'not selected': 0}
recode['CC24_420_1'] = cat_to_num(df['CC24_420_1'], _mil_map)  # oil supply
recode['CC24_420_2'] = cat_to_num(df['CC24_420_2'], _mil_map)  # terrorist camp
recode['CC24_420_3'] = cat_to_num(df['CC24_420_3'], _mil_map)  # genocide/civil war
recode['CC24_420_4'] = cat_to_num(df['CC24_420_4'], _mil_map)  # spread democracy
recode['CC24_420_5'] = cat_to_num(df['CC24_420_5'], _mil_map)  # protect allies
recode['CC24_420_6'] = cat_to_num(df['CC24_420_6'], _mil_map)  # UN/international law
recode['CC24_420_7'] = cat_to_num(df['CC24_420_7'], _mil_map)  # None [FLAG: exclusive "none"]

# --- CC24_421_1/2: ELECTION TRUST ---
# CC24_421_1: "Elections in the U.S. are fair"
# CC24_421_2: "Your state or local government conducted a fair and accurate election in 2024"
# Scale: Strongly agree → Strongly disagree (5-point).
# DIRECTION: Agree = trust elections = institutionalist/liberal in 2024 context
#   (the stolen-election narrative is strongly associated with conservative/Trump base).
#   REVERSED so higher = more distrust = more conservative direction.
# NOTE: CC24_421_2 (state elections) is heavily skewed toward trust (80%+ Agree) —
#   low variance may limit its utility. Flag if communality < 0.10.
_elec_trust_map = {'Strongly agree': 1, 'Somewhat agree': 2,
                   'Neither agree nor disagree': 3,
                   'Somewhat disagree': 4, 'Strongly disagree': 5}
recode['CC24_421_1'] = cat_to_num(df['CC24_421_1'], _elec_trust_map)
recode['CC24_421_2'] = cat_to_num(df['CC24_421_2'], _elec_trust_map)

# --- CC24_423/424: GOVERNMENT TRUST ---
# CC24_423: "How much trust in the federal government handling the nation's problems?"
# CC24_424: "How much trust in the government of the state where you live?"
# Scale: A great deal → None at all (4-point).
# DIRECTION: Low trust = anti-government. In 2024, federal distrust was concentrated
#   in the Republican base (anti-Biden administration). However, populist left ALSO
#   has low federal trust — this item may be genuinely cross-cutting.
# CODING: Higher = less trust. "A great deal"=1 ... "None at all"=4.
_govt_trust_map = {'A great deal': 1, 'A fair amount': 2,
                   'Not very much': 3, 'None at all': 4}
recode['CC24_423'] = cat_to_num(df['CC24_423'], _govt_trust_map)
recode['CC24_424'] = cat_to_num(df['CC24_424'], _govt_trust_map)

# --- CC24_312grid: JOB APPROVAL OF INSTITUTIONS ---
# "Do you approve of the way each is doing their job?"
# Scale: Strongly approve=4 / Somewhat approve=3 / Somewhat disapprove=2 /
#        Strongly disapprove=1 / Not sure=NaN.
# Higher = more approval (neutral baseline; EFA loadings reveal direction).
#
# DIRECTION WARNING by item:
#   312a (Biden) / 312i (Harris): Partisan proxies. Approve = liberal/Democratic.
#     Will load NEGATIVELY on any conservative factor. Flagged as partisan proxies.
#   312b (Congress): Bipartisan disapproval. May load on anti-institutional factor.
#   312c (SCOTUS): Conservatives approve more (conservative court post-Dobbs).
#     High approval likely loads with conservative items.
#   312d-h (Governor, state leg, House rep, Senators): STATE-SPECIFIC.
#     A Democrat approving their Democratic governor and a Republican approving
#     their Republican governor BOTH get "approve=4". Direction is not consistent
#     across respondents. These likely capture co-partisan trust, not ideology.
#     Flagged; may need to be dropped depending on EFA loadings.
_appr_map = {'Strongly approve': 4, 'Somewhat approve': 3,
             'Somewhat disapprove': 2, 'Strongly disapprove': 1,
             'Not sure': np.nan}
recode['CC24_312a'] = cat_to_num(df['CC24_312a'], _appr_map)  # Biden
recode['CC24_312b'] = cat_to_num(df['CC24_312b'], _appr_map)  # U.S. Congress
recode['CC24_312c'] = cat_to_num(df['CC24_312c'], _appr_map)  # U.S. Supreme Court
recode['CC24_312d'] = cat_to_num(df['CC24_312d'], _appr_map)  # Governor [FLAG:state-specific]
recode['CC24_312e'] = cat_to_num(df['CC24_312e'], _appr_map)  # State legislature [FLAG:same]
recode['CC24_312f'] = cat_to_num(df['CC24_312f'], _appr_map)  # House rep [FLAG:same; high DK]
recode['CC24_312g'] = cat_to_num(df['CC24_312g'], _appr_map)  # Senator 1 [FLAG:same; high DK]
recode['CC24_312h'] = cat_to_num(df['CC24_312h'], _appr_map)  # Senator 2 [FLAG:same; high DK]
recode['CC24_312i'] = cat_to_num(df['CC24_312i'], _appr_map)  # Harris [FLAG:partisan proxy]

# --- ANCILLARY VARIABLES FOR R² DIAGNOSTICS ---
recode['pid7'] = cat_to_num(df['pid7'], {
    'Strong Democrat': 1, 'Not very strong Democrat': 2, 'Lean Democrat': 3,
    'Independent': 4, 'Lean Republican': 5, 'Not very strong Republican': 6,
    'Strong Republican': 7, 'Not sure': np.nan
})
recode['ideo5'] = cat_to_num(df['ideo5'], {
    'Very liberal': 1, 'Liberal': 2, 'Moderate': 3,
    'Conservative': 4, 'Very conservative': 5, 'Not sure': np.nan
})

# Build working dataframe
data = pd.DataFrame(recode)
data['commonweight'] = df['commonweight'].values

# EFA candidates (all except pid7, ideo5, commonweight)
EFA_VARS = [v for v in data.columns
            if v not in ('pid7', 'ideo5', 'commonweight')]

print(f'  Total EFA candidate items defined: {len(EFA_VARS)}')
print(f'  Items: {EFA_VARS}')

# Quick value-range check
print('\n  Value ranges after recoding (should all be numeric):')
for v in EFA_VARS:
    s = data[v].dropna()
    print(f'    {v:18s}: min={s.min():.1f}, max={s.max():.1f}, '
          f'mean={s.mean():.2f}, N={len(s):,}, '
          f'miss={data[v].isna().sum():,}')


# =============================================================================
# STEP 3: ITEM DIAGNOSTICS — PARTYID R², VARIANCE, FLAGS
# =============================================================================
print(f'\n{DIVIDER}')
print('STEP 3: ITEM DIAGNOSTICS')
print(DIVIDER)
print("""
WHAT: For each candidate item, compute:
  (a) Weighted pseudo-R² regressing item on pid7 (partisan tribalization)
  (b) Weighted pseudo-R² regressing item on ideo5 (ideological loading)
  (c) R² difference (ideology minus party): positive = item captures ideology
      beyond partisanship; negative = item is more tribalized than ideological
  (d) Floor/ceiling check: flag if >80% responses in one category
  (e) Missing data rate

DECISION — R² metric: We use weighted OLS R² as a pseudo-measure of
partisan tribalization. OLS R² on ordinal outcomes is not ideal but is
conventional in political science item analysis and interpretable.
Alternative: ordinal logistic R² (McFadden). OLS is used here for speed
and comparability across binary and multi-point items.

FLAGGING RULES:
  - FLAG 'ceiling/floor' if >80% of weighted responses in one category
  - FLAG '441_collinearity' for CC24_441a/b per user instruction
  - FLAG '324c_not_specified' for CC24_324c (user did not explicitly keep)
  - FLAG 'binary_CC24_309d_8' for CC24_309d_8 (derived from multi-select)
  - FLAG '441efg_excluded' for CC24_441e/f/g (79% missing, non-White only)
""")

weights = data['commonweight'].values

def weighted_r2(y, x_values, w):
    """Weighted OLS R² of y on x, after listwise deletion."""
    mask = (~np.isnan(y)) & (~np.isnan(x_values)) & (~np.isnan(w))
    if mask.sum() < 100:
        return np.nan
    y_, x_, w_ = y[mask], x_values[mask], w[mask]
    # Weighted mean of y
    y_wmean = np.average(y_, weights=w_)
    ss_tot = np.sum(w_ * (y_ - y_wmean)**2)
    if ss_tot == 0:
        return np.nan
    # Weighted OLS: beta = (X'WX)^-1 X'Wy
    X = np.column_stack([np.ones(len(x_)), x_])
    W = np.diag(w_)
    try:
        XtW = X.T @ W
        beta = np.linalg.solve(XtW @ X, XtW @ y_)
        y_hat = X @ beta
        ss_res = np.sum(w_ * (y_ - y_hat)**2)
        return max(0.0, 1 - ss_res / ss_tot)
    except np.linalg.LinAlgError:
        return np.nan

pid7_vals  = data['pid7'].values
ideo5_vals = data['ideo5'].values

# Item metadata
ITEM_LABELS = {
    'CC24_330a':   'Ideology self-placement (VL=1 to VC=7)',
    'pew_churatd': 'Church attendance frequency',
    'pew_religimp':'Importance of religion in life',
    'pew_prayer':  'Prayer frequency',
    'pew_bornagain':'Born-again/evangelical identity (binary)',
    'CC24_301':    'National economy: better or worse past year',
    'CC24_302':    'Household income: change past year',
    'CC24_303':    'Perceived price change past year (rev: high=inflation)',
    'CC24_309d_8': 'Financial fragility: can\'t cover $400 emergency [binary; from multi-select]',
    'CC24_341a':   'Support extend 2017 tax cuts',
    'CC24_341b':   'Oppose raise corporate tax to 28% [rev]',
    'CC24_341c':   'Oppose allow $400k+ tax rates to rise [rev]',
    'CC24_341d':   'Oppose $150B infrastructure spending [rev]',
    'CC24_323a':   'Oppose grant legal status to working immigrants [rev]',
    'CC24_323b':   'Support increase border patrols',
    'CC24_323d':   'Oppose Dreamers pathway to citizenship [rev]',
    'CC24_321b':   'Support easier concealed carry permits',
    'CC24_321c':   'Oppose background checks on all gun sales [rev; FLAG:ceiling]',
    'CC24_321d':   'Support increase police by 10%',
    'CC24_321e':   'Oppose decrease police by 10% [rev]',
    'CC24_325':    'Abortion weeks limit reversed (40-weeks; high=restrictive)',
    'CC24_324b':   'Support permit abortion only rape/incest/life danger',
    'CC24_324c':   'Support make abortion illegal ALL circumstances [FLAG:floor]',
    'CC24_324d':   'Oppose expand abortion access [rev]',
    'CC24_340a':   'Oppose prohibit contraceptive restrictions [rev]',
    'CC24_340b':   'Oppose prohibit abortion service restrictions [rev]',
    'CC24_340c':   'Oppose require same-sex marriage recognition [rev]',
    'CC24_340e':   'Support renew post-9/11 surveillance programs',
    'CC24_340f':   'Support deny asylum at border',
    'CC24_440a':   'Disagree: white people have advantages (high=conservative)',
    'CC24_440b':   'Agree: racial problems are rare/isolated',
    'CC24_440c':   'Agree: women seek to gain power over men',
    'CC24_440d':   'Agree: women are too easily offended',
    'CC24_441a':   'Agree: Blacks should work up like other minorities [FLAG:collin]',
    'CC24_441b':   'Disagree: slavery created conditions (high=conservative) [FLAG:collin]',
    # --- NEW ADDITIONS ---
    'CC24_420_1':  'Mil. intervention: ensure oil supply [binary; r_pid=+0.145]',
    'CC24_420_2':  'Mil. intervention: destroy terrorist camp [binary; r_pid=+0.140]',
    'CC24_420_3':  'Mil. intervention: genocide/civil war [binary; r_pid=-0.052]',
    'CC24_420_4':  'Mil. intervention: spread democracy [binary; r_pid=-0.037]',
    'CC24_420_5':  'Mil. intervention: protect allies [binary; r_pid=+0.053]',
    'CC24_420_6':  'Mil. intervention: UN/intl law [binary; r_pid=-0.110]',
    'CC24_420_7':  'Mil. intervention: NONE/isolationist [binary; FLAG:exclusive-none]',
    'CC24_421_1':  'Distrust federal elections (rev: high=distrust) [5-pt]',
    'CC24_421_2':  'Distrust state elections (rev: high=distrust) [5-pt; highly skewed]',
    'CC24_423':    'Low trust in federal government (high=less trust) [4-pt]',
    'CC24_424':    'Low trust in state government (high=less trust) [4-pt]',
    'CC24_312a':   'Approve Biden [4-pt; FLAG:partisan proxy]',
    'CC24_312b':   'Approve U.S. Congress [4-pt]',
    'CC24_312c':   'Approve U.S. Supreme Court [4-pt]',
    'CC24_312d':   'Approve Governor [4-pt; FLAG:direction varies by state]',
    'CC24_312e':   'Approve State legislature [4-pt; FLAG:direction varies by state]',
    'CC24_312f':   'Approve House rep [4-pt; FLAG:direction varies; high DK]',
    'CC24_312g':   'Approve Senator 1 [4-pt; FLAG:direction varies; high DK]',
    'CC24_312h':   'Approve Senator 2 [4-pt; FLAG:direction varies; high DK]',
    'CC24_312i':   'Approve Harris [4-pt; FLAG:partisan proxy]',
}

FLAG_NOTES = {
    'CC24_321c':   'ceiling_effect: 93% Support (opposition only 7%)',
    'CC24_324c':   'floor_effect: 89% Oppose; not explicitly kept by user',
    'CC24_309d_8': 'derived_from_multiselect: user specified CC24_309d; '
                   'consider CC24_309e (5pt ordinal) as better alternative',
    'CC24_441a':   'flag_collinearity: user requests collinearity check vs Factor 1',
    'CC24_441b':   'flag_collinearity: user requests collinearity check vs Factor 1',
    # New items
    'CC24_420_7':  'exclusive_none: mutually exclusive with 420_1-6; '
                   'will create artificial negative polychoric correlations',
    'CC24_421_2':  'skewed: ~78% Strongly/Somewhat agree (state election trust very high); '
                   'low variance may suppress communality',
    'CC24_312a':   'partisan_proxy: Biden approval is near-pure party ID signal',
    'CC24_312i':   'partisan_proxy: Harris approval is near-pure party ID signal',
    'CC24_312d':   'direction_inconsistent: partisan valence varies by state governor',
    'CC24_312e':   'direction_inconsistent: partisan valence varies by state legislature',
    'CC24_312f':   'direction_inconsistent: direction varies by rep; 29% Not sure → NaN',
    'CC24_312g':   'direction_inconsistent: direction varies by senator; 22% Not sure → NaN',
    'CC24_312h':   'direction_inconsistent: direction varies by senator; 25% Not sure → NaN; '
                   'only shown when CurrentSen2Name populated',
}

rows = []
for v in EFA_VARS:
    y = data[v].values.astype(float)
    mask_valid = ~np.isnan(y)
    n_valid = mask_valid.sum()
    pct_missing = 100 * (1 - n_valid / len(y))

    # Weighted variance / floor-ceiling
    y_valid = y[mask_valid]
    w_valid = weights[mask_valid]
    w_valid_n = w_valid / w_valid.sum()
    wmean = np.sum(w_valid_n * y_valid)
    wvar  = np.sum(w_valid_n * (y_valid - wmean)**2)

    # Weighted modal category share
    vals, cnts = np.unique(y_valid, return_counts=True)
    w_by_val = {val: w_valid[y_valid == val].sum() for val in vals}
    total_w  = sum(w_by_val.values())
    modal_share = max(w_by_val.values()) / total_w

    # R² against pid7 and ideo5
    r2_pid  = weighted_r2(y, pid7_vals, weights)
    r2_ideo = weighted_r2(y, ideo5_vals, weights)
    r2_diff = (r2_ideo - r2_pid) if (r2_pid is not None and
                                     r2_ideo is not None) else np.nan

    # Determine action and reason
    flag = FLAG_NOTES.get(v, '')
    if flag:
        action = 'REVIEW'
    else:
        action = 'KEEP'

    rows.append({
        'variable_name':    v,
        'question_label':   ITEM_LABELS.get(v, ''),
        'n_valid':          n_valid,
        'pct_missing':      round(pct_missing, 1),
        'weighted_variance':round(wvar, 4),
        'modal_share_pct':  round(100 * modal_share, 1),
        'partyID_r2':       round(r2_pid, 4)  if r2_pid  is not None else np.nan,
        'ideology_r2':      round(r2_ideo, 4) if r2_ideo is not None else np.nan,
        'r2_difference':    round(r2_diff, 4) if not np.isnan(r2_diff) else np.nan,
        'redundant_with':   '',  # will be filled after polychoric
        'recommended_action': action,
        'reason':           flag if flag else 'Meets inclusion criteria',
    })

diag_df = pd.DataFrame(rows)

print('\n  Item Diagnostics Table:')
print(f'  {"Variable":18s} {"PID_R2":>7} {"IDEO_R2":>8} {"R2_DIFF":>8} '
      f'{"MODAL%":>7} {"MISS%":>6}  Action')
print('  ' + '-' * 72)
for _, r in diag_df.iterrows():
    print(f'  {r.variable_name:18s} {r.partyID_r2:7.4f} {r.ideology_r2:8.4f} '
          f'{r.r2_difference:8.4f} {r.modal_share_pct:7.1f} '
          f'{r.pct_missing:6.1f}  {r.recommended_action}')

# Also print the 3 excluded items
print('\n  EXCLUDED items (not in EFA candidate pool):')
excluded = [
    ('CC24_441e', '79.4% missing — shown only to non-White respondents'),
    ('CC24_441f', '79.4% missing — shown only to non-White respondents'),
    ('CC24_441g', '79.4% missing — shown only to non-White respondents'),
]
for vname, reason in excluded:
    print(f'    {vname}: {reason}')

print(f"""
STEP 3 SUMMARY:
  - {(diag_df.recommended_action == 'KEEP').sum()} items: KEEP
  - {(diag_df.recommended_action == 'REVIEW').sum()} items: REVIEW (flagged, not dropped)
  - 3 items EXCLUDED before EFA (CC24_441e/f/g — 79% missing)

  High partyID R² items to watch: items where PID R² > 0.30 are highly
  tribalized. R² difference < 0 means the item is MORE partisan than
  ideological — these are candidates for dropping if they load with Factor 1.

  Next: compute polychoric correlation matrix on the {len(EFA_VARS)}-item set.
""")


# =============================================================================
# STEP 4: WEIGHTED POLYCHORIC CORRELATION MATRIX
# =============================================================================
print(f'\n{DIVIDER}')
print('STEP 4: WEIGHTED POLYCHORIC CORRELATION MATRIX')
print(DIVIDER)
print("""
WHAT: Compute a weighted polychoric correlation matrix for all 35 EFA
      candidate items.

WHY:  All items are ordinal (or binary, which is the special-case tetrachoric).
      Pearson correlation assumes continuous, normally distributed data and
      systematically underestimates correlations between ordinal items,
      especially those with few categories. Polychoric correlation estimates
      the latent continuous correlation underlying the observed ordinal
      responses, which is what we want for factor analysis of attitudes.

METHOD DECISION — Weighted vs. unweighted polychoric:
  The user specified "Apply commonweight throughout using weighted correlation
  matrices." We implement weighted polychoric directly using the weighted
  frequency table approach (Olsson 1979):
    1. Build weighted 2-way contingency table n[i,j]
    2. Estimate latent thresholds from weighted marginals via normal quantiles
    3. Maximize weighted bivariate normal log-likelihood over ρ
  This correctly accounts for CES's complex survey weighting.

  ALTERNATIVE CONSIDERED: Using factor_analyzer's built-in polychoric
  (which is unweighted). Rejected because user explicitly requested weighted
  analysis. For N=60,000, weighted vs. unweighted will differ modestly, but
  the user's methodological specification takes precedence.

COMPUTATIONAL NOTE: 35 items → 35×34/2 = 595 pairs. Each pair requires
  one 1D optimization. Total runtime: approximately 3–8 minutes depending
  on hardware. Progress printed every 50 pairs.

CC24_325 (41-level quasi-continuous): With 41 ordinal levels, polychoric
  approaches Pearson. Threshold estimation from 40 marginal quantiles is
  numerically stable at N=60,000. No special handling needed.

BINARY ITEMS: Polychoric reduces to tetrachoric for 2×2 tables. The same
  algorithm handles this case naturally.
""")

# ---------------------------------------------------------------------------
# 4a. Prepare the EFA data matrix (listwise deletion)
# ---------------------------------------------------------------------------
# Drop rows where ANY EFA variable is missing (listwise deletion)
efa_data = data[EFA_VARS + ['commonweight']].dropna()
efa_weights = efa_data['commonweight'].values
efa_matrix  = efa_data[EFA_VARS].values.astype(float)
N_weighted  = efa_weights.sum()

print(f'  Listwise deletion: {len(df):,} → {len(efa_data):,} complete cases')
print(f'  Weighted N after listwise deletion: {N_weighted:,.0f}')
print(f'  ({100*len(efa_data)/len(df):.1f}% of original sample retained)')
print(f'  Computing {len(EFA_VARS)*(len(EFA_VARS)-1)//2} polychoric pairs...\n')

# ---------------------------------------------------------------------------
# 4b. Weighted polychoric correlation function
# ---------------------------------------------------------------------------
BIG = 8.0  # norm.cdf(8) ≈ 1 - 6e-16; clip -inf/+inf to ±8 for numerical safety

from scipy.stats import multivariate_normal as mvn_dist

def weighted_polychoric(x, y, w, max_iter_boundary=100):
    """
    Weighted polychoric correlation between ordinal vectors x and y.

    Algorithm:
      1. Build weighted contingency table using the sample weights w.
      2. Estimate marginal thresholds tau_x, tau_y via weighted quantiles
         mapped through the inverse normal (probit).
      3. Find rho ∈ (-1, 1) maximizing the bivariate normal log-likelihood
         L(rho) = Σ n_ij * log[P_ij(rho)] where P_ij is the bivariate normal
         probability for cell (i,j) given thresholds and correlation rho.
      4. Uses scipy.optimize.minimize_scalar (Brent method, bounded).

    Handles: binary items (tetrachoric case), many-category items,
             cells with zero weighted count (adds epsilon to avoid log(0)).
    """
    # Remove any NaN rows (should be none after listwise deletion, but safety check)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10:
        return np.nan

    x_, y_, w_ = x[mask], y[mask], w[mask]

    # Unique ordinal categories (as integers)
    x_cats = np.sort(np.unique(x_))
    y_cats = np.sort(np.unique(y_))
    nx, ny = len(x_cats), len(y_cats)

    # Edge cases
    if nx < 2 or ny < 2:
        return np.nan

    # Map values to 0-based indices
    xi = np.searchsorted(x_cats, x_)
    yi = np.searchsorted(y_cats, y_)

    # Weighted contingency table
    ct = np.zeros((nx, ny))
    for ii, jj, ww in zip(xi, yi, w_):
        ct[ii, jj] += ww
    ct = ct / ct.sum()  # normalize to proportions

    # Marginal proportions → thresholds via probit
    x_marg = ct.sum(axis=1)
    y_marg = ct.sum(axis=0)

    # Cumulative marginals (exclude last to get interior thresholds)
    x_cum = np.cumsum(x_marg)[:-1]
    y_cum = np.cumsum(y_marg)[:-1]

    # Clip to avoid norm.ppf(0) or norm.ppf(1) = ±inf
    eps_clip = 1e-6
    x_cum = np.clip(x_cum, eps_clip, 1 - eps_clip)
    y_cum = np.clip(y_cum, eps_clip, 1 - eps_clip)

    tau_x = np.concatenate([[-BIG], norm.ppf(x_cum), [BIG]])
    tau_y = np.concatenate([[-BIG], norm.ppf(y_cum), [BIG]])

    # Log-likelihood (negative, for minimization)
    def neg_log_lik(rho):
        if abs(rho) >= 0.9999:
            return 1e10
        try:
            cov = [[1.0, rho], [rho, 1.0]]
            # Build CDF grid (nx+1 × ny+1) using bivariate normal
            H, K = np.meshgrid(tau_x, tau_y, indexing='ij')
            pts = np.column_stack([H.ravel(), K.ravel()])
            cdf_flat = mvn_dist(mean=[0, 0], cov=cov).cdf(pts)
            cdf_grid = cdf_flat.reshape(nx + 1, ny + 1)
            # Cell probabilities via 2D differencing
            P = np.diff(np.diff(cdf_grid, axis=0), axis=1)
            P = np.maximum(P, 1e-12)  # floor to avoid log(0)
            return -np.sum(ct * np.log(P))
        except Exception:
            return 1e10

    result = minimize_scalar(neg_log_lik, bounds=(-0.9999, 0.9999),
                             method='bounded',
                             options={'xatol': 1e-5, 'maxiter': 500})
    return float(result.x)


# ---------------------------------------------------------------------------
# 4c. Compute the full polychoric matrix
# ---------------------------------------------------------------------------
import time

n_vars = len(EFA_VARS)
poly_matrix = np.eye(n_vars)

start_time = time.time()
pair_count  = 0
total_pairs = n_vars * (n_vars - 1) // 2

for i in range(n_vars):
    for j in range(i + 1, n_vars):
        xi = efa_matrix[:, i]
        xj = efa_matrix[:, j]
        rho = weighted_polychoric(xi, xj, efa_weights)
        poly_matrix[i, j] = rho
        poly_matrix[j, i] = rho
        pair_count += 1
        if pair_count % 50 == 0 or pair_count == total_pairs:
            elapsed = time.time() - start_time
            eta = (elapsed / pair_count) * (total_pairs - pair_count)
            print(f'    [{pair_count:3d}/{total_pairs}] '
                  f'{EFA_VARS[i]} × {EFA_VARS[j]}  '
                  f'ρ={rho:+.3f}  '
                  f'elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s')

elapsed_total = time.time() - start_time
print(f'\n  Polychoric matrix complete. Total time: {elapsed_total:.0f}s')

# Symmetry check
print(f'  Matrix is symmetric: {np.allclose(poly_matrix, poly_matrix.T)}')
print(f'  Diagonal = 1.0: {np.allclose(np.diag(poly_matrix), 1.0)}')

# Summary statistics of off-diagonal correlations
off_diag = poly_matrix[np.triu_indices(n_vars, k=1)]
print(f'  Off-diagonal r: mean={np.mean(off_diag):.3f}, '
      f'sd={np.std(off_diag):.3f}, '
      f'min={np.min(off_diag):.3f}, max={np.max(off_diag):.3f}')
print(f'  |r| > 0.50: {(np.abs(off_diag) > 0.50).sum()} pairs')
print(f'  |r| > 0.70: {(np.abs(off_diag) > 0.70).sum()} pairs  ← redundancy threshold')

# Check for any non-convergence (NaN)
nan_count = np.isnan(poly_matrix).sum()
print(f'  NaN in matrix: {nan_count}')
if nan_count > 0:
    # Replace NaN with 0 for now and report
    nan_pairs = [(EFA_VARS[i], EFA_VARS[j])
                 for i in range(n_vars) for j in range(n_vars)
                 if np.isnan(poly_matrix[i, j]) and i != j]
    print(f'  NaN pairs: {nan_pairs}')
    poly_matrix = np.where(np.isnan(poly_matrix), 0, poly_matrix)


# ---------------------------------------------------------------------------
# 4d. Flag redundant pairs (|r| > 0.70)
# ---------------------------------------------------------------------------
print('\n  --- REDUNDANT PAIRS (|r| > 0.70) ---')
redundant_pairs = []
for i in range(n_vars):
    for j in range(i + 1, n_vars):
        r = poly_matrix[i, j]
        if abs(r) > 0.70:
            vi, vj = EFA_VARS[i], EFA_VARS[j]
            ri_pid  = diag_df.loc[diag_df.variable_name == vi, 'partyID_r2'].values[0]
            rj_pid  = diag_df.loc[diag_df.variable_name == vj, 'partyID_r2'].values[0]
            ri_var  = diag_df.loc[diag_df.variable_name == vi, 'weighted_variance'].values[0]
            rj_var  = diag_df.loc[diag_df.variable_name == vj, 'weighted_variance'].values[0]
            # Retain: lower partyID R² (less tribalized) and higher variance
            # Score: lower is better (prefer lower pid_r2, higher variance)
            score_i = ri_pid - ri_var * 0.1   # lower R² preferred
            score_j = rj_pid - rj_var * 0.1
            retain = vi if score_i < score_j else vj
            drop   = vj if retain == vi else vi
            pair_rec = {
                'var_a': vi, 'var_b': vj, 'r': round(r, 4),
                'partyID_r2_a': round(ri_pid, 4), 'partyID_r2_b': round(rj_pid, 4),
                'variance_a': round(ri_var, 4),  'variance_b': round(rj_var, 4),
                'preferred_retain': retain,  'suggested_drop': drop,
                'rationale': f'Lower PID R²={min(ri_pid,rj_pid):.3f}, '
                             f'higher variance={max(ri_var,rj_var):.3f}'
            }
            redundant_pairs.append(pair_rec)
            print(f'    {vi:18s} × {vj:18s}  r={r:+.3f}  '
                  f'→ retain {retain}, review {drop}')

if not redundant_pairs:
    print('    None. No pairs exceed r=0.70 — low dimensional redundancy.')

# Update redundant_with column in diag_df
for pair in redundant_pairs:
    for va, vb in [(pair['var_a'], pair['var_b']),
                   (pair['var_b'], pair['var_a'])]:
        mask = diag_df.variable_name == va
        existing = diag_df.loc[mask, 'redundant_with'].values[0]
        diag_df.loc[mask, 'redundant_with'] = (
            (existing + '; ' if existing else '') + vb
        )
    # Update action to REVIEW if suggested drop
    for v_drop in [pair['suggested_drop']]:
        mask = diag_df.variable_name == v_drop
        if diag_df.loc[mask, 'recommended_action'].values[0] == 'KEEP':
            diag_df.loc[mask, 'recommended_action'] = 'REVIEW'
            diag_df.loc[mask, 'reason'] = (
                diag_df.loc[mask, 'reason'].values[0] +
                f'; r>{0.70:.2f} with {pair["var_a"] if v_drop == pair["var_b"] else pair["var_b"]}'
            )


# ---------------------------------------------------------------------------
# 4e. Heatmap sorted by hierarchical clustering
# ---------------------------------------------------------------------------
print('\n  Generating polychoric correlation heatmap...')

# Hierarchical clustering on the correlation matrix
# Convert to distance (1 - |r|) for clustering
dist_matrix = 1 - np.abs(poly_matrix)
np.fill_diagonal(dist_matrix, 0)
condensed = squareform(dist_matrix, checks=False)
linkage_matrix = linkage(condensed, method='ward')
leaf_order = leaves_list(linkage_matrix)

# Reorder matrix
poly_reordered = poly_matrix[np.ix_(leaf_order, leaf_order)]
labels_reordered = [EFA_VARS[i] for i in leaf_order]

# Shorter display labels
short_labels = {
    'CC24_330a': '330a_ideology', 'pew_churatd': 'pew_church',
    'pew_religimp': 'pew_rel_imp', 'pew_prayer': 'pew_prayer',
    'pew_bornagain': 'pew_bornagin', 'CC24_301': '301_econ_nat',
    'CC24_302': '302_hh_income', 'CC24_303': '303_prices',
    'CC24_309d_8': '309d_cant_pay', 'CC24_341a': '341a_taxcut',
    'CC24_341b': '341b_corp_tax', 'CC24_341c': '341c_top_rate',
    'CC24_341d': '341d_infrastr', 'CC24_323a': '323a_legal_stat',
    'CC24_323b': '323b_brd_patr', 'CC24_323d': '323d_dreamers',
    'CC24_321b': '321b_conceal', 'CC24_321c': '321c_bkgd_chk',
    'CC24_321d': '321d_+police', 'CC24_321e': '321e_-police',
    'CC24_325': '325_wks_rev', 'CC24_324b': '324b_exc_only',
    'CC24_324c': '324c_ban_all', 'CC24_324d': '324d_expand',
    'CC24_340a': '340a_contracep', 'CC24_340b': '340b_abrt_svc',
    'CC24_340c': '340c_ss_marr', 'CC24_340e': '340e_surveill',
    'CC24_340f': '340f_asylum', 'CC24_440a': '440a_wh_advan',
    'CC24_440b': '440b_race_rare', 'CC24_440c': '440c_wmn_pwr',
    'CC24_440d': '440d_wmn_off', 'CC24_441a': '441a_race_res1',
    'CC24_441b': '441b_race_res2',
}
display_labels = [short_labels.get(v, v) for v in labels_reordered]

fig, ax = plt.subplots(figsize=(18, 16))

# Custom diverging colormap: blue (negative) → white (zero) → red (positive)
cmap = plt.cm.RdBu_r

im = ax.imshow(poly_reordered, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Polychoric r')

ax.set_xticks(range(len(display_labels)))
ax.set_yticks(range(len(display_labels)))
ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(display_labels, fontsize=8)

# Overlay correlation values for cells |r| > 0.40
for ii in range(len(display_labels)):
    for jj in range(len(display_labels)):
        r_val = poly_reordered[ii, jj]
        if ii != jj and abs(r_val) > 0.40:
            color = 'white' if abs(r_val) > 0.65 else 'black'
            ax.text(jj, ii, f'{r_val:.2f}', ha='center', va='center',
                    fontsize=5.5, color=color, fontweight='bold')

ax.set_title(
    'Weighted Polychoric Correlation Matrix — CES 2024 EFA Candidates\n'
    'Sorted by Hierarchical Clustering (Ward linkage on 1−|r| distance)\n'
    'Values shown for |r| > 0.40',
    fontsize=11, pad=12
)
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, 'polychoric_heatmap.png')
fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Heatmap saved: {heatmap_path}')


# =============================================================================
# SAVE OUTPUTS
# =============================================================================
print(f'\n{DIVIDER}')
print('SAVING OUTPUT FILES')
print(DIVIDER)

# 1. efa_variable_list.csv
var_list_path = os.path.join(OUTPUT_DIR, 'efa_variable_list.csv')
diag_df.to_csv(var_list_path, index=False)
print(f'  1. Saved: {var_list_path}')

# Also save redundant pairs as a separate sheet
if redundant_pairs:
    redund_path = os.path.join(OUTPUT_DIR, 'redundant_pairs.csv')
    pd.DataFrame(redundant_pairs).to_csv(redund_path, index=False)
    print(f'     Redundant pairs: {redund_path}')

# 2. polychoric_matrix.csv
poly_df = pd.DataFrame(poly_matrix, index=EFA_VARS, columns=EFA_VARS)
poly_path = os.path.join(OUTPUT_DIR, 'polychoric_matrix.csv')
poly_df.to_csv(poly_path)
print(f'  2. Saved: {poly_path}')

# 3. efa_checkpoint_summary.txt
keep_n    = (diag_df.recommended_action == 'KEEP').sum()
review_n  = (diag_df.recommended_action == 'REVIEW').sum()
drop_n    = 0  # no items auto-dropped per user instruction

# Identify surprising collinearity results
top_pairs_by_r = sorted(
    [(EFA_VARS[i], EFA_VARS[j], poly_matrix[i,j])
     for i in range(n_vars) for j in range(i+1, n_vars)],
    key=lambda x: abs(x[2]), reverse=True
)[:10]

summary_lines = [
    'EFA PIPELINE CHECKPOINT — CES 2024 POLITICAL TYPOLOGY ANALYSIS',
    'Steps 1-4 Complete. Awaiting instruction before proceeding to EFA.',
    '=' * 68,
    '',
    'ITEM COUNTS',
    '-----------',
    f'  KEEP:    {keep_n} items',
    f'  REVIEW:  {review_n} items (flagged but included — awaiting your decision)',
    f'  EXCLUDED: 3 items (CC24_441e/f/g — 79% missing, non-White only)',
    f'  AUTO-DROPPED: 0 items (per user instruction)',
    '',
    'SAMPLE SIZE',
    '-----------',
    f'  Original N: 60,000',
    f'  After listwise deletion on 35 EFA items: {len(efa_data):,} cases',
    f'  Retained: {100*len(efa_data)/60000:.1f}% of sample',
    f'  Weighted N: {N_weighted:,.0f}',
    f'  Primary missingness drivers: CC24_440/441 series (~18% missing each)',
    '',
    'FLAGGED ITEMS (for your review)',
    '--------------------------------',
    '  CC24_321c  — CEILING EFFECT: 93% Oppose background check repeal.',
    '               Low effective variance. Communality likely low.',
    '               Recommendation: review factor loading; drop if < 0.30.',
    '',
    '  CC24_324c  — FLOOR EFFECT: 89% Oppose "abortion illegal all circs."',
    '               Not explicitly included by user (only 324b/d were kept).',
    '               Recommendation: review; likely loads with 324b on abortion',
    '               restriction factor but adds little independent signal.',
    '',
    '  CC24_309d_8 — BINARY FROM MULTI-SELECT: user specified CC24_309d.',
    '               Stored as multi-select; used sub-item 8 ("can\'t pay").',
    '               BETTER ALTERNATIVE: CC24_309e is a 5-point ordinal',
    '               financial situation scale — more appropriate for',
    '               polychoric EFA. Recommend swapping.',
    '',
    '  CC24_441a  — RACIAL RESENTMENT (user-flagged for collinearity check)',
    '  CC24_441b  — RACIAL RESENTMENT (user-flagged for collinearity check)',
    '               These will be checked against Factor 1 loading in EFA.',
    '',
    'REDUNDANT PAIRS (polychoric |r| > 0.70)',
    '-----------------------------------------',
]

if redundant_pairs:
    for p in redundant_pairs:
        summary_lines.append(
            f'  {p["var_a"]:18s} × {p["var_b"]:18s}  r={p["r"]:+.4f}'
        )
        summary_lines.append(
            f'    → Preferred retain: {p["preferred_retain"]}  '
            f'(lower PID R²={min(p["partyID_r2_a"],p["partyID_r2_b"]):.3f})'
        )
        summary_lines.append('')
else:
    summary_lines.append('  NONE — No pairs exceed r=0.70 threshold.')

summary_lines += [
    '',
    'TOP 10 POLYCHORIC CORRELATIONS (absolute value)',
    '-------------------------------------------------',
]
for va, vb, r in top_pairs_by_r:
    summary_lines.append(f'  {va:18s} × {vb:18s}  r={r:+.4f}')

summary_lines += [
    '',
    'ITEMS WITH HIGHEST PARTYID R² (most tribalized)',
    '---------------------------------------------------',
]
top_pid = diag_df.nlargest(8, 'partyID_r2')[['variable_name','partyID_r2','ideology_r2','r2_difference']]
for _, row in top_pid.iterrows():
    summary_lines.append(
        f'  {row.variable_name:18s}  PID R²={row.partyID_r2:.4f}  '
        f'IDEO R²={row.ideology_r2:.4f}  diff={row.r2_difference:+.4f}'
    )

summary_lines += [
    '',
    'ITEMS WITH LOWEST PARTYID R² (most cross-partisan)',
    '---------------------------------------------------',
]
low_pid = diag_df.nsmallest(8, 'partyID_r2')[['variable_name','partyID_r2','ideology_r2','r2_difference']]
for _, row in low_pid.iterrows():
    summary_lines.append(
        f'  {row.variable_name:18s}  PID R²={row.partyID_r2:.4f}  '
        f'IDEO R²={row.ideology_r2:.4f}  diff={row.r2_difference:+.4f}'
    )

summary_lines += [
    '',
    'SAVED FILES',
    '-----------',
    f'  1. {var_list_path}',
    f'  2. {poly_path}',
    f'  3. {os.path.join(OUTPUT_DIR, "efa_checkpoint_summary.txt")}',
    f'  4. {heatmap_path}',
    '',
    'NEXT STEP: Awaiting your instruction before proceeding to EFA (Step 5).',
    '=' * 68,
]

summary_text = '\n'.join(summary_lines)
summary_path = os.path.join(OUTPUT_DIR, 'efa_checkpoint_summary.txt')
with open(summary_path, 'w') as f:
    f.write(summary_text)
print(f'  3. Saved: {summary_path}')

print(f'\n{DIVIDER}')
print('STEP 4 COMPLETE — PIPELINE PAUSED')
print(DIVIDER)
print(summary_text)
print(f'\n{DIVIDER}')
print('All files saved. Waiting for your instruction before proceeding to EFA.')
print(DIVIDER)
