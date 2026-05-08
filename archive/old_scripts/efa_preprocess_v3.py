#!/usr/bin/env python3
"""
EFA PRE-PROCESSING UPDATE — v3
================================
Tasks:
  1. Missingness audit: % missing + marginal N gained by dropping each variable
  2. Apply explicit drops:
       - CC24_420_1 through CC24_420_7  (compositional/multi-select; invalid for EFA)
       - CC24_312a, CC24_312i           (Biden/Harris approval: near-pure partisan proxies)
       - CC24_330a                      (ideology self-placement: partisan proxy; dominates)
       - CC24_301                       (national economy eval: partisan proxy)
  3. Print question text for CC24_421_1, CC24_421_2, CC24_423, CC24_424
  4. Report new weighted N after drops
  5. Save efa_variable_list_v3.csv
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

OUTPUT_DIR = '/Users/bdecker/Documents/STV/Claude'
DATA_PATH  = ('/Users/bdecker/Documents/STV/2024 CES Base/'
              'CCES24_Common_OUTPUT_vv_topost_final.dta')
DIVIDER = '=' * 72

# =============================================================================
# LOAD DATA
# =============================================================================
print(f'\n{DIVIDER}')
print('LOADING DATA')
print(DIVIDER)
df = pd.read_stata(DATA_PATH, convert_categoricals=True,
                   convert_missing=False, convert_dates=False)
print(f'  Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns')

w = df['commonweight'].values.copy()

# =============================================================================
# RECODE — identical to efa_pipeline_step4.py, minus the DROPPED items
# We recode ALL 55 items here so we can do the missingness audit first.
# =============================================================================
def cat_to_num(series, label_map):
    return series.map(label_map).astype(float)

recode = {}

# --- IDEOLOGY SELF-PLACEMENT (will be DROPPED but recode to audit missingness)
recode['CC24_330a'] = cat_to_num(df['CC24_330a'], {
    'Very Liberal': 1, 'Liberal': 2, 'Somewhat Liberal': 3,
    'Middle of the Road': 4, 'Somewhat Conservative': 5,
    'Conservative': 6, 'Very Conservative': 7, 'Not sure': np.nan
})

# --- PEW RELIGION
recode['pew_churatd'] = cat_to_num(df['pew_churatd'], {
    'Never': 1, 'Seldom': 2, 'A few times a year': 3,
    'Once or twice a month': 4, 'Once a week': 5, 'More than once a week': 6,
    "Don't know": np.nan
})
recode['pew_religimp'] = cat_to_num(df['pew_religimp'], {
    'Not at all important': 1, 'Not too important': 2,
    'Somewhat important': 3, 'Very important': 4
})
recode['pew_prayer'] = cat_to_num(df['pew_prayer'], {
    'Never': 1, 'Seldom': 2, 'A few times a month': 3,
    'Once a week': 4, 'A few times a week': 5,
    'Once a day': 6, 'Several times a day': 7, "Don't know": np.nan
})
recode['pew_bornagain'] = cat_to_num(df['pew_bornagain'], {'No': 0, 'Yes': 1})

# --- ECONOMIC PERCEPTIONS (CC24_301 will be DROPPED but recode to audit)
recode['CC24_301'] = cat_to_num(df['CC24_301'], {
    'Gotten much better': 1, 'Gotten somewhat better': 2,
    'Stayed about the same': 3, 'Gotten somewhat worse': 4,
    'Gotten much worse': 5, 'Not sure': np.nan
})
recode['CC24_302'] = cat_to_num(df['CC24_302'], {
    'Increased a lot': 1, 'Increased somewhat': 2, 'Stayed about the same': 3,
    'Decreased somewhat': 4, 'Decreased a lot': 5
})
recode['CC24_303'] = cat_to_num(df['CC24_303'], {
    'Decreased a lot': 1, 'Decreased somewhat': 2, 'Stayed about the same': 3,
    'Increased somewhat': 4, 'Increased a lot': 5
})
recode['CC24_309d_8'] = cat_to_num(df['CC24_309d_8'], {'selected': 1, 'not selected': 0})

# --- TAX/FISCAL POLICY
recode['CC24_341a'] = cat_to_num(df['CC24_341a'], {'Support': 1, 'Oppose': 0})
recode['CC24_341b'] = cat_to_num(df['CC24_341b'], {'Support': 0, 'Oppose': 1})
recode['CC24_341c'] = cat_to_num(df['CC24_341c'], {'Support': 0, 'Oppose': 1})
recode['CC24_341d'] = cat_to_num(df['CC24_341d'], {'Support': 0, 'Oppose': 1})

# --- IMMIGRATION
recode['CC24_323a'] = cat_to_num(df['CC24_323a'], {'Support': 0, 'Oppose': 1})
recode['CC24_323b'] = cat_to_num(df['CC24_323b'], {'Support': 1, 'Oppose': 0})
recode['CC24_323d'] = cat_to_num(df['CC24_323d'], {'Support': 0, 'Oppose': 1})

# --- GUNS/POLICING
recode['CC24_321b'] = cat_to_num(df['CC24_321b'], {'Support': 1, 'Oppose': 0})
recode['CC24_321c'] = cat_to_num(df['CC24_321c'], {'Support': 0, 'Oppose': 1})
recode['CC24_321d'] = cat_to_num(df['CC24_321d'], {'Support': 1, 'Oppose': 0})
recode['CC24_321e'] = cat_to_num(df['CC24_321e'], {'Support': 0, 'Oppose': 1})

# --- ABORTION
cc325_num = pd.to_numeric(df['CC24_325'].astype(str), errors='coerce')
recode['CC24_325'] = 40.0 - cc325_num
recode['CC24_324b'] = cat_to_num(df['CC24_324b'], {'Support': 1, 'Oppose': 0})
recode['CC24_324c'] = cat_to_num(df['CC24_324c'], {'Support': 1, 'Oppose': 0})
recode['CC24_324d'] = cat_to_num(df['CC24_324d'], {'Support': 0, 'Oppose': 1})

# --- CIVIL RIGHTS/LIBERTIES
recode['CC24_340a'] = cat_to_num(df['CC24_340a'], {'Support': 0, 'Oppose': 1})
recode['CC24_340b'] = cat_to_num(df['CC24_340b'], {'Support': 0, 'Oppose': 1})
recode['CC24_340c'] = cat_to_num(df['CC24_340c'], {'Support': 0, 'Oppose': 1})
recode['CC24_340e'] = cat_to_num(df['CC24_340e'], {'Support': 1, 'Oppose': 0})
recode['CC24_340f'] = cat_to_num(df['CC24_340f'], {'Support': 1, 'Oppose': 0})

# --- RACIAL/GENDER ATTITUDES
_lib_agree = {'Strongly agree': 1, 'Somewhat agree': 2,
              'Neither agree nor disagree': 3,
              'Somewhat disagree': 4, 'Strongly disagree': 5}
_con_agree = {'Strongly agree': 5, 'Somewhat agree': 4,
              'Neither agree nor disagree': 3,
              'Somewhat disagree': 2, 'Strongly disagree': 1}
recode['CC24_440a'] = cat_to_num(df['CC24_440a'], _lib_agree)
recode['CC24_440b'] = cat_to_num(df['CC24_440b'], _con_agree)
recode['CC24_440c'] = cat_to_num(df['CC24_440c'], _con_agree)
recode['CC24_440d'] = cat_to_num(df['CC24_440d'], _con_agree)
recode['CC24_441a'] = cat_to_num(df['CC24_441a'], _con_agree)
recode['CC24_441b'] = cat_to_num(df['CC24_441b'], _lib_agree)

# --- MILITARY INTERVENTIONISM (will be DROPPED; recode to audit)
_mil_map = {'selected': 1, 'not selected': 0}
for i in range(1, 8):
    recode[f'CC24_420_{i}'] = cat_to_num(df[f'CC24_420_{i}'], _mil_map)

# --- ELECTION TRUST
_elec_trust_map = {'Strongly agree': 1, 'Somewhat agree': 2,
                   'Neither agree nor disagree': 3,
                   'Somewhat disagree': 4, 'Strongly disagree': 5}
recode['CC24_421_1'] = cat_to_num(df['CC24_421_1'], _elec_trust_map)
recode['CC24_421_2'] = cat_to_num(df['CC24_421_2'], _elec_trust_map)

# --- GOVERNMENT TRUST
_govt_trust_map = {'A great deal': 1, 'A fair amount': 2,
                   'Not very much': 3, 'None at all': 4}
recode['CC24_423'] = cat_to_num(df['CC24_423'], _govt_trust_map)
recode['CC24_424'] = cat_to_num(df['CC24_424'], _govt_trust_map)

# --- INSTITUTION APPROVAL (CC24_312a/i will be DROPPED; recode to audit)
_appr_map = {'Strongly approve': 4, 'Somewhat approve': 3,
             'Somewhat disapprove': 2, 'Strongly disapprove': 1,
             'Not sure': np.nan}
for s in list('abcdefghi'):
    recode[f'CC24_312{s}'] = cat_to_num(df[f'CC24_312{s}'], _appr_map)

# --- DIAGNOSTIC VARS
recode['pid7'] = cat_to_num(df['pid7'], {
    'Strong Democrat': 1, 'Not very strong Democrat': 2, 'Lean Democrat': 3,
    'Independent': 4, 'Lean Republican': 5, 'Not very strong Republican': 6,
    'Strong Republican': 7, 'Not sure': np.nan
})
recode['ideo5'] = cat_to_num(df['ideo5'], {
    'Very liberal': 1, 'Liberal': 2, 'Moderate': 3,
    'Conservative': 4, 'Very conservative': 5, 'Not sure': np.nan
})

data = pd.DataFrame(recode)
data['commonweight'] = w

# All 55 EFA items from previous run
ALL_55 = [v for v in data.columns if v not in ('pid7', 'ideo5', 'commonweight')]

# =============================================================================
# STEP 1: MISSINGNESS AUDIT
# =============================================================================
print(f'\n{DIVIDER}')
print('STEP 1: MISSINGNESS AUDIT')
print(DIVIDER)
print("""
For each item: raw % missing, and the MARGINAL N gain if that item were
dropped from the listwise deletion set (i.e., cases that are complete on
all other items but missing on this one).
""")

N_TOTAL = len(data)

# Baseline: listwise deletion on all 55 items
complete_55 = data[ALL_55].notna().all(axis=1)
N_baseline = complete_55.sum()
W_baseline = data.loc[complete_55, 'commonweight'].sum()

print(f'  Baseline (all 55 items): N = {N_baseline:,} ({100*N_baseline/N_TOTAL:.1f}%), '
      f'Weighted N = {W_baseline:,.0f}\n')

print(f'  {"Variable":<18} {"% Miss":>7} {"N Miss":>8} {"Marginal N Gain":>16} {"Group"}')
print(f'  {"-"*18} {"-"*7} {"-"*8} {"-"*16} {"-"*30}')

# Group labels for readability
GROUPS = {
    'CC24_330a': 'IDEOLOGY (→DROP)', 'pew_churatd': 'Religion',
    'pew_religimp': 'Religion', 'pew_prayer': 'Religion',
    'pew_bornagain': 'Religion', 'CC24_301': 'Econ Perception (→DROP)',
    'CC24_302': 'Econ Perception', 'CC24_303': 'Econ Perception',
    'CC24_309d_8': 'Econ Perception',
    'CC24_341a': 'Tax/Fiscal', 'CC24_341b': 'Tax/Fiscal',
    'CC24_341c': 'Tax/Fiscal', 'CC24_341d': 'Tax/Fiscal',
    'CC24_323a': 'Immigration', 'CC24_323b': 'Immigration', 'CC24_323d': 'Immigration',
    'CC24_321b': 'Guns/Policing', 'CC24_321c': 'Guns/Policing',
    'CC24_321d': 'Guns/Policing', 'CC24_321e': 'Guns/Policing',
    'CC24_325': 'Abortion', 'CC24_324b': 'Abortion',
    'CC24_324c': 'Abortion', 'CC24_324d': 'Abortion',
    'CC24_340a': 'Civil Rights', 'CC24_340b': 'Civil Rights',
    'CC24_340c': 'Civil Rights', 'CC24_340e': 'Civil Rights', 'CC24_340f': 'Civil Rights',
    'CC24_440a': 'Racial/Gender', 'CC24_440b': 'Racial/Gender',
    'CC24_440c': 'Racial/Gender', 'CC24_440d': 'Racial/Gender',
    'CC24_441a': 'Racial/Gender', 'CC24_441b': 'Racial/Gender',
    'CC24_420_1': 'Military (→DROP)', 'CC24_420_2': 'Military (→DROP)',
    'CC24_420_3': 'Military (→DROP)', 'CC24_420_4': 'Military (→DROP)',
    'CC24_420_5': 'Military (→DROP)', 'CC24_420_6': 'Military (→DROP)',
    'CC24_420_7': 'Military (→DROP)',
    'CC24_421_1': 'Election Trust', 'CC24_421_2': 'Election Trust',
    'CC24_423': 'Govt Trust', 'CC24_424': 'Govt Trust',
    'CC24_312a': 'Approval (→DROP)', 'CC24_312b': 'Approval',
    'CC24_312c': 'Approval', 'CC24_312d': 'Approval',
    'CC24_312e': 'Approval', 'CC24_312f': 'Approval',
    'CC24_312g': 'Approval', 'CC24_312h': 'Approval',
    'CC24_312i': 'Approval (→DROP)',
}

audit_rows = []
for v in ALL_55:
    pct_miss = 100 * data[v].isna().mean()
    n_miss = data[v].isna().sum()
    # Marginal gain: complete on all OTHER items but missing on this one
    others = [x for x in ALL_55 if x != v]
    complete_others = data[others].notna().all(axis=1)
    marginal_gain = (complete_others & data[v].isna()).sum()
    group = GROUPS.get(v, '')
    print(f'  {v:<18} {pct_miss:>6.1f}% {n_miss:>8,} {marginal_gain:>16,}   {group}')
    audit_rows.append({'variable': v, 'pct_missing': pct_miss,
                       'n_missing': n_miss, 'marginal_n_gain': marginal_gain})

audit_df = pd.DataFrame(audit_rows)

# Identify the missingness "strata" by N missing (groups that share a missingness pattern)
print(f'\n  KEY MISSINGNESS STRATA:')
strata = audit_df.groupby('n_missing')['variable'].apply(list).reset_index()
strata = strata.sort_values('n_missing', ascending=False)
for _, row in strata.iterrows():
    if row['n_missing'] > 0:
        vars_list = row['variable']
        pct = 100 * row['n_missing'] / N_TOTAL
        print(f'    N_miss={row["n_missing"]:,} ({pct:.1f}%): {", ".join(vars_list)}')

# =============================================================================
# STEP 2: QUESTION TEXT FOR 421_1, 421_2, 423, 424
# =============================================================================
print(f'\n{DIVIDER}')
print('STEP 2: QUESTION TEXT FOR CC24_421_1, CC24_421_2, CC24_423, CC24_424')
print(DIVIDER)
print("""
CC24_421 — ELECTION FAIRNESS GRID
  Stem: "Do you agree or disagree with the following statements?"
  Scale: 1=Strongly agree, 2=Somewhat agree, 3=Neither agree nor disagree,
         4=Somewhat disagree, 5=Strongly disagree
  [Pipeline codes these AS-IS: higher = more distrust/disagreement]

  CC24_421_1: "Elections in the U.S. are fair"
    → Strongly agree (trust) = 1; Strongly disagree (distrust) = 5
    → In 2024 context: election distrust is strongly associated with
      Trump/MAGA base; this item may overlap with partisan identity
      but also captures a genuinely cross-cutting anti-institutionalism.
    → Distribution: PID R² = 0.018 — VERY LOW partisan signal. Cross-partisan.

  CC24_421_2: "Your state or local government conducted a fair and accurate
               election in 2024"
    → Same scale; same direction.
    → Distribution note: ~80% Agree/Strongly agree (state elections seen as
      fairer than federal). Very limited variance; communality likely low.
    → PID R² = 0.012 — near-zero partisan signal.

CC24_423 — FEDERAL GOVERNMENT TRUST
  Stem: "How much trust do you have in the federal government in Washington
         when it comes to handling the nation's problems?"
  Scale: 1=A great deal, 2=A fair amount, 3=Not very much, 4=None at all
  [Higher = less trust]

  CC24_423: Federal government trust
    → PID R² = 0.027 — very low. Cross-partisan item (both populist left
      and MAGA right distrust Washington; libertarians, too).
    → Note: partisan asymmetry in 2024 as Biden was in office, but item
      is structurally about institutional trust, not candidate evaluation.

CC24_424 — STATE GOVERNMENT TRUST
  Stem: "How much trust do you have in the government of the state where
         you live when it comes to handling the state's problems?"
  Scale: Same 4-point scale (1=A great deal → 4=None at all)
  [Higher = less trust]

  CC24_424: State government trust
    → PID R² = 0.007 — essentially zero. Strongly cross-partisan.
    → Note: direction will vary by respondent's state party alignment.
      A Democrat in Texas and a Republican in California both distrust
      their state government, but for partisan rather than systemic reasons.
      EFA loadings will reveal whether this clusters with federal trust.
""")

# =============================================================================
# STEP 3: APPLY DROPS AND COMPUTE NEW LISTWISE N
# =============================================================================
print(f'\n{DIVIDER}')
print('STEP 3: APPLY DROPS AND COMPUTE NEW LISTWISE N')
print(DIVIDER)

DROPS = {
    'CC24_420_1': 'Compositional/multi-select data; mutually exclusive checkboxes create structural correlations invalid for EFA',
    'CC24_420_2': 'Compositional/multi-select data; mutually exclusive checkboxes create structural correlations invalid for EFA',
    'CC24_420_3': 'Compositional/multi-select data; mutually exclusive checkboxes create structural correlations invalid for EFA',
    'CC24_420_4': 'Compositional/multi-select data; mutually exclusive checkboxes create structural correlations invalid for EFA',
    'CC24_420_5': 'Compositional/multi-select data; mutually exclusive checkboxes create structural correlations invalid for EFA',
    'CC24_420_6': 'Compositional/multi-select data; mutually exclusive checkboxes create structural correlations invalid for EFA',
    'CC24_420_7': 'Compositional/multi-select data; mutually exclusive exclusive-none creates near-perfect artificial negative correlations',
    'CC24_312a': 'Partisan proxy: Biden approval (PID R²=0.568); near-pure party signal; r=0.977 with CC24_312i',
    'CC24_312i': 'Partisan proxy: Harris approval (PID R²=0.630); near-pure party signal; r=0.977 with CC24_312a',
    'CC24_330a': 'Partisan proxy: ideology self-placement (PID R²=0.568); will dominate Factor 1 and suppress cross-cutting structure',
    'CC24_301':  'Partisan proxy: national economy retrospective (PID R²=0.338); functions as Biden approval proxy in 2024',
}

EFA_VARS_V3 = [v for v in ALL_55 if v not in DROPS]

print(f'  Dropped {len(DROPS)} items:')
for v, reason in DROPS.items():
    print(f'    {v}: {reason[:80]}')

print(f'\n  Remaining items: {len(EFA_VARS_V3)}')
print(f'  {EFA_VARS_V3}')

# Compute new listwise N
complete_v3 = data[EFA_VARS_V3].notna().all(axis=1)
N_v3 = complete_v3.sum()
W_v3 = data.loc[complete_v3, 'commonweight'].sum()

print(f'\n  Listwise deletion results after v3 drops:')
print(f'    N retained:   {N_v3:,} ({100*N_v3/N_TOTAL:.1f}% of 60,000)')
print(f'    N excluded:   {N_TOTAL - N_v3:,}')
print(f'    Weighted N:   {W_v3:,.0f}')
print(f'    vs. baseline: {N_v3 - N_baseline:+,} cases ({100*(N_v3-N_baseline)/N_TOTAL:+.1f}pp)')

# Diagnose what's still driving missingness in v3 set
print(f'\n  PRIMARY MISSINGNESS DRIVERS (v3 item set):')
miss_v3 = data[EFA_VARS_V3].isna().sum().sort_values(ascending=False)
for v, n in miss_v3[miss_v3 > 0].items():
    print(f'    {v:<18}: {n:,} missing ({100*n/N_TOTAL:.1f}%)')

# =============================================================================
# STEP 4: BUILD AND SAVE efa_variable_list_v3.csv
# =============================================================================
print(f'\n{DIVIDER}')
print('STEP 4: SAVE efa_variable_list_v3.csv')
print(DIVIDER)

# Load the existing v2 variable list to inherit labels/flags
import os
v2_path = os.path.join(OUTPUT_DIR, 'efa_variable_list.csv')
v2 = pd.read_csv(v2_path)

# Add drop status
v2['v3_action'] = v2['variable_name'].apply(
    lambda x: 'DROP' if x in DROPS else v2.loc[v2['variable_name'] == x, 'recommended_action'].values[0]
    if x in v2['variable_name'].values else 'KEEP'
)
v2['v3_drop_reason'] = v2['variable_name'].apply(
    lambda x: DROPS.get(x, '')
)

# Reorder: show drops first, then remaining
v2['sort_key'] = v2['variable_name'].apply(lambda x: 0 if x in DROPS else 1)
v2 = v2.sort_values(['sort_key', 'variable_name']).drop(columns='sort_key')

# Update recommended_action for dropped items
v2.loc[v2['variable_name'].isin(DROPS), 'recommended_action'] = 'DROP'
v2.loc[v2['variable_name'].isin(DROPS), 'reason'] = v2['variable_name'].apply(
    lambda x: DROPS.get(x, '')
)

out_path = os.path.join(OUTPUT_DIR, 'efa_variable_list_v3.csv')
v2.to_csv(out_path, index=False)
print(f'  Saved: {out_path}')
print(f'  Rows: {len(v2)} (all 55 items; drops flagged)')

# =============================================================================
# SUMMARY
# =============================================================================
print(f'\n{DIVIDER}')
print('SUMMARY')
print(DIVIDER)
print(f"""
  Original item pool (v2):  55 items, N = {N_baseline:,}, Weighted N = {W_baseline:,.0f}
  Items dropped (v3):       {len(DROPS)} items
  Remaining items (v3):     {len(EFA_VARS_V3)} items
  New N (listwise):         {N_v3:,} ({100*N_v3/N_TOTAL:.1f}% of 60,000)
  New Weighted N:           {W_v3:,.0f}
  Net gain from drops:      {N_v3 - N_baseline:+,} cases

  ITEMS STILL FLAGGED FOR YOUR REVIEW (awaiting your instruction):
    CC24_421_1  — Election trust: cross-partisan (PID R²=0.018); recommend KEEP
    CC24_421_2  — State election trust: near-zero PID R²=0.012; very skewed toward
                  trust; low variance; recommend REVIEW before EFA
    CC24_423    — Federal govt trust: cross-partisan (PID R²=0.027); recommend KEEP
    CC24_424    — State govt trust: near-zero PID R²=0.007; direction varies by state
                  party alignment; recommend REVIEW before EFA

  Saved: {out_path}

  AWAITING INSTRUCTION: Decide on CC24_421_2 and CC24_424 (keep or drop),
  then proceed to polychoric recomputation + EFA.
""")
print(DIVIDER)
