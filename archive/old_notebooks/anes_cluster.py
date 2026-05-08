#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ANES 2024 Cluster Analysis
#
# Mirrors the CES clustering methodology using ANES 2024 Time Series data.
# - **Clusters on ~48 domestic policy preference variables** (NOT demographics or ideology)
# - **Profiles clusters** with demographics, ideology, partisanship, and cultural attitudes
# - Uses `V240108b` (post-election, full sample combined) as the weight variable

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Data Loading & Filtering

# %%
# Load ANES 2024 Time Series data
# Note: ANES CSV uses spaces for missing values and some decimals start with '.'
# We treat whitespace-only strings as NaN on load
anes_raw = pd.read_csv('anes data/anes_timeseries_2024_csv_20250808.csv',
                        low_memory=False, na_values=[' ', '  ', ''])
print(f"Raw dataset: {len(anes_raw):,} respondents, {anes_raw.shape[1]:,} columns")

# Convert all columns to numeric where possible (handles '.78' -> 0.78 etc.)
for col in anes_raw.columns:
    if anes_raw[col].dtype == object:
        anes_raw[col] = pd.to_numeric(anes_raw[col], errors='coerce')

print(f"After numeric conversion: {anes_raw.select_dtypes(include='number').shape[1]:,} numeric columns")

# Filter to post-election respondents only (completed both pre and post)
# V240002c: 1 = pre only, 2 = completed both pre and post
anes = anes_raw[anes_raw['V240002c'] == 2].copy()
print(f"Post-election respondents: {len(anes):,}")

# Weight variable: V240108b (post-election, full sample combined)
WEIGHT_COL = 'V240108b'
print(f"Weight variable: {WEIGHT_COL} (dtype: {anes[WEIGHT_COL].dtype})")
print(f"  Valid weights: {anes[WEIGHT_COL].gt(0).sum():,}")
print(f"  Missing weights: {anes[WEIGHT_COL].isna().sum():,}")
print(f"  Mean weight: {anes.loc[anes[WEIGHT_COL] > 0, WEIGHT_COL].mean():.4f}")

# %% [markdown]
# ## 2. Feature Definitions
#
# ### Clustering Features (~48 policy preference variables)
# Used to assign cluster membership. All are domestic/foreign policy preferences -- NO demographics or ideology.
#
# ### Descriptor Features (NOT used for clustering)
# Ideology, partisanship, demographics, and cultural attitudes -- included in profiles only.

# %%
# ============================================================
# CLUSTERING FEATURES -- used to assign cluster membership
# ============================================================

clustering_features = {
    # --- 7-Point Policy Scales (Pre-Election) ---
    'V241239': 'Govt spending & services (1=fewer..7=more)',
    'V241242': 'Defense spending (1=decrease..7=increase)',
    'V241245': 'Govt vs private health insurance (1=govt..7=private)',
    'V241248': 'Abortion permissibility (1=always..7=never)',
    'V241252': 'Guaranteed job & income (1=govt..7=on own)',
    'V241255': 'Govt assistance to Blacks (1=govt help..7=self-help)',
    'V241258': 'Environment vs business (1=regulate..7=too much)',
    'V241397': 'Urban unrest (1=solve racism..7=use force)',

    # --- Federal Budget Spending (Pre-Election, summary vars) ---
    'V241263x': 'Spending: Social Security',
    'V241266x': 'Spending: Public schools',
    'V241269x': 'Spending: Border security',
    'V241272x': 'Spending: Dealing with crime',
    'V241275x': 'Spending: Welfare programs',
    'V241278x': 'Spending: Highways',
    'V241281x': 'Spending: Aid to the poor',
    'V241284x': 'Spending: Environmental protection',

    # --- Pre-Election Issue Positions ---
    'V241302':  'Abortion categorical (1=never..4=always choice)',
    'V241308x': 'Death penalty (1=favor strongly..4=oppose strongly)',
    'V241313':  'Willingness to use military force (1=extremely..5=not at all)',
    'V241319x': 'Require photo ID to vote (1=favor..7=oppose)',
    'V241330x': 'Strong president vs checks & balances (1=helpful..7=harmful)',
    'V241366x': 'Govt action on climate (1=do more..7=do less)',
    'V241369x': 'Mandate paid parental leave (1=favor..7=oppose)',
    'V241372x': 'Transgender bathroom policy (1=favor..7=oppose)',
    'V241375x': 'Ban trans girls from sports (1=favor..7=oppose)',
    'V241378x': 'Protect gays from job discrimination (1=favor..4=oppose)',
    'V241386':  'Unauthorized immigrant policy (1=deport..4=citizenship)',
    'V241389x': 'End birthright citizenship (1=favor..7=oppose)',
    'V241395x': 'Build border wall (1=favor..7=oppose)',
    'V241400x': 'Weapons to Ukraine (1=favor..7=oppose)',
    'V241290x': 'DEI policies on campus (1=favor..7=oppose)',

    # --- Post-Election Issue Positions ---
    'V242227':  'Immigration levels (1=increase a lot..5=decrease a lot)',
    'V242234x': 'Path to citizenship (1=favor..7=oppose)',
    'V242235':  'Immigrants good/bad for economy (1=good..7=bad)',
    'V242241x': 'Preferential hiring of Blacks (1=for..4=against)',
    'V242245x': 'Affirmative action in universities (1=favor..7=oppose)',
    'V242248x': 'Less vs more government (summary)',
    'V242249':  'Govt regulation of business (1=more..7=less)',
    'V242253x': 'Reduce income inequality (1=favor..7=oppose)',
    'V242319x': 'Require vaccines in schools (1=favor..7=oppose)',
    'V242324x': 'Regulate greenhouse emissions (1=favor..7=oppose)',
    'V242325':  'Gun purchase difficulty (1=harder/2=easier/3=same)',
    'V242328x': 'Background checks at gun shows (1=favor..7=oppose)',
    'V242331x': 'Ban assault-style rifles (1=favor..7=oppose)',
    'V242346x': 'Free trade agreements (1=favor..7=oppose)',
    'V242350':  'Minimum wage (1=raise/2=keep/3=lower/4=eliminate)',
    'V242353x': 'Govt spending on health insurance (1=increase..7=decrease)',
    'V242335x': 'Govt action on opioid addiction (1=more..7=less)',
}

clustering_questions = list(clustering_features.keys())
print(f"Total clustering features: {len(clustering_questions)}")

# ============================================================
# DESCRIPTOR FEATURES -- NOT used for clustering, profiling only
# ============================================================

# Ideology & Partisanship (used for ordering and profiling)
IDEO_COL = 'V241177'    # Liberal-conservative self-placement (1=extremely liberal..7=extremely conservative, 99=DK)
PID_COL = 'V241227x'    # Party ID summary (1=strong Dem..7=strong Rep)

# Demographics -- VERIFIED against ANES 2024 codebook
demo_cols = {
    'V241458x': 'Age on election day (18-80)',
    'V241550':  'Sex (1=Male, 2=Female)',
    'V241551':  'Gender identity (1=Man, 2=Woman, 3=Nonbinary, 4=Other)',
    'V241501x': 'Race/ethnicity (1=White NH, 2=Black NH, 3=Hispanic, 4=Asian/PI NH, 5=NativeAm NH, 6=Multi NH)',
    'V241465x': 'Education 5-cat (1=<HS, 2=HS, 3=Some college, 4=BA, 5=Grad)',
    'V241567x': 'Household income 6-cat (1=<$10k, 2=$10-30k, 3=$30-60k, 4=$60-100k, 5=$100-250k, 6=$250k+)',
    'V241461x': 'Marital status (1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never married)',
    'V241445x': 'Religion major group (1=MainProt, 2=EvanProt, 3=BlackProt, 4=Catholic, 5=UndifffChristian, 6=Jewish, 7=OtherRelig, 8=NotRelig)',
    'V241442':  'Born-again Christian (1=Yes, 2=No)',
    'V241440':  'Church attendance (1=Every week..5=Never)',
    'V241420':  'Religion importance (1=Extremely..5=Not at all)',
    'V241488x': 'Employment (1=Working, 2=Laid off, 4=Unemployed, 5=Retired, 6=Disabled, 7=Homemaker, 8=Student)',
    'V243007':  'Census region (1=NE, 2=MW, 3=South, 4=West)',
}

# Cultural attitude descriptors (NOT clustering, profiling only)
cultural_descriptors = {
    'V242300': 'Racial resent: Blacks work way up w/o favors',
    'V242301': 'Racial resent: Slavery created difficult conditions',
    'V242302': 'Racial resent: Blacks gotten less than deserve',
    'V242303': 'Racial resent: If tried harder, be as well off',
    'V242254': 'Egalitarianism: Equal opportunity',
    'V242255': 'Egalitarianism: Worry less about equality',
    'V242256': 'Egalitarianism: Not big problem if some have more',
    'V242257': 'Egalitarianism: Treat more equally = fewer problems',
    'V242279x': 'Gender roles (man works/woman home)',
    'V242361x': 'Sexual harassment attention',
}

# 2024 Vote choice & turnout (for profiling)
VOTE_COL = 'V242067'    # Post-election presidential vote choice (1=Harris, 2=Trump, 4=West, 5=Stein, 6=Other)
TURNOUT_COL = 'V242065'  # Did R vote (4=Sure I voted)

all_descriptor_cols = [IDEO_COL, PID_COL, VOTE_COL, TURNOUT_COL] + list(demo_cols.keys()) + list(cultural_descriptors.keys())
print(f"Total descriptor columns: {len(all_descriptor_cols)}")

# %%
# ============================================================
# Recode missing values: ANES uses negative values for missing
# -1 = Inapplicable, -2 = Refused, -3 = Restricted/Missing,
# -4/-5/-6/-7/-8/-9 = various other missing codes
# ALSO: 99 = "Haven't thought much about this" on 7-point scales
# ============================================================

all_analysis_cols = clustering_questions + all_descriptor_cols + [WEIGHT_COL]
existing_cols = [c for c in all_analysis_cols if c in anes.columns]
missing_cols = [c for c in all_analysis_cols if c not in anes.columns]

if missing_cols:
    print(f"WARNING: {len(missing_cols)} columns not found in data:")
    for c in missing_cols:
        label = clustering_features.get(c, demo_cols.get(c, cultural_descriptors.get(c, c)))
        print(f"  {c}: {label}")

# Recode negative values to NaN for all analysis columns (except weight)
recode_cols = [c for c in clustering_questions + all_descriptor_cols if c in anes.columns]
for col in recode_cols:
    anes.loc[anes[col] < 0, col] = np.nan

# Recode 99 = "DK / Haven't thought about it" to NaN
# This affects the 8 pre-election 7-point scales AND ideology
seven_pt_scales_with_99 = [
    'V241239', 'V241242', 'V241245', 'V241248',
    'V241252', 'V241255', 'V241258', 'V241397',
    IDEO_COL,  # V241177 also has 99=DK
]
for col in seven_pt_scales_with_99:
    if col in anes.columns:
        n99 = (anes[col] == 99).sum()
        if n99 > 0:
            anes.loc[anes[col] == 99, col] = np.nan
            print(f"  Recoded {n99} '99' values to NaN in {col}")

# Update clustering questions to only those that exist
clustering_questions = [c for c in clustering_questions if c in anes.columns]
print(f"\nClustering features available: {len(clustering_questions)}")

# Check completeness: how many clustering features does each respondent have?
valid_counts = anes[clustering_questions].notna().sum(axis=1)
print(f"\nCompleteness distribution:")
print(f"  Min valid features: {valid_counts.min()}")
print(f"  Median valid features: {valid_counts.median():.0f}")
print(f"  Mean valid features: {valid_counts.mean():.1f}")
print(f"  Respondents with >= 30 features: {(valid_counts >= 30).sum():,}")
print(f"  Respondents with >= 35 features: {(valid_counts >= 35).sum():,}")
print(f"  Respondents with >= 40 features: {(valid_counts >= 40).sum():,}")

# Filter: require at least 30 of clustering features to be valid
MIN_VALID = 30
anes = anes[valid_counts >= MIN_VALID].copy()
print(f"\nAfter completeness filter (>= {MIN_VALID} features): {len(anes):,}")

# Also require valid weight
anes = anes[anes[WEIGHT_COL] > 0].copy()
print(f"After weight filter: {len(anes):,}")

# %% [markdown]
# ## 3. Clustering Pipeline
#
# 1. **Imputation**: `SimpleImputer(strategy='median')` fills remaining NaN values
# 2. **Scaling**: `StandardScaler()` normalizes to zero mean, unit variance
# 3. **PCA**: Reduce dimensionality (target ~65-70% variance explained)
# 4. **K-Means**: Test k=4-8, select based on silhouette scores
# 5. **Ordering**: Sort clusters left-to-right by mean ideology

# %%
# ============================================================
# Step 1: Imputation -- fill remaining NaN with median
# ============================================================

imputer = SimpleImputer(strategy='median')
anes_imputed = anes.copy()

# Impute clustering features
anes_imputed[clustering_questions] = imputer.fit_transform(anes[clustering_questions])
print(f"Clustering features imputed: {len(clustering_questions)}")

# Impute descriptor columns individually (some may be all-NaN or missing)
desc_cols_existing = [c for c in all_descriptor_cols if c in anes.columns]
imputed_count = 0
for col in desc_cols_existing:
    if anes[col].notna().sum() > 0:
        median_val = anes[col].median()
        anes_imputed[col] = anes[col].fillna(median_val)
        imputed_count += 1
    else:
        print(f"  WARNING: {col} is all NaN, skipping imputation")

print(f"Descriptor columns imputed: {imputed_count} of {len(desc_cols_existing)}")
print(f"Imputation complete. Shape: {anes_imputed.shape}")
print(f"NaN remaining in clustering features: {anes_imputed[clustering_questions].isna().sum().sum()}")

# %%
# ============================================================
# Step 2: Scaling
# ============================================================

X = anes_imputed[clustering_questions].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled feature matrix: {X_scaled.shape}")

# ============================================================
# Step 3: PCA -- determine number of components
# ============================================================

# Fit full PCA first to see variance explained
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

# Find n_components for various thresholds
for thresh in [60, 65, 70, 75, 80]:
    n = np.argmax(cumvar >= thresh) + 1
    print(f"  {thresh}% variance explained at {n} components")

# Plot variance explained
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(1, len(cumvar) + 1)),
    y=cumvar,
    mode='lines+markers',
    name='Cumulative Variance',
    marker=dict(size=4),
))
fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='70% threshold')
fig.update_layout(
    title='PCA: Cumulative Variance Explained',
    xaxis_title='Number of Components',
    yaxis_title='Cumulative Variance Explained (%)',
    height=400,
    width=700,
)
fig.show()

# Select n_components targeting ~67-70% variance
N_COMPONENTS = np.argmax(cumvar >= 67) + 1
print(f"\nSelected {N_COMPONENTS} PCA components ({cumvar[N_COMPONENTS-1]:.1f}% variance explained)")

# %%
# ============================================================
# Step 4: Apply PCA with selected components
# ============================================================

pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA-transformed data: {X_pca.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ============================================================
# Step 5: Silhouette analysis for k=4 through k=8
# ============================================================

print("\nSilhouette scores by k:")
sil_scores = {}
for k in range(4, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores[k] = score
    print(f"  k={k}: {score:.4f}")

# Plot silhouette scores
fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(sil_scores.keys()),
    y=list(sil_scores.values()),
    text=[f"{s:.4f}" for s in sil_scores.values()],
    textposition='outside',
))
fig.update_layout(
    title='Silhouette Score by Number of Clusters',
    xaxis_title='k (Number of Clusters)',
    yaxis_title='Silhouette Score',
    height=400,
    width=600,
)
fig.show()

# %%
# ============================================================
# Step 6: Final K-Means clustering (k=6 to mirror CES)
# ============================================================

optimal_k = 6
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
labels_raw = kmeans_final.fit_predict(X_pca)
anes_imputed['cluster_raw'] = labels_raw

# ============================================================
# Step 7: Order clusters left-to-right by mean ideology
# V241177: 1=extremely liberal .. 7=extremely conservative
# ============================================================

# Calculate mean ideology for each raw cluster
cluster_ideo = anes_imputed.groupby('cluster_raw')[IDEO_COL].mean().sort_values()
print("Raw cluster mean ideology (V241177, 1=lib..7=con):")
for cid, mean_ideo in cluster_ideo.items():
    n = (anes_imputed['cluster_raw'] == cid).sum()
    print(f"  Cluster {cid}: mean ideo = {mean_ideo:.2f}, n = {n:,}")

# Create mapping from old IDs to new IDs (sorted by ideology)
old_to_new = {old_id: new_id for new_id, old_id in enumerate(cluster_ideo.index)}
anes_imputed['cluster'] = anes_imputed['cluster_raw'].map(old_to_new)

print(f"\nReordered clusters (0=most liberal, {optimal_k-1}=most conservative):")
for new_id in range(optimal_k):
    cluster_data = anes_imputed[anes_imputed['cluster'] == new_id]
    n = len(cluster_data)
    mean_ideo = cluster_data[IDEO_COL].mean()
    total_w = cluster_data[WEIGHT_COL].sum()
    pct = total_w / anes_imputed[WEIGHT_COL].sum() * 100
    print(f"  Cluster {new_id}: n={n:,}, mean ideo={mean_ideo:.2f}, weighted %={pct:.1f}%")

# %% [markdown]
# ## 4. Cluster Profiling -- Demographics
#
# Weighted demographic breakdowns for each cluster. These variables were **NOT** used for clustering.
#

# %%
# ============================================================
# Weighted demographic profiles for each cluster
# Using VERIFIED ANES 2024 codebook variable names
# ============================================================

# Placeholder cluster names (will be updated after reviewing profiles)
cluster_names = {i: f"Cluster {i}" for i in range(optimal_k)}
cluster_order = [cluster_names[i] for i in range(optimal_k)]

# Map cluster names
anes_imputed['cluster_name'] = anes_imputed['cluster'].map(cluster_names)

def wpct(data, mask, weight_col=WEIGHT_COL):
    """Weighted percentage for a boolean mask."""
    total_w = data[weight_col].sum()
    if total_w == 0:
        return 0.0
    return (data.loc[mask, weight_col].sum() / total_w) * 100

def wmean(data, col, weight_col=WEIGHT_COL):
    """Weighted mean of a column."""
    valid = data[col].notna() & data[weight_col].notna() & (data[col] > 0)
    d = data[valid]
    if len(d) == 0 or d[weight_col].sum() == 0:
        return np.nan
    return np.average(d[col], weights=d[weight_col])

# Print detailed profiles for each cluster
for cluster_id in range(optimal_k):
    cd = anes_imputed[anes_imputed['cluster'] == cluster_id]
    tw = cd[WEIGHT_COL].sum()
    all_tw = anes_imputed[WEIGHT_COL].sum()

    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id} -- n={len(cd):,} ({tw/all_tw*100:.1f}% of weighted sample)")
    print(f"{'='*70}")

    # --- Ideology & Partisanship ---
    print(f"\n  IDEOLOGY (V241177: 1=ext liberal..7=ext conservative)")
    ideo_mean = wmean(cd, IDEO_COL)
    print(f"    Mean: {ideo_mean:.2f}")
    for val, label in [(1,'Extremely Liberal'), (2,'Liberal'), (3,'Slightly Liberal'),
                       (4,'Moderate'), (5,'Slightly Conservative'), (6,'Conservative'),
                       (7,'Extremely Conservative')]:
        pct = wpct(cd, cd[IDEO_COL] == val)
        print(f"    {label}: {pct:.1f}%")

    print(f"\n  PARTY ID (V241227x: 1=Strong Dem..7=Strong Rep)")
    print(f"    Mean: {wmean(cd, PID_COL):.2f}")
    for val, label in [(1,'Strong Democrat'), (2,'Not very strong Dem'), (3,'Lean Democrat'),
                       (4,'Independent'), (5,'Lean Republican'), (6,'Not very strong Rep'),
                       (7,'Strong Republican')]:
        pct = wpct(cd, cd[PID_COL] == val)
        print(f"    {label}: {pct:.1f}%")

    # --- Age (V241458x: actual age 18-80) ---
    print(f"\n  AGE (V241458x)")
    if 'V241458x' in cd.columns:
        mean_age = wmean(cd, 'V241458x')
        print(f"    Mean age: {mean_age:.1f}")
        print(f"    18-29: {wpct(cd, cd['V241458x'].between(18, 29)):.1f}%")
        print(f"    30-44: {wpct(cd, cd['V241458x'].between(30, 44)):.1f}%")
        print(f"    45-59: {wpct(cd, cd['V241458x'].between(45, 59)):.1f}%")
        print(f"    60+:   {wpct(cd, cd['V241458x'] >= 60):.1f}%")

    # --- Sex (V241550: 1=Male, 2=Female) ---
    print(f"\n  SEX (V241550)")
    if 'V241550' in cd.columns:
        print(f"    Male:   {wpct(cd, cd['V241550'] == 1):.1f}%")
        print(f"    Female: {wpct(cd, cd['V241550'] == 2):.1f}%")

    # --- Race/Ethnicity (V241501x) ---
    print(f"\n  RACE/ETHNICITY (V241501x)")
    if 'V241501x' in cd.columns:
        for val, label in [(1,'White non-Hispanic'), (2,'Black non-Hispanic'),
                           (3,'Hispanic'), (4,'Asian/PI non-Hispanic'),
                           (5,'Native Am/other non-Hispanic'), (6,'Multiple races non-Hispanic')]:
            pct = wpct(cd, cd['V241501x'] == val)
            if pct > 0.5:
                print(f"    {label}: {pct:.1f}%")

    # --- Education (V241465x: 1=<HS, 2=HS, 3=Some college, 4=BA, 5=Grad) ---
    print(f"\n  EDUCATION (V241465x)")
    if 'V241465x' in cd.columns:
        for val, label in [(1,'Less than HS'), (2,'HS graduate'), (3,'Some post-HS, no BA'),
                           (4,"Bachelor's degree"), (5,'Graduate degree')]:
            pct = wpct(cd, cd['V241465x'] == val)
            print(f"    {label}: {pct:.1f}%")

    # --- Income (V241567x: 6-cat) ---
    print(f"\n  HOUSEHOLD INCOME (V241567x)")
    if 'V241567x' in cd.columns:
        for val, label in [(1,'Under $10k'), (2,'$10-30k'), (3,'$30-60k'),
                           (4,'$60-100k'), (5,'$100-250k'), (6,'$250k+')]:
            pct = wpct(cd, cd['V241567x'] == val)
            if pct > 0:
                print(f"    {label}: {pct:.1f}%")

    # --- Religion ---
    print(f"\n  RELIGION")
    if 'V241445x' in cd.columns:
        for val, label in [(1,'Mainline Protestant'), (2,'Evangelical Protestant'),
                           (3,'Black Protestant'), (4,'Catholic'),
                           (5,'Undiff. Christian'), (6,'Jewish'),
                           (7,'Other religion'), (8,'Not religious')]:
            pct = wpct(cd, cd['V241445x'] == val)
            if pct > 1:
                print(f"    {label}: {pct:.1f}%")
    if 'V241442' in cd.columns:
        print(f"    Born-again Christian: {wpct(cd, cd['V241442'] == 1):.1f}%")
    if 'V241420' in cd.columns:
        print(f"    Religion extremely important: {wpct(cd, cd['V241420'] == 1):.1f}%")
        print(f"    Religion not at all important: {wpct(cd, cd['V241420'] == 5):.1f}%")
    if 'V241440' in cd.columns:
        print(f"    Attend church weekly: {wpct(cd, cd['V241440'] == 1):.1f}%")
        print(f"    Never attend: {wpct(cd, cd['V241440'] == 5):.1f}%")

    # --- Marital Status (V241461x) ---
    print(f"\n  MARITAL STATUS (V241461x)")
    if 'V241461x' in cd.columns:
        for val, label in [(1,'Married'), (2,'Widowed'), (3,'Divorced'),
                           (4,'Separated'), (5,'Never married')]:
            pct = wpct(cd, cd['V241461x'] == val)
            if pct > 1:
                print(f"    {label}: {pct:.1f}%")

    # --- Employment (V241488x) ---
    print(f"\n  EMPLOYMENT (V241488x)")
    if 'V241488x' in cd.columns:
        for val, label in [(1,'Working now'), (2,'Temp laid off'), (4,'Unemployed'),
                           (5,'Retired'), (6,'Disabled'), (7,'Homemaker'), (8,'Student')]:
            pct = wpct(cd, cd['V241488x'] == val)
            if pct > 0.5:
                print(f"    {label}: {pct:.1f}%")

    # --- Region (V243007) ---
    print(f"\n  REGION (V243007)")
    if 'V243007' in cd.columns:
        for val, label in [(1,'Northeast'), (2,'Midwest'), (3,'South'), (4,'West')]:
            pct = wpct(cd, cd['V243007'] == val)
            print(f"    {label}: {pct:.1f}%")

    # --- 2024 Vote (V242067: 1=Harris, 2=Trump) ---
    print(f"\n  2024 PRESIDENTIAL VOTE (V242067)")
    if VOTE_COL in cd.columns:
        voters = cd[cd[TURNOUT_COL] == 4]  # Only those who voted
        voters_tw = voters[WEIGHT_COL].sum()
        if voters_tw > 0:
            harris_pct = (voters.loc[voters[VOTE_COL] == 1, WEIGHT_COL].sum() / voters_tw) * 100
            trump_pct = (voters.loc[voters[VOTE_COL] == 2, WEIGHT_COL].sum() / voters_tw) * 100
            other_pct = (voters.loc[voters[VOTE_COL].isin([4,5,6]), WEIGHT_COL].sum() / voters_tw) * 100
            print(f"    Harris: {harris_pct:.1f}%")
            print(f"    Trump:  {trump_pct:.1f}%")
            if other_pct > 0.5:
                print(f"    Other:  {other_pct:.1f}%")
            turnout_pct = wpct(cd, cd[TURNOUT_COL] == 4)
            print(f"    Turnout: {turnout_pct:.1f}%")

# %%
# ============================================================
# Plotly Demographic Heatmap -- VERIFIED variable names
# ============================================================

demographics = {
    'SIZE': [
        ('Weighted %', lambda d, tw: tw / anes_imputed[WEIGHT_COL].sum() * 100),
        ('N (unweighted)', lambda d, tw: len(d)),
    ],
    'AGE': [
        ('Mean Age', lambda d, tw: wmean(d, 'V241458x') if 'V241458x' in d.columns else np.nan),
        ('18-29', lambda d, tw: wpct(d, d['V241458x'].between(18, 29)) if 'V241458x' in d.columns else np.nan),
        ('30-44', lambda d, tw: wpct(d, d['V241458x'].between(30, 44)) if 'V241458x' in d.columns else np.nan),
        ('45-59', lambda d, tw: wpct(d, d['V241458x'].between(45, 59)) if 'V241458x' in d.columns else np.nan),
        ('60+', lambda d, tw: wpct(d, d['V241458x'] >= 60) if 'V241458x' in d.columns else np.nan),
    ],
    'SEX': [
        ('Male', lambda d, tw: wpct(d, d['V241550'] == 1) if 'V241550' in d.columns else np.nan),
        ('Female', lambda d, tw: wpct(d, d['V241550'] == 2) if 'V241550' in d.columns else np.nan),
    ],
    'RACE': [
        ('White NH', lambda d, tw: wpct(d, d['V241501x'] == 1) if 'V241501x' in d.columns else np.nan),
        ('Black NH', lambda d, tw: wpct(d, d['V241501x'] == 2) if 'V241501x' in d.columns else np.nan),
        ('Hispanic', lambda d, tw: wpct(d, d['V241501x'] == 3) if 'V241501x' in d.columns else np.nan),
        ('Asian/PI NH', lambda d, tw: wpct(d, d['V241501x'] == 4) if 'V241501x' in d.columns else np.nan),
    ],
    'EDUCATION': [
        ('<HS + HS', lambda d, tw: wpct(d, d['V241465x'].isin([1,2])) if 'V241465x' in d.columns else np.nan),
        ('Some college', lambda d, tw: wpct(d, d['V241465x'] == 3) if 'V241465x' in d.columns else np.nan),
        ('BA+', lambda d, tw: wpct(d, d['V241465x'].isin([4,5])) if 'V241465x' in d.columns else np.nan),
    ],
    'INCOME': [
        ('<$30k', lambda d, tw: wpct(d, d['V241567x'].isin([1,2])) if 'V241567x' in d.columns else np.nan),
        ('$30-100k', lambda d, tw: wpct(d, d['V241567x'].isin([3,4])) if 'V241567x' in d.columns else np.nan),
        ('$100k+', lambda d, tw: wpct(d, d['V241567x'].isin([5,6])) if 'V241567x' in d.columns else np.nan),
    ],
    'PARTY ID': [
        ('Democrat (inc lean)', lambda d, tw: wpct(d, d[PID_COL].isin([1,2,3])) if PID_COL in d.columns else np.nan),
        ('Independent', lambda d, tw: wpct(d, d[PID_COL] == 4) if PID_COL in d.columns else np.nan),
        ('Republican (inc lean)', lambda d, tw: wpct(d, d[PID_COL].isin([5,6,7])) if PID_COL in d.columns else np.nan),
    ],
    'IDEOLOGY': [
        ('Mean Ideology', lambda d, tw: wmean(d, IDEO_COL)),
        ('Liberal (1-3)', lambda d, tw: wpct(d, d[IDEO_COL].isin([1,2,3])) if IDEO_COL in d.columns else np.nan),
        ('Moderate (4)', lambda d, tw: wpct(d, d[IDEO_COL] == 4) if IDEO_COL in d.columns else np.nan),
        ('Conservative (5-7)', lambda d, tw: wpct(d, d[IDEO_COL].isin([5,6,7])) if IDEO_COL in d.columns else np.nan),
    ],
    'RELIGION': [
        ('Born-again', lambda d, tw: wpct(d, d['V241442'] == 1) if 'V241442' in d.columns else np.nan),
        ('Church weekly', lambda d, tw: wpct(d, d['V241440'] == 1) if 'V241440' in d.columns else np.nan),
        ('Religion very important', lambda d, tw: wpct(d, d['V241420'] == 1) if 'V241420' in d.columns else np.nan),
    ],
    'REGION': [
        ('Northeast', lambda d, tw: wpct(d, d['V243007'] == 1) if 'V243007' in d.columns else np.nan),
        ('Midwest', lambda d, tw: wpct(d, d['V243007'] == 2) if 'V243007' in d.columns else np.nan),
        ('South', lambda d, tw: wpct(d, d['V243007'] == 3) if 'V243007' in d.columns else np.nan),
        ('West', lambda d, tw: wpct(d, d['V243007'] == 4) if 'V243007' in d.columns else np.nan),
    ],
    'VOTE 2024': [
        ('Harris', lambda d, tw: (d.loc[(d[TURNOUT_COL]==4) & (d[VOTE_COL]==1), WEIGHT_COL].sum() / d.loc[d[TURNOUT_COL]==4, WEIGHT_COL].sum() * 100) if VOTE_COL in d.columns and d.loc[d[TURNOUT_COL]==4, WEIGHT_COL].sum() > 0 else np.nan),
        ('Trump', lambda d, tw: (d.loc[(d[TURNOUT_COL]==4) & (d[VOTE_COL]==2), WEIGHT_COL].sum() / d.loc[d[TURNOUT_COL]==4, WEIGHT_COL].sum() * 100) if VOTE_COL in d.columns and d.loc[d[TURNOUT_COL]==4, WEIGHT_COL].sum() > 0 else np.nan),
    ],
}

# Build heatmap data
heatmap_rows = []
row_labels = []
category_breaks = []

for cat_name, metrics in demographics.items():
    category_breaks.append(len(row_labels))
    for metric_name, calc_func in metrics:
        row_data = []
        for cid in range(optimal_k):
            cd = anes_imputed[anes_imputed['cluster'] == cid]
            tw = cd[WEIGHT_COL].sum()
            val = calc_func(cd, tw)
            row_data.append(round(val, 1) if not np.isnan(val) else 0)
        heatmap_rows.append(row_data)
        row_labels.append(f"{cat_name}: {metric_name}")

z_data = np.array(heatmap_rows)
x_labels = [cluster_names[i] for i in range(optimal_k)]

# Row-normalize: each row gets its own 0-1 scale so the cluster
# with the highest share of each demographic lights up regardless
# of whether the raw values are 5% or 1000.
z_norm = np.zeros_like(z_data, dtype=float)
for i in range(z_data.shape[0]):
    rmin, rmax = z_data[i].min(), z_data[i].max()
    if rmax > rmin:
        z_norm[i] = (z_data[i] - rmin) / (rmax - rmin)
    else:
        z_norm[i] = 0.5

# Annotations show the actual values (not normalized)
annotations = []
for i, row in enumerate(z_data):
    for j, val in enumerate(row):
        # Pick font color for contrast against the normalized cell
        font_color = 'white' if z_norm[i][j] > 0.7 or z_norm[i][j] < 0.15 else 'black'
        # Show integers for N, one decimal for everything else
        txt = f"{val:.0f}" if 'N (' in row_labels[i] else f"{val:.1f}"
        annotations.append(
            dict(x=x_labels[j], y=row_labels[i], text=txt,
                 showarrow=False, font=dict(size=10, color=font_color))
        )

fig = go.Figure(data=go.Heatmap(
    z=z_norm,
    x=x_labels,
    y=row_labels,
    colorscale='YlOrRd',
    zmin=0,
    zmax=1,
    showscale=True,
    colorbar=dict(title='Relative<br>(row max)', tickvals=[0, 0.5, 1], ticktext=['Low', 'Mid', 'High']),
    hovertext=[[f"{row_labels[i]}<br>{x_labels[j]}: {z_data[i][j]:.1f}"
                for j in range(len(x_labels))] for i in range(len(row_labels))],
    hoverinfo='text',
))
fig.update_layout(
    title='Demographic Composition by Cluster (Row-Normalized)',
    height=max(600, len(row_labels) * 25),
    width=900,
    yaxis=dict(autorange='reversed'),
    annotations=annotations,
)
fig.show()

# %% [markdown]
# ## 5. Cluster Profiling -- Policy Positions
#
# For each policy question, shows **% who favor/agree** and **% who oppose/disagree** with the stated position, weighted by `V240108b`. All labels describe the concrete policy position, not an ideological direction.

# %%
# ============================================================
# Interpretation dictionary for all 48 clustering features
# Each entry defines: human-readable name, what "favor" means,
# which scale values count as favor vs oppose, and the scale range.
# No liberal/conservative labels -- just the concrete policy position.
# ============================================================

policy_interp = {
    # --- 7-Point Scales (1..7) ---
    'V241239': {
        'name': 'Government Services & Spending',
        'favor_label': 'More govt services',    'favor_vals': [5, 6, 7],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'Fewer govt services',   'oppose_vals': [1, 2, 3],
    },
    'V241242': {
        'name': 'Defense Spending',
        'favor_label': 'Increase defense spending', 'favor_vals': [5, 6, 7],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'Decrease defense spending', 'oppose_vals': [1, 2, 3],
    },
    'V241245': {
        'name': 'Health Insurance System',
        'favor_label': 'Govt insurance plan',    'favor_vals': [1, 2, 3],
        'neutral_label': 'Neutral', 'neutral_vals': [4],
        'oppose_label': 'Private insurance',     'oppose_vals': [5, 6, 7],
    },
    'V241248': {
        'name': 'Abortion Access',
        'favor_label': 'Always permit abortion', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'Restrict/ban abortion',  'oppose_vals': [5, 6, 7],
    },
    'V241252': {
        'name': 'Guaranteed Jobs & Income',
        'favor_label': 'Govt guarantee jobs',    'favor_vals': [1, 2, 3],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'People on their own',   'oppose_vals': [5, 6, 7],
    },
    'V241255': {
        'name': 'Government Help for Black Americans',
        'favor_label': 'Govt should help',       'favor_vals': [1, 2, 3],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'Should help themselves', 'oppose_vals': [5, 6, 7],
    },
    'V241255': {
        'name': 'Government Help for Black Americans',
        'favor_label': 'Govt should help',       'favor_vals': [1, 2, 3],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'Should help themselves', 'oppose_vals': [5, 6, 7],
    },
    'V241258': {
        'name': 'Environmental Regulation',
        'favor_label': 'Tougher regulation',     'favor_vals': [1, 2, 3],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'Too much regulation',   'oppose_vals': [5, 6, 7],
    },
    'V241397': {
        'name': 'Response to Urban Unrest',
        'favor_label': 'Address racism & police violence', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'Use force for law & order',       'oppose_vals': [5, 6, 7],
    },

    # --- Federal Spending (1=increase a lot .. 5=decrease a lot) ---
    'V241263x': {
        'name': 'Social Security Spending',
        'favor_label': 'Increase spending',  'favor_vals': [1, 2],
        'neutral_label': 'Keep about the same', 'neutral_vals': [3],
        'oppose_label': 'Decrease spending', 'oppose_vals': [4, 5],
    },
    'V241266x': {
        'name': 'Public Schools Spending',
        'favor_label': 'Increase spending',  'favor_vals': [1, 2],
        'neutral_label': 'Keep about the same', 'neutral_vals': [3],
        'oppose_label': 'Decrease spending', 'oppose_vals': [4, 5],
    },
    'V241269x': {
        'name': 'Border Security Spending',
        'favor_label': 'Increase spending',  'favor_vals': [1, 2],
        'neutral_label': 'Keep about the same', 'neutral_vals': [3],
        'oppose_label': 'Decrease spending', 'oppose_vals': [4, 5],
    },
    'V241272x': {
        'name': 'Crime Spending',
        'favor_label': 'Increase spending',  'favor_vals': [1, 2],
        'neutral_label': 'Keep about the same', 'neutral_vals': [3],
        'oppose_label': 'Decrease spending', 'oppose_vals': [4, 5],
    },
    'V241275x': {
        'name': 'Welfare Programs Spending',
        'favor_label': 'Increase spending',  'favor_vals': [1, 2],
        'neutral_label': 'Keep about the same', 'neutral_vals': [3],
        'oppose_label': 'Decrease spending', 'oppose_vals': [4, 5],
    },
    'V241278x': {
        'name': 'Highway Spending',
        'favor_label': 'Increase spending',  'favor_vals': [1, 2],
        'neutral_label': 'Keep about the same', 'neutral_vals': [3],
        'oppose_label': 'Decrease spending', 'oppose_vals': [4, 5],
    },
    'V241281x': {
        'name': 'Aid to the Poor Spending',
        'favor_label': 'Increase spending',  'favor_vals': [1, 2],
        'neutral_label': 'Keep about the same', 'neutral_vals': [3],
        'oppose_label': 'Decrease spending', 'oppose_vals': [4, 5],
    },
    'V241284x': {
        'name': 'Environmental Protection Spending',
        'favor_label': 'Increase spending',  'favor_vals': [1, 2],
        'neutral_label': 'Keep about the same', 'neutral_vals': [3],
        'oppose_label': 'Decrease spending', 'oppose_vals': [4, 5],
    },

    # --- Pre-Election Issues ---
    'V241302': {
        'name': 'Abortion (categorical)',
        'favor_label': 'Always personal choice', 'favor_vals': [3, 4],
        'oppose_label': 'Should not be permitted', 'oppose_vals': [1, 2],
    },
    'V241308x': {
        'name': 'Death Penalty',
        'favor_label': 'Favor death penalty',  'favor_vals': [1, 2],
        'oppose_label': 'Oppose death penalty', 'oppose_vals': [3, 4],
    },
    'V241313': {
        'name': 'Use of Military Force',
        'favor_label': 'Willing to use force', 'favor_vals': [1, 2, 3],
        'neutral_label': 'A little Willing', 'neutral_vals': [4],
        'oppose_label': 'Unwilling to use force', 'oppose_vals': [5],
    },
    'V241319x': {
        'name': 'Require Photo ID to Vote',
        'favor_label': 'Favor requiring ID', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose requiring ID', 'oppose_vals': [5, 6, 7],
    },
    'V241330x': {
        'name': 'Strong President vs Checks & Balances',
        'favor_label': 'President should bypass Congress/courts (helpful)', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Neutral', 'neutral_vals': [4],
        'oppose_label': 'Checks & balances are essential (harmful)',       'oppose_vals': [5, 6, 7],
    },
    'V241366x': {
        'name': 'Government Action on Climate',
        'favor_label': 'Govt do more on climate', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Govt do less on climate', 'oppose_vals': [5, 6, 7],
    },
    'V241369x': {
        'name': 'Paid Parental Leave Mandate',
        'favor_label': 'Favor mandate',  'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose mandate', 'oppose_vals': [5, 6, 7],
    },
    'V241372x': {
        'name': 'Transgender Bathroom Policy',
        'favor_label': 'Favor allowing trans bathroom choice', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose trans bathroom choice',        'oppose_vals': [5, 6, 7],
    },
    'V241375x': {
        'name': 'Ban Trans Girls from K-12 Sports',
        'favor_label': 'Favor ban',  'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose ban', 'oppose_vals': [5, 6, 7],
    },
    'V241378x': {
        'name': 'Protect Gays from Job Discrimination',
        'favor_label': 'Favor protections', 'favor_vals': [1, 2],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose protections', 'oppose_vals': [3, 4],
    },
    'V241386': {
        'name': 'Unauthorized Immigrant Policy',
        'favor_label': 'Path to citizenship/stay', 'favor_vals': [3, 4],
        'neutral_label': 'Limited Guest Worker Program', 'neutral_vals': [2],
        'oppose_label': 'Make felons & deport',     'oppose_vals': [1],
    },
    'V241389x': {
        'name': 'End Birthright Citizenship',
        'favor_label': 'Favor ending birthright', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose ending birthright', 'oppose_vals': [5, 6, 7],
    },
    'V241389x': {
        'name': 'End Birthright Citizenship',
        'favor_label': 'Favor ending birthright', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose ending birthright', 'oppose_vals': [5, 6, 7],
    },
    'V241395x': {
        'name': 'Build Border Wall',
        'favor_label': 'Favor wall',  'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose wall', 'oppose_vals': [5, 6, 7],
    },
    'V241400x': {
        'name': 'Weapons to Ukraine',
        'favor_label': 'Favor giving weapons', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose giving weapons', 'oppose_vals': [5, 6, 7],
    },
    'V241290x': {
        'name': 'DEI Policies on Campus',
        'favor_label': 'Favor DEI',  'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose DEI', 'oppose_vals': [5, 6, 7],
    },

    # --- Post-Election Issues ---
    'V242227': {
        'name': 'Immigration Levels',
        'favor_label': 'Increase immigration', 'favor_vals': [1, 2],
        'neutral_label': 'Keep about the same', 'neutral_vals': [3],
        'oppose_label': 'Decrease immigration', 'oppose_vals': [4, 5],
    },
    'V242234x': {
        'name': 'Path to Citizenship',
        'favor_label': 'Favor path to citizenship', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose path to citizenship', 'oppose_vals': [5, 6, 7],
    },
    'V242235': {
        'name': 'Immigrants\' Effect on Economy',
        'favor_label': 'Good for economy',  'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Bad for economy',  'oppose_vals': [5, 6, 7],
    },
    'V242241x': {
        'name': 'Preferential Hiring of Black Americans',
        'favor_label': 'Favor preferential hiring', 'favor_vals': [1, 2],
        'oppose_label': 'Oppose preferential hiring', 'oppose_vals': [3, 4],
    },
    'V242245x': {
        'name': 'Affirmative Action in Universities',
        'favor_label': 'Favor affirmative action', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose affirmative action', 'oppose_vals': [5, 6, 7],
    },
    'V242248x': {
        'name': 'Size of Government',
        'favor_label': 'Favor more government',  'favor_vals': [1, 2, 3],
        'oppose_label': 'Favor less government',  'oppose_vals': [4, 5, 6],
    },
    'V242249': {
        'name': 'Government Regulation of Business',
        'favor_label': 'More regulation',  'favor_vals': [1, 2, 3],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'Less regulation',  'oppose_vals': [5, 6, 7],
    },
    'V242253x': {
        'name': 'Reduce Income Inequality',
        'favor_label': 'Favor reducing inequality', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Neutral', 'neutral_vals': [4],
        'oppose_label': 'Oppose reducing inequality', 'oppose_vals': [5, 6, 7],
    },
    'V242319x': {
        'name': 'Require School Vaccines',
        'favor_label': 'Favor requiring vaccines', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Neutral', 'neutral_vals': [4],
        'oppose_label': 'Oppose requiring vaccines', 'oppose_vals': [5, 6, 7],
    },
    'V242324x': {
        'name': 'Regulate Greenhouse Emissions',
        'favor_label': 'Favor regulation',  'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose regulation', 'oppose_vals': [5, 6, 7],
    },
    'V242325': {
        'name': 'Gun Purchase Difficulty',
        'favor_label': 'Make harder to buy', 'favor_vals': [1],
        'oppose_label': 'Make easier to buy', 'oppose_vals': [2],
        'neutral_label': 'about the same', 'neutral_vals': [3],
    },
    'V242328x': {
        'name': 'Background Checks at Gun Shows',
        'favor_label': 'Favor background checks', 'favor_vals': [1, 2, 3],
        'oppose_label': 'Oppose background checks', 'oppose_vals': [5, 6, 7],
    },
    'V242331x': {
        'name': 'Ban Assault-Style Rifles',
        'favor_label': 'Favor ban',  'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose ban', 'oppose_vals': [5, 6, 7],
    },
    'V242346x': {
        'name': 'Free Trade Agreements',
        'favor_label': 'Favor free trade', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Oppose free trade', 'oppose_vals': [5, 6, 7],
    },
    'V242350': {
        'name': 'Minimum Wage',
        'favor_label': 'Raise minimum wage', 'favor_vals': [1],
        'neutral_label': 'Kept the Same', 'neutral_vals': [2],
        'oppose_label': 'Lower or eliminate', 'oppose_vals': [3, 4],
    },
    'V242353x': {
        'name': 'Govt Spending on Health Insurance',
        'favor_label': 'Increase spending',  'favor_vals': [1, 2, 3],
        'neutral_label': 'Keep about the same', 'neutral_vals': [4],
        'oppose_label': 'Decrease spending',  'oppose_vals': [5, 6, 7],
    },
    'V242335x': {
        'name': 'Govt Action on Opioid Addiction',
        'favor_label': 'Govt do more', 'favor_vals': [1, 2, 3],
        'neutral_label': 'Unsure', 'neutral_vals': [4],
        'oppose_label': 'Govt do less', 'oppose_vals': [5, 6, 7],
    },
}

print(f"Interpretation defined for {len(policy_interp)} of {len(clustering_questions)} clustering features")

# %%
# ============================================================
# Detailed policy profiles by cluster -- % favor / % oppose
# ============================================================

# Organize by domain (no duplicates)
policy_domains = {
    'Government Role & Spending': [
        'V241239', 'V241252', 'V242248x', 'V242249', 'V242253x',
    ],
    'Federal Budget Priorities': [
        'V241263x', 'V241266x', 'V241269x', 'V241272x',
        'V241275x', 'V241278x', 'V241281x', 'V241284x',
    ],
    'Healthcare': [
        'V241245', 'V242353x',
    ],
    'Abortion & Reproductive Rights': [
        'V241248', 'V241302',
    ],
    'Immigration': [
        'V241386', 'V241389x', 'V241395x', 'V241269x',
        'V242227', 'V242234x', 'V242235',
    ],
    'Guns': [
        'V242325', 'V242328x', 'V242331x',
    ],
    'Environment & Climate': [
        'V241258', 'V241284x', 'V241366x', 'V242324x',
    ],
    'Civil Rights & Social Policy': [
        'V241255', 'V241290x', 'V241372x', 'V241375x',
        'V241378x', 'V241369x', 'V242241x', 'V242245x',
    ],
    'Criminal Justice & Law Enforcement': [
        'V241397', 'V241308x', 'V241272x',
    ],
    'Foreign Policy': [
        'V241313', 'V241400x', 'V241242',
    ],
    'Voting, Institutions & Misc': [
        'V241319x', 'V241330x', 'V242319x', 'V242335x',
        'V242346x', 'V242350',
    ],
}

# Track which variables we've already printed to avoid duplicates
printed_vars = set()

for domain, vars_in_domain in policy_domains.items():
    print(f"\n{'='*100}")
    print(f"  {domain}")
    print(f"{'='*100}")

    for var in vars_in_domain:
        if var not in clustering_questions or var in printed_vars:
            continue
        if var not in policy_interp:
            continue
        printed_vars.add(var)

        info = policy_interp[var]
        print(f"\n  {info['name']}  ({var})")
        print(f"  {info['favor_label']}  vs  {info['oppose_label']} vs {info.get('neutral_label', 'Neutral')}")

        # Header
        col_width = 12
        header = "    " + "".join([f"{'C'+str(i):>{col_width}}" for i in range(optimal_k)])
        print(header)

        # % favor
        favor_row = "    "
        for cid in range(optimal_k):
            cd = anes_imputed[anes_imputed['cluster'] == cid]
            pct = wpct(cd, cd[var].isin(info['favor_vals']))
            favor_row += f"{pct:>{col_width}.1f}"
        print(f"  % {info['favor_label'][:25]:<25}")
        print(favor_row)

        # % neutral (if defined)
        neutral_row = "    "
        for cid in range(optimal_k):
            cd = anes_imputed[anes_imputed['cluster'] == cid]
            pct = wpct(cd, cd[var].isin(info.get('neutral_vals', [])))
            neutral_row += f"{pct:>{col_width}.1f}"
        print(f"  % {info.get('neutral_label', 'Neutral')[:25]:<25}")
        print(neutral_row)

        # % oppose
        oppose_row = "    "
        for cid in range(optimal_k):
            cd = anes_imputed[anes_imputed['cluster'] == cid]
            pct = wpct(cd, cd[var].isin(info['oppose_vals']))
            oppose_row += f"{pct:>{col_width}.1f}"
        print(f"  % {info['oppose_label'][:25]:<25}")
        print(oppose_row)

# Check if any clustering features were missed
missed = [v for v in clustering_questions if v not in printed_vars]
if missed:
    print(f"\n\nNOTE: {len(missed)} features not shown (no interpretation defined): {missed}")

# %%
# ============================================================
# Plotly Policy Heatmap -- % Favor by Cluster
# Shows the percentage who agree with the stated policy position
# ============================================================

heatmap_rows = []
row_labels = []
favor_descriptions = []  # for hover text

# Only include variables that have interpretation defined
for var in clustering_questions:
    if var not in policy_interp:
        continue
    info = policy_interp[var]

    row_data = []
    for cid in range(optimal_k):
        cd = anes_imputed[anes_imputed['cluster'] == cid]
        pct = wpct(cd, cd[var].isin(info['favor_vals']))
        row_data.append(round(pct, 1))

    heatmap_rows.append(row_data)
    row_labels.append(info['name'])
    favor_descriptions.append(info['favor_label'])

z_data = np.array(heatmap_rows)
x_labels = [cluster_names[i] for i in range(optimal_k)]

# Create annotation text (show % favor)
annotations = []
for i, row in enumerate(z_data):
    for j, val in enumerate(row):
        # Color text based on value for readability
        font_color = 'white' if val > 75 or val < 25 else 'black'
        annotations.append(
            dict(x=x_labels[j], y=row_labels[i], text=f"{val:.0f}%",
                 showarrow=False, font=dict(size=9, color=font_color))
        )

# Hover text shows full detail
hovertext = [[
    f"<b>{row_labels[i]}</b><br>"
    f"{x_labels[j]}<br>"
    f"% {favor_descriptions[i]}: {z_data[i][j]:.1f}%"
    for j in range(len(x_labels))] for i in range(len(row_labels))]

fig = go.Figure(data=go.Heatmap(
    z=z_data,
    x=x_labels,
    y=row_labels,
    colorscale='Viridis',
    zmin=0,
    zmax=100,
    showscale=True,
    colorbar=dict(title='% Favor'),
    hovertext=hovertext,
    hoverinfo='text',
))

fig.update_layout(
    title='Policy Support by Cluster (% Favor / Agree)',
    height=max(800, len(row_labels) * 24),
    width=1000,
    yaxis=dict(autorange='reversed'),
    annotations=annotations,
    margin=dict(l=300),
)
fig.show()

# %% [markdown]
# ## 6. Descriptor Profiles -- Cultural Attitudes
#
# These variables were deliberately **excluded** from clustering because they measure cultural identity, racial attitudes, and egalitarian values rather than specific policy preferences. They are reported here for interpretation.

# %%
# ============================================================
# Cultural Attitude Descriptors -- NOT used in clustering
# ============================================================

descriptor_domains = {
    'Racial Resentment (1=Agree strongly..5=Disagree strongly)': {
        'V242300': 'Blacks should work way up w/o special favors',
        'V242301': 'Slavery/discrimination created difficult conditions',
        'V242302': 'Blacks gotten less than they deserve',
        'V242303': 'If Blacks tried harder, be as well off',
    },
    'Egalitarianism (1=Agree strongly..5=Disagree strongly)': {
        'V242254': 'Society should ensure equal opportunity',
        'V242255': 'Better off if worried less about equality',
        'V242256': 'Not big problem if some have more chance',
        'V242257': 'Treating more equally = fewer problems',
    },
    'Gender & Social Attitudes': {
        'V242279x': 'Better if man works/woman stays home (7pt)',
        'V242361x': 'Sexual harassment attention (5pt)',
    },
}

for domain_name, vars_dict in descriptor_domains.items():
    print(f"\n{'='*80}")
    print(f"  {domain_name}")
    print(f"{'='*80}")

    header = "    " + "  ".join([f"{'C'+str(i):>8}" for i in range(optimal_k)])
    print(header)

    for var, label in vars_dict.items():
        if var not in anes_imputed.columns:
            continue
        means = []
        for cid in range(optimal_k):
            cd = anes_imputed[anes_imputed['cluster'] == cid]
            m = wmean(cd, var)
            means.append(f"{m:.2f}")

        values = "    " + "  ".join([f"{m:>8}" for m in means])
        print(f"\n  {label}")
        print(values)

# ============================================================
# Descriptor Heatmap
# ============================================================

desc_rows = []
desc_labels = []

for domain_name, vars_dict in descriptor_domains.items():
    for var, label in vars_dict.items():
        if var not in anes_imputed.columns:
            continue
        row = []
        for cid in range(optimal_k):
            cd = anes_imputed[anes_imputed['cluster'] == cid]
            m = wmean(cd, var)
            row.append(round(m, 2) if not np.isnan(m) else 0)
        desc_rows.append(row)
        desc_labels.append(f"{var}: {label}")

if desc_rows:
    z_desc = np.array(desc_rows)

    # Normalize rows
    z_desc_norm = np.zeros_like(z_desc, dtype=float)
    for i in range(z_desc.shape[0]):
        rmin, rmax = z_desc[i].min(), z_desc[i].max()
        if rmax > rmin:
            z_desc_norm[i] = (z_desc[i] - rmin) / (rmax - rmin)
        else:
            z_desc_norm[i] = 0.5

    # Annotations
    desc_annotations = []
    for i, row in enumerate(z_desc):
        for j, val in enumerate(row):
            desc_annotations.append(
                dict(x=x_labels[j], y=desc_labels[i], text=f"{val:.2f}",
                     showarrow=False, font=dict(size=10))
            )

    fig = go.Figure(data=go.Heatmap(
        z=z_desc_norm,
        x=x_labels,
        y=desc_labels,
        colorscale='viridis',
        zmid=0.5,
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(title='Relative<br>Position'),
    ))
    fig.update_layout(
        title='Cultural Attitude Descriptors by Cluster (NOT used in clustering)',
        height=max(400, len(desc_labels) * 35),
        width=900,
        yaxis=dict(autorange='reversed'),
        annotations=desc_annotations,
        margin=dict(l=350),
    )
    fig.show()

# %% [markdown]
# ## 7. Cluster Naming & Final Summary
#
# After reviewing the demographic, policy, and cultural profiles above, assign descriptive names to each cluster. Update the `cluster_names` dictionary below, then re-run subsequent cells if desired.

# %%
# ============================================================
# UPDATE THESE NAMES after reviewing profiles above
# ============================================================

# cluster_names = {
#     0: 'Progressive Left',        # <-- update based on profiles
#     1: 'Establishment Democrats',  # <-- update based on profiles
#     2: 'Christian Democrats',      # <-- update based on profiles
#     3: 'Cultural Protectionists',  # <-- update based on profiles
#     4: 'Traditional Conservatives',# <-- update based on profiles
#     5: 'Hard Right',              # <-- update based on profiles
# }

# ============================================================
# Final Summary Table
# ============================================================

print(f"{'='*80}")
print(f"  ANES 2024 CLUSTER ANALYSIS -- FINAL SUMMARY")
print(f"  {len(anes_imputed):,} respondents | {len(clustering_questions)} clustering features | {optimal_k} clusters")
print(f"  Weight: {WEIGHT_COL} | Ideology ordering: {IDEO_COL}")
print(f"{'='*80}")

print(f"\n{'Cluster':<10} {'Name':<30} {'N':>6} {'Wt%':>6} {'MeanIdeo':>10} {'MeanPID':>10}")
print(f"{'-'*72}")

for cid in range(optimal_k):
    cd = anes_imputed[anes_imputed['cluster'] == cid]
    tw = cd[WEIGHT_COL].sum()
    all_tw = anes_imputed[WEIGHT_COL].sum()
    name = cluster_names.get(cid, f'Cluster {cid}')
    mean_ideo = wmean(cd, IDEO_COL)
    mean_pid = wmean(cd, PID_COL)

    print(f"{cid:<10} {name:<30} {len(cd):>6,} {tw/all_tw*100:>5.1f}% {mean_ideo:>9.2f} {mean_pid:>9.2f}")

print(f"\nPipeline: {len(clustering_questions)} features -> StandardScaler -> PCA({N_COMPONENTS}) -> KMeans(k={optimal_k})")
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Silhouette score: {silhouette_score(X_pca, anes_imputed['cluster'].values):.4f}")

# %%
# ============================================================
# Polarization Map: Where do clusters agree vs disagree?
# Shows the standard deviation of weighted means across clusters
# for each policy question. Higher = more polarizing.
# ============================================================

# Use % favor spread instead of raw means for polarization
polarization = []
for var in clustering_questions:
    if var not in policy_interp:
        continue
    info = policy_interp[var]

    favor_pcts = []
    for cid in range(optimal_k):
        cd = anes_imputed[anes_imputed['cluster'] == cid]
        pct = wpct(cd, cd[var].isin(info['favor_vals']))
        favor_pcts.append(pct)

    polarization.append({
        'variable': var,
        'name': info['name'],
        'favor_label': info['favor_label'],
        'std_favor': np.std(favor_pcts),
        'range_favor': max(favor_pcts) - min(favor_pcts),
        'min_favor': min(favor_pcts),
        'max_favor': max(favor_pcts),
    })

pol_df = pd.DataFrame(polarization).sort_values('range_favor', ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=[f"{r['name']}" for _, r in pol_df.iterrows()],
    x=pol_df['range_favor'],
    orientation='h',
    marker_color=pol_df['range_favor'],
    marker_colorscale='Reds',
    text=[f"{r['range_favor']:.0f}pp" for _, r in pol_df.iterrows()],
    textposition='outside',
))
fig.update_layout(
    title='Policy Polarization Across Clusters<br>(Range of % Favor across clusters -- higher = more divisive)',
    xaxis_title='Range in % Favor (percentage points)',
    height=max(600, len(pol_df) * 22),
    width=1000,
    margin=dict(l=300),
)
fig.show()

print("\nTop 10 most divisive issues (largest spread in % favor across clusters):")
for _, row in pol_df.tail(10).iloc[::-1].iterrows():
    print(f"  {row['name']}: {row['favor_label']}")
    print(f"    Range: {row['min_favor']:.0f}% to {row['max_favor']:.0f}% ({row['range_favor']:.0f} pp spread)")

print("\nTop 5 consensus issues (smallest spread):")
for _, row in pol_df.head(5).iterrows():
    print(f"  {row['name']}: {row['favor_label']}")
    print(f"    Range: {row['min_favor']:.0f}% to {row['max_favor']:.0f}% ({row['range_favor']:.0f} pp spread)")

# %% [markdown]
# ## 8b. Comprehensive Cluster Profiles -- All-in-One
#
# Complete demographic, policy, and cultural attitude breakdown for every cluster, printed in a single block per cluster (mirrors the CES notebook format). Two versions:
# 1. **Original clustering** (48 policy features, including foreign policy)
# 2. **Domestic-only clustering** (45 features, excluding foreign policy) -- generated after Section 9 runs

# %%
# ============================================================
# COMPREHENSIVE CLUSTER PROFILES -- ORIGINAL (48 features)
# All demographics, policy positions, and cultural descriptors
# in one block per cluster.
# ============================================================

def print_comprehensive_profile(df, cluster_col, cluster_count, label_prefix="Cluster",
                                 feature_list=None):
    """
    Print a CES-style comprehensive profile for each cluster.

    Parameters
    ----------
    df : DataFrame with cluster assignments and all variables
    cluster_col : str, column name holding the cluster IDs (0..cluster_count-1)
    cluster_count : int, number of clusters
    label_prefix : str, how to label clusters in output
    feature_list : list of str or None; if given, only show these policy vars
    """
    all_tw = df[WEIGHT_COL].sum()

    for cid in range(cluster_count):
        cd = df[df[cluster_col] == cid]
        tw = cd[WEIGHT_COL].sum()

        print(f"\n{'='*120}")
        print(f"{label_prefix} {cid} -- n={len(cd):,} ({tw/all_tw*100:.1f}% of weighted sample)")
        print(f"{'='*120}")

        # ---- POLITICAL IDENTITY ----
        print(f"\nPOLITICAL IDENTITY:")
        ideo_mean = wmean(cd, IDEO_COL)
        print(f"  Ideology (V241177, 1=ext lib..7=ext con): Mean={ideo_mean:.2f}")
        for val, lab in [(1,'ExtLib'), (2,'Lib'), (3,'SlLib'), (4,'Mod'),
                         (5,'SlCon'), (6,'Con'), (7,'ExtCon')]:
            print(f"    {lab}={wpct(cd, cd[IDEO_COL]==val):.1f}%", end="")
        print()

        pid_mean = wmean(cd, PID_COL)
        print(f"  Party ID (V241227x, 1=StrongDem..7=StrongRep): Mean={pid_mean:.2f}")
        for val, lab in [(1,'Str Dem'), (2,'Weak Dem'), (3,'Lean Dem'),
                         (4,'Ind'), (5,'Lean Rep'), (6,'Weak Rep'), (7,'Str Rep')]:
            print(f"    {lab}={wpct(cd, cd[PID_COL]==val):.1f}%", end="")
        print()

        # ---- AGE & GENDER ----
        print(f"\nAGE & GENDER:")
        if 'V241458x' in cd.columns:
            print(f"  Mean Age: {wmean(cd, 'V241458x'):.1f}")
            print(f"  18-29={wpct(cd, cd['V241458x'].between(18,29)):.1f}%  "
                  f"30-44={wpct(cd, cd['V241458x'].between(30,44)):.1f}%  "
                  f"45-59={wpct(cd, cd['V241458x'].between(45,59)):.1f}%  "
                  f"60+={wpct(cd, cd['V241458x']>=60):.1f}%")
        if 'V241550' in cd.columns:
            print(f"  Male={wpct(cd, cd['V241550']==1):.1f}%  Female={wpct(cd, cd['V241550']==2):.1f}%")

        # ---- RACE/ETHNICITY ----
        print(f"\nRACE/ETHNICITY:")
        if 'V241501x' in cd.columns:
            for val, lab in [(1,'White NH'), (2,'Black NH'), (3,'Hispanic'),
                             (4,'Asian/PI NH'), (5,'Native Am NH'), (6,'Multi NH')]:
                pct = wpct(cd, cd['V241501x']==val)
                if pct > 0.5:
                    print(f"  {lab}={pct:.1f}%", end="")
            print()

        # ---- EDUCATION ----
        print(f"\nEDUCATION:")
        if 'V241465x' in cd.columns:
            for val, lab in [(1,'<HS'), (2,'HS'), (3,'Some college'), (4,'BA'), (5,'Grad')]:
                print(f"  {lab}={wpct(cd, cd['V241465x']==val):.1f}%", end="")
            print()

        # ---- INCOME ----
        print(f"\nHOUSEHOLD INCOME:")
        if 'V241567x' in cd.columns:
            for val, lab in [(1,'<$10k'), (2,'$10-30k'), (3,'$30-60k'),
                             (4,'$60-100k'), (5,'$100-250k'), (6,'$250k+')]:
                pct = wpct(cd, cd['V241567x']==val)
                if pct > 0:
                    print(f"  {lab}={pct:.1f}%", end="")
            print()

        # ---- EMPLOYMENT ----
        print(f"\nEMPLOYMENT:")
        if 'V241488x' in cd.columns:
            for val, lab in [(1,'Working'), (2,'Temp laid off'), (4,'Unemployed'),
                             (5,'Retired'), (6,'Disabled'), (7,'Homemaker'), (8,'Student')]:
                pct = wpct(cd, cd['V241488x']==val)
                if pct > 0.5:
                    print(f"  {lab}={pct:.1f}%", end="")
            print()

        # ---- FAMILY & GEOGRAPHY ----
        print(f"\nFAMILY & GEOGRAPHY:")
        if 'V241461x' in cd.columns:
            print(f"  Marital:", end="")
            for val, lab in [(1,'Married'), (2,'Widowed'), (3,'Divorced'),
                             (4,'Separated'), (5,'Never married')]:
                pct = wpct(cd, cd['V241461x']==val)
                if pct > 1:
                    print(f" {lab}={pct:.1f}%", end="")
            print()
        if 'V243007' in cd.columns:
            print(f"  Region:", end="")
            for val, lab in [(1,'NE'), (2,'MW'), (3,'South'), (4,'West')]:
                print(f" {lab}={wpct(cd, cd['V243007']==val):.1f}%", end="")
            print()

        # ---- RELIGION ----
        print(f"\nRELIGION:")
        if 'V241445x' in cd.columns:
            for val, lab in [(1,'Main Prot'), (2,'Evan Prot'), (3,'Black Prot'),
                             (4,'Catholic'), (5,'Undiff Chr'), (6,'Jewish'),
                             (7,'Other relig'), (8,'Not religious')]:
                pct = wpct(cd, cd['V241445x']==val)
                if pct > 1:
                    print(f"  {lab}={pct:.1f}%", end="")
            print()
        if 'V241442' in cd.columns:
            print(f"  Born-again={wpct(cd, cd['V241442']==1):.1f}%", end="")
        if 'V241420' in cd.columns:
            print(f"  Religion very important={wpct(cd, cd['V241420']==1):.1f}%"
                  f"  Not important={wpct(cd, cd['V241420']==5):.1f}%", end="")
        if 'V241440' in cd.columns:
            print(f"  Attend weekly={wpct(cd, cd['V241440']==1):.1f}%"
                  f"  Never={wpct(cd, cd['V241440']==5):.1f}%")

        # ---- 2024 VOTE ----
        print(f"\n2024 PRESIDENTIAL VOTE:")
        if VOTE_COL in cd.columns:
            voters = cd[cd[TURNOUT_COL] == 4]
            vtw = voters[WEIGHT_COL].sum()
            if vtw > 0:
                harris = (voters.loc[voters[VOTE_COL]==1, WEIGHT_COL].sum() / vtw) * 100
                trump  = (voters.loc[voters[VOTE_COL]==2, WEIGHT_COL].sum() / vtw) * 100
                other  = (voters.loc[voters[VOTE_COL].isin([4,5,6]), WEIGHT_COL].sum() / vtw) * 100
                turnout = wpct(cd, cd[TURNOUT_COL]==4)
                print(f"  Harris={harris:.1f}%  Trump={trump:.1f}%", end="")
                if other > 0.5:
                    print(f"  Other={other:.1f}%", end="")
                print(f"  (Turnout={turnout:.1f}%)")

        # ---- CULTURAL DESCRIPTORS ----
        print(f"\nCULTURAL ATTITUDES (descriptor only -- NOT used in clustering):")

        desc_blocks = {
            'Racial Resentment (1=Agree strongly..5=Disagree strongly)': {
                'V242300': 'Blacks work way up w/o favors',
                'V242301': 'Slavery created difficult conditions',
                'V242302': 'Blacks gotten less than deserve',
                'V242303': 'If tried harder, be as well off',
            },
            'Egalitarianism (1=Agree strongly..5=Disagree strongly)': {
                'V242254': 'Equal opportunity',
                'V242255': 'Worry less about equality',
                'V242256': 'Not big problem if some have more',
                'V242257': 'Treat more equally = fewer problems',
            },
            'Gender & Social': {
                'V242279x': 'Man works/woman home (7pt)',
                'V242361x': 'Sexual harassment attention (5pt)',
            },
        }
        for block_name, var_dict in desc_blocks.items():
            print(f"  {block_name}:")
            for v, lab in var_dict.items():
                if v in cd.columns:
                    m = wmean(cd, v)
                    print(f"    {lab}: {m:.2f}")

        # ---- ALL POLICY QUESTIONS ----
        use_vars = feature_list if feature_list else clustering_questions

        print(f"\nPOLICY POSITIONS (% favor / % oppose):")
        for domain, vars_in_domain in policy_domains.items():
            domain_vars = [v for v in vars_in_domain if v in use_vars and v in policy_interp]
            if not domain_vars:
                continue
            print(f"\n  {domain}:")
            for var in domain_vars:
                info = policy_interp[var]
                fav_pct = wpct(cd, cd[var].isin(info['favor_vals']))
                neut_pct = wpct(cd, cd[var].isin(info['neutral_vals'])) if 'neutral_vals' in info else 0
                opp_pct = wpct(cd, cd[var].isin(info['oppose_vals']))
                print(f"    {info['name']} ({var})")
                print(f"      {info['favor_label']}={fav_pct:.1f}%  {info.get('neutral_label', 'Neutral')}={neut_pct:.1f}%  {info['oppose_label']}={opp_pct:.1f}%")

# --- Run for original clustering ---
print("=" * 120)
print("COMPREHENSIVE CLUSTER PROFILES -- ORIGINAL (48 policy features, including foreign policy)")
print("All demographics and percentages are WEIGHTED using V240108b")
print("=" * 120)

print_comprehensive_profile(anes_imputed, 'cluster', optimal_k,
                            label_prefix="CLUSTER", feature_list=clustering_questions)

# %%
# %%
# ============================================================
# 9. Alternative Clustering: Pre-Election Domestic Policy Only
#    WITH Post-Election Attrition & DNA Analysis
# ============================================================

print("="*100)
print("SECTION 9: PRE-ELECTION CLUSTERING (Including Attrition Analysis)")
print("="*100)

# 1. DEFINE VARIABLES
# ------------------------------------------------------------
pre_election_vars = [
    'V241239', 'V241242', 'V241245', 'V241248', 'V241252', 'V241255', 'V241258', 'V241397',
    'V241263x', 'V241266x', 'V241269x', 'V241272x', 'V241275x', 'V241278x', 'V241281x', 'V241284x',
    'V241302', 'V241308x', 'V241313', 'V241319x', 'V241330x', 'V241366x', 'V241369x', 'V241372x',
    'V241375x', 'V241378x', 'V241386', 'V241389x', 'V241395x', 'V241400x', 'V241290x'
]
foreign_policy_vars = ['V241242', 'V241313', 'V241400x']

# CLUSTERING VARS: Only Pre-Election, No Foreign Policy
pre_election_domestic = [v for v in pre_election_vars if v not in foreign_policy_vars and v in anes_raw.columns]

# 2. CREATE SUPER-SET DATA (Pre-Only + Pre-Post)
# ------------------------------------------------------------
# V240002c: 1=Pre Only, 2=Pre+Post. We need BOTH to calculate attrition.
anes_all = anes_raw[anes_raw['V240002c'].isin([1, 2])].copy()

# IMPORTANT: Use Pre-Election Weight (V240108a)
PRE_WEIGHT = 'V240108a'
anes_all = anes_all[anes_all[PRE_WEIGHT] > 0].copy()

# 3. RE-APPLY DATA CLEANING
# ------------------------------------------------------------
clean_cols = clustering_questions + all_descriptor_cols
for col in clean_cols:
    if col in anes_all.columns:
        # Recode negative values to NaN
        anes_all.loc[anes_all[col] < 0, col] = np.nan
        # Recode 99 (DK) for 7-point scales
        if col in seven_pt_scales_with_99:
            anes_all.loc[anes_all[col] == 99, col] = np.nan

# 4. STRICT FILTER (Complete Cases Only)
# ------------------------------------------------------------
# ERROR FIX: PCA fails if ANY NaN exists. 
# We must strictly drop rows with missing values in the clustering variables.
print(f"Initial sample size: {len(anes_all):,}")
anes_all = anes_all.dropna(subset=pre_election_domestic).copy()

print(f"Sample size after strict dropna: {len(anes_all):,}")
print(f"Variables used for CLUSTERING: {len(pre_election_domestic)}")

# 5. CLUSTERING PIPELINE (Strictly Pre-Election)
# ------------------------------------------------------------
X_pre = anes_all[pre_election_domestic].values
scaler_pre = StandardScaler()
X_pre_scaled = scaler_pre.fit_transform(X_pre)

# PCA
pca_pre_calc = PCA(random_state=42)
pca_pre_calc.fit(X_pre_scaled)
n_comp_pre = np.argmax(np.cumsum(pca_pre_calc.explained_variance_ratio_) >= 0.67) + 1

pca_pre = PCA(n_components=n_comp_pre, random_state=42)
X_pre_pca = pca_pre.fit_transform(X_pre_scaled)

# KMeans
kmeans_pre = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_raw = kmeans_pre.fit_predict(X_pre_pca)
anes_all['cluster_raw'] = labels_raw

# Order by Ideology
pre_ideo = anes_all.groupby('cluster_raw')[IDEO_COL].mean().sort_values()
old_to_new = {old: new for new, old in enumerate(pre_ideo.index)}
anes_all['cluster'] = anes_all['cluster_raw'].map(old_to_new)

# 6. UNIFIED COMPREHENSIVE PROFILE PRINTER
# Combines the rich descriptive detail of Section 8b with the
# Attrition/DNA analysis of Section 9.
# ------------------------------------------------------------
def print_unified_profile(df, weight_col=PRE_WEIGHT):
    
    total_w_all = df[weight_col].sum()
    
    for cid in range(optimal_k):
        cd = df[df['cluster'] == cid]
        tw = cd[weight_col].sum()
        
        # =========================================================
        # HEADER & ATTRITION
        # =========================================================
        print(f"\n{'='*120}")
        print(f"CLUSTER {cid} (n={len(cd):,}, {tw/total_w_all*100:.1f}% of population)")
        # V240002c: 1=Pre Only, 2=Pre+Post
        attrition_pct = wpct(cd, cd['V240002c'] == 1, weight_col)
        print(f"POST-ELECTION ATTRITION: {attrition_pct:.1f}% dropped out before post-survey")
        print(f"{'='*120}")
        
        # =========================================================
        # PART 1: DEMOGRAPHICS & IDENTITY (Restored from 8b)
        # =========================================================
        
        # ---- POLITICAL IDENTITY ----
        print(f"\nPOLITICAL IDENTITY:")
        ideo_mean = wmean(cd, IDEO_COL, weight_col)
        print(f"  Ideology (1=Lib..7=Con): Mean={ideo_mean:.2f}")
        print("    ", end="")
        for val, lab in [(1,'ExtLib'), (2,'Lib'), (3,'SlLib'), (4,'Mod'), (5,'SlCon'), (6,'Con'), (7,'ExtCon')]:
            print(f"{lab}={wpct(cd, cd[IDEO_COL]==val, weight_col):.1f}%  ", end="")
        print()

        pid_mean = wmean(cd, PID_COL, weight_col)
        print(f"  Party ID (1=Dem..7=Rep): Mean={pid_mean:.2f}")
        print("    ", end="")
        for val, lab in [(1,'StrDem'), (2,'WkDem'), (3,'LnDem'), (4,'Ind'), (5,'LnRep'), (6,'WkRep'), (7,'StrRep')]:
            print(f"{lab}={wpct(cd, cd[PID_COL]==val, weight_col):.1f}%  ", end="")
        print()

        # ---- AGE & GENDER ----
        print(f"\nAGE & GENDER:")
        if 'V241458x' in cd.columns:
            print(f"  Mean Age: {wmean(cd, 'V241458x', weight_col):.1f}")
            print(f"  18-29={wpct(cd, cd['V241458x'].between(18,29), weight_col):.1f}%  "
                  f"30-44={wpct(cd, cd['V241458x'].between(30,44), weight_col):.1f}%  "
                  f"45-59={wpct(cd, cd['V241458x'].between(45,59), weight_col):.1f}%  "
                  f"60+={wpct(cd, cd['V241458x']>=60, weight_col):.1f}%")
        if 'V241550' in cd.columns:
            print(f"  Male={wpct(cd, cd['V241550']==1, weight_col):.1f}%  Female={wpct(cd, cd['V241550']==2, weight_col):.1f}%")

        # ---- RACE/ETHNICITY ----
        print(f"\nRACE/ETHNICITY:")
        if 'V241501x' in cd.columns:
            print("  ", end="")
            for val, lab in [(1,'White'), (2,'Black'), (3,'Hispanic'), (4,'Asian'), (5,'Native'), (6,'Multi')]:
                pct = wpct(cd, cd['V241501x']==val, weight_col)
                if pct > 0.5: print(f"{lab}={pct:.1f}%  ", end="")
            print()

        # ---- EDUCATION ----
        print(f"\nEDUCATION:")
        if 'V241465x' in cd.columns:
            print("  ", end="")
            for val, lab in [(1,'<HS'), (2,'HS'), (3,'SomeCol'), (4,'BA'), (5,'Grad')]:
                print(f"{lab}={wpct(cd, cd['V241465x']==val, weight_col):.1f}%  ", end="")
            print()

        # ---- INCOME ----
        print(f"\nHOUSEHOLD INCOME:")
        if 'V241567x' in cd.columns:
            print("  ", end="")
            for val, lab in [(1,'<$10k'), (2,'$10-30k'), (3,'$30-60k'), (4,'$60-100k'), (5,'$100-250k'), (6,'$250k+')]:
                pct = wpct(cd, cd['V241567x']==val, weight_col)
                if pct > 0: print(f"{lab}={pct:.1f}%  ", end="")
            print()

        # ---- EMPLOYMENT ----
        print(f"\nEMPLOYMENT:")
        if 'V241488x' in cd.columns:
            print("  ", end="")
            for val, lab in [(1,'Working'), (2,'LaidOff'), (4,'Unempl'), (5,'Retired'), (6,'Disabled'), (7,'Home'), (8,'Student')]:
                pct = wpct(cd, cd['V241488x']==val, weight_col)
                if pct > 0.5: print(f"{lab}={pct:.1f}%  ", end="")
            print()

        # ---- FAMILY & GEOGRAPHY ----
        print(f"\nFAMILY & GEOGRAPHY:")
        if 'V241461x' in cd.columns:
            print(f"  Marital: ", end="")
            for val, lab in [(1,'Married'), (2,'Widowed'), (3,'Divorced'), (4,'Separated'), (5,'NeverMarr')]:
                pct = wpct(cd, cd['V241461x']==val, weight_col)
                if pct > 1: print(f"{lab}={pct:.1f}% ", end="")
            print()
        if 'V243007' in cd.columns:
            print(f"  Region: ", end="")
            for val, lab in [(1,'NE'), (2,'MW'), (3,'South'), (4,'West')]:
                print(f"{lab}={wpct(cd, cd['V243007']==val, weight_col):.1f}% ", end="")
            print()

        # ---- RELIGION ----
        print(f"\nRELIGION:")
        if 'V241445x' in cd.columns:
            print("  ", end="")
            for val, lab in [(1,'MainProt'), (2,'EvanProt'), (3,'BlackProt'), (4,'Cath'), (5,'Chr'), (6,'Jew'), (7,'Oth'), (8,'None')]:
                pct = wpct(cd, cd['V241445x']==val, weight_col)
                if pct > 1: print(f"{lab}={pct:.1f}%  ", end="")
            print()
        
        print("  ", end="")
        if 'V241442' in cd.columns:
            print(f"BornAgain={wpct(cd, cd['V241442']==1, weight_col):.1f}%  ", end="")
        if 'V241420' in cd.columns:
            print(f"ReligImp={wpct(cd, cd['V241420']==1, weight_col):.1f}%  NotImp={wpct(cd, cd['V241420']==5, weight_col):.1f}%  ", end="")
        if 'V241440' in cd.columns:
            print(f"WklyCh={wpct(cd, cd['V241440']==1, weight_col):.1f}%")

        # ---- 2024 VOTE ----
        print(f"\n2024 PRESIDENTIAL VOTE (of voters who took post-survey):")
        if VOTE_COL in cd.columns:
            voters = cd[(cd[TURNOUT_COL] == 4) & (cd[VOTE_COL].notna())]
            if len(voters) > 0:
                vtw = voters[weight_col].sum()
                harris = (voters.loc[voters[VOTE_COL]==1, weight_col].sum() / vtw) * 100
                trump  = (voters.loc[voters[VOTE_COL]==2, weight_col].sum() / vtw) * 100
                other  = (voters.loc[voters[VOTE_COL].isin([4,5,6]), weight_col].sum() / vtw) * 100
                print(f"  Harris={harris:.1f}%  Trump={trump:.1f}%", end="")
                if other > 0.5: print(f"  Other={other:.1f}%", end="")
                print()
            else:
                print("  (Insufficient data due to attrition/non-voting)")

        # =========================================================
        # PART 2: CULTURAL ATTITUDES (Restored from 8b)
        # =========================================================
        print(f"\n--- CULTURAL ATTITUDES (Descriptors) ---")
        
        desc_blocks = {
            'Racial Resentment (1=Agree..5=Disagree)': {
                'V242300': 'Blacks work way up',
                'V242301': 'Slavery conditions',
                'V242302': 'Blacks gotten less',
                'V242303': 'If tried harder',
            },
            'Egalitarianism (1=Agree..5=Disagree)': {
                'V242254': 'Equal opportunity',
                'V242255': 'Worry less about equality',
                'V242256': 'Not big problem if some have more',
                'V242257': 'Treat more equally',
            },
            'Gender & Social': {
                'V242279x': 'Man works/woman home (7pt)',
                'V242361x': 'Sexual harassment attention (5pt)',
            },
        }
        
        for block_name, var_dict in desc_blocks.items():
            print(f"  {block_name}:")
            for v, lab in var_dict.items():
                if v in cd.columns:
                    m = wmean(cd, v, weight_col)
                    print(f"    {lab}: {m:.2f}")

        # =========================================================
        # PART 3: POLICY POSITIONS (With DNA Analysis)
        # =========================================================
        print(f"\n--- POLICY POSITIONS (Pre & Post) ---")
        col_w = 8
        print(f"{'VARIABLE':<55} {'FAVOR':>{col_w}} {'OPPOSE':>{col_w}} {'NEUT':>{col_w}} {'DNA':>{col_w}}")
        print("-" * 95)
        
        for var in clustering_questions:
            if var not in policy_interp: continue
            info = policy_interp[var]
            
            # Label Post-Election vars clearly
            is_post = var not in pre_election_domestic
            prefix = "[POST] " if is_post else "       "
            
            # DNA Logic: Total Weight - Valid Weight
            w_total = tw
            w_valid = cd.loc[cd[var].notna(), weight_col].sum()
            w_dna = w_total - w_valid
            
            # Response Weights
            w_fav = cd.loc[cd[var].isin(info['favor_vals']), weight_col].sum()
            w_opp = cd.loc[cd[var].isin(info['oppose_vals']), weight_col].sum()
            w_neu = 0
            if 'neutral_vals' in info:
                w_neu = cd.loc[cd[var].isin(info['neutral_vals']), weight_col].sum()
                
            # Convert to percentages
            p_fav = (w_fav / w_total) * 100
            p_opp = (w_opp / w_total) * 100
            p_neu = (w_neu / w_total) * 100
            p_dna = (w_dna / w_total) * 100
            
            # Clean Name
            name = (prefix + info['name'])[:55]
            
            # Highlight high DNA/Attrition
            dna_str = f"{p_dna:.1f}%"
            if p_dna > 35: dna_str += " (!)"
            
            print(f"{name:<55} {p_fav:>{col_w}.1f}% {p_opp:>{col_w}.1f}% {p_neu:>{col_w}.1f}% {dna_str:>{col_w}}")

# 7. RUN FULL PROFILE
# ------------------------------------------------------------
print_unified_profile(anes_all)
