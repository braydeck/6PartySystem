"""
Feature Audit for 6-Party Clustering
Evaluates which CES variables should be added to clustering algorithm.
Compares Option A (pre-election only, n=60k) vs Option B (all vars, n=49k)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("FEATURE AUDIT FOR 6-PARTY CLUSTERING")
print("=" * 100)

# ============================================================================
# STEP 1: Load all data
# ============================================================================
print("\n[1/7] Loading CES data...")

# Current clustering features (37 domestic policy questions)
clustering_questions = [
    'CC24_324a', 'CC24_324b', 'CC24_324c', 'CC24_324d', 'CC24_325',
    'CC24_323a', 'CC24_323b', 'CC24_323c', 'CC24_323d', 'CC24_340f',
    'CC24_321a', 'CC24_321b', 'CC24_321c', 'CC24_321d', 'CC24_321e', 'CC24_321f',
    'CC24_326a', 'CC24_326b', 'CC24_326c', 'CC24_326d', 'CC24_326e', 'CC24_326f',
    'CC24_341a', 'CC24_341b', 'CC24_341c', 'CC24_341d',
    'CC24_328a', 'CC24_328b', 'CC24_328c', 'CC24_328d', 'CC24_328e', 'CC24_323f',
    'CC24_340a', 'CC24_340b', 'CC24_340c', 'CC24_340d', 'CC24_340e',
]

# Candidate features to evaluate
pre_election_candidates = [
    'pew_bornagain', 'pew_churatd', 'pew_religimp', 'pew_prayer',
]

post_election_candidates = [
    # Racial attitudes
    'CC24_440a', 'CC24_440b', 'CC24_440c', 'CC24_440d',
    # Racial resentment
    'CC24_441a', 'CC24_441b',
    # Spending priorities
    'CC24_443_1', 'CC24_443_2', 'CC24_443_3', 'CC24_443_4', 'CC24_443_5',
    # State policy
    'CC24_444a', 'CC24_444b', 'CC24_444c', 'CC24_444d', 'CC24_444e', 'CC24_444f',
    # SCOTUS
    'CC24_445a', 'CC24_445b',
    # Military use
    'CC24_420_1', 'CC24_420_2', 'CC24_420_3', 'CC24_420_4', 'CC24_420_5', 'CC24_420_6', 'CC24_420_7',
    # Trust & fairness
    'CC24_421_1', 'CC24_421_2', 'CC24_423', 'CC24_424',
]

all_candidate_features = pre_election_candidates + post_election_candidates

# Feature labels for readability
feature_labels = {
    'pew_bornagain': 'Born-again Christian',
    'pew_churatd': 'Church attendance',
    'pew_religimp': 'Religion importance',
    'pew_prayer': 'Prayer frequency',
    'CC24_440a': 'White privilege exists',
    'CC24_440b': 'Racial problems rare',
    'CC24_440c': 'Women seek power over men',
    'CC24_440d': 'Women too easily offended',
    'CC24_441a': 'Blacks should work up w/o favors',
    'CC24_441b': 'Slavery created conditions',
    'CC24_443_1': 'Spend: Welfare',
    'CC24_443_2': 'Spend: Healthcare',
    'CC24_443_3': 'Spend: Education',
    'CC24_443_4': 'Spend: Law enforcement',
    'CC24_443_5': 'Spend: Transportation',
    'CC24_444a': 'Ban trans surgery for minors',
    'CC24_444b': 'Parental consent pronouns',
    'CC24_444c': 'Ban abortion pills by mail',
    'CC24_444d': 'Ban interstate abortion travel',
    'CC24_444e': 'Age verify adult content',
    'CC24_444f': 'School vouchers',
    'CC24_445a': 'SCOTUS: End affirmative action',
    'CC24_445b': 'SCOTUS: Overturn Roe',
    'CC24_420_1': 'Military: Ensure oil supply',
    'CC24_420_2': 'Military: Destroy terrorist camp',
    'CC24_420_3': 'Military: Intervene genocide',
    'CC24_420_4': 'Military: Spread democracy',
    'CC24_420_5': 'Military: Protect allies',
    'CC24_420_6': 'Military: Help UN',
    'CC24_420_7': 'Military: None of these',
    'CC24_421_1': 'US elections are fair',
    'CC24_421_2': 'Your state election was fair',
    'CC24_423': 'Trust federal government',
    'CC24_424': 'Trust state government',
}

# Load data
all_cols = clustering_questions + all_candidate_features + ['inputstate', 'commonweight', 'commonpostweight', 'tookpost', 'ideo5', 'pid3']
ces_df = pd.read_csv('dataverse_files/CCES24_Common_OUTPUT_vv_topost_final.csv', usecols=all_cols)

# Clean: drop rows missing too many clustering questions
ces_df = ces_df.dropna(subset=clustering_questions, thresh=30)
print(f"  Total respondents after cleaning: {len(ces_df):,}")

# Impute existing clustering questions
imputer = SimpleImputer(strategy='median')
ces_df[clustering_questions] = imputer.fit_transform(ces_df[clustering_questions])

# ============================================================================
# STEP 2: Reproduce current clustering (baseline)
# ============================================================================
print("\n[2/7] Reproducing current 6-cluster baseline...")

X_baseline = ces_df[clustering_questions].values
scaler_baseline = StandardScaler()
X_baseline_scaled = scaler_baseline.fit_transform(X_baseline)

kmeans_baseline = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=300)
baseline_labels = kmeans_baseline.fit_predict(X_baseline_scaled)
ces_df['baseline_cluster'] = baseline_labels

baseline_silhouette = silhouette_score(X_baseline_scaled, baseline_labels, sample_size=10000, random_state=42)
print(f"  Baseline silhouette score (37 features, k=6): {baseline_silhouette:.4f}")
print(f"  Cluster sizes: {np.bincount(baseline_labels)}")

# ============================================================================
# STEP 3: ANOVA F-statistic for each candidate feature
# ============================================================================
print("\n[3/7] Computing ANOVA F-statistics (discriminative power vs current clusters)...")

anova_results = {}
for feat in all_candidate_features:
    valid = ces_df[feat].notna()
    if valid.sum() < 1000:
        continue
    groups = [ces_df.loc[valid & (ces_df['baseline_cluster'] == c), feat].values for c in range(6)]
    groups = [g for g in groups if len(g) > 10]
    if len(groups) >= 2:
        f_stat, p_val = f_oneway(*groups)
        anova_results[feat] = {'F': f_stat, 'p': p_val, 'n_valid': valid.sum()}

# Also compute F-stats for existing features (for comparison)
existing_f_stats = []
for feat in clustering_questions:
    groups = [ces_df.loc[ces_df['baseline_cluster'] == c, feat].values for c in range(6)]
    f_stat, _ = f_oneway(*groups)
    existing_f_stats.append(f_stat)

print(f"\n  Existing features F-stat range: {min(existing_f_stats):.0f} - {max(existing_f_stats):.0f} (mean={np.mean(existing_f_stats):.0f})")
print(f"\n  {'Feature':<40} {'F-stat':>10} {'p-value':>12} {'N valid':>10} {'Label'}")
print(f"  {'-'*40} {'-'*10} {'-'*12} {'-'*10} {'-'*30}")

sorted_anova = sorted(anova_results.items(), key=lambda x: x[1]['F'], reverse=True)
for feat, vals in sorted_anova:
    label = feature_labels.get(feat, feat)
    sig = "***" if vals['p'] < 0.001 else "**" if vals['p'] < 0.01 else "*" if vals['p'] < 0.05 else ""
    print(f"  {feat:<40} {vals['F']:>10.0f} {vals['p']:>10.2e}{sig:>2} {vals['n_valid']:>10,} {label}")

# ============================================================================
# STEP 4: Mutual Information
# ============================================================================
print("\n[4/7] Computing Mutual Information scores...")

# MI between candidate features and baseline clusters
mi_results = {}
for feat in all_candidate_features:
    valid = ces_df[feat].notna()
    if valid.sum() < 1000:
        continue
    mi = mutual_info_score(ces_df.loc[valid, 'baseline_cluster'], 
                           ces_df.loc[valid, feat].astype(int))
    mi_results[feat] = mi

print(f"\n  {'Feature':<40} {'MI Score':>10} {'Label'}")
print(f"  {'-'*40} {'-'*10} {'-'*30}")
for feat, mi in sorted(mi_results.items(), key=lambda x: x[1], reverse=True):
    label = feature_labels.get(feat, feat)
    print(f"  {feat:<40} {mi:>10.4f} {label}")

# ============================================================================
# STEP 5: Option A vs B - Silhouette & BIC comparison across k values
# ============================================================================
print("\n[5/7] Comparing Option A vs B across k=4-8...")

# Option A: 37 existing + 4 religiosity (all 60k)
option_a_features = clustering_questions + pre_election_candidates
ces_df[pre_election_candidates] = SimpleImputer(strategy='median').fit_transform(
    ces_df[pre_election_candidates])

X_a = ces_df[option_a_features].values
scaler_a = StandardScaler()
X_a_scaled = scaler_a.fit_transform(X_a)

# Option B: all features, post-survey respondents only (49k)
post_mask = ces_df['tookpost'] == 2
ces_post = ces_df[post_mask].copy()
option_b_features = clustering_questions + all_candidate_features

# Impute any remaining NaN within post-survey respondents
ces_post[all_candidate_features] = SimpleImputer(strategy='median').fit_transform(
    ces_post[all_candidate_features])

X_b = ces_post[option_b_features].values
scaler_b = StandardScaler()
X_b_scaled = scaler_b.fit_transform(X_b)

print(f"\n  Option A: {len(option_a_features)} features, {X_a_scaled.shape[0]:,} respondents")
print(f"  Option B: {len(option_b_features)} features, {X_b_scaled.shape[0]:,} respondents")

# Test k=4 through k=8
print(f"\n  {'k':>3} | {'--- Option A (41 feat, 60k) ---':^35} | {'--- Option B (70 feat, 49k) ---':^35}")
print(f"  {'':>3} | {'Silhouette':>12} {'BIC':>12} {'Inertia':>10} | {'Silhouette':>12} {'BIC':>12} {'Inertia':>10}")
print(f"  {'-'*3}-+-{'-'*35}-+-{'-'*35}")

results_a = {}
results_b = {}

for k in range(4, 9):
    # Option A
    km_a = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_a = km_a.fit_predict(X_a_scaled)
    sil_a = silhouette_score(X_a_scaled, labels_a, sample_size=10000, random_state=42)
    
    gmm_a = GaussianMixture(n_components=k, random_state=42, covariance_type='diag', max_iter=200)
    gmm_a.fit(X_a_scaled)
    bic_a = gmm_a.bic(X_a_scaled)
    
    results_a[k] = {'silhouette': sil_a, 'bic': bic_a, 'inertia': km_a.inertia_, 'labels': labels_a}
    
    # Option B
    km_b = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_b = km_b.fit_predict(X_b_scaled)
    sil_b = silhouette_score(X_b_scaled, labels_b, sample_size=10000, random_state=42)
    
    gmm_b = GaussianMixture(n_components=k, random_state=42, covariance_type='diag', max_iter=200)
    gmm_b.fit(X_b_scaled)
    bic_b = gmm_b.bic(X_b_scaled)
    
    results_b[k] = {'silhouette': sil_b, 'bic': bic_b, 'inertia': km_b.inertia_, 'labels': labels_b}
    
    print(f"  {k:>3} | {sil_a:>12.4f} {bic_a:>12.0f} {km_a.inertia_:>10.0f} | {sil_b:>12.4f} {bic_b:>12.0f} {km_b.inertia_:>10.0f}")

# Best k for each option
best_k_a_sil = max(results_a, key=lambda k: results_a[k]['silhouette'])
best_k_a_bic = min(results_a, key=lambda k: results_a[k]['bic'])
best_k_b_sil = max(results_b, key=lambda k: results_b[k]['silhouette'])
best_k_b_bic = min(results_b, key=lambda k: results_b[k]['bic'])

print(f"\n  Option A best k: Silhouette={best_k_a_sil}, BIC={best_k_a_bic}")
print(f"  Option B best k: Silhouette={best_k_b_sil}, BIC={best_k_b_bic}")

# ============================================================================
# STEP 6: Stability Analysis (ARI between Options)
# ============================================================================
print("\n[6/7] Stability Analysis (Adjusted Rand Index)...")

# Compare Option A k=6 vs Option B k=6 on the overlapping 49k respondents
labels_a_post = results_a[6]['labels'][post_mask.values]
labels_b_6 = results_b[6]['labels']

ari_ab = adjusted_rand_score(labels_a_post, labels_b_6)
print(f"  ARI between Option A(k=6) and Option B(k=6) on 49k overlap: {ari_ab:.4f}")
print(f"  (1.0 = identical clusters, 0.0 = random, >0.6 = substantial agreement)")

# Also compare baseline (37 feat, k=6) vs Option A (41 feat, k=6)
ari_base_a = adjusted_rand_score(baseline_labels, results_a[6]['labels'])
print(f"  ARI between Baseline(37 feat) and Option A(41 feat): {ari_base_a:.4f}")

# Test stability across different k values for Option B
print(f"\n  Cross-k ARI for Option B:")
for k1 in range(4, 9):
    row = []
    for k2 in range(4, 9):
        if k1 == k2:
            row.append("  1.00")
        else:
            ari = adjusted_rand_score(results_b[k1]['labels'], results_b[k2]['labels'])
            row.append(f"  {ari:.2f}")
    if k1 == 4:
        print(f"       k=4   k=5   k=6   k=7   k=8")
    print(f"  k={k1} {''.join(row)}")

# ============================================================================
# STEP 7: Feature group analysis - which GROUPS of features help most?
# ============================================================================
print("\n[7/7] Feature Group Analysis...")

feature_groups = {
    'Religiosity (4)': pre_election_candidates,
    'Racial attitudes (4)': ['CC24_440a', 'CC24_440b', 'CC24_440c', 'CC24_440d'],
    'Racial resentment (2)': ['CC24_441a', 'CC24_441b'],
    'Spending priorities (5)': ['CC24_443_1', 'CC24_443_2', 'CC24_443_3', 'CC24_443_4', 'CC24_443_5'],
    'State policy (6)': ['CC24_444a', 'CC24_444b', 'CC24_444c', 'CC24_444d', 'CC24_444e', 'CC24_444f'],
    'SCOTUS opinions (2)': ['CC24_445a', 'CC24_445b'],
    'Military use (7)': ['CC24_420_1', 'CC24_420_2', 'CC24_420_3', 'CC24_420_4', 'CC24_420_5', 'CC24_420_6', 'CC24_420_7'],
    'Trust/fairness (4)': ['CC24_421_1', 'CC24_421_2', 'CC24_423', 'CC24_424'],
}

# For each group, add it to baseline and measure improvement
print(f"\n  {'Feature Group':<30} {'Silhouette':>12} {'Change':>10} {'BIC':>14} {'Change':>12}")
print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*14} {'-'*12}")

# Baseline on post-survey respondents only (for fair comparison)
X_base_post = X_baseline_scaled[post_mask.values]
km_base_post = KMeans(n_clusters=6, random_state=42, n_init=10)
base_post_labels = km_base_post.fit_predict(X_base_post)
base_post_sil = silhouette_score(X_base_post, base_post_labels, sample_size=10000, random_state=42)
gmm_base_post = GaussianMixture(n_components=6, random_state=42, covariance_type='diag', max_iter=200)
gmm_base_post.fit(X_base_post)
base_post_bic = gmm_base_post.bic(X_base_post)

print(f"  {'BASELINE (37 features)':<30} {base_post_sil:>12.4f} {'---':>10} {base_post_bic:>14.0f} {'---':>12}")

for group_name, group_features in feature_groups.items():
    # Add this group to baseline
    combined_features = clustering_questions + group_features
    ces_post_group = ces_post.copy()
    ces_post_group[group_features] = SimpleImputer(strategy='median').fit_transform(
        ces_post_group[group_features])
    
    X_group = ces_post_group[combined_features].values
    scaler_group = StandardScaler()
    X_group_scaled = scaler_group.fit_transform(X_group)
    
    km_group = KMeans(n_clusters=6, random_state=42, n_init=10)
    group_labels = km_group.fit_predict(X_group_scaled)
    sil_group = silhouette_score(X_group_scaled, group_labels, sample_size=10000, random_state=42)
    
    gmm_group = GaussianMixture(n_components=6, random_state=42, covariance_type='diag', max_iter=200)
    gmm_group.fit(X_group_scaled)
    bic_group = gmm_group.bic(X_group_scaled)
    
    sil_change = sil_group - base_post_sil
    bic_change = bic_group - base_post_bic
    
    sil_arrow = "+" if sil_change > 0 else ""
    bic_arrow = "+" if bic_change > 0 else ""
    
    print(f"  + {group_name:<28} {sil_group:>12.4f} {sil_arrow}{sil_change:>9.4f} {bic_group:>14.0f} {bic_arrow}{bic_change:>11.0f}")

# Cumulative: add ALL groups
print(f"\n  + ALL GROUPS COMBINED:")
X_all = X_b_scaled  # already computed above
sil_all = silhouette_score(X_all, results_b[6]['labels'], sample_size=10000, random_state=42)
gmm_all = GaussianMixture(n_components=6, random_state=42, covariance_type='diag', max_iter=200)
gmm_all.fit(X_all)
bic_all = gmm_all.bic(X_all)
print(f"  {'ALL 70 features':<30} {sil_all:>12.4f} {sil_all - base_post_sil:>+10.4f} {bic_all:>14.0f} {bic_all - base_post_bic:>+12.0f}")

# ============================================================================
# STEP 8: Profile the Option B k=6 clusters to see if they're meaningfully different
# ============================================================================
print("\n" + "=" * 100)
print("OPTION B CLUSTER PROFILES (k=6, 70 features, 49k respondents)")
print("=" * 100)

ces_post['optb_cluster'] = results_b[6]['labels']

for c in range(6):
    cd = ces_post[ces_post['optb_cluster'] == c]
    n = len(cd)
    tw = cd['commonpostweight'].sum() if 'commonpostweight' in cd.columns and cd['commonpostweight'].notna().any() else cd['commonweight'].sum()
    
    def wpct(mask):
        return (cd.loc[mask, 'commonweight'].sum() / cd['commonweight'].sum()) * 100
    
    print(f"\n  CLUSTER {c} (n={n:,}, {n/len(ces_post)*100:.1f}%)")
    print(f"    Party: Dem={wpct(cd['pid3']==1):.0f}% Rep={wpct(cd['pid3']==2):.0f}% Ind={wpct(cd['pid3']==3):.0f}%")
    mean_ideo = np.average(cd['ideo5'].dropna(), weights=cd.loc[cd['ideo5'].notna(), 'commonweight'])
    print(f"    Ideology: {mean_ideo:.2f} (1=VLib, 5=VCon)")
    
    # Key differentiating features
    print(f"    Born-again: {wpct(cd['pew_bornagain']==1):.0f}%")
    print(f"    Church weekly+: {wpct(cd['pew_churatd']<=2):.0f}%")
    print(f"    White privilege exists (agree): {wpct(cd['CC24_440a']<=2):.0f}%")
    print(f"    Blacks should work up no favors (agree): {wpct(cd['CC24_441a']<=2):.0f}%")
    print(f"    Ban trans surgery minors (support): {wpct(cd['CC24_444a']==1):.0f}%")
    print(f"    School vouchers (support): {wpct(cd['CC24_444f']==1):.0f}%")
    print(f"    US elections fair (agree): {wpct(cd['CC24_421_1']<=2):.0f}%")
    print(f"    Trust fed govt (great deal/fair): {wpct(cd['CC24_423']<=2):.0f}%")
    
    # Original policy positions for comparison
    print(f"    Abortion choice: {wpct(cd['CC24_324a']==1):.0f}%")
    print(f"    Ban assault rifles: {wpct(cd['CC24_321a']==1):.0f}%")
    print(f"    EPA carbon: {wpct(cd['CC24_326a']==1):.0f}%")
    print(f"    Build wall: {wpct(cd['CC24_323c']==1):.0f}%")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
