"""
cluster_item_profiles.py
------------------------
Produces per-cluster weighted means on all 24 EFA input items PLUS
supplemental demographic/contextual items from the raw CES data.

Inputs:
  typology_cluster_assignments.csv   (N=45,707; cluster labels + weights)
  CCES24_Common_OUTPUT_vv_topost_final.dta  (raw CES data)

Outputs:
  cluster_item_means.csv             (FILE 1: tidy numeric table)
  cluster_item_means_readable.md     (FILE 2: formatted markdown by domain)

Recoding: identical to efa_update.py (REV_BINARY, CC24_325=40-raw, CC24_303 check,
          CC24_423/424 value-8 → 2, ideo5 value-6 → NaN).
Weighting: commonpostweight applied to ALL means and proportions.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path("/Users/bdecker/Documents/STV/Claude")
CES_PATH = Path("/Users/bdecker/Documents/STV/2024 CES Base/"
                "CCES24_Common_OUTPUT_vv_topost_final.dta")
N_CLUSTERS = 10

# ── EFA item list (same order/set as efa_update.py) ──────────────────────────
ITEMS_25 = [
    "pew_churatd", "CC24_302",   "CC24_303",   "CC24_341a",  "CC24_341c",
    "CC24_341d",   "CC24_323a",  "CC24_323b",  "CC24_323d",  "CC24_321b",
    "CC24_321d",   "CC24_321e",  "CC24_325",   "CC24_324b",  "CC24_340a",
    "CC24_340b",   "CC24_340c",  "CC24_340e",  "CC24_340f",  "CC24_440b",
    "CC24_440c",   "CC24_421_1", "CC24_421_2", "CC24_423",   "CC24_424",
]
ITEMS_24 = [it for it in ITEMS_25 if it != "CC24_340a"]

REV_BINARY = {
    "CC24_341c", "CC24_341d",
    "CC24_323a", "CC24_323d",
    "CC24_321e",
    "CC24_340b", "CC24_340c",
}

# ── Additional demographic/contextual columns ─────────────────────────────────
EXTRA_COLS = ["pew_bornagain", "pew_religimp", "gender4", "educ",
              "race", "faminc_new", "employ"]

# ── Region mapping (Census Bureau FIPS) ──────────────────────────────────────
NORTHEAST = {9, 23, 25, 33, 34, 36, 42, 44, 50}
MIDWEST   = {17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55}
SOUTH     = {1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54}
WEST      = {2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56}

# ── Item metadata: (variable, label, scale_note, domain) ─────────────────────
# Items are shown with their POST-RECODE coding.
# For EFA items: direction per brief (LOW = conservative in polychoric matrix;
#                factor scores sign-flipped so HIGH = conservative).
# Here we report means on the recoded item scale without further flipping.
ITEM_META = [
    # ── Guns / Policing ──────────────────────────────────────────────────────
    ("CC24_321d",   "Increase police funding 10%",
     "1=Oppose, 2=Support (HIGH=con)", "Guns/Policing"),
    ("CC24_321e",   "Oppose 10% police budget cut",
     "1=Lib (support cut), 2=Con (oppose cut); HIGH=con", "Guns/Policing"),
    ("CC24_321b",   "Support easier concealed carry permits",
     "1=Oppose, 2=Support (HIGH=con)", "Guns/Policing"),
    # ── Immigration ──────────────────────────────────────────────────────────
    ("CC24_323b",   "Increase border patrols",
     "1=Oppose, 2=Support (HIGH=con)", "Immigration"),
    ("CC24_340f",   "Deny asylum at the border",
     "1=Oppose, 2=Support (HIGH=con)", "Immigration"),
    ("CC24_323a",   "Oppose legal status for working immigrants",
     "1=Lib (support legal status), 2=Con (oppose); HIGH=con", "Immigration"),
    ("CC24_323d",   "Oppose Dreamers pathway to citizenship",
     "1=Lib (support pathway), 2=Con (oppose); HIGH=con", "Immigration"),
    # ── Abortion ─────────────────────────────────────────────────────────────
    ("CC24_325",    "Abortion week limit (40−raw; high=more restrictive)",
     "0–39 (0=most permissive)", "Abortion"),
    ("CC24_324b",   "Permit abortion only rape/incest/life danger",
     "1=Oppose, 2=Support (HIGH=restrictive)", "Abortion"),
    ("CC24_340b",   "Oppose ban on abortion service restrictions",
     "1=Lib (support ban), 2=Con (oppose ban); HIGH=con", "Abortion"),
    # ── Civil Liberties ───────────────────────────────────────────────────────
    ("CC24_340e",   "Renew post-9/11 surveillance programs",
     "1=Oppose, 2=Support (HIGH=con)", "Civil Liberties"),
    ("CC24_340c",   "Oppose same-sex marriage recognition",
     "1=Lib (support SSM), 2=Con (oppose SSM); HIGH=con", "Civil Liberties"),
    # ── Institutional Trust ───────────────────────────────────────────────────
    ("CC24_421_1",  "Disagree: U.S. elections are fair [distrust]",
     "1=Agree (trust), 2=Disagree (distrust)", "Institutional Trust"),
    ("CC24_421_2",  "Disagree: 2024 state/local election was fair [distrust]",
     "1=Agree (trust), 2=Disagree (distrust)", "Institutional Trust"),
    ("CC24_423",    "Federal government trust (high=distrust)",
     "1=Trust, 2=Neutral, 3=Distrust", "Institutional Trust"),
    ("CC24_424",    "State government trust (high=distrust)",
     "1=Trust, 2=Neutral, 3=Distrust", "Institutional Trust"),
    # ── Religion ─────────────────────────────────────────────────────────────
    ("pew_churatd", "Church attendance frequency",
     "1=Never … 6=More than once/week", "Religion"),
    # ── Race / Gender ─────────────────────────────────────────────────────────
    ("CC24_440b",   "Racial problems today are rare and isolated",
     "1=Disagree, 2=Agree (HIGH=racial minimization)", "Race/Gender"),
    ("CC24_440c",   "Women seek to gain power over men",
     "1=Disagree, 2=Agree (HIGH=traditional gender)", "Race/Gender"),
    # ── Economic ─────────────────────────────────────────────────────────────
    ("CC24_341a",   "Support extending 2017 tax cuts",
     "1=Oppose, 2=Support (HIGH=con)", "Economic"),
    ("CC24_341c",   "Oppose raising $400k+ income tax rates",
     "1=Lib (support raise), 2=Con (oppose raise); HIGH=con", "Economic"),
    ("CC24_341d",   "Oppose $150B infrastructure spending",
     "1=Lib (support spending), 2=Con (oppose spending); HIGH=con", "Economic"),
    # ── Economic Context ──────────────────────────────────────────────────────
    ("CC24_303",    "Perceived price change past year (high=inflation)",
     "1–5 scale (higher=more inflation perceived)", "Economic Context"),
    ("CC24_302",    "Household income change past year",
     "1–5 scale (see CCES codebook)", "Economic Context"),
    # ── Supplemental: Demographics ────────────────────────────────────────────
    ("ideo5",            "Ideology self-placement",
     "1=Very Liberal … 5=Very Conservative", "Demographics"),
    ("pew_religimp",     "Importance of religion (flipped: HIGH=more important)",
     "1=Not at all important … 4=Very important; HIGH=more religious", "Religion"),
    # Binary / categorical items below are handled as proportions (% × 100)
    ("pew_bornagain_pct","% Born-again Christian (pew_bornagain=1)",
     "% (weighted)", "Religion"),
    ("educ",             "Education level",
     "1=No HS … 6=Post-grad", "Demographics"),
    ("faminc_new",       "Family income level",
     "1=Under $10k … 16=Over $500k", "Demographics"),
    ("gender_man_pct",   "% Man (gender4=1)",
     "% (weighted)", "Demographics"),
    ("gender_woman_pct", "% Woman (gender4=2)",
     "% (weighted)", "Demographics"),
    ("gender_nb_pct",    "% Non-binary / Other (gender4=3,4)",
     "% (weighted)", "Demographics"),
    ("race_white_pct",   "% White non-Hispanic (race=1)",
     "% (weighted)", "Demographics"),
    ("race_black_pct",   "% Black non-Hispanic (race=2)",
     "% (weighted)", "Demographics"),
    ("race_hispanic_pct","% Hispanic (race=3)",
     "% (weighted)", "Demographics"),
    ("race_asian_pct",   "% Asian (race=4)",
     "% (weighted)", "Demographics"),
    ("race_other_pct",   "% Native Am / Mid. Eastern / Multi / Other (race=5–8)",
     "% (weighted)", "Demographics"),
    ("employ_ft_pct",    "% Employed full-time (employ=1)",
     "% (weighted)", "Demographics"),
    ("employ_pt_pct",    "% Employed part-time (employ=2)",
     "% (weighted)", "Demographics"),
    ("employ_unemp_pct", "% Unemployed (employ=4)",
     "% (weighted)", "Demographics"),
    ("employ_retired_pct","% Retired (employ=5)",
     "% (weighted)", "Demographics"),
    ("employ_other_pct", "% Laid off/Disabled/Homemaker/Student/Other",
     "% (weighted)", "Demographics"),
    ("region_ne_pct",    "% Northeast",
     "% (weighted)", "Demographics"),
    ("region_mw_pct",    "% Midwest",
     "% (weighted)", "Demographics"),
    ("region_south_pct", "% South",
     "% (weighted)", "Demographics"),
    ("region_west_pct",  "% West",
     "% (weighted)", "Demographics"),
]

# ── Helper: weighted mean (NaN-safe) ─────────────────────────────────────────
def wmean(vals, wts):
    mask = ~np.isnan(vals.astype(float))
    if mask.sum() == 0:
        return np.nan
    v, w = vals[mask].astype(float), wts[mask].astype(float)
    return np.sum(v * w) / np.sum(w)

# ── Helper: weighted proportion (% where condition is True) ──────────────────
def wpct(condition, wts):
    mask = ~np.isnan(condition.astype(float))
    if mask.sum() == 0:
        return np.nan
    c, w = condition[mask].astype(float), wts[mask].astype(float)
    return 100.0 * np.sum(c * w) / np.sum(w)


# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. LOAD AND RECODE RAW DTA DATA")
print("=" * 70)

COLS_NEEDED = (ITEMS_24
               + ["pid3", "ideo5", "commonpostweight", "inputstate"]
               + EXTRA_COLS)

# Load only columns that actually exist in the DTA
print(f"  Reading {CES_PATH.name} …")
available = set(pd.read_stata(CES_PATH, iterator=True).variable_labels().keys())
missing_cols = [c for c in COLS_NEEDED if c not in available]
if missing_cols:
    print(f"  WARNING: columns not found in DTA, will be skipped: {missing_cols}")
COLS_NEEDED = [c for c in COLS_NEEDED if c in available]

df_raw = pd.read_stata(CES_PATH, columns=COLS_NEEDED, convert_categoricals=False)
print(f"  Raw rows: {len(df_raw):,}")

# Recode CC24_423/424: value 8 → 2 (midpoint; "Not sure")
for col in ["CC24_423", "CC24_424"]:
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].where(df_raw[col] != 8, other=2)

# Recode ideo5: value 6 → NaN ("Not sure")
if "ideo5" in df_raw.columns:
    df_raw["ideo5"] = df_raw["ideo5"].where(df_raw["ideo5"] != 6, other=np.nan)

# CC24_303 direction check (Pearson with CC24_341a)
mask_v = df_raw[["CC24_303", "CC24_341a", "commonpostweight"]].notna().all(axis=1)
dchk   = df_raw[mask_v]
wv     = dchk["commonpostweight"].values
x303   = dchk["CC24_303"].values.astype(float)
x341a  = dchk["CC24_341a"].values.astype(float)
mu303  = np.sum(wv * x303)  / wv.sum()
mu341a = np.sum(wv * x341a) / wv.sum()
r_raw  = (np.sum(wv * (x303 - mu303) * (x341a - mu341a)) / wv.sum()) / (
          np.sqrt(np.sum(wv * (x303 - mu303)**2) / wv.sum()) *
          np.sqrt(np.sum(wv * (x341a - mu341a)**2) / wv.sum()))
needs_rev_303 = r_raw < 0
print(f"  CC24_303 × CC24_341a weighted r = {r_raw:+.4f}  "
      f"→ {'APPLY 6−CC24_303' if needs_rev_303 else 'no flip needed'}")

df = df_raw.copy()
if needs_rev_303:
    df["CC24_303"] = 6 - df["CC24_303"]
for col in REV_BINARY:
    if col in df.columns:
        df[col] = 3 - df[col]
if "CC24_325" in df.columns:
    df["CC24_325"] = 40 - df["CC24_325"]

# Listwise deletion on 24 EFA items + commonpostweight (same as efa_update.py)
mask_complete = df[ITEMS_24 + ["commonpostweight"]].notna().all(axis=1)
df_c = df[mask_complete].copy().reset_index(drop=True)
print(f"  After listwise deletion: N = {len(df_c):,}")

# ── Normalize profile items to HIGH = conservative ────────────────────────────
# REV_BINARY items were recoded 3-x for the EFA; apply 3-x again so the
# profile table is on the original survey scale (2=conservative, 1=liberal),
# consistent with HIGH=conservative across all items.
print("\n  Normalizing items to HIGH=conservative:")
for col in REV_BINARY:
    if col in df_c.columns:
        df_c[col] = 3 - df_c[col]
        print(f"    {col}: flipped back (3-x → original scale; HIGH=con)")
# pew_religimp: 1=Very important → 4=Not at all; flip so HIGH=more religious
if "pew_religimp" in df_c.columns:
    df_c["pew_religimp"] = 5 - df_c["pew_religimp"]
    print("    pew_religimp: flipped (5-x) → HIGH=more important/religious")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. LOAD CLUSTER ASSIGNMENTS AND VERIFY ALIGNMENT")
print("=" * 70)

cl_df = pd.read_csv(DATA_DIR / "typology_cluster_assignments.csv")
print(f"  Cluster assignments: N = {len(cl_df):,}")

assert len(df_c) == len(cl_df), (
    f"Row count mismatch: raw N={len(df_c)} vs cluster N={len(cl_df)}")

# Cross-check first 10 weights
w_raw = df_c["commonpostweight"].values
w_cl  = cl_df["commonpostweight"].values
max_diff = np.max(np.abs(w_raw[:10] - w_cl[:10]))
print(f"  Weight alignment check (first 10 rows): max abs diff = {max_diff:.2e}")
if max_diff > 1e-6:
    print("  WARNING: weights do not align — check row ordering")
else:
    print("  ✓ Weights align: row ordering confirmed")

df_c["cluster"] = cl_df["cluster"].values
w = df_c["commonpostweight"].values
cl = df_c["cluster"].values
total_w = w.sum()


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. DERIVE SUPPLEMENTAL COLUMNS")
print("=" * 70)

# Region
inputstate = df_c["inputstate"].values.astype(float) if "inputstate" in df_c.columns else None
if inputstate is not None:
    def assign_region(fips):
        f = int(fips) if not np.isnan(fips) else -1
        if f in NORTHEAST: return "Northeast"
        if f in MIDWEST:   return "Midwest"
        if f in SOUTH:     return "South"
        if f in WEST:      return "West"
        return np.nan
    df_c["region"] = [assign_region(f) for f in inputstate]
    print(f"  Region: {df_c['region'].value_counts().to_dict()}")

# pew_bornagain: value 1 = yes
if "pew_bornagain" in df_c.columns:
    df_c["pew_bornagain_bin"] = (df_c["pew_bornagain"] == 1).astype(float)
    df_c.loc[df_c["pew_bornagain"].isna(), "pew_bornagain_bin"] = np.nan


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. COMPUTE WEIGHTED MEANS BY CLUSTER")
print("=" * 70)

cluster_ids = list(range(N_CLUSTERS))

# For each row in ITEM_META, compute overall mean and per-cluster mean
rows_out = []

def get_vals_wts(var, df, w, cl_mask):
    """Return (values_array, weights_array) for a given variable and mask."""
    if var not in df.columns:
        n = cl_mask.sum()
        return np.full(n, np.nan), w[cl_mask]
    return df.loc[cl_mask, var].values, w[cl_mask]

for var, label, scale_note, domain in ITEM_META:
    # Determine the actual source column
    # Handle derived proportion variables
    if var.endswith("_pct"):
        base = var[:-4]  # strip _pct
        # Map to condition
        if base == "pew_bornagain":
            src_col = "pew_bornagain_bin"
            is_pct = True
        elif base.startswith("gender_man"):
            src_col = "_gender_man"
            df_c["_gender_man"] = (df_c.get("gender4", pd.Series(dtype=float)) == 1).astype(float)
            df_c.loc[df_c.get("gender4", pd.Series(dtype=float)).isna(), "_gender_man"] = np.nan
            is_pct = True
        elif base.startswith("gender_woman"):
            src_col = "_gender_woman"
            df_c["_gender_woman"] = (df_c.get("gender4", pd.Series(dtype=float)) == 2).astype(float)
            df_c.loc[df_c.get("gender4", pd.Series(dtype=float)).isna(), "_gender_woman"] = np.nan
            is_pct = True
        elif base.startswith("gender_nb"):
            src_col = "_gender_nb"
            df_c["_gender_nb"] = df_c.get("gender4", pd.Series(dtype=float)).isin([3, 4]).astype(float)
            df_c.loc[df_c.get("gender4", pd.Series(dtype=float)).isna(), "_gender_nb"] = np.nan
            is_pct = True
        elif base.startswith("race_white"):
            src_col = "_race_white"
            df_c["_race_white"] = (df_c.get("race", pd.Series(dtype=float)) == 1).astype(float)
            df_c.loc[df_c.get("race", pd.Series(dtype=float)).isna(), "_race_white"] = np.nan
            is_pct = True
        elif base.startswith("race_black"):
            src_col = "_race_black"
            df_c["_race_black"] = (df_c.get("race", pd.Series(dtype=float)) == 2).astype(float)
            df_c.loc[df_c.get("race", pd.Series(dtype=float)).isna(), "_race_black"] = np.nan
            is_pct = True
        elif base.startswith("race_hispanic"):
            src_col = "_race_hispanic"
            df_c["_race_hispanic"] = (df_c.get("race", pd.Series(dtype=float)) == 3).astype(float)
            df_c.loc[df_c.get("race", pd.Series(dtype=float)).isna(), "_race_hispanic"] = np.nan
            is_pct = True
        elif base.startswith("race_asian"):
            src_col = "_race_asian"
            df_c["_race_asian"] = (df_c.get("race", pd.Series(dtype=float)) == 4).astype(float)
            df_c.loc[df_c.get("race", pd.Series(dtype=float)).isna(), "_race_asian"] = np.nan
            is_pct = True
        elif base.startswith("race_other"):
            src_col = "_race_other"
            df_c["_race_other"] = df_c.get("race", pd.Series(dtype=float)).isin([5, 6, 7, 8]).astype(float)
            df_c.loc[df_c.get("race", pd.Series(dtype=float)).isna(), "_race_other"] = np.nan
            is_pct = True
        elif base.startswith("employ_ft"):
            src_col = "_employ_ft"
            df_c["_employ_ft"] = (df_c.get("employ", pd.Series(dtype=float)) == 1).astype(float)
            df_c.loc[df_c.get("employ", pd.Series(dtype=float)).isna(), "_employ_ft"] = np.nan
            is_pct = True
        elif base.startswith("employ_pt"):
            src_col = "_employ_pt"
            df_c["_employ_pt"] = (df_c.get("employ", pd.Series(dtype=float)) == 2).astype(float)
            df_c.loc[df_c.get("employ", pd.Series(dtype=float)).isna(), "_employ_pt"] = np.nan
            is_pct = True
        elif base.startswith("employ_unemp"):
            src_col = "_employ_unemp"
            df_c["_employ_unemp"] = (df_c.get("employ", pd.Series(dtype=float)) == 4).astype(float)
            df_c.loc[df_c.get("employ", pd.Series(dtype=float)).isna(), "_employ_unemp"] = np.nan
            is_pct = True
        elif base.startswith("employ_retired"):
            src_col = "_employ_retired"
            df_c["_employ_retired"] = (df_c.get("employ", pd.Series(dtype=float)) == 5).astype(float)
            df_c.loc[df_c.get("employ", pd.Series(dtype=float)).isna(), "_employ_retired"] = np.nan
            is_pct = True
        elif base.startswith("employ_other"):
            src_col = "_employ_other"
            df_c["_employ_other"] = df_c.get("employ", pd.Series(dtype=float)).isin([3, 6, 7, 8, 9]).astype(float)
            df_c.loc[df_c.get("employ", pd.Series(dtype=float)).isna(), "_employ_other"] = np.nan
            is_pct = True
        elif base.startswith("region_ne"):
            src_col = "_region_ne"
            df_c["_region_ne"] = (df_c.get("region", pd.Series(dtype=object)) == "Northeast").astype(float)
            is_pct = True
        elif base.startswith("region_mw"):
            src_col = "_region_mw"
            df_c["_region_mw"] = (df_c.get("region", pd.Series(dtype=object)) == "Midwest").astype(float)
            is_pct = True
        elif base.startswith("region_south"):
            src_col = "_region_south"
            df_c["_region_south"] = (df_c.get("region", pd.Series(dtype=object)) == "South").astype(float)
            is_pct = True
        elif base.startswith("region_west"):
            src_col = "_region_west"
            df_c["_region_west"] = (df_c.get("region", pd.Series(dtype=object)) == "West").astype(float)
            is_pct = True
        else:
            src_col = var
            is_pct = False
    else:
        src_col = var
        is_pct = False

    # Compute overall weighted mean
    all_mask = np.ones(len(df_c), dtype=bool)
    if src_col in df_c.columns:
        vals_all = df_c[src_col].values.astype(float)
        overall = wmean(vals_all, w)
    else:
        vals_all = np.full(len(df_c), np.nan)
        overall = np.nan

    # Compute per-cluster weighted mean
    cluster_means = {}
    for k in cluster_ids:
        mask_k = cl == k
        if src_col in df_c.columns:
            vals_k = df_c.loc[mask_k, src_col].values.astype(float)
            cluster_means[k] = wmean(vals_k, w[mask_k])
        else:
            cluster_means[k] = np.nan

    # For binary proportion variables (0/1 → wmean returns 0–1), scale to 0–100
    if var.endswith("_pct"):
        if not np.isnan(overall):
            overall *= 100.0
        cluster_means = {
            k: (v * 100.0 if not np.isnan(v) else np.nan)
            for k, v in cluster_means.items()
        }

    # Range
    cm_vals = [v for v in cluster_means.values() if not np.isnan(v)]
    rng = (max(cm_vals) - min(cm_vals)) if len(cm_vals) >= 2 else np.nan

    row = {
        "variable":    var,
        "label":       label,
        "scale":       scale_note,
        "domain":      domain,
        "overall":     round(overall, 4) if not np.isnan(overall) else np.nan,
    }
    for k in cluster_ids:
        v = cluster_means[k]
        row[f"c{k}"] = round(v, 4) if not np.isnan(v) else np.nan
    row["range"] = round(rng, 4) if not np.isnan(rng) else np.nan
    rows_out.append(row)

profiles_df = pd.DataFrame(rows_out)
print(f"  Computed {len(profiles_df)} item rows × {len(profiles_df.columns)} columns")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. SAVE FILE 1: cluster_item_means.csv")
print("=" * 70)

out1 = DATA_DIR / "clustering_outputs" / "cluster_item_means.csv"
profiles_df.to_csv(out1, index=False)
print(f"  Saved: {out1}")
print(f"  Columns: {list(profiles_df.columns)}")

# Print top-10 discriminating items by range
print("\n  Top 15 items by range (most cluster-differentiating):")
top = profiles_df.dropna(subset=["range"]).nlargest(15, "range")
print(f"  {'Variable':<22}  {'Domain':<20}  {'Overall':>8}  {'Range':>7}")
print(f"  {'─'*22}  {'─'*20}  {'─'*8}  {'─'*7}")
for _, r in top.iterrows():
    print(f"  {r['variable']:<22}  {r['domain']:<20}  {r['overall']:>8.3f}  {r['range']:>7.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. BUILD FILE 2: cluster_item_means_readable.md")
print("=" * 70)

# Cluster column headers (abbreviated)
CL_HDRS = [f"C{k}" for k in range(N_CLUSTERS)]

# Weighted N per cluster (for header)
wn = {k: round(w[cl == k].sum(), 0) for k in range(N_CLUSTERS)}

def fmt_val(val, overall, is_pct=False):
    """Format value with deviation from overall in parentheses."""
    if np.isnan(val) or np.isnan(overall):
        return "—"
    dev = val - overall
    if is_pct:
        return f"{val:.1f} ({dev:+.1f})"
    else:
        return f"{val:.3f} ({dev:+.3f})"

def make_md_table(rows, is_pct_map):
    """
    rows: list of dicts from profiles_df for a given domain
    is_pct_map: dict {variable: bool} — True if value is a %
    Returns markdown table string.
    """
    col_w = 15   # width per cluster column
    var_w  = 22
    lbl_w  = 52
    ov_w   = 9

    # Header
    hdr  = f"| {'Variable':<{var_w}} | {'Label':<{lbl_w}} | {'Overall':>{ov_w}} "
    hdr += "".join(f"| {f'C{k}':^{col_w}} " for k in range(N_CLUSTERS))
    hdr += "| Range |\n"

    sep  = f"| {'-'*var_w} | {'-'*lbl_w} | {'-'*ov_w} "
    sep += "".join(f"| {'-'*col_w} " for k in range(N_CLUSTERS))
    sep += "| ----- |\n"

    body = ""
    for row in rows:
        var = row["variable"]
        lbl = row["label"][:lbl_w]
        ov  = row["overall"]
        rng = row["range"]
        is_pct = is_pct_map.get(var, False)

        if np.isnan(ov):
            ov_str = "n/a"
        elif is_pct:
            ov_str = f"{ov:.1f}%"   # already scaled to 0–100
        else:
            ov_str = f"{ov:.3f}"

        rng_str = f"{rng:.1f}" if (is_pct and not np.isnan(rng)) else (
                  f"{rng:.3f}" if not np.isnan(rng) else "n/a")

        line  = f"| {var:<{var_w}} | {lbl:<{lbl_w}} | {ov_str:>{ov_w}} "
        for k in range(N_CLUSTERS):
            v = row[f"c{k}"]
            if isinstance(v, float) and np.isnan(v):
                cell = "—"
            elif is_pct:
                dev = v - ov if not np.isnan(ov) else float("nan")
                # show as "47.3(+2.1)" — percent with signed deviation in pp
                cell = f"{v:.1f}({dev:+.1f})" if not np.isnan(dev) else f"{v:.1f}"
            else:
                dev = v - ov if not np.isnan(ov) else float("nan")
                cell = f"{v:.3f}({dev:+.3f})" if not np.isnan(dev) else f"{v:.3f}"
            line += f"| {cell:^{col_w}} "
        line += f"| {rng_str:>5} |\n"
        body += line

    return hdr + sep + body

# Build markdown document
md_lines = []
md_lines.append("# CES 2024 — Cluster Item Profile Report\n")
md_lines.append(f"**N = {len(df_c):,}** · Weights: `commonpostweight` · "
                f"DPGMM k=10 (all effective clusters)\n")
md_lines.append(
    "**Reading guide:** Values show weighted mean (or % for categorical items). "
    "Deviation from overall mean in parentheses `(+/-)`. "
    "**Direction: HIGH = conservative across all policy items.** "
    "REV_BINARY items have been re-normalized (3−x twice = original survey scale, 2=con). "
    "`pew_religimp` flipped (5−x) so HIGH = more religious. "
    "Non-ideological context items (`CC24_302` household income, `CC24_303` perceived inflation) "
    "retain original survey direction.\n"
)

# Cluster size header
md_lines.append("## Cluster Weighted N\n")
md_lines.append("| " + " | ".join(f"C{k}" for k in range(N_CLUSTERS)) + " |\n")
md_lines.append("| " + " | ".join("---" for _ in range(N_CLUSTERS)) + " |\n")
md_lines.append("| " + " | ".join(f"{wn[k]:,.0f}" for k in range(N_CLUSTERS)) + " |\n\n")

# Determine is_pct_map
is_pct_map = {var: var.endswith("_pct") for var, *_ in ITEM_META}

# Group by domain and render
domains_order = [
    "Guns/Policing",
    "Immigration",
    "Abortion",
    "Civil Liberties",
    "Economic",
    "Economic Context",
    "Religion",
    "Race/Gender",
    "Institutional Trust",
    "Demographics",
]

profiles_df_meta = profiles_df.copy()

for domain in domains_order:
    domain_rows = profiles_df_meta[profiles_df_meta["domain"] == domain]
    if domain_rows.empty:
        continue

    md_lines.append(f"---\n\n## {domain}\n\n")

    # Add scale note for EFA items
    non_pct_items = domain_rows[~domain_rows["variable"].str.endswith("_pct")]
    if not non_pct_items.empty:
        scales_info = non_pct_items[["variable", "scale"]].values.tolist()
        md_lines.append("**Scale notes:**\n")
        for v, s in scales_info:
            md_lines.append(f"- `{v}`: {s}\n")
        md_lines.append("\n")

    rows_for_table = domain_rows.to_dict("records")
    md_lines.append(make_md_table(rows_for_table, is_pct_map))
    md_lines.append("\n")

# Domains not in any EFA item (environment)
md_lines.append("---\n\n## Environment\n\n")
md_lines.append(
    "*No environmental policy items were included in the 24-item EFA. "
    "Items CC24_309d and CC24_421 variants were excluded at screening "
    "(high missingness / partisan proxy). "
    "Environmental attitudes cannot be profiled from this dataset.*\n\n"
)

out2 = DATA_DIR / "clustering_outputs" / "cluster_item_means_readable.md"
with open(out2, "w", encoding="utf-8") as f:
    f.writelines(md_lines)
print(f"  Saved: {out2}")
print(f"  Lines written: {len(md_lines)}")

print("\n" + "=" * 70)
print("DONE — do not aggregate yet")
print("=" * 70)
