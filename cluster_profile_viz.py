"""
cluster_profile_viz.py
----------------------
Comprehensive per-cluster profiles using the full CES 2024 policy battery.

Original survey valences throughout — no EFA direction-flipping.
  - Binary items (1=Support/2=Oppose): displayed as % Supporting
  - Likert-5 items (1=Strongly Agree … 5=Strongly Disagree): weighted mean + distribution
  - CC24_325 (abortion weeks): mean raw week number
  - Categorical/demographic items: % distribution per category

Ground truth: DTA file (not readme files).

Outputs (to stv_outputs/cluster_profiles/):
  cluster_stats.csv       — item × cluster statistics (wide format)
  cluster_report.html     — static per-cluster breakdown with domain-grouped charts
  cluster_heatmap.html    — interactive Plotly heatmap (domain filter + mode toggle)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path

# ── Check plotly ──────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:
    print("ERROR: plotly not installed. Run: pip3 install plotly")
    sys.exit(1)

# ── Paths & constants ─────────────────────────────────────────────────────────
BASE_DIR      = Path("/Users/bdecker/Documents/STV")
DTA_PATH      = BASE_DIR / "2024 CES Base" / "CCES24_Common_OUTPUT_vv_topost_final.dta"
TYPO_PATH     = BASE_DIR / "Claude" / "typology_cluster_assignments.csv"
OUTPUT_DIR    = BASE_DIR / "Claude" / "stv_outputs" / "cluster_profiles"

# From stv_config — replicate ITEMS_24 inline to avoid import path issues
ITEMS_24 = [
    "pew_churatd", "CC24_302",   "CC24_303",   "CC24_341a",  "CC24_341c",
    "CC24_341d",   "CC24_323a",  "CC24_323b",  "CC24_323d",  "CC24_321b",
    "CC24_321d",   "CC24_321e",  "CC24_325",   "CC24_324b",
    "CC24_340b",   "CC24_340c",  "CC24_340e",  "CC24_340f",  "CC24_440b",
    "CC24_440c",   "CC24_421_1", "CC24_421_2", "CC24_423",   "CC24_424",
]

CLUSTER_NAMES = {
    0: "Conservative",
    1: "Social Democrat",
    2: "Solidarity",
    3: "Nationalist",
    4: "Liberal",
    5: "Reform",
    6: "Center",
    7: "Blue Dogs",
    8: "DSA",
    9: "Progressive",
}

N_CLUSTERS = 10
CLUSTER_COLS = [f"c{i}" for i in range(N_CLUSTERS)]
CLUSTER_LABELS = [f"C{i} {CLUSTER_NAMES[i][:14]}" for i in range(N_CLUSTERS)]

MISSING_STD = {8, 9, 98, 99, 998, 97}

# ── Item metadata ─────────────────────────────────────────────────────────────
# type: "binary" (1=Support, 2=Oppose)
#       "binary_agree" (1=Agree, 2=Disagree)
#       "likert5" (1=Strongly Agree, 5=Strongly Disagree)
#       "approval4" (1=Strongly Approve, 4=Strongly Disapprove, 5=Not sure)
#       "trust" (1=Great deal, 2=Fair amount, 3=Not very much, 8=None at all)
#       "weeks" (numeric week number, 998=missing)
#       "ordinal" (numeric ordinal, direction varies)
#       "categorical" (categorical, % distribution)
#
# heatmap_stat: what scalar to use for the heatmap cell
#   "pct1"  = % where raw==1
#   "mean"  = weighted mean (original scale)
#
# For "binary" items: heatmap shows % supporting (higher = more support for policy)
# For "likert5": heatmap shows mean (lower = more agreement, i.e., 1=SA)
# For "trust": heatmap shows mean (lower = more trusting)
# For "weeks": heatmap shows mean weeks (higher = more permissive)

ITEM_META = {
    # ── Taxes & Economy ───────────────────────────────────────────────────────
    "CC24_341a": {
        "q": "Extend the 2017 tax cuts",
        "domain": "Taxes & Economy", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_341b": {
        "q": "Raise corporate income tax from 21% to 28%",
        "domain": "Taxes & Economy", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_341c": {
        "q": "Allow rates on $400k+ earners to rise to 35%",
        "domain": "Taxes & Economy", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_341d": {
        "q": "Spend $150B/yr on infrastructure",
        "domain": "Taxes & Economy", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_302": {
        "q": "Household income change past year (1=Increased a lot … 5=Decreased a lot)",
        "domain": "Taxes & Economy", "type": "ordinal",
        "cats": {1:"Increased a lot",2:"Increased somewhat",3:"Stayed same",4:"Decreased somewhat",5:"Decreased a lot"},
        "heatmap_stat": "mean", "missing": MISSING_STD,
    },
    "CC24_303": {
        "q": "Price change past year (1=Increased a lot … 5=Decreased a lot)",
        "domain": "Taxes & Economy", "type": "ordinal",
        "cats": {1:"Increased a lot",2:"Increased somewhat",3:"Stayed same",4:"Decreased somewhat",5:"Decreased a lot"},
        "heatmap_stat": "mean", "missing": MISSING_STD,
    },
    # ── Immigration ───────────────────────────────────────────────────────────
    "CC24_323a": {
        "q": "Grant legal status to long-term undocumented immigrants",
        "domain": "Immigration", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_323b": {
        "q": "Increase border patrols",
        "domain": "Immigration", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_323d": {
        "q": "Permanent status/pathway for Dreamers",
        "domain": "Immigration", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_323f": {
        "q": "Forgive up to $20k of student loan debt per person",
        "domain": "Immigration", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_340f": {
        "q": "Deny asylum for illegal border crossings",
        "domain": "Immigration", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    # ── Police & Guns ─────────────────────────────────────────────────────────
    "CC24_321a": {
        "q": "Ban assault rifles",
        "domain": "Police & Guns", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_321b": {
        "q": "Easier concealed-carry permits",
        "domain": "Police & Guns", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_321c": {
        "q": "Require background checks on all gun sales",
        "domain": "Police & Guns", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_321d": {
        "q": "Increase police by 10%",
        "domain": "Police & Guns", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_321e": {
        "q": "Decrease police by 10%, increase other funding",
        "domain": "Police & Guns", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_321f": {
        "q": "Increase spending on mental health & school safety",
        "domain": "Police & Guns", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    # ── Abortion ──────────────────────────────────────────────────────────────
    "CC24_325": {
        "q": "Abortion: how many weeks should it be legal (mean raw weeks)",
        "domain": "Abortion", "type": "weeks", "heatmap_stat": "mean",
        "missing": {998},
    },
    "CC24_324b": {
        "q": "Permit abortion only in rape/incest/life danger cases",
        "domain": "Abortion", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_340b": {
        "q": "Prohibit restrictions on abortion access (Congress bill)",
        "domain": "Abortion", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_444c": {
        "q": "Prohibit abortion-inducing drugs by mail",
        "domain": "Abortion", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_444d": {
        "q": "Prohibit women traveling to another state for abortion",
        "domain": "Abortion", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    # ── Civil Liberties ───────────────────────────────────────────────────────
    "CC24_340c": {
        "q": "Federal recognition of same-sex & interracial marriages",
        "domain": "Civil Liberties", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_340d": {
        "q": "Ban TikTok unless China sells to a US company",
        "domain": "Civil Liberties", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_340e": {
        "q": "Renew post-9/11 surveillance programs",
        "domain": "Civil Liberties", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_444a": {
        "q": "Prevent gender transition surgery for minors",
        "domain": "Civil Liberties", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_444b": {
        "q": "Parental consent for name/pronoun changes at school",
        "domain": "Civil Liberties", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_444e": {
        "q": "Age verification for adult web content",
        "domain": "Civil Liberties", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_444f": {
        "q": "School voucher subsidies",
        "domain": "Civil Liberties", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    # ── Environment & Climate ─────────────────────────────────────────────────
    "CC24_326a": {
        "q": "Give EPA power to regulate CO2 emissions",
        "domain": "Environment & Climate", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_326b": {
        "q": "Require 20% renewable electricity",
        "domain": "Environment & Climate", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_326c": {
        "q": "Strengthen Clean Air/Water Act enforcement (even if it costs jobs)",
        "domain": "Environment & Climate", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_326d": {
        "q": "Increase fossil fuel production in the U.S.",
        "domain": "Environment & Climate", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_326e": {
        "q": "Halt new oil and gas leases on federal lands",
        "domain": "Environment & Climate", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_326f": {
        "q": "Prevent government from banning gas stoves",
        "domain": "Environment & Climate", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    # ── Healthcare & Housing ──────────────────────────────────────────────────
    "CC24_328a": {
        "q": "Relax local zoning laws to allow more apartments/condos",
        "domain": "Healthcare & Housing", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_328b": {
        "q": "Expand federal tax incentives for affordable housing",
        "domain": "Healthcare & Housing", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_328c": {
        "q": "Require able-bodied Medicaid recipients under 64 to work",
        "domain": "Healthcare & Housing", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_328d": {
        "q": "Repeal the Affordable Care Act",
        "domain": "Healthcare & Housing", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_328e": {
        "q": "Expand Medicaid (income <$25k individual / <$40k family)",
        "domain": "Healthcare & Housing", "type": "binary", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    # ── Racial & Gender Attitudes ─────────────────────────────────────────────
    "CC24_440a": {
        "q": "White people have certain advantages because of their race",
        "domain": "Racial & Gender", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_440b": {
        "q": "Racial problems in the U.S. are rare, isolated situations",
        "domain": "Racial & Gender", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_440c": {
        "q": "Women seek to gain power by getting control over men",
        "domain": "Racial & Gender", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_440d": {
        "q": "Women are too easily offended",
        "domain": "Racial & Gender", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_441a": {
        "q": "Minorities should work their way up without special favors",
        "domain": "Racial & Gender", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_441b": {
        "q": "Slavery/discrimination makes it hard for Blacks to rise",
        "domain": "Racial & Gender", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_441e": {
        "q": "I resent when Whites deny racial discrimination exists",
        "domain": "Racial & Gender", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_441f": {
        "q": "Whites get away with offenses that African Americans wouldn't",
        "domain": "Racial & Gender", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_441g": {
        "q": "Whites don't understand the problems African Americans face",
        "domain": "Racial & Gender", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_445a": {
        "q": "SCOTUS: Race in admissions violates Equal Protection (Agree/Disagree)",
        "domain": "Racial & Gender", "type": "binary_agree", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    "CC24_445b": {
        "q": "SCOTUS: Constitution doesn't protect abortion; Roe overruled (Agree/Disagree)",
        "domain": "Racial & Gender", "type": "binary_agree", "heatmap_stat": "pct1",
        "missing": MISSING_STD,
    },
    # ── Elections & Trust ─────────────────────────────────────────────────────
    "CC24_421_1": {
        "q": "Elections in the U.S. are fair",
        "domain": "Elections & Trust", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_421_2": {
        "q": "2024 state/local election was fair",
        "domain": "Elections & Trust", "type": "likert5", "heatmap_stat": "mean",
        "cats": {1:"Strongly Agree",2:"Agree",3:"Neither",4:"Disagree",5:"Strongly Disagree"},
        "missing": MISSING_STD,
    },
    "CC24_423": {
        "q": "Trust in federal government (1=Great deal … 4=None at all)",
        "domain": "Elections & Trust", "type": "trust", "heatmap_stat": "mean",
        "cats": {1:"Great deal",2:"Fair amount",3:"Not very much",4:"None at all"},
        "missing": {98, 99},
    },
    "CC24_424": {
        "q": "Trust in state government (1=Great deal … 4=None at all)",
        "domain": "Elections & Trust", "type": "trust", "heatmap_stat": "mean",
        "cats": {1:"Great deal",2:"Fair amount",3:"Not very much",4:"None at all"},
        "missing": {98, 99},
    },
    # ── Approval ──────────────────────────────────────────────────────────────
    "CC24_312a": {
        "q": "Job approval — President Biden",
        "domain": "Approval", "type": "approval4", "heatmap_stat": "mean",
        "cats": {1:"Strongly approve",2:"Somewhat approve",3:"Somewhat disapprove",4:"Strongly disapprove"},
        "missing": {5, 8, 9, 98, 99},
    },
    "CC24_312b": {
        "q": "Job approval — U.S. Congress",
        "domain": "Approval", "type": "approval4", "heatmap_stat": "mean",
        "cats": {1:"Strongly approve",2:"Somewhat approve",3:"Somewhat disapprove",4:"Strongly disapprove"},
        "missing": {5, 8, 9, 98, 99},
    },
    "CC24_312c": {
        "q": "Job approval — U.S. Supreme Court",
        "domain": "Approval", "type": "approval4", "heatmap_stat": "mean",
        "cats": {1:"Strongly approve",2:"Somewhat approve",3:"Somewhat disapprove",4:"Strongly disapprove"},
        "missing": {5, 8, 9, 98, 99},
    },
    # ── Religion ──────────────────────────────────────────────────────────────
    "pew_churatd": {
        "q": "Church attendance (1=More than once/week … 6=Never)",
        "domain": "Religion", "type": "ordinal", "heatmap_stat": "mean",
        "cats": {1:"More than once/week",2:"Once/week",3:"1–2x/month",4:"Few times/year",5:"Seldom",6:"Never"},
        "missing": {7, 98, 99},
    },
    "pew_religimp": {
        "q": "Importance of religion (1=Very important … 4=Not at all important)",
        "domain": "Religion", "type": "ordinal", "heatmap_stat": "mean",
        "cats": {1:"Very important",2:"Somewhat important",3:"Not too important",4:"Not at all important"},
        "missing": {98, 99},
    },
    "pew_bornagain": {
        "q": "Born-again or evangelical Christian",
        "domain": "Religion", "type": "binary", "heatmap_stat": "pct1",
        "missing": {98, 99},
    },
    # ── Demographics / Identity ───────────────────────────────────────────────
    "ideo5": {
        "q": "Ideological self-identification (1=Very Liberal … 5=Very Conservative)",
        "domain": "Demographics", "type": "ordinal", "heatmap_stat": "mean",
        "cats": {1:"Very Liberal",2:"Liberal",3:"Moderate",4:"Conservative",5:"Very Conservative"},
        "missing": {6, 8, 9},
    },
    "CC24_330a": {
        "q": "Ideology self-placement 7-pt (1=Very Liberal … 7=Very Conservative)",
        "domain": "Demographics", "type": "ordinal", "heatmap_stat": "mean",
        "cats": {1:"Very Liberal",2:"Liberal",3:"Somewhat Liberal",4:"Moderate",
                 5:"Somewhat Conservative",6:"Conservative",7:"Very Conservative"},
        "missing": {8, 9, 98, 99},
    },
    "pid3": {
        "q": "Party identification (3-point)",
        "domain": "Demographics", "type": "categorical",
        "cats": {1:"Democrat",2:"Republican",3:"Independent",4:"Other",5:"Not sure"},
        "heatmap_stat": None, "missing": {98, 99},
    },
    "gender4": {
        "q": "Gender",
        "domain": "Demographics", "type": "categorical",
        "cats": {1:"Man",2:"Woman",3:"Non-binary",4:"Other"},
        "heatmap_stat": None, "missing": {98, 99},
    },
    "race": {
        "q": "Race/ethnicity",
        "domain": "Demographics", "type": "categorical",
        "cats": {1:"White",2:"Black",3:"Hispanic",4:"Asian",5:"Native American",6:"Multiracial",7:"Other",8:"Middle Eastern"},
        "heatmap_stat": None, "missing": {98, 99},
    },
    "educ": {
        "q": "Education level",
        "domain": "Demographics", "type": "categorical",
        "cats": {1:"No HS",2:"HS grad",3:"Some college",4:"2-year degree",5:"4-year degree",6:"Post-grad"},
        "heatmap_stat": None, "missing": {98, 99},
    },
    "urbancity": {
        "q": "Type of area (1=City … 4=Rural)",
        "domain": "Demographics", "type": "categorical",
        "cats": {1:"City",2:"Suburb",3:"Town",4:"Rural area",5:"Other"},
        "heatmap_stat": None, "missing": {98, 99},
    },
    "region": {
        "q": "Region",
        "domain": "Demographics", "type": "categorical",
        "cats": {1:"Northeast",2:"Midwest",3:"South",4:"West"},
        "heatmap_stat": None, "missing": {98, 99},
    },
}

# ── Items for the heatmap (those with a scalar heatmap_stat) ─────────────────
HEATMAP_ITEMS = [v for v, m in ITEM_META.items() if m.get("heatmap_stat") is not None]

# ── Domain order ──────────────────────────────────────────────────────────────
DOMAIN_ORDER = [
    "Taxes & Economy", "Immigration", "Police & Guns", "Abortion",
    "Civil Liberties", "Environment & Climate", "Healthcare & Housing",
    "Racial & Gender", "Elections & Trust", "Approval", "Religion", "Demographics",
]

# ── Stats helpers ─────────────────────────────────────────────────────────────

def _clean(series, missing):
    """Return (values, weights=None) bool mask of non-missing values."""
    return ~series.isin(missing) & series.notna()

def pct1(series, weight, missing):
    """% weighted respondents where raw == 1."""
    mask = _clean(series, missing)
    s, w = series[mask], weight[mask]
    if w.sum() < 1e-9:
        return np.nan
    return float((w[s == 1].sum() / w.sum()) * 100)

def wmean(series, weight, missing):
    """Weighted mean, excluding missing."""
    mask = _clean(series, missing)
    s, w = series[mask].astype(float), weight[mask]
    if w.sum() < 1e-9:
        return np.nan
    return float((s * w).sum() / w.sum())

def wdist(series, weight, cats, missing):
    """Weighted % in each category (dict: cat_int → pct)."""
    mask = _clean(series, missing)
    s, w = series[mask], weight[mask]
    total = w.sum()
    if total < 1e-9:
        return {c: np.nan for c in cats}
    return {c: float((w[s == c].sum() / total) * 100) for c in cats}


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    """
    Load DTA + cluster assignments. Join by positional alignment.
    Returns DataFrame with cluster_id, commonpostweight, all policy/demo columns.
    """
    print("Loading DTA metadata to determine available columns...")
    _, meta = pyreadstat.read_dta(DTA_PATH, metadataonly=True)
    available = set(meta.column_names)

    # All items we want to load
    wanted = set(ITEMS_24)
    for v in ITEM_META:
        wanted.add(v)
    wanted.add("commonpostweight")

    # Filter to those that exist in DTA
    missing_cols = wanted - available
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} columns not in DTA, skipping: {sorted(missing_cols)}")

    read_cols = sorted(wanted & available)
    # Ensure no duplicates
    read_cols = list(dict.fromkeys(read_cols))

    print(f"  Reading {len(read_cols)} columns from DTA...")
    df_dta, _ = pyreadstat.read_dta(DTA_PATH, usecols=read_cols, apply_value_formats=False)
    print(f"  DTA rows: {len(df_dta):,}")

    # Apply listwise deletion (same as STV pipeline)
    efa_present = [c for c in ITEMS_24 if c in df_dta.columns]
    mask_cols = efa_present + ["commonpostweight"]
    mask = df_dta[mask_cols].notna().all(axis=1)
    df = df_dta.loc[mask].reset_index(drop=True)
    print(f"  After listwise deletion: {len(df):,} rows")
    assert len(df) == 45_707, f"Expected 45,707 rows, got {len(df)}"

    # Recode trust items: 8 (None at all) → 4
    for col in ["CC24_423", "CC24_424"]:
        if col in df.columns:
            df[col] = df[col].replace(8, 4)

    # Load cluster assignments
    print("  Loading cluster assignments...")
    df_typo = pd.read_csv(TYPO_PATH)
    assert len(df_typo) == len(df), f"Cluster CSV length mismatch: {len(df_typo)} vs {len(df)}"

    if "cluster_id" in df_typo.columns:
        df["cluster_id"] = df_typo["cluster_id"].values
    else:
        # Compute as argmax of prob columns
        prob_cols = [c for c in df_typo.columns if c.startswith("prob_cluster_")]
        prob_cols = sorted(prob_cols, key=lambda x: int(x.split("_")[-1]))
        df["cluster_id"] = df_typo[prob_cols].values.argmax(axis=1)

    print(f"  Cluster distribution: {df['cluster_id'].value_counts().sort_index().to_dict()}")
    return df


# ── Compute stats ─────────────────────────────────────────────────────────────

def compute_stats(df):
    """
    Compute per-cluster (+ overall) statistics for all items in ITEM_META.
    Returns dict: var → {
        "overall": scalar or {cat: pct},
        "clusters": [scalar_or_dict for c in 0..9],
        "type": ..., "domain": ..., "q": ...
    }
    Also returns cluster_n dict: cluster → N weighted.
    """
    w = df["commonpostweight"]
    stats = {}

    for var, meta in ITEM_META.items():
        if var not in df.columns:
            continue

        s = df[var]
        itype = meta["type"]
        missing = meta["missing"]
        cats = meta.get("cats", {})

        result = {"type": itype, "domain": meta["domain"], "q": meta["q"]}

        if itype in ("binary", "binary_agree"):
            # Scalar: % supporting (code 1)
            overall = pct1(s, w, missing)
            by_cluster = [
                pct1(s[df["cluster_id"] == k], w[df["cluster_id"] == k], missing)
                for k in range(N_CLUSTERS)
            ]
            result["stat_label"] = "% Supporting" if itype == "binary" else "% Agreeing"
            result["overall"] = overall
            result["clusters"] = by_cluster

        elif itype == "likert5":
            overall_mean = wmean(s, w, missing)
            by_cluster_mean = [
                wmean(s[df["cluster_id"] == k], w[df["cluster_id"] == k], missing)
                for k in range(N_CLUSTERS)
            ]
            overall_dist = wdist(s, w, cats.keys(), missing)
            by_cluster_dist = [
                wdist(s[df["cluster_id"] == k], w[df["cluster_id"] == k], cats.keys(), missing)
                for k in range(N_CLUSTERS)
            ]
            result["stat_label"] = "Mean (1=Strongly Agree, 5=Strongly Disagree)"
            result["overall"] = overall_mean
            result["clusters"] = by_cluster_mean
            result["overall_dist"] = overall_dist
            result["clusters_dist"] = by_cluster_dist
            result["cats"] = cats

        elif itype in ("ordinal", "trust", "approval4"):
            overall_mean = wmean(s, w, missing)
            by_cluster_mean = [
                wmean(s[df["cluster_id"] == k], w[df["cluster_id"] == k], missing)
                for k in range(N_CLUSTERS)
            ]
            if cats:
                overall_dist = wdist(s, w, cats.keys(), missing)
                by_cluster_dist = [
                    wdist(s[df["cluster_id"] == k], w[df["cluster_id"] == k], cats.keys(), missing)
                    for k in range(N_CLUSTERS)
                ]
                result["overall_dist"] = overall_dist
                result["clusters_dist"] = by_cluster_dist
                result["cats"] = cats
            result["stat_label"] = "Weighted mean"
            result["overall"] = overall_mean
            result["clusters"] = by_cluster_mean

        elif itype == "weeks":
            week_missing = meta["missing"]
            overall_mean = wmean(s, w, week_missing)
            by_cluster_mean = [
                wmean(s[df["cluster_id"] == k], w[df["cluster_id"] == k], week_missing)
                for k in range(N_CLUSTERS)
            ]
            result["stat_label"] = "Mean weeks"
            result["overall"] = overall_mean
            result["clusters"] = by_cluster_mean

        elif itype == "categorical":
            overall_dist = wdist(s, w, cats.keys(), missing)
            by_cluster_dist = [
                wdist(s[df["cluster_id"] == k], w[df["cluster_id"] == k], cats.keys(), missing)
                for k in range(N_CLUSTERS)
            ]
            result["stat_label"] = "% Distribution"
            result["overall"] = overall_dist
            result["clusters"] = by_cluster_dist
            result["cats"] = cats

        stats[var] = result

    # Cluster Ns
    cluster_n = {k: int((df["cluster_id"] == k).sum()) for k in range(N_CLUSTERS)}
    cluster_wn = {k: float(w[df["cluster_id"] == k].sum()) for k in range(N_CLUSTERS)}
    total_n = int(len(df))

    return stats, cluster_n, cluster_wn, total_n


# ── Build cluster_stats.csv ───────────────────────────────────────────────────

def build_stats_csv(stats, cluster_n):
    """Wide-format CSV: variable × cluster + overall."""
    rows = []
    for var, res in stats.items():
        itype = res["type"]
        if itype in ("binary", "binary_agree", "likert5", "ordinal", "trust", "approval4", "weeks"):
            row = {
                "variable": var,
                "domain": res["domain"],
                "type": itype,
                "stat_label": res["stat_label"],
                "question": res["q"],
                "overall": round(res["overall"], 4) if res["overall"] is not None else None,
            }
            for k in range(N_CLUSTERS):
                v = res["clusters"][k]
                row[f"c{k}"] = round(v, 4) if v is not None and not np.isnan(v) else None
            rows.append(row)

            # Add distribution rows for items that have them
            if "overall_dist" in res:
                cats = res["cats"]
                for cat_code, cat_label in cats.items():
                    drow = {
                        "variable": var,
                        "domain": res["domain"],
                        "type": f"{itype}_dist",
                        "stat_label": f"% {cat_label}",
                        "question": res["q"],
                        "overall": round(res["overall_dist"].get(cat_code, np.nan), 2),
                    }
                    for k in range(N_CLUSTERS):
                        v = res["clusters_dist"][k].get(cat_code, np.nan)
                        drow[f"c{k}"] = round(v, 2) if not np.isnan(v) else None
                    rows.append(drow)

        elif itype == "categorical":
            cats = res["cats"]
            for cat_code, cat_label in cats.items():
                row = {
                    "variable": var,
                    "domain": res["domain"],
                    "type": "categorical_dist",
                    "stat_label": f"% {cat_label}",
                    "question": res["q"],
                    "overall": round(res["overall"].get(cat_code, np.nan), 2),
                }
                for k in range(N_CLUSTERS):
                    v = res["clusters"][k].get(cat_code, np.nan)
                    row[f"c{k}"] = round(v, 2) if not np.isnan(v) else None
                rows.append(row)

    df_out = pd.DataFrame(rows)
    return df_out


# ── Build heatmap HTML ────────────────────────────────────────────────────────

def build_heatmap_html(stats, cluster_n):
    """
    Interactive Plotly heatmap.
    Rows = items (those with scalar heatmap_stat), grouped by domain.
    Cols = clusters.
    Controls: domain dropdown + z-scored ↔ absolute toggle.
    """
    # Collect scalar values for all heatmap items (in domain order)
    ordered_items = []
    for domain in DOMAIN_ORDER:
        for var in HEATMAP_ITEMS:
            if ITEM_META[var]["domain"] == domain and var in stats:
                ordered_items.append(var)

    n_items = len(ordered_items)
    n_cols  = N_CLUSTERS

    # Build value matrix: rows=items, cols=clusters
    vals_abs = np.full((n_items, n_cols), np.nan)
    for i, var in enumerate(ordered_items):
        res = stats[var]
        for k in range(n_cols):
            v = res["clusters"][k]
            if v is not None and not np.isnan(v):
                vals_abs[i, k] = v

    # Z-score per row (across clusters)
    vals_z = np.full_like(vals_abs, np.nan)
    for i in range(n_items):
        row = vals_abs[i, :]
        valid = ~np.isnan(row)
        if valid.sum() > 1:
            mu = np.nanmean(row)
            sigma = np.nanstd(row)
            if sigma > 1e-9:
                vals_z[i, :] = (row - mu) / sigma
            else:
                vals_z[i, :] = 0.0

    # Row labels (short) and hover data
    row_labels = []
    hover_abs = []
    hover_z   = []
    for i, var in enumerate(ordered_items):
        meta = ITEM_META[var]
        short = meta["q"][:55] + ("…" if len(meta["q"]) > 55 else "")
        row_labels.append(short)

        overall = stats[var]["overall"]
        overall_str = f"{overall:.1f}" if overall is not None and not np.isnan(overall) else "N/A"
        stat_label = stats[var]["stat_label"]

        ha_row = []
        hz_row = []
        for k in range(n_cols):
            v_abs = vals_abs[i, k]
            v_z   = vals_z[i, k]
            cluster_lbl = f"C{k} {CLUSTER_NAMES[k]}"
            if not np.isnan(v_abs):
                ha_row.append(f"<b>{cluster_lbl}</b><br>{meta['q']}<br>{stat_label}: {v_abs:.1f}<br>National avg: {overall_str}")
                hz_row.append(f"<b>{cluster_lbl}</b><br>{meta['q']}<br>{stat_label}: {v_abs:.1f}<br>National avg: {overall_str}<br>z = {v_z:+.2f}")
            else:
                ha_row.append(f"<b>{cluster_lbl}</b><br>{meta['q']}<br>N/A")
                hz_row.append(f"<b>{cluster_lbl}</b><br>{meta['q']}<br>N/A")
        hover_abs.append(ha_row)
        hover_z.append(hz_row)

    # Domain boundary line positions (y positions between domain groups)
    domain_breaks = []
    domain_label_pos = {}
    prev_domain = None
    domain_start = {}
    for i, var in enumerate(ordered_items):
        d = ITEM_META[var]["domain"]
        if d != prev_domain:
            if prev_domain is not None:
                domain_breaks.append(i - 0.5)
            domain_start[d] = i
            prev_domain = d
    for d, start in domain_start.items():
        items_in_d = [v for v in ordered_items if ITEM_META[v]["domain"] == d]
        domain_label_pos[d] = start + len(items_in_d) / 2 - 0.5

    # Build per-domain item index sets for dropdown
    domain_masks = {}
    for domain in DOMAIN_ORDER:
        idxs = [i for i, var in enumerate(ordered_items) if ITEM_META[var]["domain"] == domain]
        if idxs:
            domain_masks[domain] = idxs

    def slice_data(idxs, z_data):
        return z_data[idxs, :].tolist()

    def slice_hover(idxs, hover):
        return [hover[i] for i in idxs]

    def slice_labels(idxs):
        return [row_labels[i] for i in idxs]

    # Build figure with initial trace (all items, z-scored)
    all_idxs = list(range(n_items))
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=vals_z.tolist(),
        x=CLUSTER_LABELS,
        y=row_labels,
        hoverinfo="text",
        hovertext=hover_z,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="z-score", thickness=15, len=0.5),
        name="z-scored",
    ))

    # Precompute all domain+mode combinations for updatemenus
    # We'll embed them as JSON and use JavaScript to restyle
    precomputed = {
        "All_z":    {"z": vals_z.tolist(), "y": row_labels, "hover": hover_z},
        "All_abs":  {"z": vals_abs.tolist(), "y": row_labels, "hover": hover_abs},
    }
    for domain, idxs in domain_masks.items():
        precomputed[f"{domain}_z"]   = {
            "z": slice_data(idxs, vals_z),
            "y": slice_labels(idxs),
            "hover": slice_hover(idxs, hover_z),
        }
        precomputed[f"{domain}_abs"] = {
            "z": slice_data(idxs, vals_abs),
            "y": slice_labels(idxs),
            "hover": slice_hover(idxs, hover_abs),
        }

    n_cluster_labels = "\n".join([f"C{k}: N={cluster_n[k]:,}" for k in range(N_CLUSTERS)])

    fig.update_layout(
        title=dict(text="CES 2024 Cluster Policy Profiles", font_size=18),
        height=max(600, 22 * n_items + 150),
        width=1200,
        margin=dict(l=400, r=80, t=80, b=40),
        xaxis=dict(side="top", tickfont_size=11),
        yaxis=dict(autorange="reversed", tickfont_size=10),
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
    )

    # Get the base HTML
    base_html = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True},
    )

    # Inject JavaScript for domain/mode controls and embed precomputed data
    controls_js = f"""
<script>
var _data = {json.dumps(precomputed)};
var _mode = 'z';
var _domain = 'All';

function _getKey() {{ return _domain + '_' + _mode; }}

function updateHeatmap() {{
    var key = _getKey();
    var d = _data[key];
    if (!d) return;

    var colorscale = _mode === 'z' ? 'RdBu' : 'Viridis';
    var zmid_val = _mode === 'z' ? 0 : null;
    var cbar_title = _mode === 'z' ? 'z-score' : 'Value';

    var div = document.getElementById('heatmap-div');
    Plotly.restyle(div, {{
        z: [d.z],
        y: [d.y],
        hovertext: [d.hover],
        colorscale: [colorscale],
        zmid: [zmid_val],
        'colorbar.title': [cbar_title],
    }}, [0]);
    Plotly.relayout(div, {{
        'yaxis.autorange': 'reversed',
        height: Math.max(600, 22 * d.y.length + 150),
    }});
}}

function setDomain(domain) {{
    _domain = domain;
    document.querySelectorAll('.domain-btn').forEach(function(b) {{
        b.classList.toggle('active', b.dataset.domain === domain);
    }});
    updateHeatmap();
}}

function setMode(mode) {{
    _mode = mode;
    document.querySelectorAll('.mode-btn').forEach(function(b) {{
        b.classList.toggle('active', b.dataset.mode === mode);
    }});
    updateHeatmap();
}}
</script>
"""

    domain_buttons_html = "\n".join([
        f'<button class="domain-btn{" active" if d == "All" else ""}" data-domain="{d}" onclick="setDomain(\'{d}\')">{d}</button>'
        for d in ["All"] + DOMAIN_ORDER
        if d == "All" or d in domain_masks
    ])
    mode_buttons_html = """
        <button class="mode-btn active" data-mode="z" onclick="setMode('z')">Z-scored (relative)</button>
        <button class="mode-btn" data-mode="abs" onclick="setMode('abs')">Absolute value</button>
    """

    controls_css = """
<style>
.heatmap-controls { font-family: Arial, sans-serif; padding: 12px; background: #f0f2f5; border-bottom: 1px solid #ddd; }
.heatmap-controls h2 { margin: 0 0 10px 0; font-size: 16px; color: #333; }
.control-row { margin-bottom: 8px; }
.control-row label { font-weight: bold; font-size: 13px; margin-right: 8px; color: #555; }
button.domain-btn, button.mode-btn {
    margin: 2px; padding: 4px 10px; font-size: 12px; cursor: pointer;
    border: 1px solid #bbb; border-radius: 4px; background: #fff; color: #333;
}
button.domain-btn.active, button.mode-btn.active {
    background: #1f77b4; color: white; border-color: #1f77b4;
}
#heatmap-div { margin-top: 0 !important; }
</style>
"""

    controls_html = f"""
{controls_css}
<div class="heatmap-controls">
  <h2>CES 2024 — 10 Cluster Policy Profiles (Interactive Heatmap)</h2>
  <div class="control-row">
    <label>Domain:</label>
    {domain_buttons_html}
  </div>
  <div class="control-row">
    <label>Color mode:</label>
    {mode_buttons_html}
  </div>
  <div style="font-size:11px;color:#888;">
    Z-scored: color shows each cluster's deviation from the cross-cluster mean (row-standardized).<br>
    Absolute: raw % supporting or mean score. Hover for question text and exact values.
  </div>
</div>
"""

    # Inject controls + JS into the HTML
    # Find the plotly div, rename its ID everywhere (div tag + Plotly init script),
    # then inject controls before it.
    insert_before = '<div id="'
    idx = base_html.find(insert_before)
    if idx >= 0:
        # Extract the generated UUID div id
        id_start = idx + len(insert_before)
        div_end = base_html.find('"', id_start)
        div_id = base_html[id_start:div_end]
        # Replace ALL occurrences (div attribute + Plotly.newPlot("uuid", ...) reference)
        base_html = base_html.replace(div_id, "heatmap-div")
        # Now inject controls before the renamed div
        new_idx = base_html.find('<div id="heatmap-div"')
        base_html = (
            base_html[:new_idx]
            + controls_html
            + base_html[new_idx:]
        )
    base_html = base_html.replace("</body>", controls_js + "</body>")

    return base_html


# ── Build report HTML ─────────────────────────────────────────────────────────

def _make_binary_chart(stats, cluster_id, domain):
    """Horizontal bar chart for binary items in a domain, for one cluster."""
    items = [v for v, m in ITEM_META.items()
             if m["domain"] == domain and m["type"] in ("binary", "binary_agree")
             and v in stats]
    if not items:
        return None

    cluster_vals = [stats[v]["clusters"][cluster_id] for v in items]
    overall_vals = [stats[v]["overall"] for v in items]
    labels = [ITEM_META[v]["q"][:60] for v in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cluster_vals,
        y=labels,
        orientation="h",
        name=f"C{cluster_id}",
        marker_color="#1f77b4",
        text=[f"{v:.1f}%" if v is not None and not np.isnan(v) else "N/A"
              for v in cluster_vals],
        textposition="outside",
    ))
    # Overall reference as scatter
    fig.add_trace(go.Scatter(
        x=overall_vals,
        y=labels,
        mode="markers",
        marker=dict(color="gray", size=8, symbol="diamond"),
        name="National avg",
    ))

    fig.update_layout(
        title=dict(text=domain, font_size=13),
        height=max(200, len(items) * 35 + 80),
        width=780,
        margin=dict(l=400, r=80, t=40, b=20),
        xaxis=dict(title="% Supporting", range=[0, 105], ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
    )
    return fig


def _make_likert_chart(stats, cluster_id, domain):
    """Stacked bar for likert5/ordinal/trust items in a domain."""
    items = [v for v, m in ITEM_META.items()
             if m["domain"] == domain
             and m["type"] in ("likert5", "trust", "approval4")
             and "clusters_dist" in stats.get(v, {})
             and v in stats]
    if not items:
        return None

    # Use 5-category color palette
    cat_colors_5 = ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#d6604d"]
    cat_colors_4 = ["#2166ac", "#92c5de", "#f4a582", "#d6604d"]
    # approval reverse: 1=approve (green), 4=disapprove (red)
    cat_colors_appr = ["#4dac26", "#b8e186", "#f4a582", "#d01c8b"]

    fig = go.Figure()
    for i, var in enumerate(items):
        res = stats[var]
        cats = ITEM_META[var].get("cats", {})
        dist = res["clusters_dist"][cluster_id]
        itype = ITEM_META[var]["type"]

        if itype == "approval4":
            colors = cat_colors_appr
        elif len(cats) == 4:
            colors = cat_colors_4
        else:
            colors = cat_colors_5

        label = ITEM_META[var]["q"][:55]
        x_cumul = 0
        for j, (cat_code, cat_label) in enumerate(cats.items()):
            v = dist.get(cat_code, 0)
            if np.isnan(v):
                v = 0
            color = colors[j % len(colors)]
            fig.add_trace(go.Bar(
                x=[v],
                y=[label],
                orientation="h",
                marker_color=color,
                name=cat_label if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate=f"{cat_label}: %{{x:.1f}}%<extra></extra>",
            ))

    fig.update_layout(
        barmode="stack",
        title=dict(text=f"{domain} — Distribution", font_size=13),
        height=max(150, len(items) * 45 + 100),
        width=780,
        margin=dict(l=400, r=80, t=40, b=60),
        xaxis=dict(title="%", range=[0, 105], ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.15, font_size=10),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
    )
    return fig


def _make_weeks_chart(stats, all_cluster_ids):
    """Bar chart showing mean abortion weeks per cluster."""
    if "CC24_325" not in stats:
        return None
    res = stats["CC24_325"]
    vals = [res["clusters"][k] for k in all_cluster_ids]
    overall = res["overall"]
    labels = [f"C{k}" for k in all_cluster_ids]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=vals,
        marker_color=[
            "#2166ac" if (v is not None and not np.isnan(v) and v > (overall or 20)) else "#d6604d"
            for v in vals
        ],
        text=[f"{v:.1f} wks" if v is not None and not np.isnan(v) else "N/A" for v in vals],
        textposition="outside",
    ))
    if overall is not None and not np.isnan(overall):
        fig.add_hline(y=overall, line_dash="dash", line_color="gray",
                      annotation_text=f"National avg: {overall:.1f} wks")
    fig.update_layout(
        title="Abortion: Mean weeks legal",
        height=320, width=780,
        margin=dict(l=60, r=60, t=50, b=40),
        yaxis=dict(title="Weeks", range=[0, 42]),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
    )
    return fig


def _make_demo_chart(stats, cluster_id, var):
    """Stacked horizontal bar for one categorical/ordinal demographic variable."""
    if var not in stats:
        return None
    res = stats[var]
    itype = ITEM_META[var]["type"]
    cats = ITEM_META[var].get("cats", {})

    if itype == "categorical":
        dist_c = res["clusters"][cluster_id]
        dist_o = res["overall"]
    elif "clusters_dist" in res:
        dist_c = res["clusters_dist"][cluster_id]
        dist_o = res["overall_dist"]
    else:
        return None

    labels = list(cats.values())
    vals_c = [dist_c.get(k, 0) or 0 for k in cats.keys()]
    vals_o = [dist_o.get(k, 0) or 0 for k in cats.keys()]

    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f",
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals_c, y=["Cluster"], orientation="h",
        marker_color=palette[:len(labels)],
        text=[f"{v:.0f}%" for v in vals_c],
        textposition="inside",
        hovertext=[f"{l}: {v:.1f}%" for l, v in zip(labels, vals_c)],
        hoverinfo="text",
        showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=vals_o, y=["National"], orientation="h",
        marker_color=palette[:len(labels)],
        text=[f"{v:.0f}%" for v in vals_o],
        textposition="inside",
        opacity=0.5,
        showlegend=False,
        hovertext=[f"{l}: {v:.1f}%" for l, v in zip(labels, vals_o)],
        hoverinfo="text",
    ))

    # Legend annotations
    annotations = []
    for j, label in enumerate(labels):
        annotations.append(dict(
            x=1.01, y=j / max(len(labels), 1),
            xref="paper", yref="paper",
            text=f"<span style='color:{palette[j % len(palette)]}'> ■</span> {label}",
            showarrow=False, font_size=10, xanchor="left",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text=ITEM_META[var]["q"][:70], font_size=11),
        height=130, width=780,
        margin=dict(l=80, r=180, t=35, b=5),
        xaxis=dict(ticksuffix="%", range=[0, 105]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
        annotations=annotations,
    )
    return fig


def build_report_html(stats, cluster_n, cluster_wn, total_n):
    """Build self-contained static report HTML with one section per cluster."""

    DEMO_VARS = ["ideo5", "CC24_330a", "pid3", "gender4", "race", "educ",
                 "urbancity", "region"]
    POLICY_DOMAINS = [d for d in DOMAIN_ORDER if d != "Demographics"]

    figures_js = []  # collected plotly div + JS
    cluster_sections = []

    def fig_to_div(fig, div_id):
        html = fig.to_html(full_html=False, include_plotlyjs=False,
                           div_id=div_id, config={"responsive": True})
        return html

    all_divs = []
    for k in range(N_CLUSTERS):
        name = CLUSTER_NAMES[k]
        n = cluster_n[k]
        section_divs = [f'<div class="cluster-section" id="cluster-{k}">']
        section_divs.append(
            f'<h2>C{k} — {name}</h2>'
            f'<p class="cluster-meta">N = {n:,} respondents'
        )

        # Quick stats row
        for var in ["ideo5", "pid3"]:
            if var in stats and stats[var]["type"] in ("categorical", "ordinal"):
                res = stats[var]
                if res["type"] == "categorical":
                    dist = res["clusters"][k]
                else:
                    dist = res.get("clusters_dist", [None]*N_CLUSTERS)[k] if "clusters_dist" in res else None
                if dist:
                    top_cat_code = max(dist, key=lambda x: dist[x] or 0)
                    top_label = ITEM_META[var]["cats"].get(top_cat_code, str(top_cat_code))
                    top_pct = dist[top_cat_code] or 0
                    section_divs.append(f" | {ITEM_META[var]['q'].split('(')[0].strip()}: {top_label} ({top_pct:.0f}%)")

        section_divs.append("</p>")

        # Policy domain charts
        section_divs.append('<h3>Policy Positions</h3>')
        for domain in POLICY_DOMAINS:
            # Binary items chart
            fig_bin = _make_binary_chart(stats, k, domain)
            if fig_bin:
                div_id = f"fig_bin_c{k}_{domain.replace(' ', '_').replace('&','')}"
                section_divs.append(fig_to_div(fig_bin, div_id))

            # Likert/ordinal distribution chart
            fig_lik = _make_likert_chart(stats, k, domain)
            if fig_lik:
                div_id = f"fig_lik_c{k}_{domain.replace(' ', '_').replace('&','')}"
                section_divs.append(fig_to_div(fig_lik, div_id))

        # Abortion weeks chart (shown once per cluster separately)
        if "CC24_325" in stats:
            fig_weeks = _make_weeks_chart(stats, [k])
            if fig_weeks:
                div_id = f"fig_weeks_c{k}"
                # simple single bar — show mean weeks for this cluster vs national
                res = stats["CC24_325"]
                cluster_val = res["clusters"][k]
                overall_val = res["overall"]
                fig_wk = go.Figure()
                fig_wk.add_trace(go.Bar(
                    x=[cluster_val or 0],
                    y=[f"C{k} {name[:20]}"],
                    orientation="h",
                    marker_color="#1f77b4",
                    text=[f"{cluster_val:.1f} wks" if cluster_val and not np.isnan(cluster_val) else "N/A"],
                    textposition="outside",
                ))
                if overall_val is not None and not np.isnan(overall_val):
                    fig_wk.add_vline(x=overall_val, line_dash="dash", line_color="gray",
                                     annotation_text=f"Natl avg: {overall_val:.1f}")
                fig_wk.update_layout(
                    title="Abortion: Mean weeks legal",
                    height=130, width=780,
                    margin=dict(l=200, r=120, t=40, b=10),
                    xaxis=dict(title="Weeks", range=[0, 44]),
                    plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
                )
                section_divs.append(fig_to_div(fig_wk, div_id))

        # Demographics
        section_divs.append('<h3>Demographics</h3>')
        for var in DEMO_VARS:
            if var in stats:
                fig_d = _make_demo_chart(stats, k, var)
                if fig_d:
                    div_id = f"fig_demo_c{k}_{var}"
                    section_divs.append(fig_to_div(fig_d, div_id))

        section_divs.append("</div>")  # end cluster-section
        cluster_sections.append("\n".join(section_divs))

    # Tab navigation
    tabs_html = "\n".join([
        f'<button class="tab-btn" onclick="showCluster({k})" id="tab-{k}">C{k}: {CLUSTER_NAMES[k][:16]}</button>'
        for k in range(N_CLUSTERS)
    ])

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CES 2024 Cluster Profiles</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f8f9fa; }}
  .header {{ background: #2c3e50; color: white; padding: 16px 24px; }}
  .header h1 {{ margin: 0; font-size: 22px; }}
  .header p {{ margin: 4px 0 0 0; font-size: 13px; opacity: 0.8; }}
  .tab-bar {{ background: #ecf0f1; border-bottom: 2px solid #bdc3c7; padding: 8px 12px; display: flex; flex-wrap: wrap; gap: 4px; }}
  .tab-btn {{ padding: 6px 12px; border: 1px solid #bbb; border-radius: 4px 4px 0 0; cursor: pointer;
              background: white; font-size: 12px; color: #333; }}
  .tab-btn.active {{ background: #1f77b4; color: white; border-color: #1f77b4; }}
  .cluster-section {{ display: none; padding: 16px 24px; }}
  .cluster-section.visible {{ display: block; }}
  .cluster-section h2 {{ color: #2c3e50; border-bottom: 2px solid #1f77b4; padding-bottom: 6px; }}
  .cluster-section h3 {{ color: #555; font-size: 15px; margin-top: 24px; border-bottom: 1px solid #eee; }}
  .cluster-meta {{ color: #666; font-size: 13px; margin-top: -6px; }}
  @media (max-width: 900px) {{ .cluster-section {{ padding: 8px; }} }}
</style>
</head>
<body>
<div class="header">
  <h1>CES 2024 — 10 Cluster Political Profiles</h1>
  <p>Full policy battery · Original survey valences · Weighted by commonpostweight · N = {total_n:,}</p>
</div>
<div class="tab-bar">
{tabs_html}
</div>
<div id="report-body">
{"".join(cluster_sections)}
</div>
<script>
function showCluster(k) {{
  document.querySelectorAll('.cluster-section').forEach(function(el) {{
    el.classList.remove('visible');
  }});
  document.querySelectorAll('.tab-btn').forEach(function(el) {{
    el.classList.remove('active');
  }});
  document.getElementById('cluster-' + k).classList.add('visible');
  document.getElementById('tab-' + k).classList.add('active');
  // Trigger plotly resize for any hidden charts now visible
  setTimeout(function() {{
    window.dispatchEvent(new Event('resize'));
  }}, 100);
}}
// Show first cluster by default
showCluster(0);
</script>
</body>
</html>"""

    return page_html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()

    print("=" * 65)
    print("CLUSTER PROFILE VISUALIZATION")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df = load_data()

    # Compute statistics
    print("\nComputing per-cluster statistics...")
    stats, cluster_n, cluster_wn, total_n = compute_stats(df)
    print(f"  Computed stats for {len(stats)} items")

    # Save CSV
    print("\nBuilding cluster_stats.csv...")
    df_stats = build_stats_csv(stats, cluster_n)
    csv_path = OUTPUT_DIR / "cluster_stats.csv"
    df_stats.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}  ({len(df_stats)} rows)")

    # Heatmap
    print("\nBuilding cluster_heatmap.html...")
    heatmap_html = build_heatmap_html(stats, cluster_n)
    hmap_path = OUTPUT_DIR / "cluster_heatmap.html"
    with open(hmap_path, "w", encoding="utf-8") as f:
        f.write(heatmap_html)
    print(f"  Saved: {hmap_path}  ({len(heatmap_html)//1024} KB)")

    # Report
    print("\nBuilding cluster_report.html...")
    report_html = build_report_html(stats, cluster_n, cluster_wn, total_n)
    report_path = OUTPUT_DIR / "cluster_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"  Saved: {report_path}  ({len(report_html)//1024} KB)")

    print(f"\n{'='*65}")
    print(f"DONE  ({time.time()-t0:.1f}s)")
    print(f"{'='*65}")
    print(f"\nOutputs in: {OUTPUT_DIR}")
    print(f"  cluster_stats.csv")
    print(f"  cluster_heatmap.html  (open in browser)")
    print(f"  cluster_report.html   (open in browser)")


if __name__ == "__main__":
    main()
