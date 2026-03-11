#!/usr/bin/env python3
"""
senate_chamber_profile.py
--------------------------
Produces seat-weighted policy and demographic profiles for the simulated senate,
in the same format as blend_stats.csv.

Sources
-------
  blend_stats.csv          - existing profiles for 12 blended senate types
  cluster_stats.csv        - source for 4 pure types + 3 new senate-only blends
  senate_composition.csv   - Condorcet seat counts (senator_label)
  senate_irv_composition.csv - IRV seat counts (winner_label)
  senate_voting_blocs.csv  - 4-bloc assignments per scenario

Output
------
  Claude/outputs/senate/senate_chamber_profile.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE        = Path("/Users/bdecker/Documents/STV")
PROFILE_DIR = BASE / "Claude" / "outputs" / "profiles"
SENATE_DIR  = BASE / "Claude" / "outputs" / "senate"

# ── Pure type columns: label → cluster_stats column ────────────────────────
PURE_TYPES = {
    "CON": "c0",
    "SD":  "c1",
    "STY": "c2",
    "REF": "c5",
}

# ── New senate blends not in blend_stats.csv ────────────────────────────────
# (label, primary_col, secondary_col, w_primary)
NEW_BLENDS = [
    ("CON/REF", "c0", "c5", 0.6900),
    ("CON/NAT", "c0", "c3", 0.5100),
    ("LIB/CTR", "c4", "c6", 0.6000),
]

# ── Existing blends to pull from blend_stats.csv ───────────────────────────
EXISTING_BLENDS = [
    "CON/CTR", "CON/SD", "CON/STY",
    "SD/STY",  "SD/CON", "SD/CTR", "SD/LIB",
    "STY/REF", "STY/SD", "STY/CON",
    "REF/STY",
]

# Column order for output
TYPE_COLS = (
    list(PURE_TYPES.keys()) +
    EXISTING_BLENDS +
    [label for label, *_ in NEW_BLENDS]
)

META_COLS = ["variable", "domain", "type", "stat_label", "question", "overall"]


def weighted_avg(profiles: dict, seats: dict) -> pd.Series:
    """
    Compute seat-weighted average across types present in both dicts.
    profiles: {type_label: Series of numeric values}
    seats:    {type_label: int}
    """
    total = sum(seats.get(t, 0) for t in profiles)
    if total == 0:
        return pd.Series(np.nan, index=next(iter(profiles.values())).index)
    result = sum(
        seats.get(t, 0) * profiles[t]
        for t in profiles
        if seats.get(t, 0) > 0
    )
    return (result / total).round(4)


def main():
    # ── Load base data ──────────────────────────────────────────────────────
    cluster = pd.read_csv(PROFILE_DIR / "cluster_stats.csv")
    blends  = pd.read_csv(PROFILE_DIR / "blend_stats.csv")

    print(f"cluster_stats: {len(cluster)} rows")
    print(f"blend_stats:   {len(blends)} rows")

    # ── Build type profiles DataFrame (same row index as cluster_stats) ─────
    out = cluster[META_COLS].copy()

    # Pure types
    for label, col in PURE_TYPES.items():
        out[label] = cluster[col].round(4)

    # Existing blends — pull directly from blend_stats
    for label in EXISTING_BLENDS:
        out[label] = blends[label]

    # New senate-only blends — compute from cluster_stats
    for label, pc, sc, wp in NEW_BLENDS:
        ws = 1.0 - wp
        out[label] = (wp * cluster[pc] + ws * cluster[sc]).round(4)

    # Dict of type → Series for weighted averaging
    type_profiles = {t: out[t] for t in TYPE_COLS}

    # ── Load seat counts ────────────────────────────────────────────────────
    cond_seats = (
        pd.read_csv(SENATE_DIR / "senate_composition.csv")["senator_label"]
        .value_counts()
        .to_dict()
    )
    irv_seats = (
        pd.read_csv(SENATE_DIR / "senate_irv_composition.csv")["winner_label"]
        .value_counts()
        .to_dict()
    )

    cond_total = sum(cond_seats.values())
    irv_total  = sum(irv_seats.values())
    print(f"\nCondorcet: {cond_total} senators across {len(cond_seats)} types")
    print(f"IRV:       {irv_total} senators across {len(irv_seats)} types")

    # Warn on any type not covered
    for label, seats in cond_seats.items():
        if label not in TYPE_COLS:
            print(f"  ⚠ Condorcet type '{label}' ({seats} seats) not in TYPE_COLS")
    for label, seats in irv_seats.items():
        if label not in TYPE_COLS:
            print(f"  ⚠ IRV type '{label}' ({seats} seats) not in TYPE_COLS")

    # ── Chamber aggregates ──────────────────────────────────────────────────
    out["cond_chamber"] = weighted_avg(type_profiles, cond_seats)
    out["irv_chamber"]  = weighted_avg(type_profiles, irv_seats)

    # ── Voting bloc aggregates ──────────────────────────────────────────────
    blocs_df = pd.read_csv(SENATE_DIR / "senate_voting_blocs.csv")

    for scenario, seats_dict, prefix in [
        ("Condorcet", cond_seats, "cond"),
        ("IRV",       irv_seats,  "irv"),
    ]:
        subset = blocs_df[(blocs_df["scenario"] == scenario) &
                          (blocs_df["n_blocs"] == 4)]
        for _, row in subset.iterrows():
            b = int(row["bloc"])
            members = row["members"].split("|")
            bloc_seats = {m: seats_dict.get(m, 0) for m in members}
            bloc_profiles = {m: type_profiles[m] for m in members if m in type_profiles}
            col = f"{prefix}_bloc{b}"
            out[col] = weighted_avg(bloc_profiles, bloc_seats)

    # ── Assemble final column order ─────────────────────────────────────────
    agg_cols = (
        ["cond_chamber", "irv_chamber"] +
        [f"cond_bloc{b}" for b in range(1, 5)] +
        [f"irv_bloc{b}"  for b in range(1, 5)]
    )
    final_cols = META_COLS + TYPE_COLS + agg_cols
    out = out[final_cols]

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = SENATE_DIR / "senate_chamber_profile.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows × {len(out.columns)} cols → {out_path}")

    # ── Spot-checks ─────────────────────────────────────────────────────────
    # CON should equal c0
    check = out[out["variable"] == "ideo5"] if "ideo5" in out["variable"].values else out.head(1)
    if not check.empty:
        row = check.iloc[0]
        c0_val = cluster.loc[cluster["variable"] == row["variable"], "c0"]
        if not c0_val.empty:
            match = "✓" if abs(row["CON"] - c0_val.iloc[0]) < 0.001 else "✗"
            print(f"\nSpot-check CON == c0 for '{row['variable']}': "
                  f"CON={row['CON']:.4f}  c0={c0_val.iloc[0]:.4f}  {match}")

        print(f"\nChamber ideo5 (Condorcet): {row['cond_chamber']:.3f}")
        print(f"Chamber ideo5 (IRV):       {row['irv_chamber']:.3f}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
