#!/usr/bin/env python3
"""
senate_voting_blocs.py
----------------------
Groups senate members into ideological voting blocs using hierarchical
agglomerative clustering (Ward linkage) in 5D factor space.

Only includes candidate types that actually won seats in each scenario.
Reports 3-, 4-, and 5-bloc divisions for both Condorcet and IRV chambers.

Output
------
  Claude/outputs/senate/senate_voting_blocs.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

BASE    = Path("/Users/bdecker/Documents/STV")
OUT_DIR = BASE / "Claude" / "outputs" / "senate"

FACTOR_COLS = [
    "F1_security_order",
    "F2_electoral_skepticism",
    "F3_government_distrust",
    "F4_religious_traditionalism",
    "F5_populist_conservatism",
]
FACTOR_SHORT = {
    "F1_security_order":           "SecOrd",
    "F2_electoral_skepticism":     "ElecSkep",
    "F3_government_distrust":      "GovtDis",
    "F4_religious_traditionalism": "ReligTrad",
    "F5_populist_conservatism":    "PopCons",
}


def factor_profile(centroid: dict, threshold: float = 0.25) -> str:
    """Readable string for dominant factor directions."""
    signals = []
    for col, val in centroid.items():
        if abs(val) >= threshold:
            direction = "+" if val > 0 else "−"
            signals.append(f"{direction}{FACTOR_SHORT[col]}({val:+.2f})")
    return "  ".join(signals) if signals else "near center (all factors < 0.25)"


def run_scenario(comp_path: Path, label_col: str, scenario_name: str,
                 centroids: pd.DataFrame, prox_wide: pd.DataFrame) -> pd.DataFrame:

    print(f"\n{'═'*62}")
    print(f"  {scenario_name.upper()} SENATE")
    print(f"{'═'*62}")

    # ── Seat counts ────────────────────────────────────────────────────
    comp  = pd.read_csv(comp_path)
    seats = (comp[label_col]
             .value_counts()
             .rename_axis("candidate_label")
             .reset_index(name="seats"))

    # ── Merge with factor centroids ────────────────────────────────────
    df = seats.merge(centroids, on="candidate_label", how="inner")
    missing = set(seats["candidate_label"]) - set(df["candidate_label"])
    if missing:
        print(f"  ⚠ No centroid found for: {missing} — excluded from clustering")

    df = df.sort_values("seats", ascending=False).reset_index(drop=True)
    n  = len(df)
    total = int(df["seats"].sum())
    candidates = df["candidate_label"].tolist()

    print(f"\n  Composition: {total} seats across {n} candidate types")
    print(f"  {'Type':<14} {'Seats':>5}  {'Pct':>5}")
    print(f"  {'─'*14} {'─'*5}  {'─'*5}")
    for _, row in df.iterrows():
        pct = 100 * row["seats"] / total
        print(f"  {row['candidate_label']:<14} {int(row['seats']):>5}  {pct:>4.1f}%")

    # ── Build sub-distance-matrix ─────────────────────────────────────
    sub = prox_wide.reindex(index=candidates, columns=candidates).values  # n×n
    if np.any(np.isnan(sub)):
        # Fill missing pairs with max observed distance
        max_d = np.nanmax(sub)
        sub   = np.where(np.isnan(sub), max_d, sub)
        print("  ⚠ Some pairwise distances missing — filled with max observed distance")
    np.fill_diagonal(sub, 0.0)

    condensed = squareform(sub, checks=False)
    Z = linkage(condensed, method="ward")

    # ── Bloc cuts ─────────────────────────────────────────────────────
    all_rows = []
    for n_blocs in [3, 4, 5]:
        raw_labels = fcluster(Z, n_blocs, criterion="maxclust")
        df["bloc"] = raw_labels

        # Re-number blocs from most-conservative to most-progressive
        # based on seat-weighted mean F5 (PopCons) descending
        bloc_f5 = (df.groupby("bloc")
                     .apply(lambda g: np.average(g["F5_populist_conservatism"],
                                                  weights=g["seats"]))
                     .sort_values(ascending=False))
        rank_map = {old: new for new, old in enumerate(bloc_f5.index, start=1)}
        df["bloc"] = df["bloc"].map(rank_map)

        print(f"\n  ── {n_blocs}-Bloc Division ──────────────────────────────────")

        for b in sorted(df["bloc"].unique()):
            sub_df = df[df["bloc"] == b].sort_values("seats", ascending=False)
            members     = sub_df["candidate_label"].tolist()
            bloc_seats  = int(sub_df["seats"].sum())
            pct_chamber = 100 * bloc_seats / total
            w           = sub_df["seats"].values.astype(float)

            centroid = {col: float(np.average(sub_df[col].values, weights=w))
                        for col in FACTOR_COLS}
            profile  = factor_profile(centroid)

            print(f"\n  Bloc {b}  [{bloc_seats} seats / {pct_chamber:.0f}%]")
            print(f"    Members:  {', '.join(members)}")
            print(f"    Profile:  {profile}")

            all_rows.append({
                "scenario":   scenario_name,
                "n_blocs":    n_blocs,
                "bloc":       b,
                "seats":      bloc_seats,
                "pct_chamber": round(pct_chamber, 1),
                "members":    "|".join(members),
                **{f"centroid_{FACTOR_SHORT[col]}": round(v, 3)
                   for col, v in centroid.items()},
            })

    return pd.DataFrame(all_rows)


def main():
    # ── Load centroids and proximity matrix ───────────────────────────
    centroids = pd.read_csv(
        OUT_DIR / "senate_candidate_factor_centroids.csv"
    )[["candidate_label"] + FACTOR_COLS]

    prox      = pd.read_csv(OUT_DIR / "candidate_proximity.csv")
    prox_wide = prox.pivot(index="candidate_a", columns="candidate_b",
                           values="euclidean_dist")

    # ── Run both scenarios ────────────────────────────────────────────
    results = pd.concat([
        run_scenario(
            OUT_DIR / "senate_composition.csv",
            "senator_label",
            "Condorcet",
            centroids, prox_wide,
        ),
        run_scenario(
            OUT_DIR / "senate_irv_composition.csv",
            "winner_label",
            "IRV",
            centroids, prox_wide,
        ),
    ], ignore_index=True)

    out_path = OUT_DIR / "senate_voting_blocs.csv"
    results.to_csv(out_path, index=False)
    print(f"\n\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
