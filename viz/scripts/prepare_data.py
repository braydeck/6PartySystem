#!/usr/bin/env python3
"""Prepare CSV data from simulation outputs into JSON files for the React viz app."""

import csv
import json
import os
from pathlib import Path
from collections import defaultdict

OUTPUTS  = Path(__file__).parent.parent.parent / "data" / "outputs"
PURE_DIR = OUTPUTS / "pure_only"
RESULTS  = Path(__file__).parent.parent.parent / "results"
DATA_OUT = Path(__file__).parent.parent / "src" / "data"
DATA_OUT.mkdir(parents=True, exist_ok=True)

CLUSTER_TO_PARTY = {
    "0": "CON", "1": "SD", "2": "STY", "3": "NAT",
    "4": "LIB", "5": "REF", "6": "CTR", "8": "DSA", "9": "PRG",
}

# Map platonic candidate short codes to party abbreviations
CANDIDATE_TO_PARTY = {
    "RH": "CON", "MW": "SD", "MRJ": "STY", "BE": "NAT",
    "CO": "LIB", "DH": "REF", "LK": "CTR", "ZN": "DSA", "JR": "PRG",
}

def normalize_candidate_code(code: str) -> str:
    """Convert candidate code to display code: short names → party, underscores → slashes."""
    code = code.strip()
    if code in CANDIDATE_TO_PARTY:
        return CANDIDATE_TO_PARTY[code]
    return code.replace("_", "/")

PARTY_NAMES = {
    "CON": "Conservative", "SD": "Social Democrat", "STY": "Solidarity",
    "NAT": "Nationalist", "LIB": "Liberal", "REF": "Reform",
    "CTR": "Center", "DSA": "Democratic Socialists", "PRG": "Progressive",
}

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(data, name):
    path = DATA_OUT / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  Wrote {path.name} ({os.path.getsize(path):,} bytes)")


# ---------- primary.json ----------
def build_primary():
    rows = read_csv(OUTPUTS / "primary_results_2028.csv")
    centroids = {r["candidate_code"]: r for r in read_csv(OUTPUTS / "candidate_factor_centroids.csv")}

    stages_order = ["After_Retail_Six", "After_Pod_A", "After_Pod_C", "After_Pod_BD"]
    stage_labels = {
        "After_Retail_Six": "Retail + Bench States",
        "After_Pod_A": "After Pod A (West)",
        "After_Pod_C": "After Pod C (South)",
        "After_Pod_BD": "After Pods B+D (Final)",
    }

    by_candidate = defaultdict(dict)
    quota_by_stage = {}
    for row in rows:
        stage = row["winnowing_point"]
        code = row["candidate_code"]
        by_candidate[code][stage] = {
            "voteTotal": float(row["vote_total"]),
            "votePct": float(row["vote_pct"]),
            "status": row["status"],
            "quotaThreshold": float(row["quota_threshold"]),
        }
        quota_by_stage[stage] = float(row["quota_threshold"])

    candidates = []
    for raw_code, stages in by_candidate.items():
        display_code = normalize_candidate_code(raw_code)
        c = centroids.get(raw_code, {})
        name = c.get("candidate_name", display_code) or display_code
        entry = {
            "code": display_code,
            "name": name,
            "F1": float(c.get("F1_security_order", 0)),
            "F2": float(c.get("F2_electoral_skepticism", 0)),
            "F3": float(c.get("F3_government_distrust", 0)),
            "F4": float(c.get("F4_religious_traditionalism", 0)),
            "F5": float(c.get("F5_populist_conservatism", 0)),
            "stages": {s: stages.get(s, {"voteTotal": 0, "votePct": 0, "status": "previously_eliminated", "quotaThreshold": quota_by_stage.get(s, 0)}) for s in stages_order},
        }
        candidates.append(entry)

    output = {
        "stagesOrder": stages_order,
        "stageLabels": stage_labels,
        "quotaByStage": quota_by_stage,
        "candidates": candidates,
    }
    write_json(output, "primary.json")


# ---------- primaryStateWinners.json ----------
# Maps each state's presidential primary winner to pods so the map can reveal by stage.
STAGE_TO_PODS = {
    "After_Retail_Six": {"Retail"},
    "After_Pod_A":       {"Retail", "A"},
    "After_Pod_C":       {"Retail", "A", "C"},
    "After_Pod_BD":      {"Retail", "A", "B", "C", "D"},
}

def build_primary_state_winners():
    state_rows = read_csv(OUTPUTS / "irv" / "irv_presidential_states_2028.csv")
    pod_rows = read_csv(OUTPUTS / "state_pod_assignments.csv")
    profile_rows = read_csv(OUTPUTS / "state_candidate_profiles.csv")

    pod_by_fips = {r["state_fips"].zfill(2): r["pod"] for r in pod_rows}

    # Build per-state first-choice shares from all ~20 primary candidates
    first_choice_by_fips = {}
    for r in profile_rows:
        fips = r["state_fips"].zfill(2)
        fc_cols = [k for k in r.keys() if k.startswith("first_choice_")]
        shares = {}
        for col in fc_cols:
            raw_code = col.replace("first_choice_", "")
            display_code = normalize_candidate_code(raw_code)
            val = float(r.get(col) or 0)
            if val > 0:
                shares[display_code] = shares.get(display_code, 0) + val
        total = sum(shares.values())
        if total > 0:
            first_choice_by_fips[fips] = {k: round(v / total, 4) for k, v in shares.items()}

    out = {}
    for r in state_rows:
        fips = r["state_fips"].zfill(2)
        winner = r["winner_code"].replace("_", "/")
        runner_up = r["runner_up_code"].replace("_", "/")
        pod = pod_by_fips.get(fips, "D")
        out[fips] = {
            "stateAbbr": r["state_abbr"],
            "winnerCode": winner,
            "runnerUpCode": runner_up,
            "pod": pod,
            "nRespondents": int(r["n_respondents"]),
            "shares": first_choice_by_fips.get(fips, {}),
        }
    write_json(out, "primaryStateWinners.json")


# ---------- senate*.json ----------
def _extract_senate_condorcet(rows):
    out = []
    for r in rows:
        out.append({
            "stateFips": r["state_fips"].zfill(2),
            "stateAbbr": r["state_abbr"],
            "senatorCode": r.get("senator_code", ""),
            "senatorLabel": r.get("senator_label", ""),
            "senatorType": r.get("senator_type", ""),
            "primaryCluster": r.get("primary_cluster", ""),
            "secondaryCluster": r.get("secondary_cluster", ""),
        })
    return out


def _extract_senate_irv(rows):
    out = []
    for r in rows:
        out.append({
            "stateFips": r["state_fips"].zfill(2),
            "stateAbbr": r["state_abbr"],
            "senatorCode": r.get("winner_code", ""),
            "senatorLabel": r.get("winner_label", ""),
            "senatorType": r.get("winner_type", ""),
            "primaryCluster": r.get("winner_primary_cluster", ""),
            "secondaryCluster": r.get("winner_secondary_cluster", ""),
        })
    return out


def build_senate():
    cond_rows = read_csv(OUTPUTS / "senate" / "senate_composition.csv")
    irv_rows  = read_csv(OUTPUTS / "senate" / "senate_irv_composition.csv")
    write_json(_extract_senate_condorcet(cond_rows), "senateCondorcet.json")
    write_json(_extract_senate_irv(irv_rows), "senateIRV.json")


def build_senate_pure():
    cond_rows = read_csv(PURE_DIR / "senate" / "senate_composition.csv")
    irv_rows  = read_csv(PURE_DIR / "senate" / "senate_irv_composition.csv")
    write_json(_extract_senate_condorcet(cond_rows), "senateCondorcetPure.json")
    write_json(_extract_senate_irv(irv_rows), "senateIRVPure.json")


# ---------- senateVoteModel.json ----------
def build_senate_vote_model():
    rows = read_csv(RESULTS / "vote_model.csv")

    # Compute SD/CON (Condorcet blended president) signing decisions from chamber profile
    senate_prof_rows = read_csv(OUTPUTS / "senate" / "senate_chamber_profile.csv")
    sdcon_pct = {}
    for r in senate_prof_rows:
        if r.get("stat_label") == "% Supporting" and r.get("variable", "").startswith("CC24_"):
            try:
                sdcon_pct[r["variable"]] = float(r["SD/CON"])
            except (KeyError, ValueError):
                pass

    out = []
    for r in rows:
        var = r["variable"]
        sdcon_support = sdcon_pct.get(var)
        pres_mixed_cond_pct = round(sdcon_support, 2) if sdcon_support is not None else float(r.get("pres_mixed_support_pct", 0))
        pres_mixed_cond_signs = ("SIGN" if sdcon_support > 50 else "VETO") if sdcon_support is not None else r["pres_mixed_signs"]

        out.append({
            "variable": var,
            "domain": r["domain"],
            "question": r["question"],
            "overallPct": float(r["overall_pct"]),
            # Mixed senate scenarios (new keys + legacy aliases for UnifiedBillTable)
            "condMixedProbPass": float(r["senate_cond_prob_pass"]),
            "condMixedVerdict": r["senate_cond_verdict"],
            "irvMixedProbPass": float(r["senate_irv_prob_pass"]),
            "irvMixedVerdict": r["senate_irv_verdict"],
            # Legacy aliases (UnifiedBillTable reads these)
            "condProbPass": float(r["senate_cond_prob_pass"]),
            "condVerdict": r["senate_cond_verdict"],
            "irvProbPass": float(r["senate_irv_prob_pass"]),
            "irvVerdict": r["senate_irv_verdict"],
            # Pure senate scenarios (new)
            "condPureProbPass": float(r["senate_cond_pure_prob_pass"]),
            "condPureVerdict": r["senate_cond_pure_verdict"],
            "irvPureProbPass": float(r["senate_irv_pure_prob_pass"]),
            "irvPureVerdict": r["senate_irv_pure_verdict"],
            # Presidential sign + support pct (used in 3-way comparison table)
            "presMixedSigns": r["pres_mixed_signs"],         # CON/SD (IRV winner)
            "presMixedPct":   float(r.get("pres_mixed_support_pct", 0)),
            "presMixedCondSigns": pres_mixed_cond_signs,     # SD/CON (Condorcet winner)
            "presMixedCondPct": pres_mixed_cond_pct,
            "presPureSigns": r["pres_pure_signs"],           # STY (pure IRV winner)
            "presPurePct":   float(r.get("pres_pure_support_pct", 0)),
        })
    write_json(out, "senateVoteModel.json")


# ---------- houseSeats.json ----------
def build_house_seats():
    rows = read_csv(OUTPUTS / "No_C7_canonical" / "stv_seat_summary.csv")
    out = []
    for r in rows:
        if int(r["party"]) == 7:  # skip Blue Dogs (C7 — merged into adjacent clusters)
            continue
        out.append({
            "party": int(r["party"]),
            "partyName": r["party_name"],
            "urban": int(r["URBAN"]),
            "suburban": int(r["SUBURBAN"]),
            "rural": int(r["RURAL"]),
            "national": int(r["NATIONAL"]),
            "pctNational": float(r["pct_national"]),
        })
    write_json(out, "houseSeats.json")


# ---------- houseVoteModel.json ----------
def build_house_vote_model():
    rows = read_csv(OUTPUTS / "house_vote_model.csv")
    out = []
    for r in rows:
        out.append({
            "variable": r["variable"],
            "domain": r["domain"],
            "question": r["question"],
            "overallPct": float(r["overall_pct"]),
            "probPass": float(r["house_prob_pass"]),
            "verdict": r["house_verdict"],
        })
    write_json(out, "houseVoteModel.json")


# ---------- houseStateMap.json ----------
def build_house_state_map():
    """Aggregate house STV results by state to find plurality party per state."""
    rows = read_csv(OUTPUTS / "No_C7_canonical" / "stv_results_by_district.csv")
    pod_rows = read_csv(OUTPUTS / "state_pod_assignments.csv")
    abbr_by_fips = {r["state_fips"].zfill(2): r["state_abbr"] for r in pod_rows}

    state_seats = defaultdict(lambda: defaultdict(int))
    for row in rows:
        fips = row["state_fips"].zfill(2)
        for i in range(7):
            v = row.get(f"elected_party_{i}", "").strip()
            if v:
                cid = str(int(float(v)))
                party = CLUSTER_TO_PARTY.get(cid, "")
                if party:
                    state_seats[fips][party] += 1

    out = {}
    for fips, counts in state_seats.items():
        total = sum(counts.values())
        plurality = max(counts, key=counts.get)
        out[fips] = {
            "stateAbbr": abbr_by_fips.get(fips, fips),
            "pluralityParty": plurality,
            "totalSeats": total,
            "seats": dict(counts),
        }
    write_json(out, "houseStateMap.json")


# ---------- coalitionProfiles.json ----------
def build_coalition_profiles():
    rows = read_csv(OUTPUTS / "coalitions" / "coalition_type_profiles.csv")
    out = []
    for r in rows:
        out.append({
            "type": r["type"],
            "chamber": r["chamber"],
            "F1": float(r["F1_security_order"]),
            "F2": float(r["F2_electoral_skepticism"]),
            "F3": float(r["F3_government_distrust"]),
            "F4": float(r["F4_religious_traditionalism"]),
            "F5": float(r["F5_populist_conservatism"]),
            "seatsHouse": int(r["seats_house"]),
            "seatsSenateCondorcet": int(r.get("seats_senate_cond", 0)),
            "seatsSenateIRV": int(r.get("seats_senate_irv", 0)),
        })
    write_json(out, "coalitionProfiles.json")


# ---------- transferMatrix.json ----------
def build_transfer_matrix():
    rows = read_csv(OUTPUTS / "No_C7_canonical" / "transfer_matrix_10party.csv")
    parties = [k for k in rows[0].keys() if k != "party_a"]
    matrix = {}
    for row in rows:
        src = row["party_a"]
        if not src:
            continue
        matrix[src] = {p: float(row[p]) for p in parties if row.get(p, "0") != "0"}
    write_json({"parties": parties, "matrix": matrix}, "transferMatrix.json")


# ---------- clusterProfiles.json ----------
KEY_VARS = [
    "CC24_341a", "CC24_341b", "CC24_341c", "CC24_341d",
    "CC24_321a", "CC24_321b", "CC24_321c", "CC24_321d",
    "CC24_322a", "CC24_322b", "CC24_323a", "CC24_323b",
    "CC24_324b", "CC24_327a", "CC24_327b",
    "CC24_421_1", "CC24_421_2", "CC24_423", "CC24_424",
    "CC24_440a", "pew_churatd",
]

def compute_key_positions(rows, cid, n=4):
    """Return top-n data-driven policy positions that most differentiate this cluster."""
    binary_rows = [r for r in rows if r["type"] == "binary"]
    diffs = []
    for r in binary_rows:
        try:
            overall = float(r["overall"])
            val = float(r[f"c{cid}"]) if r.get(f"c{cid}") else overall
            diff = val - overall
            diffs.append((abs(diff), diff, r["question"], val))
        except (ValueError, KeyError):
            pass
    diffs.sort(reverse=True)
    out = []
    for _, diff, question, pct in diffs[:n]:
        out.append({
            "question": question,
            "pct": round(pct, 1),
            "direction": "supports" if diff > 0 else "opposes",
            "diffPp": round(diff, 1),
        })
    return out


def compute_key_positions_vs_neighbors(rows, cid, cluster_factors, n=4, min_diff=15):
    """Return top-n positions most distinguishing this cluster from its 2 nearest neighbors.
    Falls back to overall-diff approach if fewer than n positions pass the threshold."""
    me = cluster_factors.get(cid)
    if not me:
        return compute_key_positions(rows, cid, n)

    factor_keys = ["F1", "F2", "F3", "F4", "F5"]
    distances = []
    for other_cid, other in cluster_factors.items():
        if other_cid == cid:
            continue
        dist = sum((me[f] - other[f]) ** 2 for f in factor_keys) ** 0.5
        distances.append((dist, other_cid))
    distances.sort()
    neighbor_cids = [c2 for _, c2 in distances[:2]]

    binary_rows = [r for r in rows if r["type"] == "binary"]
    diffs = []
    for r in binary_rows:
        try:
            val = float(r[f"c{cid}"])
            neighbor_vals = [float(r[f"c{nc}"]) for nc in neighbor_cids if r.get(f"c{nc}")]
            if not neighbor_vals:
                continue
            avg_neighbor = sum(neighbor_vals) / len(neighbor_vals)
            diff = val - avg_neighbor
            if abs(diff) >= min_diff:
                diffs.append((abs(diff), diff, r["question"], val))
        except (ValueError, KeyError):
            pass
    diffs.sort(reverse=True)
    out = []
    for _, diff, question, pct in diffs[:n]:
        out.append({
            "question": question,
            "pct": round(pct, 1),
            "direction": "supports" if diff > 0 else "opposes",
            "diffPp": round(diff, 1),
        })
    # Fall back to overall-diff for any remaining slots
    if len(out) < n:
        seen = {p["question"] for p in out}
        fallback = compute_key_positions(rows, cid, n * 2)
        for pos in fallback:
            if pos["question"] not in seen:
                out.append(pos)
                seen.add(pos["question"])
            if len(out) >= n:
                break
    return out

def build_cluster_profiles():
    rows = read_csv(OUTPUTS / "profiles" / "cluster_stats.csv")
    clusters = {str(i): {"id": str(i), "variables": {}} for i in range(10) if str(i) != "7"}

    for row in rows:
        var = row["variable"]
        if var not in KEY_VARS:
            continue
        for cid in clusters:
            val = row.get(f"c{cid}", "")
            if val:
                clusters[cid]["variables"][var] = {
                    "pct": float(val),
                    "question": row.get("question", var),
                    "domain": row.get("domain", ""),
                }

    coalition_rows = read_csv(OUTPUTS / "coalitions" / "coalition_type_profiles.csv")
    party_to_cluster = {v: k for k, v in CLUSTER_TO_PARTY.items()}
    cluster_factors = {}
    for r in coalition_rows:
        party = r["type"]
        cid = party_to_cluster.get(party)
        if cid and cid in clusters:
            clusters[cid]["party"] = party
            clusters[cid]["partyName"] = PARTY_NAMES.get(party, party)
            clusters[cid]["F1"] = float(r["F1_security_order"])
            clusters[cid]["F2"] = float(r["F2_electoral_skepticism"])
            clusters[cid]["F3"] = float(r["F3_government_distrust"])
            clusters[cid]["F4"] = float(r["F4_religious_traditionalism"])
            clusters[cid]["F5"] = float(r["F5_populist_conservatism"])
            clusters[cid]["seatsHouse"] = int(r["seats_house"])
            cluster_factors[cid] = {
                "F1": float(r["F1_security_order"]),
                "F2": float(r["F2_electoral_skepticism"]),
                "F3": float(r["F3_government_distrust"]),
                "F4": float(r["F4_religious_traditionalism"]),
                "F5": float(r["F5_populist_conservatism"]),
            }

    # Fix pew_churatd: store % weekly+ attendance instead of last row (% Never)
    church_weekly_more = {}
    church_weekly = {}
    for row in rows:
        if row["variable"] == "pew_churatd":
            label = row.get("stat_label", "")
            if "More than once/week" in label:
                for cid in clusters:
                    church_weekly_more[cid] = float(row.get(f"c{cid}") or 0)
            elif "Once/week" in label and "More" not in label:
                for cid in clusters:
                    church_weekly[cid] = float(row.get(f"c{cid}") or 0)
    for cid in clusters:
        if "pew_churatd" in clusters[cid]["variables"]:
            weekly_total = church_weekly_more.get(cid, 0) + church_weekly.get(cid, 0)
            clusters[cid]["variables"]["pew_churatd"]["pct"] = round(weekly_total, 1)
            clusters[cid]["variables"]["pew_churatd"]["question"] = "Weekly church attendance"

    # Add key positions vs nearest neighbors
    for cid in clusters:
        clusters[cid]["keyPositions"] = compute_key_positions_vs_neighbors(rows, cid, cluster_factors)

    write_json(list(clusters.values()), "clusterProfiles.json")


# ---------- blendProfiles.json ----------
def build_blend_profiles():
    """Build profiles for blended senator types that appear in the senate simulations."""
    blend_rows = read_csv(OUTPUTS / "profiles" / "blend_stats.csv")
    cluster_rows = read_csv(OUTPUTS / "profiles" / "cluster_stats.csv")
    centroid_rows = read_csv(OUTPUTS / "senate" / "senate_candidate_factor_centroids.csv")
    senate_prof_rows = read_csv(OUTPUTS / "senate" / "senate_chamber_profile.csv")
    cond_rows = read_csv(OUTPUTS / "senate" / "senate_composition.csv")
    irv_rows = read_csv(OUTPUTS / "senate" / "senate_irv_composition.csv")
    # Also read pure senate compositions to capture parties that only appear there (e.g. REF)
    pure_cond_rows = read_csv(PURE_DIR / "senate" / "senate_composition.csv")
    pure_irv_rows  = read_csv(PURE_DIR / "senate" / "senate_irv_composition.csv")

    # Cluster-to-party mapping for pure party key positions
    PARTY_TO_CID = {'CON': '0', 'SD': '1', 'STY': '2', 'NAT': '3',
                    'LIB': '4', 'REF': '5', 'CTR': '6', 'DSA': '8', 'PRG': '9'}

    # Factor scores indexed by blend label
    centroids = {r["candidate_label"]: r for r in centroid_rows}

    # Count senate seats per code (blended scenarios)
    seat_counts_cond = defaultdict(int)
    for r in cond_rows:
        seat_counts_cond[r["senator_code"]] += 1
    seat_counts_irv = defaultdict(int)
    for r in irv_rows:
        seat_counts_irv[r["winner_code"]] += 1

    # All unique blended codes (contain '/') across both chambers
    all_codes = set()
    for r in cond_rows:
        if "/" in r["senator_code"]:
            all_codes.add(r["senator_code"])
    for r in irv_rows:
        if "/" in r["winner_code"]:
            all_codes.add(r["winner_code"])

    # Blend stats has column names like 'CON/CTR', 'CON/SD', etc.
    blend_cols = [k for k in blend_rows[0].keys() if "/" in k] if blend_rows else []
    # Senate chamber profile binary rows — fallback for codes not in blend_stats
    senate_binary_rows = [r for r in senate_prof_rows
                          if r.get("stat_label") == "% Supporting"
                          and r.get("variable", "").startswith("CC24_")]

    def _diffs_from_rows(rows, col, overall_col="overall"):
        diffs = []
        for r in rows:
            try:
                overall = float(r[overall_col])
                val = float(r[col])
                diff = val - overall
                diffs.append((abs(diff), diff, r["question"], val))
            except (ValueError, KeyError):
                pass
        diffs.sort(reverse=True)
        return diffs

    def compute_blend_positions(blend_code, n=4):
        # Prefer blend_stats columns; fall back to senate_chamber_profile
        if blend_code in blend_cols:
            binary = [r for r in blend_rows if r["type"] == "binary"]
            diffs = _diffs_from_rows(binary, blend_code)
        elif senate_binary_rows and blend_code in (senate_binary_rows[0] if senate_binary_rows else {}):
            diffs = _diffs_from_rows(senate_binary_rows, blend_code)
        else:
            return []
        pos_out = []
        for _, diff, question, pct in diffs[:n]:
            pos_out.append({
                "question": question,
                "pct": round(pct, 1),
                "direction": "supports" if diff > 0 else "opposes",
                "diffPp": round(diff, 1),
            })
        return pos_out

    def compute_blend_variables(blend_code, max_vars=40):
        """Return top-max_vars most differentiating binary variables for a blended party."""
        if blend_code not in blend_cols:
            return {}
        binary = [r for r in blend_rows if r["type"] == "binary"]
        result = {}
        for r in binary:
            try:
                overall = float(r["overall"])
                val = float(r[blend_code])
                diff = val - overall
                result[r["variable"]] = {
                    "pct": round(val, 1),
                    "question": r["question"],
                    "domain": r.get("domain", ""),
                    "diffPp": round(diff, 1),
                }
            except (ValueError, KeyError):
                pass
        sorted_vars = sorted(result.items(), key=lambda x: abs(x[1]["diffPp"]), reverse=True)
        return dict(sorted_vars[:max_vars])

    def compute_pure_blend_variables(cid, max_vars=40):
        """Return top-max_vars most differentiating binary variables for a pure-party cluster."""
        binary = [r for r in cluster_rows if r["type"] == "binary"]
        result = {}
        for r in binary:
            try:
                overall = float(r["overall"])
                val = float(r[f"c{cid}"])
                diff = val - overall
                result[r["variable"]] = {
                    "pct": round(val, 1),
                    "question": r["question"],
                    "domain": r.get("domain", ""),
                    "diffPp": round(diff, 1),
                }
            except (ValueError, KeyError):
                pass
        sorted_vars = sorted(result.items(), key=lambda x: abs(x[1]["diffPp"]), reverse=True)
        return dict(sorted_vars[:max_vars])

    out = []
    for code in sorted(all_codes):
        c = centroids.get(code, {})
        profile = {
            "code": code,
            "seatsCond": seat_counts_cond.get(code, 0),
            "seatsIRV": seat_counts_irv.get(code, 0),
            "F1": float(c.get("F1_security_order", 0)) if c else 0,
            "F2": float(c.get("F2_electoral_skepticism", 0)) if c else 0,
            "F3": float(c.get("F3_government_distrust", 0)) if c else 0,
            "F4": float(c.get("F4_religious_traditionalism", 0)) if c else 0,
            "F5": float(c.get("F5_populist_conservatism", 0)) if c else 0,
            "keyPositions": compute_blend_positions(code),
            "variables": compute_blend_variables(code),
        }
        out.append(profile)

    # Collect all pure-party codes across blended AND pure senate compositions
    pure_codes = set()
    for r in cond_rows + pure_cond_rows:
        code = r.get("senator_code", "")
        if code and "/" not in code:
            pure_codes.add(code)
    for r in irv_rows + pure_irv_rows:
        code = r.get("winner_code", "")
        if code and "/" not in code:
            pure_codes.add(code)

    for code in sorted(pure_codes):
        c = centroids.get(code, {})
        cid = PARTY_TO_CID.get(code)
        key_positions = compute_key_positions(cluster_rows, cid) if cid else []
        out.append({
            "code": code,
            "isPure": True,
            "seatsCond": seat_counts_cond.get(code, 0),
            "seatsIRV": seat_counts_irv.get(code, 0),
            "F1": float(c.get("F1_security_order", 0)) if c else 0,
            "F2": float(c.get("F2_electoral_skepticism", 0)) if c else 0,
            "F3": float(c.get("F3_government_distrust", 0)) if c else 0,
            "F4": float(c.get("F4_religious_traditionalism", 0)) if c else 0,
            "F5": float(c.get("F5_populist_conservatism", 0)) if c else 0,
            "keyPositions": key_positions,
            "variables": compute_pure_blend_variables(cid) if cid else {},
        })

    out.sort(key=lambda x: -(x["seatsCond"] + x["seatsIRV"]))
    write_json(out, "blendProfiles.json")


# ---------- quizQuestions.json ----------
# Quiz variables: (variable, factor, row_selection_strategy)
# row_selection_strategy:
#   "binary"       → use type=="binary" row (% Supporting)
#   "likert_agree" → sum % Strongly Agree + % Agree rows
#   "trust_none"   → use % None at all row
#   "church_never" → use % Never row
QUIZ_VARS = [
    ("CC24_323b", "F1", "binary"),
    ("CC24_321d", "F1", "binary"),
    ("CC24_421_1", "F2", "likert_agree"),
    ("CC24_421_2", "F2", "likert_agree"),
    ("CC24_423",   "F3", "trust_none"),
    ("CC24_424",   "F3", "trust_none"),
    ("CC24_324b",  "F4", "binary"),
    ("pew_churatd","F4", "church_never"),
    ("CC24_323a",  "F5", "binary"),
    ("CC24_440a",  "F5", "likert_agree"),
]

QUIZ_QUESTION_OVERRIDES = {
    'CC24_423': 'I have very little trust in the federal government.',
    'CC24_424': 'I have very little trust in my state government.',
    'pew_churatd': 'I rarely or never attend religious services.',
}

CIDS = ["0","1","2","3","4","5","6","8","9"]

def build_quiz():
    rows = read_csv(OUTPUTS / "profiles" / "cluster_stats.csv")

    # Group rows by variable
    by_var = defaultdict(list)
    for r in rows:
        by_var[r["variable"]].append(r)

    questions = []
    for var, factor, strategy in QUIZ_VARS:
        var_rows = by_var.get(var, [])
        if not var_rows:
            print(f"  WARNING: quiz var {var} not found in cluster_stats.csv")
            continue

        cluster_pcts = {}
        question_text = QUIZ_QUESTION_OVERRIDES.get(var)
        domain = var_rows[0].get("domain", "")

        if strategy == "binary":
            row = next((r for r in var_rows if r.get("type") == "binary"), var_rows[0])
            if not question_text:
                question_text = row.get("question", var)
            for cid in CIDS:
                val = row.get(f"c{cid}", "")
                cluster_pcts[cid] = float(val) / 100 if val else 0.5

        elif strategy == "likert_agree":
            sa_row = next((r for r in var_rows if r.get("stat_label", "") == "% Strongly Agree"), None)
            a_row  = next((r for r in var_rows if r.get("stat_label", "") == "% Agree"), None)
            if not question_text:
                question_text = var_rows[0].get("question", var)
            for cid in CIDS:
                sa = float(sa_row.get(f"c{cid}", 0) or 0) if sa_row else 0
                a  = float(a_row.get(f"c{cid}", 0)  or 0) if a_row  else 0
                cluster_pcts[cid] = round((sa + a) / 100, 4)

        elif strategy == "trust_none":
            row = next((r for r in var_rows if "None at all" in r.get("stat_label", "")), var_rows[-1])
            if not question_text:
                question_text = row.get("question", var)
            for cid in CIDS:
                val = row.get(f"c{cid}", "")
                cluster_pcts[cid] = float(val) / 100 if val else 0.5

        elif strategy == "church_never":
            row = next((r for r in var_rows if r.get("stat_label", "") == "% Never"), var_rows[-1])
            if not question_text:
                question_text = row.get("question", var)
            for cid in CIDS:
                val = row.get(f"c{cid}", "")
                cluster_pcts[cid] = float(val) / 100 if val else 0.5

        questions.append({
            "variable": var,
            "factor": factor,
            "question": question_text or var,
            "domain": domain,
            "clusterSupport": cluster_pcts,
        })

    write_json(questions, "quizQuestions.json")


# ---------- presidentialElection.json ----------
def build_presidential_election():
    # IRV national rounds
    irv_rows = read_csv(OUTPUTS / "irv" / "irv_presidential_national_2028.csv")
    rounds_by_num = defaultdict(list)
    for r in irv_rows:
        rounds_by_num[int(r["round"])].append(r)

    irv_rounds = []
    irv_winner = None
    for rnum in sorted(rounds_by_num.keys()):
        candidates = []
        for r in rounds_by_num[rnum]:
            code = r["candidate_code"].replace("_", "/")
            eliminated = r["eliminated"].strip().lower() == "true"
            winner = r["winner"].strip().lower() == "true"
            if winner and not eliminated:
                irv_winner = code
            candidates.append({
                "code": code,
                "name": r["candidate_name"],
                "pct": round(float(r["vote_pct"]), 2),
                "votes": round(float(r["vote_total"]), 0),
                "eliminated": eliminated,
                "winner": winner,
            })
        irv_rounds.append({"round": rnum, "candidates": candidates})

    # Condorcet matchups from primary_diagnostics
    diag_rows = read_csv(OUTPUTS / "primary_diagnostics_2028.csv")
    condorcet_matchups = []
    condorcet_winner = None
    for r in diag_rows:
        if r.get("diagnostic") != "condorcet":
            continue
        a = r["candidate_a"].replace("_", "/")
        b = r["candidate_b"].replace("_", "/")
        votes_a = float(r["votes_a_beats_b"])
        votes_b = float(r["votes_b_beats_a"])
        total = votes_a + votes_b
        a_wins_pct = round(votes_a / total * 100, 3) if total > 0 else 50.0
        winner = r["winner"].replace("_", "/")
        margin_pct = round(float(r["margin_pct"]), 3)
        condorcet_matchups.append({
            "candidateA": a,
            "candidateB": b,
            "aWinsPct": a_wins_pct,
            "margin": margin_pct,
            "winner": winner,
        })
        if r.get("rp_winner_overall"):
            condorcet_winner = r["rp_winner_overall"].replace("_", "/")

    # State winners + shares
    state_rows = read_csv(OUTPUTS / "irv" / "irv_presidential_states_2028.csv")
    pod_rows = read_csv(OUTPUTS / "state_pod_assignments.csv")
    pod_by_fips = {r["state_fips"].zfill(2): r["pod"] for r in pod_rows}

    irv_state_winners = {}
    for r in state_rows:
        fips = r["state_fips"].zfill(2)
        winner = r["winner_code"].replace("_", "/")
        r1_cols = [k for k in r.keys() if k.startswith("r1_pct_")]
        raw_shares = {}
        for col in r1_cols:
            code = col.replace("r1_pct_", "").replace("_", "/")
            val = float(r.get(col) or 0)
            if val > 0:
                raw_shares[code] = val
        total = sum(raw_shares.values())
        shares = {k: round(v / total, 4) for k, v in raw_shares.items()} if total > 0 else {}
        irv_state_winners[fips] = {
            "stateAbbr": r["state_abbr"],
            "winner": winner,
            "pod": pod_by_fips.get(fips, "D"),
            "nRespondents": int(r["n_respondents"]),
            "shares": shares,
        }

    write_json({
        "irvRounds": irv_rounds,
        "irvWinner": irv_winner,
        "condorcetMatchups": condorcet_matchups,
        "condorcetWinner": condorcet_winner,
        "irvStateWinners": irv_state_winners,
    }, "presidentialElection.json")


# ---------- presidentialElectionPure.json ----------
def build_presidential_election_pure():
    # IRV national rounds
    irv_rows = read_csv(PURE_DIR / "irv" / "irv_presidential_national_2028.csv")
    rounds_by_num = defaultdict(list)
    for r in irv_rows:
        rounds_by_num[int(r["round"])].append(r)

    irv_rounds = []
    irv_winner = None
    for rnum in sorted(rounds_by_num.keys()):
        candidates = []
        for r in rounds_by_num[rnum]:
            code = normalize_candidate_code(r["candidate_code"])
            eliminated = r["eliminated"].strip().lower() == "true"
            winner = r["winner"].strip().lower() == "true"
            if winner and not eliminated:
                irv_winner = code
            candidates.append({
                "code": code,
                "name": normalize_candidate_code(r["candidate_name"]),
                "pct": round(float(r["vote_pct"]), 2),
                "votes": round(float(r["vote_total"]), 0),
                "eliminated": eliminated,
                "winner": winner,
            })
        irv_rounds.append({"round": rnum, "candidates": candidates})

    # Condorcet matchups from pure primary_diagnostics
    diag_rows = read_csv(PURE_DIR / "primary_diagnostics_2028.csv")
    condorcet_matchups = []
    condorcet_winner = None
    for r in diag_rows:
        if r.get("diagnostic") != "condorcet":
            continue
        a = normalize_candidate_code(r["candidate_a"])
        b = normalize_candidate_code(r["candidate_b"])
        votes_a = float(r["votes_a_beats_b"])
        votes_b = float(r["votes_b_beats_a"])
        total = votes_a + votes_b
        a_wins_pct = round(votes_a / total * 100, 3) if total > 0 else 50.0
        winner = normalize_candidate_code(r["winner"])
        margin_pct = round(float(r["margin_pct"]), 3)
        condorcet_matchups.append({
            "candidateA": a,
            "candidateB": b,
            "aWinsPct": a_wins_pct,
            "margin": margin_pct,
            "winner": winner,
        })
        if r.get("rp_winner_overall"):
            condorcet_winner = normalize_candidate_code(r["rp_winner_overall"])

    # State winners + shares
    state_rows = read_csv(PURE_DIR / "irv" / "irv_presidential_states_2028.csv")
    pod_rows = read_csv(OUTPUTS / "state_pod_assignments.csv")
    pod_by_fips = {r["state_fips"].zfill(2): r["pod"] for r in pod_rows}

    irv_state_winners = {}
    for r in state_rows:
        fips = r["state_fips"].zfill(2)
        winner = normalize_candidate_code(r["winner_code"])
        r1_cols = [k for k in r.keys() if k.startswith("r1_pct_")]
        raw_shares = {}
        for col in r1_cols:
            raw_code = col.replace("r1_pct_", "")
            code = normalize_candidate_code(raw_code)
            val = float(r.get(col) or 0)
            if val > 0:
                raw_shares[code] = raw_shares.get(code, 0) + val
        total = sum(raw_shares.values())
        shares = {k: round(v / total, 4) for k, v in raw_shares.items()} if total > 0 else {}
        irv_state_winners[fips] = {
            "stateAbbr": r["state_abbr"],
            "winner": winner,
            "pod": pod_by_fips.get(fips, "D"),
            "nRespondents": int(r["n_respondents"]),
            "shares": shares,
        }

    write_json({
        "irvRounds": irv_rounds,
        "irvWinner": irv_winner,
        "condorcetMatchups": condorcet_matchups,
        "condorcetWinner": condorcet_winner,
        "irvStateWinners": irv_state_winners,
    }, "presidentialElectionPure.json")


# ---------- primaryTransfers.json ----------
def build_primary_transfers():
    diag_rows = read_csv(OUTPUTS / "primary_diagnostics_2028.csv")
    out = []
    for r in diag_rows:
        if r.get("diagnostic") != "transfer_analysis":
            continue
        elim_code = r.get("eliminated_code", "").strip()
        dest_code = r.get("dest_code", "").strip()
        transferred = r.get("transferred_votes", "").strip()
        pct = r.get("pct_of_eliminated_total", "").strip()
        winnowing = r.get("winnowing_point", "").strip()
        transfer_type = r.get("transfer_type", "").strip()
        if not elim_code or not dest_code or not transferred:
            continue
        out.append({
            "source": normalize_candidate_code(elim_code),
            "target": normalize_candidate_code(dest_code),
            "votes": round(float(transferred), 1),
            "pct": round(float(pct), 2) if pct else 0,
            "round": winnowing,
            "type": transfer_type,
        })
    write_json(out, "primaryTransfers.json")


# ---------- primarySankey.json ----------
def build_primary_sankey():
    """Build stage-by-stage Sankey: 5 columns (initial → retail → pod A → pod C → final 5)."""
    profiles = read_csv(OUTPUTS / "state_candidate_profiles.csv")
    diag_rows = read_csv(OUTPUTS / "primary_diagnostics_2028.csv")

    # First-choice national percentages for all 20 candidates (stage 0)
    fc_totals = {}
    for row in profiles:
        n = float(row.get("total_weighted_respondents") or 0)
        for k, v in row.items():
            if k.startswith("first_choice_"):
                raw = k.replace("first_choice_", "")
                code = normalize_candidate_code(raw)
                fc_totals[code] = fc_totals.get(code, 0) + n * float(v or 0)
    total_fc = sum(fc_totals.values())
    fc_pct = {code: round(v / total_fc * 100, 3) for code, v in fc_totals.items()}

    # Trajectory vote_pcts keyed by (norm_code, stageIdx)
    stage_order = ["After_Retail_Six", "After_Pod_A", "After_Pod_C", "After_Pod_BD"]
    stage_to_idx = {s: i + 1 for i, s in enumerate(stage_order)}
    traj_rows = [r for r in diag_rows if r["diagnostic"] == "trajectories"]

    active_at = {i: [] for i in range(1, 5)}  # stageIdx → [norm_code]
    vote_pct_at = {}  # (norm_code, stageIdx) → pct
    for r in traj_rows:
        stage_idx = stage_to_idx.get(r["phase"])
        if stage_idx is None:
            continue
        code = normalize_candidate_code(r["candidate_code"])
        pct = float(r["vote_pct"] or 0)
        if r["status"] in ("active", "surviving", "elected") and pct > 0:
            active_at[stage_idx].append(code)
            vote_pct_at[(code, stage_idx)] = pct

    # Elimination transfers: elim_xfers[stageIdx][eliminated_code] = [(dest_code, pct)]
    elim_xfers = {i: {} for i in range(1, 5)}
    for r in diag_rows:
        if r["diagnostic"] != "transfer_analysis" or r.get("transfer_type") != "elimination":
            continue
        stage_idx = stage_to_idx.get(r["winnowing_point"])
        if stage_idx is None:
            continue
        e_code = normalize_candidate_code(r["eliminated_code"])
        d_code = normalize_candidate_code(r["dest_code"])
        pct = float(r["pct_of_eliminated_total"] or 0)
        if e_code not in elim_xfers[stage_idx]:
            elim_xfers[stage_idx][e_code] = []
        elim_xfers[stage_idx][e_code].append((d_code, pct))

    # Build nodes
    nodes = []
    # Stage 0: all 20 initial candidates
    for code, pct in sorted(fc_pct.items(), key=lambda x: -x[1]):
        nodes.append({"id": f"{code}__0", "label": code, "stageIdx": 0, "pct": pct})

    # Stages 1–4: active candidates per stage
    for stage_idx in range(1, 5):
        for code in active_at[stage_idx]:
            pct = vote_pct_at.get((code, stage_idx), 0)
            nodes.append({"id": f"{code}__{stage_idx}", "label": code, "stageIdx": stage_idx, "pct": pct})

    # Collect all valid node ids
    node_ids = {n["id"] for n in nodes}

    # Build links
    links = []

    def add_link(src_id, tgt_id, value):
        if src_id in node_ids and tgt_id in node_ids and value > 0.01:
            links.append({"source": src_id, "target": tgt_id, "value": round(value, 3)})

    # Stage 0 → Stage 1 (Retail)
    retail_active = set(active_at[1])
    retail_elim_codes = set(elim_xfers[1].keys())

    for code, pct in fc_pct.items():
        src = f"{code}__0"
        if code in retail_active:
            add_link(src, f"{code}__1", pct)
        elif code in retail_elim_codes:
            for dest_code, xfer_pct in elim_xfers[1][code]:
                add_link(src, f"{dest_code}__1", pct * xfer_pct / 100)

    # Stages 1→2, 2→3, 3→4
    # Eliminations happen AT stage N+1 (winnowing_point=After_Pod_X)
    # So candidate is active in stage N and eliminated in stage N+1
    elim_stage_map = {
        2: 1,  # After_Pod_A elims happened between stage 1 and 2
        3: 2,
        4: 3,
    }
    for dst_idx in range(2, 5):
        src_idx = dst_idx - 1
        dst_active = set(active_at[dst_idx])

        # Survivors from src to dst
        for code in active_at[src_idx]:
            src_pct = vote_pct_at.get((code, src_idx), 0)
            if code in dst_active:
                add_link(f"{code}__{src_idx}", f"{code}__{dst_idx}", src_pct)
            elif code in elim_xfers[dst_idx]:
                # Eliminated between src and dst — transfer votes
                for dest_code, xfer_pct in elim_xfers[dst_idx][code]:
                    add_link(f"{code}__{src_idx}", f"{dest_code}__{dst_idx}", src_pct * xfer_pct / 100)
            # else: no links (exhausted/missing transfer data)

    stage_labels = [
        "Initial Slate (20)",
        "After Retail (12)",
        "After Pod A (10)",
        "After Pod C (8)",
        "Final Five",
    ]
    write_json({"stageLabels": stage_labels, "nodes": nodes, "links": links}, "primarySankey.json")


# ---------- primaryRaw.json ----------
def build_primary_raw():
    rows = read_csv(PURE_DIR / "primary_results_2028.csv")
    centroids = {r["candidate_code"]: r for r in read_csv(OUTPUTS / "candidate_factor_centroids.csv")}

    stages_order = ["After_Retail_Six", "After_Pod_A", "After_Pod_C", "After_Pod_BD"]
    stage_labels = {
        "After_Retail_Six": "Retail + Bench States",
        "After_Pod_A": "After Pod A (West)",
        "After_Pod_C": "After Pod C (South)",
        "After_Pod_BD": "After Pods B+D (Final)",
    }

    by_candidate = defaultdict(dict)
    quota_by_stage = {}
    for row in rows:
        stage = row["winnowing_point"]
        raw_code = row["candidate_code"]
        by_candidate[raw_code][stage] = {
            "voteTotal": float(row["vote_total"]),
            "votePct": float(row["vote_pct"]),
            "status": row["status"],
            "quotaThreshold": float(row["quota_threshold"]),
        }
        quota_by_stage[stage] = float(row["quota_threshold"])

    candidates = []
    for raw_code, stages in by_candidate.items():
        display_code = normalize_candidate_code(raw_code)
        c = centroids.get(raw_code, {})
        name = c.get("candidate_name", display_code) or display_code
        entry = {
            "code": display_code,
            "name": name,
            "F1": float(c.get("F1_security_order", 0)),
            "F2": float(c.get("F2_electoral_skepticism", 0)),
            "F3": float(c.get("F3_government_distrust", 0)),
            "F4": float(c.get("F4_religious_traditionalism", 0)),
            "F5": float(c.get("F5_populist_conservatism", 0)),
            "stages": {s: stages.get(s, {"voteTotal": 0, "votePct": 0, "status": "previously_eliminated", "quotaThreshold": quota_by_stage.get(s, 0)}) for s in stages_order},
        }
        candidates.append(entry)

    output = {
        "stagesOrder": stages_order,
        "stageLabels": stage_labels,
        "quotaByStage": quota_by_stage,
        "candidates": candidates,
    }
    write_json(output, "primaryRaw.json")


# ---------- primaryStateWinnersRaw.json ----------
def build_primary_state_winners_raw():
    state_rows = read_csv(PURE_DIR / "irv" / "irv_presidential_states_2028.csv")
    pod_rows = read_csv(OUTPUTS / "state_pod_assignments.csv")

    pod_by_fips = {r["state_fips"].zfill(2): r["pod"] for r in pod_rows}

    out = {}
    for r in state_rows:
        fips = r["state_fips"].zfill(2)
        winner = normalize_candidate_code(r["winner_code"].replace("_", "/"))
        runner_up = normalize_candidate_code(r["runner_up_code"].replace("_", "/"))
        # r1_pct_* columns are already party codes in the pure run
        r1_cols = [k for k in r.keys() if k.startswith("r1_pct_")]
        shares = {}
        for col in r1_cols:
            code = col.replace("r1_pct_", "")
            val = float(r.get(col) or 0)
            if val > 0:
                shares[code] = val
        total = sum(shares.values())
        if total > 0:
            shares = {k: round(v / total, 4) for k, v in shares.items()}
        pod = pod_by_fips.get(fips, "D")
        out[fips] = {
            "stateAbbr": r["state_abbr"],
            "winnerCode": winner,
            "runnerUpCode": runner_up,
            "pod": pod,
            "nRespondents": int(r["n_respondents"]),
            "shares": shares,
        }
    write_json(out, "primaryStateWinnersRaw.json")


# ---------- primarySankeyRaw.json ----------
def build_primary_sankey_raw():
    """Build stage-by-stage Sankey for pure (9-candidate) primary run."""
    diag_rows = read_csv(PURE_DIR / "primary_diagnostics_2028.csv")
    results_rows = read_csv(PURE_DIR / "primary_results_2028.csv")

    # Stage 0: initial pcts from first-stage results (After_Retail_Six)
    fc_pct = {}
    for row in results_rows:
        if row["winnowing_point"] == "After_Retail_Six":
            code = normalize_candidate_code(row["candidate_code"])
            fc_pct[code] = float(row["vote_pct"])

    stage_order = ["After_Retail_Six", "After_Pod_A", "After_Pod_C", "After_Pod_BD"]
    stage_to_idx = {s: i + 1 for i, s in enumerate(stage_order)}
    traj_rows = [r for r in diag_rows if r["diagnostic"] == "trajectories"]

    active_at = {i: [] for i in range(1, 5)}
    vote_pct_at = {}
    for r in traj_rows:
        stage_idx = stage_to_idx.get(r["phase"])
        if stage_idx is None:
            continue
        code = normalize_candidate_code(r["candidate_code"])
        pct = float(r["vote_pct"] or 0)
        if r["status"] in ("active", "surviving", "elected") and pct > 0:
            active_at[stage_idx].append(code)
            vote_pct_at[(code, stage_idx)] = pct

    elim_xfers = {i: {} for i in range(1, 5)}
    for r in diag_rows:
        if r["diagnostic"] != "transfer_analysis" or r.get("transfer_type") != "elimination":
            continue
        stage_idx = stage_to_idx.get(r["winnowing_point"])
        if stage_idx is None:
            continue
        e_code = normalize_candidate_code(r["eliminated_code"])
        d_code = normalize_candidate_code(r["dest_code"])
        pct = float(r["pct_of_eliminated_total"] or 0)
        if e_code not in elim_xfers[stage_idx]:
            elim_xfers[stage_idx][e_code] = []
        elim_xfers[stage_idx][e_code].append((d_code, pct))

    nodes = []
    for code, pct in sorted(fc_pct.items(), key=lambda x: -x[1]):
        nodes.append({"id": f"{code}__0", "label": code, "stageIdx": 0, "pct": pct})
    for stage_idx in range(1, 5):
        for code in active_at[stage_idx]:
            pct = vote_pct_at.get((code, stage_idx), 0)
            nodes.append({"id": f"{code}__{stage_idx}", "label": code, "stageIdx": stage_idx, "pct": pct})

    node_ids = {n["id"] for n in nodes}

    links = []

    def add_link(src_id, tgt_id, value):
        if src_id in node_ids and tgt_id in node_ids and value > 0.01:
            links.append({"source": src_id, "target": tgt_id, "value": round(value, 3)})

    retail_active = set(active_at[1])
    retail_elim_codes = set(elim_xfers[1].keys())
    for code, pct in fc_pct.items():
        src = f"{code}__0"
        if code in retail_active:
            add_link(src, f"{code}__1", pct)
        elif code in retail_elim_codes:
            for dest_code, xfer_pct in elim_xfers[1][code]:
                add_link(src, f"{dest_code}__1", pct * xfer_pct / 100)

    for dst_idx in range(2, 5):
        src_idx = dst_idx - 1
        dst_active = set(active_at[dst_idx])
        for code in active_at[src_idx]:
            src_pct = vote_pct_at.get((code, src_idx), 0)
            if code in dst_active:
                add_link(f"{code}__{src_idx}", f"{code}__{dst_idx}", src_pct)
            elif code in elim_xfers[dst_idx]:
                for dest_code, xfer_pct in elim_xfers[dst_idx][code]:
                    add_link(f"{code}__{src_idx}", f"{dest_code}__{dst_idx}", src_pct * xfer_pct / 100)

    stage_labels = [
        "Initial Slate (9)",
        "After Retail",
        "After Pod A",
        "After Pod C",
        "Final",
    ]
    write_json({"stageLabels": stage_labels, "nodes": nodes, "links": links}, "primarySankeyRaw.json")


# ---------- statePodAssignments.json ----------
def build_state_pods():
    rows = read_csv(OUTPUTS / "state_pod_assignments.csv")
    out = {}
    for r in rows:
        out[r["state_fips"].zfill(2)] = {
            "stateAbbr": r["state_abbr"],
            "pod": r["pod"],
            "bench": r["bench"] == "True",
            "retail": r["retail_2028"] == "True",
        }
    write_json(out, "statePodAssignments.json")


if __name__ == "__main__":
    print("Preparing data...")
    build_primary()
    build_primary_state_winners()
    build_presidential_election()
    build_presidential_election_pure()
    build_primary_transfers()
    build_primary_sankey()
    build_primary_raw()
    build_primary_state_winners_raw()
    build_primary_sankey_raw()
    build_senate()
    build_senate_pure()
    build_senate_vote_model()
    build_house_seats()
    build_house_vote_model()
    build_house_state_map()
    build_coalition_profiles()
    build_transfer_matrix()
    build_cluster_profiles()
    build_blend_profiles()
    build_quiz()
    build_state_pods()
    print("Done.")
