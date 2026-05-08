# Candidate Generation: Open Methodology Question

**Status:** Tabled — senate simulation will inform the decision.

---

## Current presidential approach

9 **platonic** candidates (one per active cluster, C7 dissolved) + 9 **hand-curated straddlers**
with manually assigned cluster pairs and weights (55/45 or 50/50).

The platonic candidates are unambiguously principled: each is the pure ideological archetype of its cluster.

The straddlers are where the question lies.

---

## What the data says

Cluster pair distances in F1-F5 factor space — closest = most natural bridge:

| Closest pairs | Distance | Current straddler? |
|---------------|----------|--------------------|
| C3–C5 (NAT/REF) | 0.692 | **No — missing** |
| C1–C4 (SD/LIB) | 0.818 | AW ✓ |
| C4–C9 (LIB/PRG) | 0.919 | PN ✓ |
| C4–C6 (LIB/CTR) | 0.980 | **No — missing** |
| C0–C6 (CON/CTR) | 0.988 | TW ✓ |
| C2–C8 (STY/DSA) | 1.000 | **No — missing** |
| C1–C2 (SD/STY) | 1.008 | JC ✓ |
| C1–C8 (SD/DSA) | 1.068 | SC ✓ |

The hand-curated set covers 5 of the 8 closest pairs but misses C3–C5 (the CLOSEST pair).

Current straddlers that span longer distances:

| Candidate | Pair | Distance | Notes |
|-----------|------|----------|-------|
| DM | STY/CTR | 1.70 | Long stretch; poor primary performer |
| FM | CON/LIB | 1.62 | Long stretch; won Ranked Pairs (centrist institutionalist) |
| MR | REF/CON | 1.34 | Moderate stretch |

---

## Three alternative approaches considered

1. **Current (ad hoc):** Hand-pick straddlers based on political intuition. Allows intentional cross-aisle
   candidates (like FM) that test interesting dynamics, but leaves obvious gaps.

2. **Data-driven co-occurrence:** For each respondent find their top-2 clusters; build a co-occurrence
   rate matrix; select the highest-rate pairs as straddlers with data-driven weights. Fully principled
   but may over-cluster in ideologically adjacent regions and miss cross-aisle candidates.

3. **Factor-space positioning:** Define candidates as F1-F5 coordinates; compute cluster affinity via
   inverse distance weighting. Most flexible but loses the clean cluster→party mapping.

---

## Why FM winning matters

FM (CON/LIB, d=1.62) is the longest-distance straddler and won Ranked Pairs precisely *because*
it occupies a "nobody strongly opposes me" position. A purely data-driven approach would likely
not generate this candidate (few voters naturally co-occur at C0 and C4). This is an argument for
keeping some intentional cross-aisle "wild card" straddlers alongside data-driven ones.

---

## Resolution: per-state senate simulation

The senate simulation uses a **hybrid approach per state**:
- **Pure candidates** from each cluster exceeding a 5% state share
- **Co-occurrence straddlers** from actual top-2 cluster co-occurrence within that state
- **Wild card cross-aisle straddlers** when two clusters both have ≥15% share and their
  factor-space distance ≥ 1.40

This provides empirical data on which types of candidates win in which states, which will
inform whether (and how) to revise the presidential candidate roster in a future iteration.
