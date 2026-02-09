# STV Electoral System Simulation

A comprehensive simulation of alternative electoral systems using real survey data from the 2024 Cooperative Election Study (CES).

## Overview

This project explores what American politics might look like under different electoral rules:
- **Multi-party proportional representation** instead of winner-take-all
- **Single Transferable Vote (STV)** for multi-member districts
- **A rolling presidential primary** with geographic balance
- **Condorcet-based general elections** with proportional elector allocation

## Key Findings

### 1. Six Natural Political Clusters

Using K-Means clustering on 37 policy questions, we identified **6 distinct political groupings** that better represent American ideological diversity than the current two-party system:

| Cluster | Description | ~% of Electorate |
|---------|-------------|------------------|
| **The Left** | Progressive on all issues, strong support for abortion rights, climate action, wealth redistribution | ~15% |
| **Liberals** | Socially liberal, moderate on economics, educated suburban voters | ~18% |
| **Working Families** | Economically populist, moderate on social issues, union-friendly | ~17% |
| **Security Centrists** | Moderate across the board, pragmatic, swing voters | ~16% |
| **Conservatives** | Traditional conservative values, pro-business, religious | ~17% |
| **MAGA** | Nationalist populist, anti-immigration, anti-establishment | ~17% |

### 2. Proportional House Representation

Simulated a **"Doubled House"** (870 seats instead of 435) with multi-member districts:
- **5-seat districts** preferred (best proportionality)
- **3-seat districts** to fill gaps
- **2-seat statewide districts** only for small states

**Result**: All 6 clusters win meaningful representation in nearly every state, compared to winner-take-all where ~40% of voters have no representative aligned with their views.

### 3. Rolling Presidential Primary ("The American Mosaic")

A 4-round primary system designed to test candidates across diverse regions:

| Round | Format | Purpose |
|-------|--------|---------|
| **Round 1 (May)** | 6 small "Retail" states from different cultural regions | Tests retail politics, winnows 24→12 |
| **Round 2 (June)** | One geographic Pod (with major anchor state) | Tests scalability, 12→10 |
| **Round 3 (July)** | Another Pod | Tests broad appeal, 10→8 |
| **Round 4 (August)** | Final two Pods together | The finale, 8→5 finalists |

**Key insight**: STV naturally winnows out pure ideologues in favor of candidates with broader appeal. The "Final 5" typically includes 1-2 ideological voices (for representation) but is dominated by candidates near the median voter.

### 4. Condorcet General Election

Simulated **"Bottom-Up Condorcet with Proportional Runoff"**:

1. **Build pairwise matrices**: For each state, calculate head-to-head preferences between all Final 5 candidates
2. **Sum nationally**: Create a national pairwise matrix weighted by population
3. **Find Top 2**: Use Copeland method (count pairwise victories) to identify finalists
4. **Proportional allocation**: Each state splits its electors proportionally based on the Top 2 head-to-head margin

**Result**: The Condorcet winner (candidate who would beat any other head-to-head) almost always wins. Electoral inversions become nearly impossible because proportional allocation eliminates the "efficiency gap."

## Methodology

### Data Source
- **2024 Cooperative Election Study (CES)** - 60,000+ respondents
- 37 policy questions covering abortion, immigration, guns, climate, economics, healthcare, etc.
- Weighted to be nationally representative

### Clustering
- **Weighted K-Means** (approximated via resampling)
- Questions coded as Support (1) or Oppose (2)
- Cluster support rates calculated directly from weighted survey responses

### Vote Simulation
- **"Base + Appeal" model**: Candidates scored on 12 key polarizing questions
- Alignment calculated as: how well does candidate position match cluster majority?
- Transformed via `tanh` to create realistic vote share distributions (5-95% range)

### STV Implementation
- Full Droop quota calculation
- Vote transfers based on position similarity between candidates
- Round-by-round elimination with detailed tracking

## Visualizations

The notebook includes interactive Plotly visualizations:
- **Pairwise heatmaps**: Head-to-head margins between candidates
- **Choropleth maps**: State-by-state electoral allocation
- **Sankey diagrams**: Electoral vote flows from regions to candidates
- **Radar charts**: Candidate ideological profiles across clusters
- **Parliament semicircles**: Seat distribution by party

## Files

| File | Description |
|------|-------------|
| `five_member_proportional.ipynb` | Main analysis notebook |
| `DemRep.ipynb` | Baseline Democratic/Republican analysis |
| `ces_questions_VERIFIED.md` | Documentation of policy questions used |
| `dataverse_files/` | CES survey data |

## Key Takeaways

1. **America is not binary**: Six distinct political clusters exist, but the two-party system forces them into uncomfortable coalitions.

2. **STV produces centrist winners**: Multi-round elimination naturally favors broadly acceptable candidates over factional champions.

3. **Proportional allocation fixes the Electoral College**: When states split electors proportionally, the national popular vote winner almost always wins.

4. **Representation vs. Governance tradeoff**: More parties = better representation but potentially harder coalition-building. The 6-cluster system might naturally form 2-3 governing coalitions.

5. **Geographic pods ensure cultural balance**: No single region can dominate a primary when states are grouped to offset each other's biases.

## Requirements

```
pandas
numpy
scikit-learn
plotly
```

## Usage

Run the Jupyter notebook cells in order. The analysis builds progressively:
1. Data loading and cleaning
2. Cluster analysis
3. State-level party share calculations
4. Seat allocation simulation
5. Presidential primary simulation
6. Condorcet general election simulation

## License

For educational and research purposes.

---

*"The test of a first-rate intelligence is the ability to hold two opposed ideas in mind at the same time and still retain the ability to function."* — F. Scott Fitzgerald

This simulation holds six.
