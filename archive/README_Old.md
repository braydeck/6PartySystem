# STV Electoral System Simulation

A comprehensive simulation of alternative electoral systems using real survey data from the 2024 Cooperative Election Study (CES). The project models what American politics might look like under proportional representation, multi-party primaries, and alternative voting methods.

## Overview

This project explores what American politics might look like under different electoral rules:
- **Multi-party proportional representation** instead of winner-take-all
- **Single Transferable Vote (STV)** primaries with Droop quota surplus transfers
- **A rolling presidential primary** with geographic balance across 4 rounds
- **Instant Runoff Voting (IRV)** for senate general elections
- **Condorcet/Copeland general elections** with proportional elector allocation

---

## Data Source

- **2024 Cooperative Election Study (CES)** -- 60,000+ respondents
- Filtered to post-election respondents (`tookpost == 2`)
- All calculations weighted using `commonweight` for national representativeness
- Policy questions coded as Support (1) or Oppose (2), with some continuous scales (e.g., abortion weeks)

---

## Methodology

### Step 1: Clustering -- Identifying 6 Political Parties

We use unsupervised machine learning to discover natural political groupings in the American electorate.

#### Feature Sets

**50 Clustering Questions** (used for cluster assignment AND vote transfers):

| Domain | Count | Variables |
|--------|-------|-----------|
| Abortion | 5 | CC24_324a, CC24_324b, CC24_324c, CC24_324d, CC24_325 |
| Immigration | 5 | CC24_323a, CC24_323b, CC24_323c, CC24_323d, CC24_340f |
| Guns & Policing | 6 | CC24_321a through CC24_321f |
| Environment & Climate | 6 | CC24_326a through CC24_326f |
| Taxation | 4 | CC24_341a through CC24_341d |
| Healthcare & Medicaid | 3 | CC24_328c, CC24_328d, CC24_328e |
| Student Loans | 1 | CC24_323f |
| Housing | 2 | CC24_328a, CC24_328b |
| Civil Rights | 3 | CC24_340a, CC24_340b, CC24_340c |
| Surveillance & Tech | 2 | CC24_340d, CC24_340e |
| Spending Priorities | 5 | CC24_443_1 through CC24_443_5 |
| Trans/Vouchers/Age Policy | 6 | CC24_444a through CC24_444f |
| SCOTUS Decisions | 2 | CC24_445a, CC24_445b |

**21 Descriptor Columns** (used for cluster profiling only, NOT clustering):

| Domain | Count | Variables |
|--------|-------|-----------|
| Religiosity | 4 | pew_bornagain, pew_churatd, pew_religimp, pew_prayer |
| Institutional Trust | 4 | CC24_421_1, CC24_421_2, CC24_423, CC24_424 |
| Racial/Cultural Attitudes | 6 | CC24_440a-d, CC24_441a-b |
| Military Interventionism | 7 | CC24_420_1 through CC24_420_7 |

These descriptors are deliberately excluded from clustering because they measure cultural identity and trust rather than domestic policy preferences. They are reported in cluster profiles for interpretation.

#### Pipeline

1. **Imputation**: `SimpleImputer(strategy='median')` fills missing values across all 50 clustering questions
2. **Scaling**: `StandardScaler()` normalizes features to zero mean, unit variance
3. **PCA**: `PCA(n_components=17)` reduces 50 features to 17 principal components (retaining maximum variance while reducing noise)
4. **K-Means**: `KMeans(n_clusters=6, n_init=10, max_iter=300)` on PCA-transformed data
5. **Cluster Ordering**: Clusters sorted left-to-right by mean `ideo5` (1=Very Liberal to 5=Very Conservative)

#### The Six Clusters

| # | Party | ~% of Electorate | Description |
|---|-------|-------------------|-------------|
| 0 | **Progressive Left** | 19.7% | Secular, young, defund police, halt oil, ban assault weapons |
| 1 | **Establishment Democrats** | 20.5% | Older Dems, high trust, pro-institution, pro-Dreamers, pro-Ukraine |
| 2 | **Christian Democrats** | 21.1% | Cross-partisan moderates, religious, mixed on immigration, pro-Medicaid |
| 3 | **Cultural Protectionists** | 12.4% | Socially conservative, economically populist, pro-wall but pro-spending |
| 4 | **Traditional Conservatives** | 19.5% | Older, white, pro-wall, pro-fossil fuel, more police, repeal ACA |
| 5 | **Hard Right** | 8.8% | Male 70%, concealed carry, repeal ACA, anti-EPA, anti-student loans |

---

### Step 2: Candidate Model -- 18 Fictional Candidates

Each cluster gets three candidate types to represent the range of political strategy:

| Type | Count | Description |
|------|-------|-------------|
| **Purists** | 6 | Strong ideological consistency with home cluster. Crossover appeal ≤ 0.30. One per cluster. |
| **Triangulators** | 6 | Cross-cutting positions that borrow from adjacent clusters. Crossover 0.50-0.70. One per cluster. |
| **Near-Clones** | 6 | ~80% aligned with cluster but with some crossover positions. One per cluster. |

#### Candidate Roster

| Candidate | Type | Home Cluster |
|-----------|------|--------------|
| Green Crusader | Purist | Progressive Left |
| Pragmatic Progressive | Triangulator | Progressive Left |
| Progressive Ally | Near-Clone | Progressive Left |
| Beltway Institutionalist | Purist | Establishment Democrats |
| Security Liberal | Triangulator | Establishment Democrats |
| Liberal Hawk | Near-Clone | Establishment Democrats |
| Community Matriarch | Purist | Christian Democrats |
| Kitchen Table Democrat | Triangulator | Christian Democrats |
| Rural Progressive | Near-Clone | Christian Democrats |
| Southern Preacher | Purist | Cultural Protectionists |
| Faith & Family Governor | Triangulator | Cultural Protectionists |
| Hawkish Centrist | Near-Clone | Cultural Protectionists |
| Chamber Conservative | Purist | Traditional Conservatives |
| Reform Republican | Triangulator | Traditional Conservatives |
| Compassionate Conservative | Near-Clone | Traditional Conservatives |
| Hard Right Warrior | Purist | Hard Right |
| Populist Firebrand | Triangulator | Hard Right |
| Populist Nationalist | Near-Clone | Hard Right |

**Position creation:**
- `create_cluster_champion_positions(cluster_name)` -- generates positions that exactly match a cluster's majority on all 50 questions
- `create_nuanced_positions(primary_cluster, overrides)` -- starts from the cluster champion, then overrides specific questions to create crossover appeal

---

### Step 3: Vote Calculation

Votes are calculated in two stages: **first-choice allocation** (for STV round 1) and **vote share scoring** (for general elections and alignment).

#### First-Choice Allocation (STV Primaries)

Function: `calculate_cluster_first_support(candidates, state)`

Uses a **home loyalty model**:
- **HOME_LOYALTY = 0.75**: 75% of each cluster's weighted votes go to candidates from that cluster
- **SPILLOVER = 0.25**: The remaining 25% is distributed across all candidates weighted by alignment
- If a cluster has no home candidate, 100% spills over
- Intra-cluster allocation splits the home vote by each candidate's `CANDIDATE_CLUSTER_ALIGNMENTS` score

This ensures that, e.g., Progressive Left voters mostly support Progressive Left candidates but a fraction supports well-aligned candidates from Establishment Democrats or Christian Democrats.

#### Vote Share Scoring (General Elections)

Function: `calculate_cluster_vote_share(candidate_positions, cluster_support_rates, ...)`

Uses a **tanh transformation model** on 12 key polarizing questions:

1. For each question, calculate alignment: how well the candidate's position matches the cluster's majority
2. Weight by intensity: questions where the cluster feels strongly (far from 50%) matter more
3. Average alignment transformed via:

```
vote_share = 50 + 45 * tanh(avg_alignment * 3)
```

This produces a realistic S-curve: candidates near a cluster's views get 70-95%, misaligned candidates get 5-30%, and moderately aligned get 40-60%.

**Identity modifiers** add a bonus/penalty based on ideological distance between the candidate's home cluster and the voter's cluster:

| Distance | Modifier |
|----------|----------|
| 0 (same cluster) | +12 |
| 1 (adjacent) | +4 |
| 2 | 0 |
| 3 | -8 |
| 4 | -14 |
| 5 (opposite end) | -20 |

Final vote share is clipped to [5, 95].

---

### Step 4: Vote Transfer Logic

Function: `calculate_transfer_preferences(eliminated_candidate, active_candidates)`

When a candidate is eliminated (or has surplus votes redistributed) in STV, their votes transfer to remaining candidates based on **policy similarity across all 50 clustering questions**.

For each remaining candidate:
1. Count **agreements** (positions match exactly) and **disagreements** (positions differ by > 0.5)
2. Special handling for CC24_325 (abortion weeks): difference < 5 weeks = agree, > 15 weeks = disagree
3. Calculate similarity ratio: `agree_count / max(1, agree_count + disagree_count)`
4. **Square the similarity** to amplify differences:

| Similarity | Transfer Weight |
|------------|----------------|
| 90% similar | 0.81 |
| 70% similar | 0.49 |
| 50% similar | 0.25 |
| 30% similar | 0.09 |

5. Normalize weights to sum to 1.0

Using all 50 clustering questions (rather than a smaller subset) ensures transfers respect the full ideological spectrum, preventing disproportionate vote leakage to centrist candidates.

---

### Step 5: STV with Droop Quota

Both the presidential primary and senate primaries use Single Transferable Vote with proper Droop quota surplus transfers.

#### Droop Quota Formula

```
droop_quota = total_votes / (seats + 1)
```

For a 5-seat senate primary: quota = total_votes / 6 ≈ 16.67%
For the presidential primary Round 1 (18 to 12): quota = total_votes / 13 ≈ 7.7%

#### STV Algorithm

Each round proceeds as follows:

1. **Check for quota exceeders**: If any candidate's vote total >= Droop quota:
   - **Elect** that candidate
   - Calculate **surplus** = votes - quota
   - Transfer surplus to remaining candidates using `calculate_transfer_preferences()`
   - Remove elected candidate from active pool
   - Repeat step 1 (multiple candidates may exceed quota in sequence)

2. **If no one exceeds the quota**: Eliminate the candidate with the fewest votes
   - Transfer **all** of their votes using `calculate_transfer_preferences()`
   - Remove eliminated candidate from active pool

3. **Repeat** until the target number of candidates are elected

4. **Early termination**: If remaining active candidates <= remaining seats, elect all remaining

#### Implementation

- **Presidential primary**: `run_stv_elimination(candidates, states, target_survivors, seed)`
  - Aggregates votes across multiple states (population-weighted)
  - Used across all 4 rounds with shrinking candidate pools

- **Senate primary**: `run_senate_stv(candidates, state, target_survivors=5, seed)`
  - Runs per-state with state-specific cluster distributions
  - Each state independently reduces 18 candidates to 5 finalists

---

### Step 6: Rolling Presidential Primary ("The American Mosaic")

Function: `run_full_primary_cycle(cycle_year, seed=42)`

A 4-round primary system that tests candidates across diverse geographic and demographic regions:

| Round | Name | When | States | Candidates |
|-------|------|------|--------|------------|
| 1 | The Retail Six | May | 6 small states from different cultural regions | 18 -> 12 |
| 2 | The Expansion | June | One geographic pod (with major anchor state) | 12 -> 10 |
| 3 | The Gauntlet | July | Another geographic pod | 10 -> 8 |
| 4 | The Finale | August | Final two pods combined | 8 -> **5 Finalists** |

Each round uses STV with Droop quota. Eliminated candidates are removed; surviving candidates carry forward. The geographic pods are designed so no single region can dominate -- each pod contains a mix of urban/rural, coastal/interior, and different demographic profiles.

---

### Step 7: Senate Elections

#### Senate Primary (per state)

- All 18 candidates compete in each state
- STV with Droop quota reduces to **5 finalists** per state
- State-specific cluster distributions drive results (e.g., Massachusetts favors Progressive Left candidates, Alabama favors Traditional Conservatives)
- Each state uses a unique seed for tie-breaking

#### Senate General Election (per state)

- **Instant Runoff Voting (IRV)**: The 5 finalists compete in each state
- Process:
  1. Calculate first-choice votes from `calculate_cluster_first_support()`
  2. If no candidate has >= 50%, eliminate the candidate with the fewest votes
  3. Transfer eliminated votes via `calculate_transfer_preferences()`
  4. Repeat until one candidate has >= 50% or only 2 remain
- Produces a single winner per state (50 senate seats total)

---

### Step 8: Presidential General Election

The Final 5 from the presidential primary compete in a national Condorcet election:

1. **Build pairwise matrices**: For each state, calculate head-to-head preferences between all 5 candidates using cluster vote shares
2. **Sum nationally**: Create a national pairwise matrix weighted by state population (electoral votes)
3. **Find Condorcet winner**: Check if any candidate beats all others in pairwise comparisons
4. **Copeland scoring**: Count pairwise victories to rank candidates if no pure Condorcet winner exists
5. **Proportional elector allocation**: Each state splits its electors proportionally based on the top-2 head-to-head margin

The Condorcet winner (the candidate who would beat any other head-to-head) almost always wins. Electoral inversions become nearly impossible because proportional allocation eliminates the efficiency gap inherent in winner-take-all.

---

### Step 9: Horserace Analysis

The notebook also includes a **Cluster Horserace Analysis** that correlates the 6 clusters with real-world political data from the CES:
- Biden/Harris/SCOTUS/Congress approval ratings by cluster
- Economic perceptions (national economy, household income, prices) by cluster
- 2024 presidential preference (Trump vs Harris) by cluster
- Cross-tabulations identifying persuadable voters and loyalty patterns

---

## Visualizations

The notebook produces interactive Plotly visualizations throughout:
- **Demographic heatmaps**: Race, education, religion, industry, age, etc. by cluster
- **Policy profile tables**: Full text output of all 50 clustering questions + 21 descriptors per cluster
- **Pairwise heatmaps**: Head-to-head margins between candidates
- **Choropleth maps**: State-by-state electoral allocation
- **Sankey diagrams**: Vote transfer flows during STV elimination rounds
- **Radar charts**: Candidate ideological profiles across clusters
- **Parliament semicircles**: Seat distribution by party

---

## Files

| File | Description |
|------|-------------|
| `five_member_proportional.ipynb` | Main analysis notebook (all clustering, elections, and visualizations) |
| `DemRep.ipynb` | Baseline Democratic/Republican two-party analysis |
| `ces_questions_VERIFIED.md` | Verified CES question text and coding (reference) |
| `ces_questions_COMPLETE.md` | Complete CES question text across all dimensions (reference) |
| `ces_question_selection.md` | Early-stage feature selection analysis (historical) |
| `feature_audit.py` | Script for auditing CES feature availability |
| `dataverse_files/` | CES 2024 survey data and codebooks |

---

## Key Takeaways

1. **America is not binary**: Six distinct political clusters exist, but the two-party system forces them into uncomfortable coalitions.

2. **STV produces proportional primaries**: The Droop quota ensures that clusters with significant support get candidates through to the general, while elimination rounds remove fringe candidates.

3. **Vote transfers matter**: Using all 50 policy questions for transfer similarity (rather than a small subset) prevents disproportionate leakage to centrist candidates and preserves ideological proportionality.

4. **IRV favors broad-appeal candidates**: Senate general elections (IRV) produce winners who are acceptable to a majority, though not necessarily the first choice of any plurality.

5. **Condorcet finds the true center**: Presidential general elections using Condorcet consistently select the candidate who would beat any other in a head-to-head matchup.

6. **Proportional allocation fixes the Electoral College**: When states split electors proportionally, the national popular vote winner almost always wins, eliminating inversions.

7. **Geographic pods ensure cultural balance**: No single region can dominate a primary when states are grouped to offset each other's biases.

---

## Requirements

```
pandas
numpy
scikit-learn
plotly
scipy
jupyter
```

## Usage

Run the Jupyter notebook cells in order. The analysis builds progressively:

1. **Data loading and cleaning** (Cells 1-10)
2. **Feature selection and clustering** (Cells 11-21)
3. **Cluster profiling and demographics** (Cells 22-34)
4. **State-level party share calculations** (Cells 35-48)
5. **Candidate creation and vote model** (Cells 49-53)
6. **Presidential primary simulation** (Cells 54-60)
7. **Presidential general election** (Cells 61-70)
8. **Senate primary and general elections** (Cells 71-80)
9. **Horserace analysis and cross-tabs** (Cells 81-82)

## License

For educational and research purposes.

---

*"The test of a first-rate intelligence is the ability to hold two opposed ideas in mind at the same time and still retain the ability to function."* -- F. Scott Fitzgerald

This simulation holds six.
