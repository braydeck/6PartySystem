# CES 2024 Policy Question Selection for Clustering Analysis

## Executive Summary

**Total questions analyzed:** 126 valid policy questions  
**Redundant pairs (r>0.70):** Only 8 pairs - minimal redundancy  
**Recommendation:** Use **40-50 questions** across 5 thematic dimensions

## Key Finding: Cross-Cutting Populist Faction

Using just TWO questions (CC24_312a for economics, CC24_323c for social issues), we found:

### **Voter Quadrants:**
- **Populist (Econ Left + Social Right): 36.9%** ⭐ **LARGEST GROUP**
- **Libertarian (Econ Right + Social Left): 36.4%**
- **Progressive (Econ Left + Social Left): 8.3%**
- **Conservative (Econ Right + Social Right): 7.0%**
- **Moderate (middle on both): 11.4%**

**The "socially conservative, economically liberal" faction is real and MASSIVE (37%)!**

Demographics of Populist voters:
- 72% identify as Democrats
- 60% self-identify as liberal/very liberal on ideology scale
- BUT they hold socially conservative positions
- This group is currently forced into the Democratic coalition despite value misalignment

---

## Recommended Question Set (48 Questions)

### **Dimension 1: SOCIAL/CULTURAL ISSUES** (12 questions)

**High-salience social issues:**
- `CC24_301` (r=+0.46) - Likely abortion or religious freedom
- `CC24_323c` (r=-0.49) - Key social dimension
- `CC24_323a` (r=+0.37) 
- `CC24_323d` (r=+0.36)
- `CC24_323f` (r=+0.39)
- `CC24_323b` (r=-0.26)

**Immigration/border:**
- `CC24_326a` (r=+0.42)
- `CC24_326b` (r=+0.38)
- `CC24_326c` (r=+0.43)
- `CC24_326d` (r=-0.41)
- `CC24_326e` (r=+0.41)
- `CC24_326f` (r=-0.20)

### **Dimension 2: ECONOMIC POLICY** (12 questions)

**Government role in economy:**
- `CC24_312a` (r=+0.53) - Core economic question
- `CC24_312i` (r=+0.54) - Highly correlated with 312a
- `CC24_312c` (r=-0.28)
- `CC24_312g` (r=+0.15)

**Specific economic policies:**
- `CC24_321a` (r=+0.41)
- `CC24_321b` (r=-0.36)
- `CC24_321c` (r=+0.16)
- `CC24_321d` (r=-0.33)
- `CC24_321e` (r=+0.31)
- `CC24_321f` (r=+0.22)

**Taxation/redistribution:**
- `CC24_328b` (r=+0.28)
- `CC24_328d` (r=-0.42)

### **Dimension 3: GOVERNANCE/POPULISM** (10 questions)

**Government trust/competence:**
- `CC24_324a` (r=+0.44)
- `CC24_324b` (r=-0.33)
- `CC24_324c` (r=-0.19)
- `CC24_324d` (r=+0.45)

**Institutional trust:**
- `CC24_341a` (r=-0.34)
- `CC24_341b` (r=+0.40)
- `CC24_341c` (r=+0.35)
- `CC24_341d` (r=+0.23)

**Trade/globalization (populist dimension):**
- `CC24_328c` (r=-0.22)
- `CC24_328e` (r=+0.27)

### **Dimension 4: CRIMINAL JUSTICE/LAW & ORDER** (8 questions)

**Law enforcement:**
- `CC24_340a` (r=+0.22)
- `CC24_340b` (r=+0.31)
- `CC24_340c` (r=+0.41)
- `CC24_340d` (r=-0.16)
- `CC24_340e` (r=-0.04)
- `CC24_340f` (r=-0.40)

**Justice reform:**
- `CC24_361b` (r=+0.32)
- `CC24_364b` (r=+0.37)

### **Dimension 5: FOREIGN POLICY/INTERNATIONALISM** (6 questions)

**Military/defense:**
- `CC24_308a_1` (r=-0.24)
- `CC24_308a_2` (r=+0.32)
- `CC24_308a_3` (r=+0.28)
- `CC24_308a_4` (r=+0.31)
- `CC24_308a_5` (r=+0.26)
- `CC24_308a_8` (r=+0.32)

---

## Question Batteries Identified

Based on correlation analysis:

1. **CC24_300 series** (5 questions) - Favorability/approval ratings (LOW ideology correlation ~0.07)
2. **CC24_301** (1 question) - Core social issue (HIGH correlation 0.46)
3. **CC24_302-303** (2 questions) - Policy opinions (MEDIUM correlation 0.22-0.25)
4. **CC24_305 series** (13 questions) - Issue priorities (LOW correlation ~0.04)
5. **CC24_308a series** (8 questions) - Foreign policy battery (MEDIUM-HIGH avg 0.24)
6. **CC24_308b series** (9 questions) - Domestic policy battery (MEDIUM-HIGH avg 0.19)
7. **CC24_312 series** (9 questions) - Economic policy (HIGH correlation, 312a/i at 0.53)
8. **CC24_321 series** (6 questions) - Economic specifics (MEDIUM-HIGH 0.16-0.41)
9. **CC24_323 series** (5 questions) - Social issues (VERY HIGH 0.26-0.49)
10. **CC24_324 series** (4 questions) - Governance (HIGH 0.19-0.45)
11. **CC24_326 series** (6 questions) - Immigration (HIGH 0.20-0.43)
12. **CC24_328 series** (5 questions) - Mixed economic/trade (MEDIUM 0.14-0.42)
13. **CC24_330 series** (11 questions) - Candidate choice (330a is HIGHEST at 0.83!)
14. **CC24_340-341 series** (10 questions) - Criminal justice (MEDIUM-HIGH 0.04-0.41)

---

## Redundancy Removal

**High correlation pairs to address (r>0.70):**

1. CC24_330c ↔ CC24_330d (r=0.89) - **Drop one**
2. CC24_312a ↔ CC24_312i (r=0.88) - **Keep both** (core economic)
3. CC24_310c ↔ CC24_310d (r=0.88) - **Drop one**
4. CC24_330d ↔ CC24_330f (r=0.84) - **Drop 330d** (already correlated with 330c)
5. CC24_330c ↔ CC24_330f (r=0.83) - **Drop 330c**
6. CC24_300d_3 ↔ CC24_300d_6 (r=-0.81) - **Keep both** (opposite poles)
7. CC24_324a ↔ CC24_324d (r=0.72) - **Keep both** (governance is key)
8. CC24_310a ↔ CC24_310b (r=0.71) - **Drop one**

**Final recommendation:** Remove CC24_330c, CC24_330d, CC24_310c, CC24_310a from selection

---

## Why This Selection Works

1. **Comprehensive coverage:** 5 distinct policy dimensions
2. **Cross-cutting issues:** Captures economic-left/social-right voters
3. **Minimal redundancy:** Only 8 pairs >0.70 correlation
4. **Balanced:** ~8-12 questions per dimension
5. **Empirically validated:** Each question shows meaningful ideology variation
6. **Populist-sensitive:** Includes trade, immigration, institutional trust questions that separate populists from progressives

---

## Expected Cluster Outcomes

Based on this question set, we expect to find:

1. **Progressive Left** (~10-15%): Econ left + social left + internationalist
2. **Establishment Liberal** (~20-25%): Moderate econ + social left + pro-institution
3. **Populist/Working Class** (~20-30%): Econ left + social right + anti-trade
4. **Libertarian** (~10-15%): Econ right + social left + small government
5. **Conservative** (~15-20%): Econ right + social right + traditional
6. **Moderate/Centrist** (~10-15%): Middle on most dimensions

The **Populist faction** (currently 37% in 2D analysis) will likely split into a true "populist party" and some moderates once we add more dimensions.

---

## Next Steps

1. Load these 48 questions into clustering algorithm
2. Standardize all variables (z-scores)
3. Run k-means for k=4,5,6,7
4. Use silhouette scores and substantive interpretation to select optimal k
5. Profile each cluster by demographics, geography, and policy positions
6. Assign party labels based on cluster characteristics
