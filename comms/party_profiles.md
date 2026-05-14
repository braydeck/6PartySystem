# American Political Party Profiles

*A guide to the 23 political types in a proportional representation simulation based on CES 2024 survey data (N=45,707)*

---

## Introduction

This document profiles the political parties and coalition types that emerge when you run American survey data through a proportional representation system. The underlying data comes from the 2024 Cooperative Election Study, one of the largest academic surveys of American political opinion, with responses from over 45,000 adults weighted to match the national population. These parties were not invented — they were discovered. Researchers applied clustering algorithms to how Americans actually answered 172 policy and demographic questions, and the nine parties described below are the natural groupings that fell out. The question this project asks is simple: if Americans could vote for a party that actually matched their views, what would the map look like?

Five underlying dimensions replace the single left-right axis that dominates American political conversation. The first is **Security & Order** — whether someone wants more police, more border enforcement, and more surveillance. The second is **Electoral Skepticism** — whether someone believes American elections are run fairly. The third is **Government Distrust** — general low trust in federal and state institutions (this dimension turns out to be nearly identical across all winning parties and does not differentiate them). The fourth is **Religious Traditionalism** — the weight someone gives to church attendance, traditional views on abortion, and traditional views on marriage. The fifth is **Populist Conservatism** — the cluster of attitudes around immigration restriction, fiscal conservatism, and racial traditionalism that defines the populist right. Some party profiles look contradictory at first glance. They usually stop looking contradictory once you see all five dimensions at once.

No party holds a majority. The simulation allocates 873 total seats across the House and Senate, and the three largest parties — Social Democrat (166 seats, 19%), Conservative (164 seats, 18.8%), and Solidarity (160 seats, 18.3%) — collectively hold just over half. Every legislative majority requires coalition-building across parties that disagree on at least two or three fundamental issues. That is the design. The system forces the negotiation that the current two-party structure conceals behind partisan loyalty.

These parties do not map cleanly onto Democrat or Republican. The Reform Party supports raising the corporate tax rate above what most Democrats have managed to pass. Solidarity has near-MAGA-level distrust of elections but supports labor unions, DACA, and opposes increased police funding. The Liberal Party supports universal background checks and has the highest gun ownership rate of any left-leaning party. The Center Party voted for Trump by a plurality (41.9% Trump, 33.2% Biden) — yet it is the most trusting of elections of any party in the entire system. The coalitions are real. They are just new.

---

## The Presidency

**IRV and Condorcet produce different presidents — but they are nearly identical candidates.**

Under Instant Runoff Voting, the winner is **CON/SD** (50.43% vs. CON/STY 49.57% in the final round): a Conservative whose coalition must also speak to Social Democrat voters. Under Ranked-Pairs (Condorcet), the winner is **SD/CON**: a Social Democrat whose coalition must speak to Conservative voters. SD/CON beats CON/SD in a head-to-head matchup by 50.49% to 49.51% — a margin of 223 votes out of 45,000. The two candidates sit at nearly the same position in factor space; the difference is which party claims the dominant identity on the ticket.

The divergence illustrates a structural property of the two methods. IRV rewards first-choice accumulation: CON/SD, running with the Conservative label leading, attracts more first-choice votes from the CON base and survives elimination rounds until it faces CON/STY in the final. Ranked-Pairs rewards global head-to-head performance: SD/CON, with the slightly more centrist SD-leading identity, wins marginally more pairwise contests across the full field. The result is two different presidents who would govern almost identically.

Both CON/SD and SD/CON sit at the genuine cross-aisle midpoint of the factor space, slightly conservative on Security & Order, trusting of elections, moderately religious, and near-zero on Populist Conservatism. In practice either president signs increased border patrol funding, asylum denial at the border, extension of the 2017 tax structure, universal background checks, the $150B infrastructure bill, and permanent DACA reform. Either vetoes outright ACA repeal — the SD coalition partner cannot support dismantling coverage for tens of millions — and declines to sign the post-9/11 surveillance renewal or the gas stove ban. Conservatives call the DACA signature a betrayal; progressives call the border enforcement a capitulation. The multi-party arithmetic calls it the only governing position with majority support. The legislative vote model below uses **CON/SD** (the IRV winner) as the reference president for bill-signing outcomes.

**CON/SD factor profile:** F1 +0.24 (security-oriented) · F2 −0.03 (trusts elections) · F4 −0.03 (moderate-secular) · F5 −0.01 (true cross-aisle centrist)
**SD/CON factor profile:** F1 +0.15 · F2 −0.03 · F4 −0.07 · F5 −0.08 (slightly more center-left across all dimensions)

**Pure-party scenario:** When candidates are restricted to pure single-party nominees — no coalition blends — both methods converge on the same president: **Solidarity (STY)**. STY wins as the spatial median of a nine-party field. Under IRV, LIB is eliminated first, then CON, then SD, leaving STY to prevail over CTR 50.54% to 49.46% in the final round. Under Ranked-Pairs, STY is also the winner. The STY president signs progressive economic measures, Medicaid expansion, infrastructure, and DACA, while vetoing immigration restriction bills and tax-cut extensions. The three scenarios — mixed IRV, mixed Condorcet, and pure-party — all produce different presidents, which illustrates how much both the candidate field structure and the counting method matter before a single governing vote is cast.

---

## Part I: The Nine Parties

---

### Progressive (PRG) — 8 House Seats (0.9%)

Progressives are the smallest party in the system, but they are not an outlier — they are a pole. They sit at the extreme end of nearly every dimension: the most anti-enforcement, the most secular, the most economically redistributive, the most permissive on immigration. What animates them is not just policy positions but a coherent worldview in which government's primary function is to actively reduce inequality — in income, in race, in gender, in access to healthcare. They want federal power used aggressively and unapologetically toward those ends. The fact that only 8 House seats correspond to this profile reflects not that these views are rare, but that they are concentrated: young, urban, highly educated, and dispersed across districts that already elect other left-leaning candidates.

**Who they are:**
- Median age 41 · 53.1% male · 76.3% white · 60.2% 4-yr+ degree · 32.6% city
- Income: upper-middle ($70–80k) · 43.9% married · 11.4% own a gun · 28.1% LGBTQ+ · 4.3% born-again evangelical
- 9.3% current union members · 4.3% veterans

**Where they stand:**
- Support DACA/Dreamer pathway: 96.9%
- Renewable energy mandate: 97.3%
- Medicaid expansion: 97.6%
- Assault rifle ban: 87.2%
- Raise corporate taxes: 90.6%

**Unusually for American politics:**
- Only 23.8% say national elections are run unfairly — the most institutionally trusting of any far-left party. Despite holding radical positions on nearly every policy issue, Progressives show a deep commitment to the democratic process itself, distinguishing them sharply from DSA, where electoral skepticism runs twice as high.
- Despite averaging age 41 and being 28.1% LGBTQ+, PRG is 53% male — a reminder that progressive politics draws heavily from educated men alongside its more visible feminist and queer constituencies.

**Senate seats:** 0 as a pure party in all four scenarios. PRG's support is too concentrated in a handful of urban House districts to command a state-level plurality. Senate presence comes only through the PRG/DSA coalition blend, which captures Vermont under mixed-scenario IRV.

---

### Democratic Socialists (DSA) — 26 House Seats (3.0%)

The Democratic Socialists are ideologically close to Progressives on economic and social policy, but they come from a different cultural place. DSA has the highest LGBTQ+ share of any party — nearly four in ten members identify as LGBTQ+ — and the youngest median age (33) in the system. What distinguishes DSA from PRG is not what they want from government but how they feel about the government they have. DSA members are deeply skeptical that American elections are run fairly, a sentiment that sits in sharp tension with their progressive politics but makes sense when you consider a community with long experience of disenfranchisement and structural exclusion. They want universal healthcare, aggressive redistribution, and an end to immigration enforcement — and they do not trust the current system to deliver any of it.

**Who they are:**
- Median age 33 · 32.8% male · 65.9% white · 42.7% 4-yr+ degree · 30.0% city
- Income: middle ($60–70k) · 29.0% married · 9.8% own a gun · 38.6% LGBTQ+ · 7.3% born-again evangelical
- 7.1% current union members · 3.0% veterans

**Where they stand:**
- Support DACA/Dreamer pathway: 93.1%
- Medicaid expansion: 95.5%
- EPA CO2 regulation: 87.1%
- Raise corporate taxes: 87.0%
- Assault rifle ban: 79.4%

**Unusually for American politics:**
- 52.7% say national elections are not run fairly — the highest of any explicitly left-wing party, putting DSA in the same bracket as Nationalist (63.1%) and Reform (61.6%). Electoral distrust is not a right-wing monopoly; for a party that is 38.6% LGBTQ+ and 34.1% non-white, there are independent reasons to doubt whether the system delivers equal treatment.
- DSA has the lowest marriage rate of any party (29.0%) and among the highest non-voting rates in 2020 (27.5% did not vote), a combination that reflects a young, non-traditional left base that remains substantially disengaged from electoral participation despite deep political conviction.

**Senate seats:** 0 as a pure party in all four scenarios. Like PRG, DSA's vote share never commands a state-level plurality alone; its senate presence comes only through the PRG/DSA coalition blend in Vermont under mixed-scenario IRV.

---

### Liberal (LIB) — 100 House Seats (11.5%)

Liberals are the third-largest left-leaning party, and in many ways the most surprising. They are the oldest left-leaning party (median age 55), whiter and more male than their ideological neighbors, and the least LGBTQ+ of the non-conservative parties. Their urban share (29.7% city) is nearly identical to Social Democrat (30.7%) — LIB is not particularly rural, just suburban and spread across smaller metros rather than dense urban cores. What unifies them is a civil libertarian streak: they want government out of the bedroom and the boardroom alike, they trust electoral institutions, and they are skeptical of both police power and immigration enforcement. Their economics are genuinely left — support for taxing high earners, EPA regulation, renewable mandates — but they are also the most ambivalent left-leaning party on a range of social issues, from transgender policy to abortion access. They are older, relatively prosperous, and more likely to personally own a gun than PRG or DSA, while still supporting background checks at 86.5%.

**Who they are:**
- Median age 55 · 52.0% male · 69.4% white · 48.5% 4-yr+ degree · 29.7% city
- Income: upper-middle ($70–80k) · 49.1% married · 17.0% own a gun · 14.3% LGBTQ+ · 14.1% born-again evangelical
- 8.0% current union members · 8.0% veterans

**Where they stand:**
- Universal background checks: 86.5%
- Support DACA/Dreamer pathway: 81.3%
- EPA CO2 regulation: 73.1%
- Renewable energy mandate: 76.9%
- Medicaid expansion: 71.7%

**Unusually for American politics:**
- 86.5% support universal background checks, and 17.0% personally own guns — the most gun-owning left-leaning party in the system by a wide margin, and notably higher than PRG (11.4%) or DSA (9.8%). Their libertarianism is real but it cuts both ways: they trust individuals with firearms while also trusting elections (only 24.2% say national elections are unfair, the lowest of any non-CTR party).
- Only 29.9% support gender-transition surgery for minors — far below the PRG (68%) and DSA (57.5%) levels — reflecting a genuine social conservatism on gender issues that coexists with broadly progressive economics.

**Senate seats — mixed (coalition blends allowed):** Condorcet 1 *(Rhode Island)* · IRV 1 *(Rhode Island)* · **Pure-party:** Condorcet 0 · IRV 0. LIB's civil-libertarian profile wins Rhode Island as a pure senator in both mixed-scenario methods but cannot command a plurality in any state once blended candidates are removed from the field.

---

### Social Democrat (SD) — 166 House Seats (19.0%)

Social Democrats are the largest party in the system and, in many ways, the most recognizable to European observers of American politics. They want an expanded welfare state, regulated markets, labor protections, and immigration reform — the European center-left platform, applied to American conditions. They are disproportionately Black, Hispanic, and working-class, with a median age matching the national center and ideological self-placement of 2.9 on a 7-point scale. They trust elections more than most parties, support Medicaid expansion at near-universal rates, and are the strongest backers of clean energy in the system. What makes them politically complicated is their working-class base, which produces genuine ambivalence on immigration — not hostility, but a calculation about labor market competition and community change that distinguishes them from their left-leaning neighbors.

**Who they are:**
- Median age 43 · 44.6% male · 56.7% white · 35.1% 4-yr+ degree · 30.7% city
- Income: middle ($60–70k) · 37.5% married · 18.2% own a gun · 14.6% LGBTQ+ · 19.2% born-again evangelical
- 6.4% current union members · 6.7% veterans

**Where they stand:**
- Medicaid expansion: 93.5%
- Renewable energy mandate: 91.0%
- Support DACA/Dreamer pathway: 89.5%
- Infrastructure spending: 88.7%
- Universal background checks: 94.7%

**Unusually for American politics:**
- 43.2% support increasing border patrols — nearly half — reflecting a working-class base more ambivalent on immigration than a "liberal" label would suggest. SD simultaneously supports DACA at 89.5% while backing border enforcement at nearly the same rate as Center (65.9%), a combination that makes sense as a labor-protection rather than nativist position.
- SD has the highest combined union affiliation of any party — 6.4% currently active, with an additional 15.8% who are former union members (22.2% total union household experience) — yet only 4.3% voted Trump in 2020. SD's labor identity is real and shapes its economic politics even among members who no longer hold active membership.

**Senate seats — mixed (coalition blends allowed):** Condorcet 6 *(IL, MA, MI, OR, SC, WA)* · IRV 6 *(IL, MA, MI, OR, WA, WI)* · **Pure-party:** Condorcet 16 *(AK, CA, CT, DE, IL, KS, MD, MA, MS, MT, NJ, NM, OR, PA, RI, WA)* · IRV 14 *(AK, CA, CT, DE, FL, IL, KS, MD, MA, NM, OR, PA, RI, WA)*. SD is the dominant party in the pure-party scenario, nearly tripling its seat count when coalition blends are removed and the electorate sorts to its spatial center.

---

### Solidarity (STY) — 160 House Seats (18.3%)

Solidarity is the most politically disorienting party in the system, and arguably the most important one to understand. They are the youngest working-class party (median age 38), majority female (60.7%), majority non-white (56.6%), lowest household income of any party in the system, and — most surprisingly — 37.7% born-again evangelical. Their economics lean center-left: they support infrastructure spending and Medicaid expansion, but express ambivalence on immigration and are more skeptical of corporate tax hikes than their left-leaning neighbors. What makes STY genuinely novel is their combination of economic marginalization with deep institutional distrust: 55.8% say national elections are not run fairly, a level matching Nationalist and Reform. Nearly half (44%) didn't vote in 2020. They are the party of Americans most left behind by the economy — low income, limited education, religiously conservative, and alienated from both political parties.

**Who they are:**
- Median age 38 · 39.3% male · 43.4% white · 23.0% 4-yr+ degree · 36.3% city
- Income: working-class ($40–50k) · 34.2% married · 18.8% own a gun · 11.7% LGBTQ+ · 37.7% born-again evangelical
- 4.4% current union members · 5.5% veterans

**Where they stand:**
- Universal background checks: 89.4%
- Support DACA/Dreamer pathway: 72.6%
- Medicaid expansion: 83.0%
- Raise corporate taxes: 73.1%
- Oppose increasing police funding: 63.4%

**Unusually for American politics:**
- Despite 55.8% believing elections aren't fair — a rate matching Nationalist (63.1%) and Reform (61.6%) — STY voted 88% for non-Trump candidates in 2020 and has a 44% non-voting rate. Only 16.3% voted Trump. STY's electoral skepticism comes from the left, not the right: for a majority non-white, lowest-income, female-majority party, distrust of institutions reflects lived experience of exclusion rather than election denialism.
- STY is the only party that combines majority non-white membership, majority female membership, the lowest income in the system, and 37.7% born-again evangelical identity — a demographic profile that fits no existing political coalition and explains why they are estranged from both parties rather than firmly attached to either.

**Senate seats — mixed (coalition blends allowed):** Condorcet 5 *(AZ, UT, VT, WI, WY)* · IRV 2 *(UT, WY)* · **Pure-party:** Condorcet 10 *(CO, GA, LA, ME, MI, NV, NH, NY, SC, TX)* · IRV 8 *(CO, LA, MI, MN, NV, NH, SC, WY)*. STY's seat count doubles in the pure-party scenario as its spatial median position prevails across a wider set of states once coalition blends are no longer available to absorb its voters.

---

### Center (CTR) — 102 House Seats (11.7%)

The Center Party occupies a genuinely unusual political position: moderate on almost everything, deeply religious, and the most trusting of elections of any party in the entire system. Its members describe themselves as slightly conservative (3.6 on the 7-point scale) and voted 41.9% for Trump and 33.2% for Biden in 2020 — making them a Trump-plurality party that nonetheless trusts elections more than any other constituency in the simulation. They are significantly evangelical (43% born-again), older (median age 51), homeowning, and married — the religious moderate who is economically pragmatic, culturally traditional without being extreme, and the genuine swing-right voter the two-party system has long fought over. They support background checks, Medicaid expansion, and infrastructure spending, but also back border enforcement and are split on abortion.

**Who they are:**
- Median age 51 · 54.4% male · 66.2% white · 29.0% 4-yr+ degree · 27.2% city
- Income: lower-middle ($50–60k) · 47.5% married · 29.0% own a gun · 5.5% LGBTQ+ · 43.0% born-again evangelical
- 5.4% current union members · 8.7% veterans

**Where they stand:**
- Universal background checks: 89.4%
- Medicaid expansion: 74.8%
- Infrastructure spending: 74.5%
- Increase border patrols: 65.9%
- Support same-sex marriage: 77.7%

**Unusually for American politics:**
- CTR has the lowest electoral skepticism in the entire system — only 17.5% say national elections are not run fairly, lower even than Social Democrat (19.7%). Yet 41.9% voted for Trump in 2020 and only 33.2% for Biden, making CTR the most genuinely swing-right party in the system, not the "reluctant Democrat" the two-party framing would suggest. They lean Trump but trust the elections Biden won — a combination that is genuinely unusual.
- Despite being 43% born-again evangelical and favoring stricter abortion limits (median 16 weeks), 77.7% support federal recognition of same-sex marriage — suggesting a religious conservatism focused on reproductive rather than sexual ethics.

**Senate seats — mixed (coalition blends allowed):** Condorcet 1 *(AK)* · IRV 2 *(AK, SC)* · **Pure-party:** Condorcet 15 *(AZ, AR, DC, FL, HI, ID, IA, MN, MO, NE, TN, UT, VT, WV, WY)* · IRV 13 *(AZ, AR, DC, HI, ID, IA, MS, NE, NY, TN, UT, VT, WV)*. CTR's moderate, trusting profile sits at the spatial center of the pure-party field, making it the second-largest senate party in both pure scenarios despite winning only 1–2 seats in the mixed pipeline where coalition blends absorb its voters.

---

### Conservative (CON) — 164 House Seats (18.8%)

Conservatives are the second-largest party and the dominant force on the center-right. They are high on Security & Order and Religious Traditionalism, low on Electoral Skepticism, and sitting at a moderate +0.46 on Populist Conservatism — meaningfully right but not at the populist extreme. They are older, whiter, and more married than the national average, with above-average incomes, and near gender parity (49% male). Their politics are recognizable as the Eisenhower-to-Reagan Republican tradition: law enforcement, border security, religious values, low taxes, and a belief that American institutions basically work. They are hawkish on immigration and police funding, opposed to most climate regulation, and split nearly in half on some social issues. What distinguishes them from NAT is not just intensity but institutional orientation — Conservatives trust elections and, to a striking degree, trust background checks.

**Who they are:**
- Median age 59 · 49.0% male · 78.4% white · 30.3% 4-yr+ degree · 20.5% city
- Income: middle ($60–70k) · 56.2% married · 39.2% own a gun · 3.4% LGBTQ+ · 42.6% born-again evangelical
- 4.1% current union members · 13.6% veterans

**Where they stand:**
- Extend 2017 tax cuts: 89.2%
- Increase border patrols: 91.2%
- Increase police funding: 84.6%
- Deny asylum seekers: 80.1%
- ACA repeal: 73.9%

**Unusually for American politics:**
- 72.3% of Conservatives support universal background checks on gun sales — nearly three in four. This is the signature cross-cutting finding for CON: a party that is 39.2% gun-owning and 91.2% pro-border-patrol nonetheless supports a policy their partisan analog in the two-party system has blocked for decades. Their gun politics are about ownership and rights, not about blocking all regulation.
- Only 31.7% say national elections are unfair — nearly identical to the overall national average — reflecting an institutional conservatism that CON shares with CTR but not with REF (61.6%) or NAT (63.1%).

**Senate seats — mixed (coalition blends allowed):** Condorcet 3 *(LA, OK, TN)* · IRV 3 *(LA, OK, TN)* · **Pure-party:** Condorcet 9 *(AL, IN, KY, NC, OH, OK, SD, VA, WI)* · IRV 12 *(AL, GA, IN, KY, MO, MT, NJ, OH, SD, TX, VA, WI)*. CON's seat count triples in the pure-party scenario, and unusually it gains more seats under IRV than Condorcet — reflecting that CON accumulates first-choice votes in contested states more effectively than it wins head-to-head matchups against the centrist field.

---

### Reform (REF) — 125 House Seats (14.3%)

Reform is the Tea Party and MAGA populism stripped of the Republican establishment — economically skeptical of concentrated corporate power, deeply suspicious that elections are run fairly, intensely restrictionist on immigration, and socially conservative without being particularly evangelical. Reform members are older (median age 46), majority male (51.2%), rural, and low-education relative to income. They feel economically left behind — among the highest perceived inflation of any party — and they channel that anxiety into distrust of both elections and elites. What makes Reform genuinely different from Conservative is their populist skepticism: 61.6% say national elections are unfair, and their support for corporate taxation runs higher than most centrist parties. They are not pro-business. They are pro-themselves, against a system they believe is rigged by people who are not like them.

**Who they are:**
- Median age 46 · 51.2% male · 70.2% white · 25.8% 4-yr+ degree · 23.1% city
- Income: lower-middle ($50–60k) · 45.5% married · 37.3% own a gun · 7.1% LGBTQ+ · 39.9% born-again evangelical
- 4.6% current union members · 9.5% veterans

**Where they stand:**
- Increase border patrols: 88.9%
- Deny asylum seekers: 77.4%
- Extend 2017 tax cuts: 83.1%
- ACA repeal: 61.4%
- Elections not fair (national): 61.6%

**Unusually for American politics:**
- 26.4% of Reform members support raising the corporate tax rate — higher than what most centrist Democrats manage in Republican-leaning districts, and notable for a party with near-universal immigration restriction. Reform is economically populist, not pro-business: they want the corporate elite taxed even as they support the rest of the conservative policy agenda.
- 76.7% support universal background checks despite being the most immigration-restrictionist party outside of NAT — suggesting gun regulation opposition in the two-party system reflects partisan packaging more than genuine voter preference.

**Senate seats — mixed (coalition blends allowed):** Condorcet 0 · IRV 0 · **Pure-party:** Condorcet 1 *(ND)* · IRV 4 *(ME, NC, ND, OK)*. REF wins no senate seats in the mixed pipeline — its voters are absorbed into CON/REF blends in states where it runs competitively. In the pure-party scenario REF breaks through, particularly under IRV where its first-choice plurality in low-competition states survives elimination rounds.

---

### Nationalist (NAT) — 22 House Seats (2.5%)

The Nationalist Party is the populist far-right pole of the system. At +1.51 on Populist Conservatism — a full standard deviation above the next highest type — they are an outlier even within a system designed to find outliers. They are the oldest party (median age 60), the most male (63.6%), the most rural (only 15.9% city), and the most evangelical (47.7% born-again) — but they are also the second highest income party in the simulation, and more college-educated (39.3%) than Conservative (30.3%) or Reform (25.8%). NAT is not the economically precarious working class; it is prosperous older white rural Christians who have organized their politics around immigration restriction and cultural anxiety rather than economic grievance. They want borders enforced, firearms secured, tradition maintained, and a specific vision of American national identity preserved. Their 89.4% Trump support in 2020 and near-universal immigration restriction coexist with household incomes that benefit substantially from the tax cuts they demand.

**Who they are:**
- Median age 60 · 63.6% male · 81.2% white · 39.3% 4-yr+ degree · 15.9% city
- Income: upper-middle ($70–80k) · 63.1% married · 54.9% own a gun · 3.3% LGBTQ+ · 47.7% born-again evangelical
- 5.5% current union members · 18.1% veterans

**Where they stand:**
- Increase border patrols: 97.1%
- Deny asylum seekers: 94.3%
- Extend 2017 tax cuts: 93.8%
- Oppose same-sex marriage: 65.0% (only 35% in favor)
- Elections not fair (national): 63.1%

**Unusually for American politics:**
- Despite being the most evangelical party (47.7% born-again) and the most rural (15.9% city), NAT ranks second in household income — a profile of prosperous older white rural Christians, not the economically precarious working class its anti-establishment rhetoric might suggest. NAT is more college-educated (39.3%) than Conservative (30.3%) or Reform (25.8%), and their near-universal support for the 2017 tax cuts aligns with their actual economic position rather than contradicting it.
- 63.3% support universal background checks — a majority, even in the most nativist and gun-owning party in the system.

**Senate seats:** 0 in all four scenarios. NAT's votes are absorbed into CON/NAT-adjacent blends in the mixed pipeline, and in the pure-party scenario its extreme F5 position makes it a non-plurality candidate in every state — including the rural states where it has the strongest House presence.

---

## Part II: Senate Coalition Types

Senate candidates in this simulation emerge when two voter clusters co-occur frequently enough within a state that a politician must credibly appeal to both simultaneously. A CON/SD senator represents a state where enough voters hold a mixture of conservative and social-democratic views that no pure-party candidate can win. The first-named party is the dominant orientation, carrying roughly 55–67% of the weight; the second names the constituency that pulls the candidate toward positions their base party would not normally hold. Each blend type is listed with seat counts under both the IRV and Condorcet runoff methods; the two methods produce meaningfully different outcomes in 9 states. Several states also elect pure single-party senators — those are listed at the end of this section. The presidency (CON/SD) is addressed in its own section above.

---

### SD/STY — IRV: 8 seats · Condorcet: 8 seats

*IRV: California, DC, Hawaii, Iowa, Maryland, Minnesota, Nevada, New York*
*Condorcet: California, DC, Hawaii, Iowa, Maryland, Minnesota, Nevada, New York*

The SD/STY senator represents the dominant coalitional type on the center-left: a Social Democrat whose base is large enough that they must also speak to Solidarity voters — more working-class, more religiously observant, more skeptical that elections are fair. Compared to a pure SD candidate, the SD/STY blend is noticeably less likely to support same-sex marriage recognition (the biggest policy shift), more skeptical of elections (17 points of range shift toward distrust), and more ambivalent on abortion restrictions. Economically, the blend moves slightly toward fiscal caution — less support for the full $150B infrastructure package, slightly less support for upper-bracket tax increases. What remains stable is the core: immigration reform, labor support, and climate policy. This is the senator who wins California or Minnesota by holding together educated liberals and working-class voters of color.

The surprising combination: despite moving toward Solidarity on nearly every cultural axis, the SD/STY blend is *more secular* than pure SD — attendance frequency shifts down — because STY's high Black church attendance is paired with more political independence from church doctrine.

**Factor profile:** Low F1 (anti-enforcement), Medium-Low F2 (modest electoral skepticism), Low F4 (secular-leaning), Low F5 (progressive-populist).

---

### CON/SD — IRV: 6 seats · Condorcet: 6 seats

*IRV: Colorado, Georgia, Nebraska, New Jersey, Pennsylvania, Virginia*
*Condorcet: Colorado, Georgia, Nebraska, New Jersey, Pennsylvania, Virginia*

The CON/SD senator is the most counterintuitive type in the system: a Conservative who must also speak to Social Democrats in a purple-to-blue-leaning state. The blend moves CON dramatically toward SD on police funding (the largest single shift: +30.5% of range toward more police skepticism), same-sex marriage recognition (CON moves −29 points toward SD's near-universal support), church attendance, abortion timing, and Dreamer support. Electoral skepticism stays essentially unchanged — neither CON nor SD is particularly skeptical about elections, so the blend inherits that stability. The result is a Conservative who sounds like a mainstream Republican on immigration and taxes but who breaks sharply with the right on social and policing issues. This is the senator who can win Pennsylvania or Virginia by running culturally conservative on economics and culturally moderate on social issues.

The surprising combination: CON/SD is the most significant cross-ideological blend in the system — moving a full 29 points of range on same-sex marriage and 30 points on police funding while keeping the core of Conservative fiscal policy largely intact.

**Factor profile:** Medium F1 (moderate on security), Medium F2 (trusts elections), Medium F4 (moderate religious traditionalism), Medium F5 (near-center on populism).

---

### CON/STY — IRV: 4 seats · Condorcet: 4 seats

*IRV: Alabama, Arkansas, Connecticut, New Mexico*
*Condorcet: Alabama, Arkansas, Connecticut, New Mexico*

The CON/STY senator holds the dominant center-right coalition in Southern and border states: a Conservative who must also speak to Solidarity voters — the working-class, electorally skeptical, Black-and-rural constituency that makes up a substantial part of the Southern electorate. Compared to pure CON, this blend is more skeptical of elections (the second-largest shift, +22.7% on local election fairness), more supportive of police skepticism, more attentive to immigration enforcement (moving slightly toward STY's ambivalence on asylum), and more focused on economic anxiety. The blend is meaningfully less evangelical than pure CON, reflecting Solidarity's more secular working-class base. This is the senator who wins Alabama or Connecticut by sounding tough on immigration and crime while acknowledging that the system has not always worked fairly for everyone.

The surprising combination: CON/STY registers the highest electoral skepticism shift of any CON-dominant blend, pushing a Conservative senator toward a Solidarity-level distrust of election fairness without moving at all toward left economics — a posture that mirrors Trump-era Republican rhetoric but comes from a different demographic base.

**Factor profile:** High F1 (security-oriented), Medium F2 (modest electoral skepticism), Medium F4 (moderate religious traditionalism), Medium F5 (center-right).

---

### STY/SD — IRV: 1 seat · Condorcet: 0 seats

*IRV: Arizona*
*Condorcet: none — Arizona elects a pure STY senator under Condorcet*

The STY/SD senator is Solidarity-dominant but pulled toward Social Democrat — representing a state where the working-class populist base is larger than the educated progressive base. The blend's most significant shifts from pure STY are toward less electoral skepticism (election distrust drops as SD's institutional trust moderates the blend) and toward more support for spending programs. This is a Solidarity senator who had to earn Social Democrat votes by softening on abortion, spending more on infrastructure, and reducing their anti-establishment rhetoric. What stays fixed is the economic core: labor support, immigration ambivalence, police skepticism. Arizona produces this type under IRV because second-choice transfers from the SD base accumulate toward the STY candidate; under Condorcet, the head-to-head matchup advantage goes to a pure STY nominee who commands broader cross-party appeal.

The surprising combination: STY/SD is *more* religious than pure STY, reflecting the high Black church attendance within the Social Democrat constituency that pulls the blend back toward religious traditionalism even as it moves left on economic and electoral issues.

**Factor profile:** Low F1 (anti-enforcement), Medium F2 (moderate electoral skepticism), Medium F4 (moderate religious traditionalism), Low F5 (progressive-leaning).

---

### CON/CTR — IRV: 6 seats · Condorcet: 5 seats

*IRV: Florida, Idaho, Kentucky, Mississippi, North Carolina, South Dakota*
*Condorcet: Florida, Kentucky, North Carolina, Ohio, South Dakota*

The CON/CTR senator is the institutional center-right — a Conservative who must also speak to Center Party voters, who are the most trusting of elections of any constituency in the system. The defining feature of this blend is what disappears: compared to pure CON, the CON/CTR senator is dramatically less likely to say the government is untrustworthy, less skeptical of elections (−18 points on fairness), and more likely to support police funding (the CTR base brings its law-and-order sensibility). The result is a Conservative who sounds more mainstream, more institutionalist, and less populist — a Midwestern or Southern senator who will oppose Medicaid expansion and EPA regulation but will not say the 2020 election was stolen. Under IRV, second-choice transfers from CTR voters push the CON/CTR blend into additional states; the Condorcet method favors a slightly different set by surfacing head-to-head matchup winners.

The surprising combination: the top differentiators for this blend are entirely about trust — federal government trust, state government trust, election fairness — not about policy substance. CON/CTR senators look almost identical to pure CON on immigration, abortion, and economics, but they sound fundamentally different on whether the system works.

**Factor profile:** High F1 (security-oriented), Low F2 (trusts elections), Medium F4 (moderate religious traditionalism), High F5 (center-right populist).

---

### CON/REF — IRV: 5 seats · Condorcet: 6 seats

*IRV: Indiana, Kansas, Missouri, Montana, Ohio*
*Condorcet: Idaho, Indiana, Kansas, Missouri, Mississippi, Montana*

The CON/REF senator combines Conservative institutional conservatism with Reform's populist economic grievance and deep electoral skepticism. This is the blend that most closely resembles Trump-era Republicanism: high F1 (security and enforcement), elevated F2 (doubts about elections), strongly conservative on F5. Compared to pure CON, CON/REF moves substantially toward border hardness, immigration restriction, and concealed carry, while also absorbing Reform's greater sense of economic anxiety. Church attendance stays nearly identical. Electoral distrust increases meaningfully — this blend will express doubts about election integrity that pure CON would not. Indiana and Montana produce this type: states where the suburban Conservative base must be held together with the exurban Reform voter who feels left behind.

The surprising combination: CON/REF sits at a notably higher F5 than CON alone, meaning the blend is more economically populist despite having a Conservative-dominant base — suggesting Reform voters who blend with CON push the ticket toward economic grievance rather than toward fiscal orthodoxy.

**Factor profile:** High F1, Medium-High F2 (moderately skeptical), Medium F4, High F5 (populist-conservative).

---

### SD/LIB — IRV: 1 seat · Condorcet: 2 seats

*IRV: New Hampshire*
*Condorcet: Maine, New Hampshire*

The SD/LIB senator is the most institutionally trusting left-wing type in the system. SD/LIB combines Social Democrat's economic program with Liberal's deep trust in elections, skepticism of government surveillance, and civil libertarian streak. The dominant policy shifts from pure SD are all about institutional trust: election skepticism drops 26 points, state and local government distrust drops 24 points, and federal government distrust drops 14 points. The blend is slightly more permissive on abortion, slightly higher on income (reflecting LIB's professional-class skew), and almost unchanged on immigration. This senator wins in states where the educated liberal professional community is large enough to moderate the Social Democrat base's working-class skepticism. New Hampshire is a natural home; Maine reaches this blend under Condorcet when the head-to-head matchup advantage goes to the candidate most trusted across the center-left spectrum.

The surprising combination: the defining feature of SD/LIB is not any policy shift but a wholesale change in the senator's relationship to institutions — moving from SD's mild ambivalence to LIB's pronounced confidence that elections are fair and government surveillance is the bigger threat.

**Factor profile:** Low F1 (anti-enforcement), Low F2 (trusts elections strongly), Low F4 (secular), Very Low F5 (strongly progressive).

---

### STY/REF — IRV: 1 seat · Condorcet: 1 seat

*IRV: West Virginia*
*Condorcet: West Virginia*

The STY/REF senator pairs Solidarity's working-class economics and deep electoral skepticism with Reform's immigration restrictionism and fiscal conservatism. The result is the highest F2 score of any left-leaning-dominant blend — electoral skepticism near the maximum — combined with a sharp rightward shift on immigration (opposing legal status for undocumented workers increases by 28 points, Dreamer opposition rises 27 points). The blend remains skeptical of police expansion, reflects above-average concern about inflation, and sits at lower concealed-carry support than pure REF. West Virginia is the natural home: an electorate with a significant working-class populist constituency that combines labor instincts with immigration ambivalence and a distrust of how elections are administered.

The surprising combination: STY/REF has higher F2 (electoral skepticism) than even the CON/REF blend — the most electorally distrustful senator type outside pure REF itself, despite having a Solidarity-dominant base that is pro-labor and opposed to police expansion.

**Factor profile:** Medium F1 (moderate enforcement), Very High F2 (deeply skeptical of elections), Medium F4, High F5 (leans populist-conservative).

---

### SD/CON — IRV: 1 seat · Condorcet: 1 seat

*IRV: Delaware*
*Condorcet: Delaware*

The SD/CON senator is the mirror of CON/SD: Social Democrat-dominant but obligated to speak to Conservative voters in an unusual state configuration. The blend represents a center-left candidate who has absorbed Conservative views on police (dropping 33 points toward pro-police), asylum (dropping 29 points toward denial), and church attendance (dropping 31 points toward more frequent attendance). Electoral skepticism is essentially unchanged — neither SD nor CON is particularly skeptical, so the blend inherits their shared trust in institutions. Delaware is a small state where the SD coalition is large enough to anchor the ticket but where enough Conservative voters exist to require real accommodation on culture and policing.

The surprising combination: SD/CON and CON/SD have nearly identical factor scores despite being ordered differently — they represent a genuinely centrist position that falls almost exactly between the two parent parties, and either parent can claim the senator depending on how the state's electorate leans.

**Factor profile:** Medium F1 (moderate on security), Medium F2 (trusts elections), Medium F4 (moderate religious traditionalism), Near-zero F5 (true centrist).

---

### STY/CON — IRV: 1 seat · Condorcet: 1 seat

*IRV: Texas*
*Condorcet: Texas*

The STY/CON senator represents the Southern cross-pressure type: a Solidarity-dominant senator who must speak to Conservative voters in a large state where the working-class multi-racial coalition anchors the ticket but the CON constituency is too large to ignore. Compared to pure STY, the blend moves toward police skepticism (−26.5% — less pro-police than pure STY, as CON's police skepticism moderates STY's), less skeptical of elections (−22 points), and more restrictionist on immigration. Texas's Black and Hispanic working-class majority produces a Solidarity base large enough to anchor the ticket, but the CON constituency requires movement toward immigration restriction and away from STY's most anti-establishment positions.

The surprising combination: STY/CON moves *away* from police support compared to pure STY, because the CON base — despite its general law-and-order lean — provides less police-expansion pressure than STY's working-class base already carries. The blend actually softens the pro-police stance.

**Factor profile:** Medium F1 (moderate enforcement), Medium F2 (moderate electoral skepticism), Medium F4, Medium F5 (center-right leaning).

---

### REF/STY — IRV: 1 seat · Condorcet: 0 seats

*IRV: Maine*
*Condorcet: none — Maine elects an SD/LIB senator under Condorcet*

REF/STY wins one seat under Instant Runoff Voting but zero under Condorcet. Maine's electorate contains a significant working-class populist constituency that combines labor instincts with immigration ambivalence and deep electoral distrust — enough first-preference support to survive IRV's elimination rounds. Under Condorcet, however, the head-to-head matchup advantage goes to the SD/LIB candidate, who is broadly acceptable across the center-left spectrum in a way that REF/STY is not. The blend is Reform-dominant but pulled toward Solidarity's working-class economics — high electoral skepticism, but with somewhat less immigration restriction and more attentiveness to economic precarity than pure REF.

The surprising combination: REF/STY moves *away* from REF's anti-asylum, anti-immigration restrictionism when pulled toward STY — but maintains near-maximum electoral skepticism. The combination of high election distrust with moderated immigration restriction defines a voter type that feels the system is rigged economically, not just demographically.

**Factor profile:** Medium F1, Very High F2 (near-maximum election skepticism), Medium F4, High F5 (populist-conservative).

---

### REF/SD — IRV: 0 seats · Condorcet: 1 seat

*Condorcet: North Dakota*
*IRV: none — North Dakota elects a CTR/LIB senator under IRV*

REF/SD wins zero seats under Instant Runoff Voting but captures one seat under Condorcet. North Dakota's electorate contains a Reform-dominant base pulled toward Social Democrat on labor and economic issues — a combination that emerges as the Condorcet winner when head-to-head matchups filter out the more extreme options on both sides. Under IRV, the same state produces a CTR/LIB winner as second-choice transfers from moderate voters accumulate toward the most institutionally trusted candidate. REF/SD represents the economically populist voter who distrusts elections, wants border enforcement, but also wants labor protections and is skeptical of pure-party Conservative orthodoxy.

**Factor profile:** Medium-Low F1, High F2 (electoral skepticism), Medium F4, High F5 (populist-conservative).

---

### CTR/LIB — IRV: 1 seat · Condorcet: 0 seats

*IRV: North Dakota*
*Condorcet: none — North Dakota elects a REF/SD senator under Condorcet*

CTR/LIB wins one seat under IRV but zero under Condorcet. North Dakota's unusual political geography — a deeply conservative state with a significant libertarian-leaning rural tradition — produces this blend as second-choice transfers accumulate from the center during IRV counting. CTR/LIB combines Center's deep trust in elections and moderate religiosity with Liberal's civil libertarian skepticism of government surveillance and distrust of concentrated police power. The result is the most institutionally trusting senator type in the system, holding moderate positions on social issues while opposing post-9/11 surveillance programs and backing background checks. Under Condorcet, the Reform/SD candidate's head-to-head matchup strength prevails instead, reflecting how differently the state electorate resolves when ranked preferences are fully counted.

**Factor profile:** Low F1, Very Low F2 (maximum election trust), Low-Medium F4, Low F5 (moderate-progressive).

---

### PRG/DSA — IRV: 1 seat · Condorcet: 0 seats

*IRV: Vermont*
*Condorcet: none — Vermont elects a pure STY senator under Condorcet*

PRG/DSA wins one seat under IRV but zero under Condorcet. Vermont's left-dominated electorate produces a Progressive/Democratic Socialist blend as the IRV winner — a state where the combined first-preference support for the two far-left parties is large enough to survive elimination rounds and consolidate. Under Condorcet, however, Vermont produces a pure Solidarity senator: the STY candidate beats every other nominee in head-to-head matchups, because Solidarity's working-class economics and institutional skepticism find common ground with Vermont's rural populist voters in ways that a pure PRG/DSA stance does not. The PRG/DSA senator would be the most economically redistributive and socially progressive senator in the chamber, with near-universal support for Medicare for All, aggressive climate regulation, and full immigration liberalization.

**Factor profile:** Very Low F1 (strongly anti-enforcement), Low F2 (trusts elections), Very Low F4 (secular), Very Low F5 (far-progressive).

---

### Single-Party Senate Seats

Several states elect a senator from a single party with no co-occurrence blend required — meaning one party holds enough of a plurality in that state that a pure candidate wins without needing to appeal to a second cluster. Numbers here are for the **mixed-coalition pipeline** (blend candidates compete alongside pure-party nominees); see the pure-party scenario below.

**Solidarity (STY)** — IRV: 2 seats *(Utah, Wyoming)* · Condorcet: 5 seats *(Arizona, Utah, Vermont, Wisconsin, Wyoming)*
Under Condorcet, five states surface a pure STY winner: working-class, majority non-white electorates where Solidarity's combination of economic populism and institutional distrust commands an outright plurality when all head-to-head matchups are resolved. Under IRV, Utah and Wyoming return pure STY; Arizona shifts to STY/SD as SD second-choice transfers move toward a blend, Vermont goes to PRG/DSA, and Wisconsin goes to a pure SD winner.

**Social Democrat (SD)** — IRV: 6 seats *(Illinois, Massachusetts, Michigan, Oregon, Washington, Wisconsin)* · Condorcet: 6 seats *(Illinois, Massachusetts, Michigan, Oregon, South Carolina, Washington)*
Illinois, Massachusetts, Michigan, Oregon, and Washington return pure SD senators under both methods. Wisconsin shifts from SD (IRV) to STY (Condorcet) as the Condorcet head-to-head matchup favors the more electorally skeptical working-class candidate. South Carolina shifts from CTR (IRV) to SD (Condorcet).

**Conservative (CON)** — IRV: 3 seats *(Louisiana, Oklahoma, Tennessee)* · Condorcet: 3 seats *(Louisiana, Oklahoma, Tennessee)*
All three states elect a pure Conservative senator under both methods — states where CON holds a dominant first-preference plurality and the head-to-head matchup advantage holds regardless of counting method.

**Center (CTR)** — IRV: 2 seats *(Alaska, South Carolina)* · Condorcet: 1 seat *(Alaska)*
Alaska returns a pure CTR senator under both methods. South Carolina, where Center's moderate religious pragmatism commands enough first-preference support under IRV, shifts to pure SD under Condorcet.

**Liberal (LIB)** — IRV: 1 seat *(Rhode Island)* · Condorcet: 1 seat *(Rhode Island)*
Rhode Island returns a pure Liberal senator under both methods — the only state where LIB's civil libertarian, institutionally trusting profile commands an outright plurality in both counting systems.

---

### Pure-Party Scenario Senate Seats

When the field is restricted to the nine core parties (no coalition blends), the map changes dramatically. No seat goes to LIB, NAT, PRG, or DSA in any state. The five parties that win seats are:

**Social Democrat (SD)** — Condorcet: 16 seats · IRV: 14 seats
*Both methods:* AK, CA, CT, DE, IL, KS, MD, MA, NM, OR, PA, RI, WA *(13 states)*
*Condorcet only:* MS, MT, NJ *(3 additional)*
*IRV only:* FL *(1 additional)*
SD is the plurality party in the pure-party field, dominating the coasts, Midwest, and Mountain West states where its center-left economic platform outpolls both STY and CTR in head-to-head and IRV rounds.

**Center (CTR)** — Condorcet: 15 seats · IRV: 13 seats
*Both methods:* AZ, AR, DC, HI, ID, IA, NE, TN, UT, VT, WV *(11 states)*
*Condorcet only:* FL, MN, MO, WY *(4 additional)*
*IRV only:* MS, NY *(2 additional)*
CTR's position at the spatial center of the nine-party field makes it the Condorcet winner in states where no single party holds an outright plurality — the states where moderate-religious swing voters decide the outcome.

**Solidarity (STY)** — Condorcet: 10 seats · IRV: 8 seats
*Both methods:* CO, LA, MI, NV, NH, SC *(6 states)*
*Condorcet only:* GA, ME, NY, TX *(4 additional)*
*IRV only:* MN, WY *(2 additional)*
STY wins states with large working-class multi-racial electorates where its combination of economic populism and institutional distrust commands a plurality. Under Condorcet it reaches a wider set; under IRV it loses some contested states to CON when first-choice accumulation favors the Republican-leaning candidate.

**Conservative (CON)** — Condorcet: 9 seats · IRV: 12 seats
*Both methods:* AL, IN, KY, OH, SD, VA, WI *(7 states)*
*Condorcet only:* NC *(1 additional)*
*IRV only:* GA, MO, MT, NJ, TX *(5 additional)*
CON's seat count is notably higher under IRV than Condorcet — it accumulates first-choice votes effectively in closely contested states but loses more head-to-head matchups to the centrist field.

**Reform (REF)** — Condorcet: 1 seat · IRV: 4 seats
*Both methods:* ND *(1 state)*
*IRV only:* ME, NC, OK *(3 additional)*
REF wins only North Dakota under both methods. Under IRV it breaks through in three additional low-turnout, highly populist states where its first-choice plurality survives elimination rounds — but those same states go to CON or CTR under Condorcet head-to-head resolution.

---

## Electoral Breakdown

### Senate Seat Totals by Method — Mixed Scenario (coalition blends allowed)

| Type | IRV | Condorcet | Difference |
|------|-----|-----------|------------|
| SD/STY | 8 | 8 | — |
| CON/SD | 6 | 6 | — |
| CON/CTR | 6 | 5 | +1 IRV |
| SD (pure) | 6 | 6 | — |
| CON/REF | 5 | 6 | +1 Condorcet |
| CON/STY | 4 | 4 | — |
| CON (pure) | 3 | 3 | — |
| STY (pure) | 2 | 5 | +3 Condorcet |
| CTR (pure) | 2 | 1 | +1 IRV |
| STY/SD | 1 | 0 | IRV only |
| SD/CON | 1 | 1 | — |
| REF/STY | 1 | 0 | IRV only |
| CTR/LIB | 1 | 0 | IRV only |
| SD/LIB | 1 | 2 | +1 Condorcet |
| STY/CON | 1 | 1 | — |
| PRG/DSA | 1 | 0 | IRV only |
| STY/REF | 1 | 1 | — |
| LIB (pure) | 1 | 1 | — |
| REF/SD | 0 | 1 | Condorcet only |

The Condorcet method produces more pure-party Solidarity seats (5 vs. 2) and more CON/REF seats (6 vs. 5), while IRV produces more mixed-blend diversity — CTR/LIB, STY/SD, REF/STY, and PRG/DSA each win seats under IRV that disappear under Condorcet. The two methods agree on the dominant coalition types — SD/STY, CON/SD, CON/STY, SD, CON, SD/CON, STY/CON, STY/REF, LIB — and diverge in 9 states where the IRV second-choice dynamics produce different winners than the Condorcet head-to-head matchup.

### Senate Seat Totals — Pure-Party Scenario (no coalition blends)

When blended coalition candidates are removed and voters must choose among the nine core parties, the map consolidates around the spatial center. LIB, PRG, DSA, and NAT win no senate seats under either method; every seat goes to SD, CTR, STY, CON, or REF.

| Party | Condorcet | IRV | Difference |
|-------|-----------|-----|------------|
| SD | 16 | 14 | +2 Condorcet |
| CTR | 15 | 13 | +2 Condorcet |
| STY | 10 | 8 | +2 Condorcet |
| CON | 9 | 12 | +3 IRV |
| REF | 1 | 4 | +3 IRV |

Condorcet favors the three spatial-center parties (SD, CTR, STY) that win head-to-head matchups across a broader coalition. IRV favors CON and REF, which accumulate strong first-choice pluralities in contested right-leaning states even when they lose more head-to-head contests against centrist candidates. The 13-state divergence between methods (vs. 9 states in the mixed scenario) reflects a more competitive field where no candidate has the coalition-bridging advantage that blends provided.

### State-by-State Senate Comparison

States where the two methods agree are shown in plain text. The 9 states where outcomes differ are **bold**.

| State | IRV | Condorcet |
|-------|-----|-----------|
| Alabama | CON/STY | CON/STY |
| Alaska | CTR | CTR |
| **Arizona** | **STY/SD** | **STY** |
| Arkansas | CON/STY | CON/STY |
| California | SD/STY | SD/STY |
| Colorado | CON/SD | CON/SD |
| Connecticut | CON/STY | CON/STY |
| Delaware | SD/CON | SD/CON |
| DC | SD/STY | SD/STY |
| Florida | CON/CTR | CON/CTR |
| Georgia | CON/SD | CON/SD |
| Hawaii | SD/STY | SD/STY |
| **Idaho** | **CON/CTR** | **CON/REF** |
| Illinois | SD | SD |
| Indiana | CON/REF | CON/REF |
| Iowa | SD/STY | SD/STY |
| Kansas | CON/REF | CON/REF |
| Kentucky | CON/CTR | CON/CTR |
| Louisiana | CON | CON |
| **Maine** | **REF/STY** | **SD/LIB** |
| Maryland | SD/STY | SD/STY |
| Massachusetts | SD | SD |
| Michigan | SD | SD |
| Minnesota | SD/STY | SD/STY |
| **Mississippi** | **CON/CTR** | **CON/REF** |
| Missouri | CON/REF | CON/REF |
| Montana | CON/REF | CON/REF |
| Nebraska | CON/SD | CON/SD |
| Nevada | SD/STY | SD/STY |
| New Hampshire | SD/LIB | SD/LIB |
| New Jersey | CON/SD | CON/SD |
| New Mexico | CON/STY | CON/STY |
| New York | SD/STY | SD/STY |
| North Carolina | CON/CTR | CON/CTR |
| **North Dakota** | **CTR/LIB** | **REF/SD** |
| **Ohio** | **CON/REF** | **CON/CTR** |
| Oklahoma | CON | CON |
| Oregon | SD | SD |
| Pennsylvania | CON/SD | CON/SD |
| Rhode Island | LIB | LIB |
| **South Carolina** | **CTR** | **SD** |
| South Dakota | CON/CTR | CON/CTR |
| Tennessee | CON | CON |
| Texas | STY/CON | STY/CON |
| Utah | STY | STY |
| Virginia | CON/SD | CON/SD |
| **Vermont** | **PRG/DSA** | **STY** |
| Washington | SD | SD |
| **Wisconsin** | **SD** | **STY** |
| West Virginia | STY/REF | STY/REF |
| Wyoming | STY | STY |

### State-by-State Pure-Party Comparison

States where IRV and Condorcet agree are shown in plain text. The 13 states where outcomes differ are **bold**.

| State | IRV | Condorcet |
|-------|-----|-----------|
| Alabama | CON | CON |
| Alaska | SD | SD |
| Arizona | CTR | CTR |
| Arkansas | CTR | CTR |
| California | SD | SD |
| Colorado | STY | STY |
| Connecticut | SD | SD |
| Delaware | SD | SD |
| DC | CTR | CTR |
| **Florida** | **SD** | **CTR** |
| **Georgia** | **CON** | **STY** |
| Hawaii | CTR | CTR |
| Idaho | CTR | CTR |
| Illinois | SD | SD |
| Indiana | CON | CON |
| Iowa | CTR | CTR |
| Kansas | SD | SD |
| Kentucky | CON | CON |
| Louisiana | STY | STY |
| **Maine** | **REF** | **STY** |
| Maryland | SD | SD |
| Massachusetts | SD | SD |
| Michigan | STY | STY |
| **Minnesota** | **STY** | **CTR** |
| **Mississippi** | **CTR** | **SD** |
| **Missouri** | **CON** | **CTR** |
| **Montana** | **CON** | **SD** |
| Nebraska | CTR | CTR |
| Nevada | STY | STY |
| New Hampshire | STY | STY |
| **New Jersey** | **CON** | **SD** |
| New Mexico | SD | SD |
| **New York** | **CTR** | **STY** |
| **North Carolina** | **REF** | **CON** |
| North Dakota | REF | REF |
| Ohio | CON | CON |
| **Oklahoma** | **REF** | **CON** |
| Oregon | SD | SD |
| Pennsylvania | SD | SD |
| Rhode Island | SD | SD |
| South Carolina | STY | STY |
| South Dakota | CON | CON |
| Tennessee | CTR | CTR |
| **Texas** | **CON** | **STY** |
| Utah | CTR | CTR |
| Vermont | CTR | CTR |
| Virginia | CON | CON |
| Washington | SD | SD |
| West Virginia | CTR | CTR |
| Wisconsin | CON | CON |
| **Wyoming** | **STY** | **CTR** |

---

## Legislative Outlook

The vote simulation models 37 binary policy items across all four scenarios (mixed/pure × Condorcet/IRV). Senate composition alone does not determine what becomes law — the president's signature is decisive on five bills, and the two pipelines elect different presidents.

**Four distinct presidents across the scenarios:**
- Mixed IRV → **CON/SD** (Conservative/Social Democrat; IRV accumulates CON first-choice votes)
- Mixed Condorcet → **SD/CON** (Social Democrat/Conservative; wins more head-to-head matchups)
- Pure Condorcet → **STY** (Solidarity; spatial median of the nine-party field)
- Pure IRV → **STY** (same result under both methods in the pure scenario)

CON/SD and SD/CON are different candidates from different parties, though their factor profiles are nearly identical — both sit at the genuine cross-aisle centroid, separated by less than 0.1 on every dimension. The vote model uses **CON/SD as the presidential reference for both mixed scenarios**, since the two would likely govern indistinguishably on most legislation; a full separation would require running separate bill-signing models for each. Bills where a CON-leaning vs. SD-leaning instinct might plausibly diverge are flagged below.

**Mixed pipeline president (CON/SD as reference):** Signs border enforcement, police expansion, Medicaid work requirements, and the TikTok ban. Vetoes student loan forgiveness. Governs center-right on security with cross-aisle accommodation on immigration pathways, infrastructure, and healthcare preservation. An SD/CON president would likely sign the same set, though the work requirements and TikTok ban represent the margin calls most likely to flip under an SD-leading identity.

**Pure-party pipeline president (STY):** Signs student loan forgiveness, legal status for undocumented immigrants, and the full progressive economic package. Vetoes asylum denial at the border, the police funding increase, Medicaid work requirements, and the TikTok ban. Governs center-left, anchored by the working-class multi-racial coalition that dominates the pure-party senate.

Thirty-two of the 37 bills produce the same outcome across all four scenarios. The five that differ are driven entirely by the presidential signature difference between the mixed and pure pipelines, not by senate composition — senate verdicts are identical within each pipeline regardless of counting method.

### What Passes in All Four Scenarios

**Taxes** — The congress simultaneously extends the 2017 tax cuts, raises the corporate rate from 21% to 28%, and allows top-bracket rates on incomes over $400k to rise to 35%. All three pass the House and all four senate configurations. In a two-party system these are mutually exclusive party platforms; in a proportional congress they represent three separate majority coalitions forming around the same vote schedule.

**Immigration** — Legal status for long-term undocumented immigrants (PASS all scenarios), permanent DACA pathway for Dreamers (PASS all scenarios), and increased border patrols (PASS all scenarios) all pass both chambers and are signed by both presidents. Denying asylum at the border passes both chambers in all scenarios but is signed only by CON/SD — STY vetoes it (see presidential split below).

**Environment** — EPA authority to regulate CO2 (PASS), 20% renewable electricity requirement (PASS), strengthening the Clean Air and Water Act even at some job cost (PASS), increasing fossil fuel production (PASS), and preventing the government from banning gas stoves (PASS) all clear both chambers under both presidents. The renewable mandate and the gas stove protection pass through entirely different party coalitions — and both pass.

**Police & Public Safety** — The 10% police funding increase passes both chambers in all scenarios and is signed by CON/SD; STY vetoes it (see below). A 10% cut fails in all chambers. Mental health and school safety spending passes. The assault rifle ban passes. Universal background checks pass with near-certainty. Easier concealed carry fails.

**Healthcare** — Medicaid expansion passes in all four scenarios. ACA repeal fails clearly — it cannot pass either chamber, and both presidents would veto it in any case. Work requirements for able-bodied Medicaid recipients pass both chambers in all scenarios but are signed only by CON/SD — STY vetoes them (see below).

**Social Issues** — Federal recognition of same-sex and interracial marriages passes (66.1% global support). Congressional protection of abortion access passes in all four scenarios; restrictions on abortion-inducing drugs by mail and interstate travel bans for abortion both fail in all four. Preventing gender transition surgery for minors passes (60.1% global support). Parental consent requirements for school name/pronoun changes passes. Age verification for adult web content passes. School voucher subsidies pass.

**Civil Liberties** — The conditional TikTok ban passes both chambers in all scenarios but is signed only by CON/SD — STY vetoes it (see below). Post-9/11 surveillance renewal fails in all four scenarios. Affordable housing tax incentives pass in all four.

### Where the President Decides (the Five Split Bills)

These bills pass both chambers in all four scenarios. Whether they become law depends entirely on which president sits in the Oval Office. The mixed column uses CON/SD as reference; SD/CON would govern nearly identically but the starred bills (★) represent positions where an SD-leading president might plausibly diverge.

| Bill | CON/SD ref (mixed) | STY (pure) |
|------|-------------------|------------|
| Deny asylum at the border | **Signs → Law** | Vetoes → Fails |
| Increase police funding 10% | **Signs → Law** ★ | Vetoes → Fails |
| Medicaid work requirements | **Signs → Law** ★ | Vetoes → Fails |
| Conditional TikTok ban | **Signs → Law** ★ | Vetoes → Fails |
| Student loan forgiveness up to $20k | Vetoes → Fails ★ | **Signs → Law** |

The asymmetry is revealing. CON/SD signs four enforcement/conditionality bills and vetoes the redistributive one. STY signs the redistributive bill and vetoes all four enforcement measures. The starred bills are where an SD/CON president's center-left identity might flip the outcome — but the model does not resolve this, and the policy distance between the two mixed presidents is small enough that the chamber majorities would likely hold in any case.

### What Fails in All Four Scenarios

The items that fail share a common structure: they represent the maximum position of a single ideological bloc that cannot find coalition partners.

- **Easier concealed carry** — fails in all chambers; even CON and NAT senators face gun-owning constituencies that support background checks
- **Decrease police 10%** — no cross-party coalition reaches majority; DSA and PRG cannot build one alone
- **Halt new oil/gas leases on federal lands** — the SD/LIB/PRG coalition is too small; CON and STY-dominant types cannot support it (though STY would sign it if it passed)
- **ACA repeal** — fails clearly in all chambers; even REF and NAT senators represent constituents who rely on Medicaid, and both presidents veto it
- **Renew post-9/11 surveillance programs** — LIB, DSA, and PRG provide the opposition margin; STY's institutional distrust reinforces it; both presidents veto it
- **Prohibit abortion-inducing drugs by mail** — the majority willing to restrict abortion access cannot reach 50% even with CON and NAT support; both presidents veto it
- **Restrict interstate abortion travel** — fails decisively; even CON senators will not restrict citizens' physical movement across state lines; both presidents veto it

### The One Toss-Up

Relaxing local zoning laws to allow more apartments and condos sits at almost exactly 50/50 across all four scenarios (global support: 50.2%). This is the rare item where the electorate is genuinely split rather than cross-pressured — urban SD and LIB types support it, rural CON and REF types oppose it, and the suburban swing types (CTR, CON/CTR) are evenly divided. The CON/SD president would veto it; STY would sign it — but it never clears the senate to reach either desk.

---

## Appendix: The Five Ideological Dimensions

### Factor Overview

The five factors underlying this typology were identified through Exploratory Factor Analysis (EFA) of 24 survey items from the 2024 CES, using oblique (Promax) rotation. N=45,707 after listwise deletion.

**Tier thresholds:** Very High > +0.75 | High +0.25 to +0.75 | Medium −0.25 to +0.25 | Low −0.75 to −0.25 | Very Low < −0.75

---

### F1 — Security & Order

High scorers favor increasing police by 10%, expanding border patrols, denying asylum to Central American migrants, renewing post-9/11 surveillance programs, and opposing legal status for undocumented immigrants. Low scorers oppose all of the above. This is the law enforcement and national security axis — it is distinct from fiscal conservatism (which loads on F5) and from religious values (which load on F4). F1 and F4 are strongly correlated (+0.55), reflecting the traditional "socially conservative" combination, but they are not the same thing: CON scores Very High on F1 but only Medium on F4.

---

### F2 — Electoral Skepticism

High scorers disagree that U.S. elections are run fairly and disagree that their 2024 state/local elections were fair. This factor is near-orthogonal to partisan identity (Cramér's V ≈ 0.15). Critically, REF (+0.76), STY (+0.66), and DSA (+0.50) all score High on F2 despite being maximally opposed on F5. Electoral skepticism cuts across the left-right divide. CTR (−0.82) and LIB/CTR (−0.77) are the most trusting types; REF and REF/STY are the most skeptical.

---

### F3 — Government Distrust

High scorers distrust the federal and state governments generally. This factor shares items with F2 but captures general institutional distrust rather than specifically electoral skepticism. Critically: all 23 winning types score Medium on F3 (range: −0.21 to +0.13). Government distrust does not differentiate winning coalition types from each other — it is a background condition shared broadly across the electorate.

---

### F4 — Religious Traditionalism

High scorers attend church frequently, favor stricter limits on abortion, and oppose federal recognition of same-sex marriages. Church attendance and abortion week limits have the joint-highest loadings (+0.69 each). This is genuinely the religious values axis, not just social conservatism broadly: NAT (+0.46) and CON/NAT (+0.34) are the only High-scoring types; all others are Medium or Low.

---

### F5 — Populist Conservatism

High scorers agree that racial problems are rare and isolated, oppose Dreamer pathways to citizenship, oppose raising top-bracket tax rates, oppose legal status for undocumented immigrants, and disagree with progressive racial attitudes. Most conservative items load negatively because the CES coded liberal responses as higher numbers; a high F5 score predicts the conservative response. NAT at +1.51 is a full standard deviation above the next type (REF at +0.99). PRG (−0.99) and LIB (−0.95) anchor the opposite pole.

---

### Full Factor Score Table — All 23 Types (sorted by F5, most populist-conservative first)

| Type | Chamber | F1 Sec/Ord | F2 ElecSkep | F3 GovtDis | F4 ReligTrad | F5 PopCons |
|------|---------|-----------|-------------|------------|--------------|------------|
| NAT | House | +0.737 | +0.428 | −0.208 | +0.457 | **+1.510** |
| REF | Both | +0.202 | +0.759 | −0.206 | +0.147 | +0.990 |
| CON/NAT | Senate | +0.752 | +0.198 | −0.045 | +0.336 | +0.966 |
| CON/REF | Senate | +0.592 | +0.219 | +0.013 | +0.196 | +0.612 |
| REF/STY | Senate | −0.038 | +0.722 | −0.081 | +0.153 | +0.601 |
| STY/REF | Senate | −0.154 | +0.704 | −0.019 | +0.157 | +0.411 |
| CON | Both | +0.767 | −0.024 | +0.111 | +0.219 | +0.442 |
| CON/CTR | Senate | +0.577 | −0.325 | +0.002 | +0.185 | +0.289 |
| CON/STY | Senate | +0.258 | +0.263 | +0.120 | +0.196 | +0.230 |
| STY/CON | Senate | +0.076 | +0.365 | +0.124 | +0.188 | +0.155 |
| CTR | Both | +0.266 | −0.817 | −0.174 | +0.130 | +0.039 |
| CON/SD | Senate | +0.236 | −0.027 | +0.102 | −0.035 | −0.011 |
| STY | Both | −0.446 | +0.658 | +0.133 | +0.165 | −0.062 |
| SD/CON | Senate | +0.153 | −0.028 | +0.101 | −0.074 | −0.081 |
| SD/CTR | Senate | −0.122 | −0.369 | −0.023 | −0.141 | −0.305 |
| STY/SD | Senate | −0.430 | +0.313 | +0.112 | −0.090 | −0.313 |
| SD/STY | Senate | −0.425 | +0.196 | +0.105 | −0.177 | −0.398 |
| SD | Both | −0.414 | −0.032 | +0.091 | −0.345 | −0.564 |
| LIB/CTR | Senate | −0.171 | −0.773 | −0.121 | −0.142 | −0.554 |
| SD/LIB | Senate | −0.438 | −0.381 | +0.004 | −0.334 | −0.753 |
| DSA | House | −1.303 | +0.504 | +0.076 | −0.387 | −0.874 |
| LIB | Both | −0.462 | −0.744 | −0.086 | −0.323 | −0.950 |
| PRG | House | −1.260 | −0.634 | −0.206 | −0.387 | −0.990 |

---

*Simulation based on CES 2024 data (N=45,707). House seats allocated by Single Transferable Vote; Senate seats allocated by Instant Runoff Voting (IRV) and Condorcet methods. Party cluster assignments derived from k-means clustering on EFA factor scores.*
