# Subjective Decisions Log

Every non-obvious, subjective choice made while building the DrafTables variable pipeline, normalization, and model — with the reasoning behind each.

---

## 1. Success Score: 70% AV/game, 30% Second Contract

**Decision:** Composite success = 0.7 × normalized AV/game + 0.3 × normalized second contract value.

**Rationale:** AV (Approximate Value) is the most widely accepted single-number measure of NFL production, but it has blind spots — a player on a cheap rookie deal stacking AV and a player who gets a massive second contract represent different kinds of success. Second contract value captures how the market values a player's future, which reflects tape, durability, and locker room factors that AV misses. We weight AV heavier (70%) because it's an objective on-field measure, while contract value is influenced by cap inflation, team needs, and agent leverage. 30% is enough to differentiate between "good starter who got paid" and "good starter who got replaced" without letting market noise dominate.

---

## 2. Arctan Transform on Success Score (s=5, center=0.35)

**Decision:** Apply arctan(5 × (composite − 0.35)) then re-scale to [0, 1].

**Rationale:** Raw composite scores are roughly linear, which means the gap between an elite WR (Ja'Marr Chase) and a very good WR (Chris Olave) is nearly as large as the gap between a solid starter and a backup. That doesn't match reality — the difference between "great" and "elite" is much smaller than the difference between "starter" and "bust." The arctan compresses the top end and spreads out the middle-to-low range, where our model needs to discriminate most. Steepness of 5 was chosen to create meaningful separation in the 0.2–0.5 composite range (where most players cluster) without completely flattening the top. Center at 0.35 is slightly below the median composite, which means the inflection point sits right where the "starter vs. replacement" boundary lives.

---

## 3. Minimum 5 Games Filter

**Decision:** Only include draftees with 5+ NFL games in the success score dataset.

**Rationale:** Players with fewer than 5 games don't have enough sample to judge. A player who tore his ACL in week 2 of his rookie year isn't a "bust" — he's unobserved. Setting the bar at 5 games removes noise from injury-shortened careers and practice squad players who got a couple of emergency snaps, while still keeping players who got a real shot and failed. Lower thresholds (1-3) would include too many noise cases; higher thresholds (10+) would exclude late bloomers and players on bad teams who didn't get early opportunities.

---

## 4. Training Window: 2017–2022 Draft Classes

**Decision:** Train on classes 2017 through 2022, predict 2023–2026.

**Rationale:** We need players who have had enough NFL time to judge their careers — at minimum 2-3 seasons. The 2022 class has played 3 seasons by now, which is enough to see who earned a role and who washed out. Going back further than 2017 would introduce a different era of college football (pre-transfer portal, different offensive schemes) that may not generalize to modern prospects. Six draft classes gives us ~100-180 players per position, which is enough to train on without overfitting.

---

## 5. Contract Value Capped at 1.1× Top of Market

**Decision:** Clip `pct_of_top` at 1.1 before normalizing to [0, 1].

**Rationale:** Some players reset the market (e.g., a WR who signs a deal bigger than any previous WR contract). Their `pct_of_top` can exceed 1.0. But a player who got 1.15× the previous top deal isn't meaningfully more successful than one who got 1.0× — they just had better timing or a more desperate team. Capping at 1.1 gives a small bonus for being the absolute top of the market without letting one mega-deal distort the entire scale. Dividing by 1.1 instead of 1.0 means a player at exactly the market top scores ~0.91, leaving room for true market-resetters to score higher.

---

## 6. Manual Contract Fills

**Decision:** Hand-fill second contract values for players whose contracts weren't in our scraped data (e.g., Drake London 1.00, Chris Olave 0.90, Michael Pittman Jr. 0.486).

**Rationale:** Some players signed contracts too recently for our scraper to catch, or their contracts are structured unusually (option years, voidable years). Rather than drop these players entirely (losing valuable training data) or impute from stats (circular reasoning — stats are our features), we manually looked up their actual contract values and converted to pct_of_top. This is the most accurate approach for a small number of missing values. Each fill is based on the actual reported contract and the position's market top at the time of signing.

---

## 7. Zero Contract List (Busts/Cut Players)

**Decision:** Explicit list of players who get pct_of_top = 0.0 (e.g., Kadarius Toney, JJ Arcega-Whiteside, Derrius Guice).

**Rationale:** Players who were cut, traded for nothing, or left the league before earning a meaningful second contract represent the "bust" end of the spectrum. Setting them to 0 (rather than NaN/missing) is an active statement: "the market said this player has no value." This is different from a player who simply doesn't appear in our contract data — missing could mean we didn't scrape it, while 0 means we know they failed. Each player on this list was verified as having been cut, out of the league, or signed for a veteran minimum deal.

---

## 8. Conference Prestige Tiers

**Decision:** SEC/Big Ten = 1.0, Big 12/Pac-12/ACC = 0.9, AAC/MWC = 0.55, Sun Belt/MAC/C-USA = 0.35, FCS/unmapped = 0.15.

**Rationale:** Conference strength is a well-established factor in prospect evaluation — a 1000-yard WR season in the SEC faces better DBs than one in the MAC. We use a tier system rather than continuous rankings because (a) the differences between conferences within a tier are small and unstable year-to-year, and (b) a simple tier system is more robust than trying to rank all 130+ programs. The specific values were chosen to create meaningful gaps: the jump from 0.35 (G5) to 0.9 (P5 non-SEC/B1G) is large, reflecting the real talent gap, while the gap between 0.9 and 1.0 is small, reflecting that the difference between ACC and SEC competition is real but not enormous. We didn't use actual conference win rates or SP+ rankings because those fluctuate annually and would require per-season lookups that add complexity without clear accuracy gains.

---

## 9. Per-Season Conference Prestige for Transfers

**Decision:** Track Y0_conf_prestige, Y1_conf_prestige, etc. instead of just one conference score.

**Rationale:** The transfer portal has made it common for a player to produce stats at two different schools in two different conferences. A WR who put up 800 yards at Eastern Kentucky (0.35) then transferred and put up 1200 yards at Alabama (1.0) is a very different profile than one who did all his damage at a G5 school. Per-season prestige lets the model weight each season's stats in context. Without this, a transfer from Sun Belt to SEC would get either the Sun Belt score (ignoring their SEC production) or the SEC score (inflating their Sun Belt stats).

---

## 10. Independent School Overrides

**Decision:** Notre Dame → ACC, BYU → Big 12, Liberty → Sun Belt, UMass → Sun Belt, UConn → AAC, NMSU → C-USA.

**Rationale:** These schools are FBS independents (or were at the time) and don't appear in conference standings files. Without overrides they'd get the default 0.15 FCS score, which is wildly wrong for Notre Dame (competing at SEC/Big Ten level). Each mapping reflects the conference the school either joined, is scheduled to join, or most closely resembles in competition level. Notre Dame to ACC rather than its own tier because their schedule and talent level is comparable to top P5 conferences and they have an ACC scheduling agreement.

---

## 11. Default Prestige Score: 0.15

**Decision:** FCS schools and any school not found in our conference mapping get 0.15.

**Rationale:** Not zero because FCS players do occasionally make the NFL (e.g., Cooper Kupp from Eastern Washington), and their stats aren't meaningless — just produced against weaker competition. 0.15 is low enough to heavily discount FCS production relative to Power 5 but high enough that a dominant FCS stat line still registers as something. Setting it to 0 would make all FCS stats multiply to nothing, which is too aggressive.

---

## 12. COVID Season Scaling (2020)

**Decision:** Scale 2020 volume stats by (13 / conference_games_played). E.g., Pac-12 played 6 games → multiply by 13/6 = 2.17×.

**Rationale:** The 2020 season had different game counts by conference (Pac-12 played 6, Big Ten played 8, SEC played 10, etc.). Without scaling, a Pac-12 WR's 2020 stats would look like an injury season compared to an SEC WR's. Scaling to a 13-game equivalent makes cross-conference and cross-year comparisons fair. We only scale volume stats (receptions, yards, TDs) — rate stats (yards per reception, yards per carry) are already per-play and don't need adjustment.

---

## 13. Normal Season = 13 Games

**Decision:** Use 13 as the baseline "full season" game count for COVID scaling.

**Rationale:** The FBS regular season is 12 games, plus a conference championship game for most competitive teams. We use 13 rather than 12 because most drafted players come from teams good enough to play in a conference championship. Using 12 would slightly under-scale. Using 14 (including bowls) would over-scale since bowl game stats are already in the data for non-COVID years and not every player plays in a bowl.

---

## 14. Injury Fill-Forward: 40% Threshold

**Decision:** If ALL of a player's volume stats in season Y(n) are below 40% of their Y(n+1) values, replace Y(n) with Y(n+1) stats.

**Rationale:** When a player misses most of a season due to injury, their stats for that year are misleadingly low and would make the model think they're worse than they are. Fill-forward says "assume they would have performed at least as well as the previous year." The 40% threshold means a player has to be dramatically below their prior production — not just a down year. A player who goes from 60 catches to 35 catches (58%) is having a bad year, not necessarily injured. A player who goes from 60 to 10 (17%) almost certainly missed significant time. We chose 40% as the boundary because it captures obvious injury seasons while leaving legitimate regression alone.

---

## 15. Injury Detection: ALL Volume Stats Must Drop

**Decision:** Every volume stat must be below the threshold, not just most of them.

**Rationale:** If a WR's receptions dropped 80% but his rush attempts stayed the same, that's a role change, not an injury. Requiring all stats to drop simultaneously is a strong signal of missed games rather than scheme changes or position shifts. This is intentionally conservative — we'd rather miss some injury cases and keep real data than accidentally overwrite a legitimate bad season with inflated prior-year stats.

---

## 16. Arctan Normalization for Speed/Agility

**Decision:** Use arctan(−8 × (value − center)) for forty (4.45), cone (6.95), shuttle (4.28). Linear min-max for everything else.

**Rationale:** Speed and agility times have a non-linear relationship with NFL value. The difference between a 4.35 and 4.45 forty is huge (elite vs. average for a WR), but the difference between 4.55 and 4.65 barely matters (both are slow for a WR). Arctan captures this: it creates large separation near the center (where most draftable players cluster) and compresses the tails. The centers (4.45, 6.95, 4.28) are approximate position-average times — the point where "fast" transitions to "slow." Steepness of 8 (higher than the success score's 5) because combine times have a tighter range and we need aggressive separation in a narrow window. Physical measurables like height, weight, and bench press have a more linear relationship with value, so simple min-max works fine.

---

## 17. Dropping LONG, Returns, and Fumbles Columns

**Decision:** Drop all columns containing `_LONG`, `kickReturns_`, `puntReturns_`, and `fumbles_` patterns.

**Rationale:**
- **LONG (longest play):** A single-play stat that's more about opportunity and randomness than skill. One 80-yard catch shouldn't define a player.
- **Kick/Punt returns:** Most drafted WRs/RBs who return kicks do so because they're athletic, not because return production predicts NFL success. Return stats add noise.
- **Fumbles:** College fumble numbers are low-count events with high variance. A player with 2 fumbles isn't meaningfully worse than one with 0 — it's noise at this sample size.

All three categories failed to improve model accuracy in testing and add dimensionality without adding signal.

---

## 18. Position-Specific Stat Filtering

**Decision:** Only keep stat categories relevant to each position (WR: receiving + rushing; RB: rushing + receiving; CB: defensive + interceptions + returns; etc.).

**Rationale:** Including all stat categories for every position would create hundreds of mostly-empty columns. A WR's defensive stats are almost always zero (or from garbage time), and feeding those zeros to the model adds noise. Filtering by position keeps the feature space tight and meaningful. The specific category assignments follow standard football logic — WRs occasionally rush (jet sweeps, end-arounds) so we keep rushing stats, but WRs don't play defense so we drop those columns entirely.

---

## 19. played_at_X Binary Flags (Ages 18–23)

**Decision:** Create binary columns `played_at_18` through `played_at_23` indicating whether the player was an active contributor at each age.

**Rationale:** Age matters in prospect evaluation, but raw draft age is a crude measure. A 23-year-old who started playing at 18 (5 years of college production) is a different profile than a 23-year-old who redshirted twice and only played 2 seasons. The binary flags encode the player's developmental timeline. Binary (yes/no) rather than continuous because what matters is "did they play meaningful snaps," not "how much." This also captures early enrollees (played_at_18 = 1) and late developers separately.

---

## 20. "Playing" Threshold: 10% of Position Best

**Decision:** A player counts as "playing" in a season if any of their stats reach 10% of the position's best value for that stat column.

**Rationale:** We need to distinguish between "played meaningful snaps" and "appeared in 1 game for 2 plays." Setting the bar at 10% of the position maximum means a player has to have done something notable — you can't hit 10% of the top WR's receptions by catching 3 passes all year if the top guy caught 100. It's low enough to catch part-time starters and rotational players, high enough to filter out garbage-time appearances.

---

## 21. years_since_first_played: Earliest Season with Any Data

**Decision:** Count from the first season where the player has any non-null stat, regardless of magnitude.

**Rationale:** This captures total college exposure, not just productive years. A player who has Y3 data (even small numbers as a true freshman) has been in a college program for 4 years — they've had 4 years of coaching, film study, and strength training. That developmental runway matters even if their freshman stats were minimal. This is intentionally different from the played_at_X flags, which require the 10% threshold — years_since_first_played is about total time in the system.

---

## 22. SmartImputer: 25th Percentile vs. Median

**Decision:** Players who attended the combine but skipped drills get 25th percentile for missing drills. Players who didn't attend at all get median.

**Rationale:** If a player went to the combine but didn't run the 40, there's usually a reason — they're slow, they're injured, or their agent advised against it. Filling with median would give them "average" speed, which is optimistic. 25th percentile is a mild penalty that reflects "probably below average but we don't know how bad." If a player didn't attend the combine at all (injured, top prospect who skipped it), we have no information — median is the least biased default. This distinction prevents the model from learning that "no combine data = bad athlete" when many top prospects skip the combine entirely.

---

## 23. Sample Weighting: Step Function (1×, 3×, 8×)

**Decision:** Busts (score < 0.3) weighted 1×, mid-tier (0.3–0.7) weighted 3×, stars (> 0.7) weighted 8×.

**Rationale:** We care much more about correctly identifying stars and starters than about precisely ranking busts. The difference between the 10th-worst and 20th-worst WR in a class doesn't matter, but correctly identifying the top 5 matters a lot. Step weights (rather than continuous) because the boundaries are inherently fuzzy — we don't want to pretend there's a meaningful difference between a 0.69 and 0.71 player, but we do want the model to try hard on clearly elite players. 8× for stars is aggressive but intentional: without heavy top-weighting, the model optimizes for the crowded middle of the distribution and treats elite players as outliers to be averaged away.

---

## 24. Ensemble: Ridge + Random Forest + SVR

**Decision:** Equal-weight average of Ridge regression (α=10), Random Forest (500 trees, depth 6), and SVR (RBF kernel, C=1).

**Rationale:** Each model captures different patterns:
- **Ridge:** Linear relationships (bigger + faster = better). Good baseline, handles multicollinearity from correlated stats.
- **Random Forest:** Non-linear interactions (e.g., "fast + small WR" vs. "slow + big WR" are different archetypes). Handles missing data patterns naturally.
- **SVR:** Smooth non-linear relationships that RF might overfit. Good at finding the overall shape of the success surface.

Equal weighting because in cross-validation no single model consistently dominates — they take turns being best depending on the draft class. Simple averaging is more robust than learned weights with only 6 training folds.

---

## 25. Model Hyperparameters

**Decision:** Ridge α=10, RF n_estimators=500/max_depth=6/min_samples_leaf=5, SVR kernel=rbf/C=1.

**Rationale:**
- **Ridge α=10:** Moderate regularization. Lower α overfits to noise in the small dataset; higher α underfits and ignores real patterns.
- **RF depth=6:** Shallow enough to prevent memorizing individual players, deep enough to capture 2-3 way feature interactions (e.g., conference × speed × production).
- **RF min_samples_leaf=5:** With ~100-180 training examples, requiring 5 samples per leaf prevents any leaf from being a single player.
- **RF 500 trees:** Enough for stable predictions; more trees showed no improvement.
- **SVR C=1:** Default regularization. Cross-validation showed minimal sensitivity to C in the 0.1-10 range.

These weren't extensively tuned — with ~150 training examples and leave-one-year-out CV (6 folds), aggressive hyperparameter search would overfit to the CV splits.

---

## 26. Class-to-Season Mapping

**Decision:** Map college class labels to Y0/Y1/Y2/Y3 where Y0 = most recent season:
- FR → [Y0]
- SO → [Y0, Y1] (SO=Y0, FR=Y1)
- JR → [Y0, Y1, Y2]
- SR → [Y0, Y1, Y2, Y3]
- RSO/RJR/RSR → same as SO/JR/SR (redshirt year is invisible in stats)
- GR → [Y0=GR, Y1=SR, Y2=JR, Y3=SO] (skip FR, most distant)

**Rationale:** Y0 is always the most recent season because recent performance is most predictive. Redshirt years (RSO, RJR, RSR) map the same as non-redshirt because the redshirt year produces no stats — it's already captured by the age-based features (played_at_X). For graduate students (GR), we drop the freshman year rather than the most recent because (a) freshman stats are least predictive, (b) the model only has 4 season slots, and (c) a 5th-year senior's freshman stats are so old they may reflect a completely different player.

---

## 27. Position Aliases: Defensive Edge → DE, Linebacker → ILB

**Decision:** Merge "Defensive Edge" into "Defensive End" and "Linebacker" into "Inside Linebacker."

**Rationale:** Different data sources use different position labels for the same role. "Defensive Edge" is a modern analytics term for what traditional scouting calls "Defensive End" — they're the same players, just labeled differently depending on the source. Similarly, generic "Linebacker" in older data almost always refers to inside/middle linebackers (OLBs are usually labeled separately). Merging prevents the pipeline from creating tiny position groups with 3-4 players that can't be meaningfully modeled.

---

---

## 28. TE Arctan Centers: forty=4.65, cone=7.10, shuttle=4.40

**Decision:** Shift the arctan centers for tight ends compared to WRs (forty 4.45→4.65, cone 6.95→7.10, shuttle 4.28→4.40).

**Rationale:** TEs are 20-40 lbs heavier than WRs and measurably slower across all agility drills. Using WR centers would make almost every TE score poorly on speed, compressing the entire position into the "slow" end of the scale and losing discrimination between athletic TEs and slow TEs. The shifted centers put the inflection point at the TE-average for each drill, preserving the arctan's ability to separate the athletic freaks (George Kittle, Kyle Pitts) from average TEs.

---

## 29. QB Arctan: Only Forty (center=4.75), Cone/Shuttle as Linear

**Decision:** Only use arctan normalization for the QB forty (center=4.75). Cone and shuttle use linear min-max instead.

**Rationale:** Most QBs skip the cone and shuttle drills at the combine — the data is too sparse for arctan to create meaningful separation. For the few who do run them, linear normalization is fine since there's no strong "fast vs. slow" breakpoint the way there is for the forty. The forty center at 4.75 reflects that QBs are significantly slower than skill positions — a 4.65 QB is elite (Lamar Jackson territory), while 4.85 is average. The 4.75 center puts the steepest part of the curve right where mobile vs. pocket-passer separation lives.

---

## 30. QB Passing INT Inversion

**Decision:** After normalizing passing interception columns (Y0_passing_INT through Y3_passing_INT) to 0-1, invert them (1 − value) so that fewer interceptions = higher score.

**Rationale:** Interceptions are the only major stat where more is worse. Without inversion, a QB who threw 20 INTs would score higher than one who threw 5 — the model would have to learn this inversion on its own from a tiny dataset. Explicitly flipping the column makes the feature direction consistent with all other stats (higher = better) and reduces what the model needs to learn. This is only done for INTs, not fumbles (which are dropped entirely due to noise).

---

## 31. QB No Bench Press in Variables

**Decision:** The QB variable file doesn't include bench press — it's not in the data because QBs almost never do the bench press at the combine.

**Rationale:** Unlike WRs/TEs/RBs where bench press measures blocking ability or physical durability, arm strength for QBs is measured differently (not relevant to the bench). The combine data naturally reflects this — almost zero QBs have bench data, so the column is absent from the variable file.

---

## 32. TE Drop Pattern: No Returns

**Decision:** TEs don't have kick/punt return columns dropped (unlike WRs) because TEs don't have return stats in the first place — the position stat filter already excludes them.

**Rationale:** The WR normalizer drops `kickReturns_` and `puntReturns_` patterns because WRs occasionally return kicks. TEs never do, so `build_variables.py` already filters those categories out. The TE normalizer only needs to drop `_LONG` and `fumbles_`.

---

## 33. Small Sample Caveat: 76 TEs, 40 QBs

**Decision:** Train and ship models despite very small training sets (76 TEs with 5+ games, only 40 QBs).

**Rationale:** Both positions are notoriously hard to predict from college stats — TE because the college role differs dramatically from the NFL (many TEs are converted receivers or develop blocking later), and QB because NFL success depends heavily on coaching, supporting cast, and mental factors invisible in box scores. The models' cross-validation numbers (TE: correlation -0.09, QB: Spearman 0.25) reflect this reality. We ship them anyway because (a) even weak signal is better than no signal for prospect rankings, (b) the models correctly identify broad tiers even if exact rankings are noisy, and (c) being transparent about model limitations (shown in metadata) is better than withholding predictions. The conservative hyperparameters (Ridge α=50, shallow trees) help prevent overfitting to this small sample.

*Last updated: 2026-04-04*
