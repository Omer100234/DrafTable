# DrafTable - Where We Left Off

## What's Done

### Data Collection (scrapers/)
- **eligible_players_stats_fetcher.py** - Fetches college stats for P4 conferences + Notre Dame (2014-2025)
- **missing_stats_fetcher.py** - Smart fetcher that scans draftees, finds missing stats, and only makes necessary API calls. Covers all conferences + independent teams.
- **draftees_nfl_stats_scraper.py** - Downloads NFL career stats from nflverse (career totals only)
- **draftees_fetcher.py** - Fetches draft pick data
- **draft_prospects_scraper.py** - Fetches 2026 prospect rankings
- **eligible_p4_players_fetcher.py** - Fetches eligible P4 players list
- **season_av_scraper.py** - Downloads per-season Approximate Value from PFR source, matches to draftees
- **contracts_scraper.py** - Downloads historical contracts from nflverse/OverTheCap, identifies second contracts, calculates % of position top money
- **combine_scraper.py** - Downloads NFL Combine measurables from nflverse

### Raw Data (data/raw/)
- `college_stats/` - ~200+ CSVs of college player stats by conference/team and year (2013-2025)
- `draftees_2015.csv` through `draftees_2025.csv` - Draft picks per year
- `draftees_nfl_career_stats.csv` - NFL career totals (games, AV, stats) for 2015-2025 draftees
- `nfl per season av/` - Per-season AV data (2015-2025), both raw yearly CSVs and matched to draftees
- `contracts_all.csv` - All historical NFL contracts from OverTheCap
- `second_contracts.csv` - Second contracts for 2015-2022 draftees with pct_of_top and market_reset flag
- `combine.csv` - NFL Combine measurables (2000-2026)
- `prospects_2026.csv` - 2026 draft prospect rankings
- `eligible_players.csv` - Eligible P4 players

### Processed Data (data/processed/)
- **process.py** - Matches 2026 prospects to college stats by class year (FR/SO/JR/SR/etc.), tracks transfers
- `prospects_stats.csv` - Raw output
- `prospects_clean_stats.csv` - Cleaned (irrelevant stats removed by position)
- `ambiguous_names.json` - 3 prospects with name collisions (Cameron Ball, Ryan Davis, Anthony Smith)
- **process_draftees.py** - Matches draftees (2017-2025) to college stats using collegeAthleteId. Labels as Y0 (draft_year-1), Y1, Y2, Y3. Gaps from opt-outs/injuries show as empty.
- `draftees_stats.csv` - Raw output (2014 draftees matched)
- `draftees_clean_stats.csv` - Cleaned by position
- `draftees_clean_2017.csv` through `draftees_clean_2025.csv` - Per-year splits
- **clean_stats.py** - Cleans prospect stats by position
- **clean_draftees_stats.py** - Cleans draftee stats by position
- `variables_class_YEAR_POSITION.csv` - Per-position variable files for ML (college stats + combine + age-based features)
- `variables_POSITION.csv` - Combined training data per position (2017-2022 draftees)
- `normalized_wide_receiver.csv` - Normalized WR training data (182 WRs, 54 columns)
- `success_wide_receiver.csv` - WR NFL success scores (171 WRs, 0-1 scale)

### Models (models/)
- **build_variables.py** - Builds per-position variable files for each draft class. Merges college stats with combine measurables and age data. Adds derived features:
  - `years_since_first_played` - How many years of college data exist
  - `played_at_18` through `played_at_23` - Age-based starter flags (10% of position best threshold, cascades forward once triggered)
- **build_success_score.py** - Builds NFL success scores for WRs (2017-2022):
  - 70% AV/game (linear normalized) + 30% second contract value (capped at 1.1, normalized)
  - Arctan transform (s=5, center=0.35) to compress elite tier
  - Manual contract fills for 21 players without second contract data
  - Outputs success_wide_receiver.csv (171 WRs scored)
- **normalization/preprocess.py** - Preprocessing before normalization:
  - COVID scaling: 2020 season volume stats scaled up per conference game count (SEC 1.30x, Big Ten 1.62x, Pac-12 2.17x, etc.)
  - Applies to Y0 for 2021 draft class, Y1 for 2022 draft class
  - Injury fill-forward: if all volume stats drop below 20% of previous season, carry forward previous season stats
- **normalization/normalize_wide_receiver.py** - WR normalization:
  - Arctan (s=8, lower-is-better): forty (center=4.45), cone (center=6.95), shuttle (center=4.28)
  - Linear min-max: height, weight, age, bench, vertical, broad_jump
  - Linear min-max: all Y0-Y3 college stats
  - Drop: LONG, kickReturns, puntReturns, fumbles columns
  - Conference prestige tiers: SEC/Big Ten=1.0, Big12/Pac12/ACC=0.9, AAC/MWC=0.55, Sun Belt/MAC/CUSA=0.35, default=0.15
  - College overrides for independents (Notre Dame→ACC, BYU→Big 12, etc.)

### App (app/)
- Streamlit app with 9 pages: Eligible Players, Past Drafts, Prospect Stats, NFL AV Leaders, Second Contracts, Draft Class Variables, Training Data, Normalized Data, Success Scores
- NFL AV Leaders: AV per game rankings by draft class (min 5 games)
- Second Contracts: % of position top money on second contract, market reset detection
- Draft Class Variables: Browse per-position variable files with all features
- Success Scores: View NFL success scores ranked by position

## What's Next

### Immediate: Build WR ML Model
- Training data ready: normalized_wide_receiver.csv (features) + success_wide_receiver.csv (target)
- 155 WRs with both features and success scores
- Plan: XGBoost with native NaN handling for combine missing values
- 49 features, need aggressive regularization for small dataset
- Prediction target: 2026 WR prospects

### Then: Other Positions
- Apply lessons from WR model to normalize and build models for other positions
- Each position needs its own success score, normalization, and model
- Preprocessing (COVID scaling, injury fill-forward) is position-agnostic and reusable

### Open Issues
- 298 draftees (mostly small school) have no college stats match
- 3 ambiguous prospect names to resolve
- 2024-2025 draftees too recent for reliable NFL outcomes (excluded from training)
- Only 155 WRs in training set — small dataset, overfitting risk
