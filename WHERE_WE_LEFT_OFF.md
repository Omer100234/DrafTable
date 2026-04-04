# DrafTable - Where We Left Off

## What's Done

### Data Collection (data_collection/)
- **eligible_players_stats_fetcher.py** - Fetches college stats for P4 conferences + Notre Dame (2014-2025)
- **missing_stats_fetcher.py** - Smart fetcher that scans draftees, finds missing stats, and only makes necessary API calls
- **draftees_nfl_stats_scraper.py** - Downloads NFL career stats from nflverse
- **draftees_fetcher.py** - Fetches draft pick data
- **draft_prospects_scraper.py** - Fetches 2026 prospect rankings
- **eligible_p4_players_fetcher.py** - Fetches eligible P4 players list
- **season_av_scraper.py** - Downloads per-season Approximate Value from PFR
- **contracts_scraper.py** - Downloads historical contracts from nflverse/OverTheCap
- **combine_scraper.py** - Downloads NFL Combine measurables from nflverse
- **process.py** - Matches 2026 prospects to college stats by class year
- **process_draftees.py** - Matches draftees (2017-2025) to college stats using collegeAthleteId
- **clean_stats.py** / **clean_draftees_stats.py** - Cleans stats by position

### Processed Data (data/processed/)
- `draftees_clean_2017.csv` through `draftees_clean_2025.csv` - Per-year draftee data
- `variables_class_YEAR_POSITION.csv` - Per-position variable files (2017-2026, all positions)
- `variables_wide_receiver.csv` / `variables_running_back.csv` / `variables_tight_end.csv` / `variables_quarterback.csv` - Combined training data (2017-2022)
- `normalized_*.csv` - Normalized training data for WR, RB, TE, QB
- `normalized_class_2023-2026_*.csv` - Normalized prediction data for WR, RB, TE, QB
- `success_wide_receiver.csv` (171 WRs) / `success_running_back.csv` (124 RBs) / `success_tight_end.csv` (84 TEs) / `success_quarterback.csv` (44 QBs)

### Models (models/)
- **build_variables.py** - Builds per-position variable files from draftee data. Merges college stats + combine + age. Adds `years_since_first_played`, `played_at_18-23`, and per-season `Y*_college` columns for transfer tracking.
- **build_prospect_variables.py** - Same as above but for 2026 prospects. Converts class labels (FR/SO/JR/SR/etc.) to Y0-Y3 format, looks up conference from college_stats files.
- **build_success_score.py** - Generalized for WR, RB, TE, QB. 70% AV/game + 30% second contract, arctan s=5 center=0.35. Manual contract fills per position. Zero-contract lists for busts.
- **normalization/preprocess.py** - COVID scaling + injury fill-forward (40% threshold)
- **normalization/normalize_wide_receiver.py** - fit_normalize/transform pattern with per-season conference prestige (Y0-Y3_conf_prestige from Y*_college columns). Arctan for forty(4.45), cone(6.95), shuttle(4.28).
- **normalization/normalize_running_back.py** - Same pattern, imports shared utilities from WR normalizer
- **normalization/normalize_tight_end.py** - TE-specific arctan centers: forty(4.65), cone(7.10), shuttle(4.40). Drops LONG + fumbles (no returns for TEs).
- **normalization/normalize_quarterback.py** - QB-specific: arctan only for forty(4.75), cone/shuttle as linear (too sparse). Inverts passing INT columns (more INTs = worse). Drops LONG + fumbles.
- **train_model.py** - Generalized trainer, takes position as CLI arg. Ensemble: Ridge(50) + RF(1000,d8,leaf3) + GBR(300,d2,lr=0.03). SmartImputer with position-specific stat fill percentile (WR=20th, RB=30th, default=25th).
- **trained/** - Trained ensemble models for WR, RB, TE, QB (`.pkl` + `_meta.json`)
- **experiment.py** / **experiment2.py** / **experiment3.py** - Model experimentation scripts

### CV Results
| Position | MAE | Correlation | Spearman | Training Size |
|----------|------|-------------|----------|---------------|
| WR | 0.309 | 0.279 | 0.271 | 155 |
| RB | 0.235 | 0.566 | 0.551 | 107 |
| TE | 0.324 | -0.094 | 0.007 | 76 |
| QB | 0.340 | 0.172 | 0.252 | 40 |

### App (app/)
- Streamlit app with 10 pages including Predictions
- Predictions page: shows model stats (MAE, Correlation, Spearman, Training Size), loads ensemble, runs predictions, ranks prospects
- Supports WR, RB, TE, and QB positions (auto-discovered from model files)
- Running at localhost:8513

### Documentation
- **DECISIONS.md** - 33 documented subjective decisions with rationale covering: success score construction, conference prestige, COVID scaling, injury fill-forward, normalization strategies, model design, and TE/QB-specific choices.

## What's Next

### Immediate: More Positions
- Apply same pipeline to defensive positions (CB, S, EDGE, DT, LB)
- Each needs: success scores (manual contract fills), normalization, training
- `build_success_score.py` and `train_model.py` are already generalized
- Just need per-position normalizer and manual contract research
- Defensive positions have different stat categories (tackles, sacks, INTs) — normalizers need different drop/keep patterns

### Model Improvement
- TE and QB models have weak predictive power (TE correlation ~0, QB Spearman 0.25) due to small samples and inherent position unpredictability
- Could explore: feature engineering (efficiency metrics, advanced stats), expanding training window, or different model architectures for small-N positions
- Consider whether TE/QB models should use a different success score formula (e.g., QB might benefit from more weight on contract value since QB market is distinct)

### Open Issues
- 298 draftees (mostly small school) have no college stats match
- 3 ambiguous prospect names to resolve
- 2024-2025 draftees too recent for reliable NFL outcomes (excluded from training)
- `train_wide_receiver.py` is now outdated — `train_model.py` replaces it (but kept for backwards compat with old pkl import in app)
- Streamlit deprecation warning: `use_container_width` should be replaced with `width='stretch'`
