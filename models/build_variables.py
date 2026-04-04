"""
Build variables_class_X.csv files for each draft class, split by position.
Merges pre-draft college stats (draftees_clean_X.csv) with
combine measurables (combine.csv) and age data from nflverse,
then outputs per-position files with only relevant stat categories.
"""

import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DRAFT_PICKS_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"

# Map alternate position names to standard names
POSITION_ALIASES = {
    "Defensive Edge": "Defensive End",
    "Linebacker": "Inside Linebacker",
}

# Stat categories relevant to each position group
POSITION_STATS = {
    "Quarterback": ["passing", "rushing", "fumbles"],
    "Running Back": ["rushing", "receiving", "fumbles"],
    "Fullback": ["rushing", "receiving", "fumbles"],
    "Wide Receiver": ["receiving", "rushing", "fumbles", "kickReturns", "puntReturns"],
    "Tight End": ["receiving", "rushing", "fumbles"],
    "Offensive Tackle": [],
    "Offensive Guard": [],
    "Center": [],
    "Defensive End": ["defensive", "fumbles"],
    "Defensive Tackle": ["defensive", "fumbles"],
    "Outside Linebacker": ["defensive", "fumbles", "interceptions"],
    "Inside Linebacker": ["defensive", "fumbles", "interceptions"],
    "Cornerback": ["defensive", "interceptions", "puntReturns", "kickReturns"],
    "Safety": ["defensive", "interceptions", "fumbles"],
    "Place Kicker": ["kicking"],
    "Punter": ["punting"],
}


def load_combine() -> pd.DataFrame:
    """Load combine data from local file."""
    filepath = os.path.join(RAW_DIR, "combine.csv")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} combine entries")
    return df


def load_draft_ages() -> pd.DataFrame:
    """Load draft pick ages from nflverse."""
    logger.info("Downloading draft picks for age data...")
    df = pd.read_csv(DRAFT_PICKS_URL)
    return df[["season", "pick", "pfr_player_name", "age"]].rename(
        columns={"season": "draft_year", "pick": "overall_pick"}
    )


def add_derived_features(df: pd.DataFrame, position: str, draft_year: int, age_df: pd.DataFrame) -> pd.DataFrame:
    """Add years_since_first_played and played_at_X age-based starter flags."""
    normalized = POSITION_ALIASES.get(position, position)
    stat_categories = POSITION_STATS.get(normalized, [])

    seasons = ["Y3", "Y2", "Y1", "Y0"]

    # For each season, collect relevant stat columns
    season_stat_cols = {}
    for season in seasons:
        cols = [c for c in df.columns if c.startswith(f"{season}_")
                and c.split("_")[1] in stat_categories]
        season_stat_cols[season] = cols

    # years_since_first_played — check ALL stat columns, not just position-relevant
    season_years = {"Y3": 4, "Y2": 3, "Y1": 2, "Y0": 1}
    all_season_cols = {}
    for season in seasons:
        all_season_cols[season] = [c for c in df.columns if c.startswith(f"{season}_")
                                   and not c.endswith("_college")]

    df["years_since_first_played"] = 0
    for _, row in df.iterrows():
        for season in seasons:
            cols = all_season_cols[season]
            if cols and row[cols].notna().any():
                df.loc[row.name, "years_since_first_played"] = season_years[season]
                break

    # Merge age from nflverse (match on draft year + overall pick)
    year_ages = age_df[age_df["draft_year"] == draft_year][["overall_pick", "age"]]
    df = df.merge(year_ages, on="overall_pick", how="left")
    age_matched = df["age"].notna().sum()
    logger.info(f"    Age matched: {age_matched}/{len(df)}")

    # Compute position max for each stat column (use all stat cols for threshold)
    pos_maxes = {}
    for season in seasons:
        for col in all_season_cols[season]:
            pos_maxes[col] = df[col].max()

    # played_at_X columns (18 through 23)
    # Age at each season: Y0 = draft_year-1 so age ~ draft_age-1, Y1 ~ draft_age-2, etc.
    age_offsets = {"Y0": 1, "Y1": 2, "Y2": 3, "Y3": 4}
    age_cols = [f"played_at_{a}" for a in range(18, 24)]
    for col in age_cols:
        df[col] = 0

    for idx, row in df.iterrows():
        if pd.isna(row["age"]):
            continue
        draft_age = int(row["age"])

        # Check each season from earliest (Y3) to latest (Y0)
        started = False
        first_started_age = None
        for season in seasons:
            season_age = draft_age - age_offsets[season]
            if season_age < 18 or season_age > 23:
                continue

            if started:
                # Once started, all later ages get 1
                pass
            else:
                # Check if player hit 10% of position best in any stat
                for col in all_season_cols[season]:
                    max_val = pos_maxes.get(col)
                    if pd.notna(max_val) and max_val > 0 and pd.notna(row[col]):
                        if row[col] >= 0.10 * max_val:
                            started = True
                            first_started_age = season_age
                            break

            if started:
                df.loc[idx, f"played_at_{season_age}"] = 1

        # Fill in all ages between first_started_age and draft_age-1 (Y0 age)
        if started and first_started_age is not None:
            y0_age = draft_age - 1
            for a in range(first_started_age, y0_age + 1):
                if 18 <= a <= 23:
                    df.loc[idx, f"played_at_{a}"] = 1

    return df


def filter_position_cols(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Keep only stat columns relevant to the position."""
    normalized = POSITION_ALIASES.get(position, position)
    stat_categories = POSITION_STATS.get(normalized, [])

    keep_cols = ["name", "position", "college", "conference", "height", "weight",
                 "forty", "bench", "vertical", "broad_jump", "cone", "shuttle",
                 "age", "years_since_first_played",
                 "played_at_18", "played_at_19", "played_at_20",
                 "played_at_21", "played_at_22", "played_at_23"]

    # Keep Y*_ columns only if they match a relevant stat category or are college names
    for col in df.columns:
        if col in keep_cols:
            continue
        if col[:3] in ["Y0_", "Y1_", "Y2_", "Y3_"]:
            if col.endswith("_college"):
                keep_cols.append(col)
                continue
            stat_part = col[3:]
            category = stat_part.split("_")[0]
            if category in stat_categories:
                keep_cols.append(col)

    return df[[c for c in keep_cols if c in df.columns]]


def build_class(year: int, combine_df: pd.DataFrame) -> pd.DataFrame:
    """Build combined variables for a single draft class."""
    filepath = os.path.join(PROCESSED_DIR, f"draftees_clean_{year}.csv")
    if not os.path.exists(filepath):
        logger.warning(f"No file for {year}, skipping")
        return None

    df = pd.read_csv(filepath)
    logger.info(f"{year}: {len(df)} draftees")

    # Drop non-variable columns (draft outcomes, metadata)
    # Keep overall_pick for now — needed for age matching
    drop_cols = ["draft_year", "round", "pick", "nfl_team",
                 "pre_draft_ranking", "pre_draft_grade"]
    drop_cols += [c for c in df.columns if c.endswith("_season")]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Get combine measurables for this draft class
    combine_year = combine_df[combine_df["season"] == year][[
        "player_name",
        "forty", "bench", "vertical", "broad_jump", "cone", "shuttle",
    ]].copy()
    combine_year = combine_year.drop_duplicates(subset="player_name", keep="first")

    df = df.merge(
        combine_year,
        left_on="name",
        right_on="player_name",
        how="left",
    )
    df = df.drop(columns=["player_name"], errors="ignore")

    matched = df["forty"].notna().sum()
    logger.info(f"{year}: {matched}/{len(df)} matched to combine data")

    return df


if __name__ == "__main__":
    combine_df = load_combine()
    age_df = load_draft_ages()

    for year in range(2017, 2026):
        full_df = build_class(year, combine_df)
        if full_df is None:
            continue

        for position in full_df["position"].unique():
            pos_df = full_df[full_df["position"] == position].copy()
            pos_df = add_derived_features(pos_df, position, year, age_df)
            pos_df = pos_df.drop(columns=["overall_pick"], errors="ignore")
            pos_df = filter_position_cols(pos_df, position)

            # Drop columns that are entirely empty for this group
            pos_df = pos_df.dropna(axis=1, how="all")

            pos_label = position.lower().replace(" ", "_")
            outpath = os.path.join(PROCESSED_DIR, f"variables_class_{year}_{pos_label}.csv")
            pos_df.to_csv(outpath, index=False)
            logger.info(f"  {position}: {len(pos_df)} players, {len(pos_df.columns)} cols -> {outpath}")

    logger.info("Done!")
