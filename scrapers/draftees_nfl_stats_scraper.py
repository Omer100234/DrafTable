"""
Draftees NFL Stats Scraper
Downloads draft pick data with career AV and NFL stats from nflverse.
Matches against our draftees files (2015-2025) and saves per-season AV.

Data source: https://github.com/nflverse/nflverse-data
No API key needed — single CSV download.
"""

import pandas as pd
import logging
import os
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NFLVERSE_DRAFT_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
NFLVERSE_ROSTERS_URL = "https://raw.githubusercontent.com/leesharpe/nfldata/master/data/rosters.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_draft_data() -> pd.DataFrame:
    """Download full draft picks dataset with career AV from nflverse."""
    logger.info("Downloading nflverse draft picks data...")
    df = pd.read_csv(NFLVERSE_DRAFT_URL)
    logger.info(f"Got {len(df)} draft picks ({df.season.min()}-{df.season.max()})")
    return df


def download_season_av() -> pd.DataFrame:
    """Download per-season roster data with AV from nflverse (2006-2019)."""
    logger.info("Downloading nflverse rosters data (has per-season AV)...")
    df = pd.read_csv(NFLVERSE_ROSTERS_URL)
    logger.info(f"Got {len(df)} roster entries ({df.season.min()}-{df.season.max()})")
    return df


def build_career_av(draft_df: pd.DataFrame, roster_df: pd.DataFrame, start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    """
    For each draftee, build their season-by-season AV.
    Uses roster data for per-season AV where available,
    and draft data for career totals.
    """
    # Filter to our draft range
    draftees = draft_df[(draft_df.season >= start_year) & (draft_df.season <= end_year)].copy()
    logger.info(f"Filtered to {len(draftees)} draftees from {start_year}-{end_year}")

    # Build per-season AV from roster data
    season_av = roster_df[["season", "playerid", "full_name", "team", "av", "games", "starts"]].copy()
    season_av = season_av.rename(columns={"playerid": "pfr_player_id"})

    # Merge draftees with their season-by-season AV
    merged = draftees.merge(
        season_av,
        on="pfr_player_id",
        how="inner",
        suffixes=("_draft", "_season"),
    )

    # Calculate which NFL season number this is (1st, 2nd, 3rd, etc.)
    merged["nfl_season_num"] = merged["season_season"] - merged["season_draft"] + 1
    merged = merged[merged["nfl_season_num"] > 0]

    result = merged[[
        "season_draft", "round", "pick", "pfr_player_name", "position",
        "college", "team_season", "season_season", "nfl_season_num",
        "av", "games", "starts",
        "car_av", "w_av", "allpro", "probowls",
    ]].rename(columns={
        "season_draft": "draft_year",
        "pfr_player_name": "name",
        "team_season": "team",
        "season_season": "season",
    })

    logger.info(f"Built {len(result)} season-AV rows")
    return result


def save(df: pd.DataFrame, filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")


if __name__ == "__main__":
    # Download both datasets
    draft_df = download_draft_data()
    roster_df = download_season_av()

    # Save the full career stats for our draftees (2015-2025)
    career = draft_df[
        (draft_df.season >= 2015) & (draft_df.season <= 2025)
    ][[
        "season", "round", "pick", "pfr_player_name", "position", "college",
        "age", "to", "allpro", "probowls", "seasons_started",
        "w_av", "car_av", "dr_av", "games",
        "pass_completions", "pass_attempts", "pass_yards", "pass_tds", "pass_ints",
        "rush_atts", "rush_yards", "rush_tds",
        "receptions", "rec_yards", "rec_tds",
        "def_solo_tackles", "def_ints", "def_sacks",
    ]].rename(columns={"season": "draft_year", "pfr_player_name": "name"})

    save(career, "draftees_nfl_career_stats.csv")

    # Build and save per-season AV
    season_av = build_career_av(draft_df, roster_df, 2015, 2025)
    if not season_av.empty:
        save(season_av, "draftees_season_av.csv")
    else:
        logger.warning("No per-season AV data matched")

    logger.info("Done!")
