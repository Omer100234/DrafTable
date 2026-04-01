"""
Per-Season Approximate Value Scraper
Downloads per-season AV from theedgepredictor/nfl-madden-data (PFR source),
matches to our draftees using pfr_player_id, and saves the result.

Data source: https://github.com/theedgepredictor/nfl-madden-data
No API key needed.
"""

import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AV_URL_TEMPLATE = "https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/main/data/pfr/approximate_value/{year}.csv"
DRAFT_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


AV_DIR = os.path.join(DATA_DIR, "nfl per season av")


def download_season_av(start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    """Download per-season AV CSVs, save each individually, and combine them."""
    os.makedirs(AV_DIR, exist_ok=True)
    frames = []
    for year in range(start_year, end_year + 1):
        url = AV_URL_TEMPLATE.format(year=year)
        try:
            df = pd.read_csv(url)
            df.to_csv(os.path.join(AV_DIR, f"{year}.csv"), index=False)
            frames.append(df)
            logger.info(f"{year}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"{year}: failed to download - {e}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total: {len(combined)} season-AV rows")
    return combined


def match_to_draftees(av_df: pd.DataFrame, draft_years: tuple = (2015, 2025)) -> pd.DataFrame:
    """Match season AV data to our draftees using pfr_player_id."""
    logger.info("Downloading nflverse draft picks for matching...")
    draft_df = pd.read_csv(DRAFT_URL)

    # Filter to our draft range
    draftees = draft_df[
        (draft_df.season >= draft_years[0]) & (draft_df.season <= draft_years[1])
    ][["season", "round", "pick", "pfr_player_id", "pfr_player_name", "position", "college"]].copy()
    draftees = draftees.rename(columns={"season": "draft_year", "pfr_player_name": "name"})

    logger.info(f"{len(draftees)} draftees in {draft_years[0]}-{draft_years[1]}")

    # Merge: draftees x season AV
    merged = draftees.merge(
        av_df.drop(columns=["name"]),
        left_on="pfr_player_id",
        right_on="player_id",
        how="inner",
    )

    # Calculate NFL season number (rookie = 1)
    merged["nfl_season_num"] = merged["season"] - merged["draft_year"] + 1
    merged = merged[merged["nfl_season_num"] > 0]

    # Clean up columns
    result = merged[[
        "draft_year", "round", "pick", "name", "position", "college",
        "pfr_player_id", "team", "season", "nfl_season_num", "approximate_value",
    ]].sort_values(["draft_year", "pick", "season"]).reset_index(drop=True)

    logger.info(f"Matched {result.pfr_player_id.nunique()} draftees across {len(result)} season rows")
    return result


def save(df: pd.DataFrame, filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")


if __name__ == "__main__":
    # Download per-season AV (2015-2025 NFL seasons)
    av_df = download_season_av(2015, 2025)

    # Match to our draftees
    result = match_to_draftees(av_df)
    save(result, os.path.join("nfl per season av", "draftees_season_av.csv"))
    save(av_df, os.path.join("nfl per season av", "all_players_season_av.csv"))

    # Quick summary
    logger.info("\nSample:")
    logger.info(result.head(10).to_string(index=False))
    logger.info(f"\nDraftees matched: {result.pfr_player_id.nunique()}")
    logger.info(f"Seasons covered: {result.season.min()}-{result.season.max()}")
    logger.info("Done!")
