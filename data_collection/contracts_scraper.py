"""
NFL Contracts Scraper
Downloads historical contract data from nflverse (sourced from OverTheCap),
identifies second contracts for draftees, and calculates % of position top money.

Data source: https://github.com/nflverse/nflverse-data (contracts release)
No API key needed.
"""

import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONTRACTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/contracts/historical_contracts.parquet"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def download_contracts() -> pd.DataFrame:
    """Download all historical contracts from nflverse."""
    logger.info("Downloading contracts data...")
    df = pd.read_parquet(CONTRACTS_URL)
    logger.info(f"Got {len(df)} contracts ({df['year_signed'].min()}-{df['year_signed'].max()})")
    return df


def build_second_contracts(df: pd.DataFrame, draft_start: int = 2015, draft_end: int = 2022) -> pd.DataFrame:
    """
    For each drafted player, find their second contract and calculate
    what % of the position's top APY they got at the time.
    """
    # Only drafted players in our range
    drafted = df[(df["draft_year"] >= draft_start) & (df["draft_year"] <= draft_end)].copy()
    drafted = drafted[drafted["draft_round"].notna()]

    # Second contract = first contract signed after draft year
    drafted = drafted[drafted["year_signed"] > drafted["draft_year"]].copy()
    drafted = drafted.sort_values(["otc_id", "year_signed"])
    second = drafted.drop_duplicates(subset="otc_id", keep="first")
    logger.info(f"{len(second)} second contracts found")

    # For each position+year, find the max APY from ALL prior years (not including current year)
    # This way if a player resets the market, their pct_of_top will be > 1.0
    position_max = df.groupby(["position", "year_signed"])["apy"].max().reset_index()
    position_max = position_max.rename(columns={"apy": "pos_top_apy"})
    position_max = position_max.sort_values(["position", "year_signed"])
    position_max["pos_top_apy_prior"] = position_max.groupby("position")["pos_top_apy"].cummax().shift(1)
    # Fill first year per position with that year's own max as fallback
    position_max["pos_top_apy_prior"] = position_max.groupby("position")["pos_top_apy_prior"].ffill()

    second = second.merge(
        position_max[["position", "year_signed", "pos_top_apy_prior"]],
        on=["position", "year_signed"],
        how="left",
    )

    second["pct_of_top"] = (second["apy"] / second["pos_top_apy_prior"]).round(4)
    second["market_reset"] = second["pct_of_top"] > 1.0

    result = second[[
        "player", "position", "team", "year_signed", "years", "value", "apy",
        "pos_top_apy_prior", "pct_of_top", "market_reset",
        "draft_year", "draft_round", "draft_overall", "draft_team", "college",
    ]].rename(columns={
        "pos_top_apy_prior": "pos_top_apy",
    }).sort_values("pct_of_top", ascending=False).reset_index(drop=True)

    logger.info(f"Sample:\n{result.head(10).to_string(index=False)}")
    return result


def save(df: pd.DataFrame, filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")


if __name__ == "__main__":
    contracts = download_contracts()
    save(contracts, "contracts_all.parquet")

    # Also save as CSV for the app
    contracts.to_csv(os.path.join(DATA_DIR, "contracts_all.csv"), index=False)
    logger.info("Saved full contracts CSV")

    second = build_second_contracts(contracts)
    second_path = os.path.join(PROCESSED_DIR, "second_contracts.csv")
    second.to_csv(second_path, index=False)
    logger.info(f"Saved {len(second)} second contracts to {second_path}")

    logger.info("Done!")
