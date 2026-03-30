"""
Draftees Fetcher
Fetches NFL draft pick data from the CFBD API.
1 API call per year.
"""

import requests
import pandas as pd
import logging
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CFBD_URL = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFBD_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}


def fetch_draft_picks(year: int) -> pd.DataFrame:
    """Fetch all NFL draft picks for a given year. 1 API call."""
    logger.info(f"Fetching {year} draft picks...")
    resp = requests.get(
        f"{CFBD_URL}/draft/picks",
        headers=HEADERS,
        params={"year": year},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"Got {len(data)} draft picks")
    return pd.DataFrame(data)


def save(df: pd.DataFrame, filename: str):
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "raw", filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")


if __name__ == "__main__":
    for year in range(2015, 2026):
        picks_df = fetch_draft_picks(year)
        if not picks_df.empty:
            save(picks_df, f"draftees_{year}.csv")
        else:
            logger.warning(f"No draft data available for {year}")
