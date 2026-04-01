"""
Eligible Players Stats Fetcher
Fetches player season stats from the CFBD API for P4 conferences + Notre Dame.

API call budget: 5 calls (4 conferences + Notre Dame)
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



def fetch_stats_by_conference(year: int, conference: str) -> list:
    """Fetch all player season stats for a conference. 1 API call."""
    logger.info(f"Fetching stats for {conference}...")
    resp = requests.get(
        f"{CFBD_URL}/stats/player/season",
        headers=HEADERS,
        params={"year": year, "conference": conference},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"  Got {len(data)} stat rows")
    return data


def fetch_stats_by_team(year: int, team: str) -> list:
    """Fetch all player season stats for a single team. 1 API call."""
    logger.info(f"Fetching stats for {team}...")
    resp = requests.get(
        f"{CFBD_URL}/stats/player/season",
        headers=HEADERS,
        params={"year": year, "team": team},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"  Got {len(data)} stat rows")
    return data


CONF_FILE_NAMES = {
    "SEC": "sec_players_stats.csv",
    "B1G": "big_ten_players_stats.csv",
    "B12": "big_12_players_stats.csv",
    "ACC": "acc_players_stats.csv",
}


CONF_DISPLAY_NAMES = {
    "SEC": "sec",
    "B1G": "big_ten",
    "B12": "big_12",
    "ACC": "acc",
}


def save(df: pd.DataFrame, filename: str, subdir: str = None):
    parts = [os.path.dirname(__file__), "..", "data", "raw"]
    if subdir:
        parts.append(subdir)
    filepath = os.path.join(*parts, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")


if __name__ == "__main__":
    START_YEAR = 2014
    END_YEAR = 2025
    api_calls = 0

    for year in range(START_YEAR, END_YEAR + 1):
        logger.info(f"--- Fetching {year} season ---")

        # Conference stats (4 API calls per year)
        for conf, display_name in CONF_DISPLAY_NAMES.items():
            data = fetch_stats_by_conference(year, conf)
            api_calls += 1
            if data:
                save(pd.DataFrame(data), f"{display_name}_{year}.csv", subdir="college_stats")

        # Notre Dame stats (1 API call per year)
        nd_data = fetch_stats_by_team(year, "Notre Dame")
        api_calls += 1
        if nd_data:
            save(pd.DataFrame(nd_data), f"notre_dame_{year}.csv", subdir="college_stats")

    logger.info(f"Done! Total API calls used: {api_calls}")
