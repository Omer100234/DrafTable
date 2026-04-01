"""
NFL Combine Measurables Scraper
Downloads combine data from nflverse.

Data source: https://github.com/nflverse/nflverse-data
No API key needed.
"""

import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMBINE_URL = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_combine() -> pd.DataFrame:
    """Download combine measurables from nflverse."""
    logger.info("Downloading combine data...")
    df = pd.read_csv(COMBINE_URL)
    logger.info(f"Got {len(df)} combine entries ({df['season'].min()}-{df['season'].max()})")
    return df


if __name__ == "__main__":
    df = download_combine()
    filepath = os.path.join(DATA_DIR, "combine.csv")
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info("Done!")
