"""
Draft Prospects Scraper
Scrapes 2026 NFL Draft prospect rankings from DraftTek's Big Board.
No API key needed. ~500 prospects across 5 pages.
"""

import requests
import pandas as pd
import logging
import os
import time
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.drafttek.com/2026-NFL-Draft-Big-Board/Top-NFL-Draft-Prospects-2026-Page-{}.asp"
PAGES = range(1, 6)  # Pages 1-5 (1-500 prospects, page 6 is "coming soon")


def scrape_page(page_num: int) -> list:
    """Scrape a single page of prospects."""
    url = BASE_URL.format(page_num)
    logger.info(f"Scraping page {page_num}: {url}")

    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    players = []

    # Find all table rows with prospect data
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 7:
            continue

        text = [c.get_text(strip=True) for c in cells]

        # Skip header rows or empty rows
        try:
            rank = int(text[0])
        except (ValueError, IndexError):
            continue

        players.append({
            "rank": rank,
            "name": text[2],
            "college": text[3],
            "position": text[4],
            "height": text[5],
            "weight": text[6],
            "class": text[7] if len(text) > 7 else "",
        })

    logger.info(f"  Found {len(players)} prospects")
    return players


def scrape_all() -> pd.DataFrame:
    """Scrape all pages of prospects."""
    all_players = []

    for page in PAGES:
        players = scrape_page(page)
        all_players.extend(players)
        time.sleep(2)  # Be respectful

    df = pd.DataFrame(all_players)
    logger.info(f"Total prospects scraped: {len(df)}")
    return df


def save(df: pd.DataFrame, filename: str = "prospects_2026.csv"):
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "raw", filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} prospects to {filepath}")


if __name__ == "__main__":
    df = scrape_all()
    if not df.empty:
        save(df)
    else:
        logger.warning("No prospects scraped")
