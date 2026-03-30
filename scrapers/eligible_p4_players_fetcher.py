"""
College Football Players Scraper
Fetches junior/senior player data from Power 4 conferences + Notre Dame via ESPN's public API
"""

import requests
import pandas as pd
import time
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"


STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"

P4_CONF_IDS = {
    8: "SEC",
    5: "Big Ten",
    4: "Big 12",
    1: "ACC",
}
NOTRE_DAME_TEAM_ID = "87"


def get_teams(season: int = 2025) -> List[Dict]:
    """Fetch teams from all Power 4 conferences plus Notre Dame using standings API"""
    teams = []
    seen_ids = set()

    for conf_id, conf_name in P4_CONF_IDS.items():
        resp = requests.get(
            STANDINGS_URL, params={"group": conf_id, "season": season}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        for entry in data.get("standings", {}).get("entries", []):
            team = entry.get("team", {})
            tid = str(team.get("id", ""))
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                teams.append({"id": tid, "name": team.get("displayName", "")})

        logger.info(f"Found {len(data.get('standings', {}).get('entries', []))} teams for {conf_name}")

    # Add Notre Dame (independent)
    if NOTRE_DAME_TEAM_ID not in seen_ids:
        teams.append({"id": NOTRE_DAME_TEAM_ID, "name": "Notre Dame Fighting Irish"})

    logger.info(f"Total: {len(teams)} teams (Power 4 + Notre Dame)")
    return teams


ELIGIBLE_CLASSES = {"Junior", "Senior"}


def get_roster(team_id: str, team_name: str) -> List[Dict]:
    """Fetch juniors and seniors from a single team's roster"""
    try:
        resp = requests.get(f"{BASE_URL}/teams/{team_id}/roster", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch roster for {team_name}: {e}")
        return []

    players = []
    for group in data.get("athletes", []):
        for athlete in group.get("items", []):
            experience = athlete.get("experience", {}).get("displayValue", "")
            if experience not in ELIGIBLE_CLASSES:
                continue
            players.append({
                "name": athlete.get("fullName", ""),
                "position": athlete.get("position", {}).get("abbreviation", ""),
                "jersey": athlete.get("jersey", ""),
                "height": athlete.get("displayHeight", ""),
                "weight": athlete.get("displayWeight", ""),
                "experience": experience,
                "college": team_name,
            })

    return players


def scrape_all(season: int = 2025, delay: float = 1.0) -> pd.DataFrame:
    """Scrape rosters from all teams for a given season"""
    teams = get_teams(season=season)
    all_players = []

    for i, team in enumerate(teams):
        logger.info(f"[{i+1}/{len(teams)}] {team['name']}")
        players = get_roster(team["id"], team["name"])
        all_players.extend(players)
        time.sleep(delay)

    df = pd.DataFrame(all_players)
    logger.info(f"Total players scraped: {len(df)}")
    return df


def save(df: pd.DataFrame, filename: str = "eligible_players.csv"):
    filepath = f"../data/raw/{filename}"
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} players to {filepath}")


if __name__ == "__main__":
    df = scrape_all()
    if not df.empty:
        save(df)
    else:
        logger.warning("No player data collected")
