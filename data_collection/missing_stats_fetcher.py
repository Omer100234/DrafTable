"""
Missing Stats Fetcher
Scans draftees (2017-2025) and fetches college stats only for players
not already in our data. Uses conference-level calls where possible,
team-level calls for independents and small conferences.
Skips any (conference, season) or (team, season) already fetched.
"""

import requests
import pandas as pd
import logging
import os
import glob
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

STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "college_stats")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Conferences large enough to fetch by conference (1 call = all teams)
CONF_API_NAMES = {
    "SEC": "SEC",
    "Big Ten": "B1G",
    "Big 12": "B12",
    "ACC": "ACC",
    "Pac-12": "PAC",
    "American Athletic": "AAC",
    "Mountain West": "MWC",
    "Conference USA": "CUSA",
    "Mid-American": "MAC",
    "Sun Belt": "SBC",
    "MVFC": "MVFC",
    "CAA": "CAA",
    "Big Sky": "Big Sky",
    "SWAC": "SWAC",
    "Southern": "Southern",
    "Southland": "Southland",
    "OVC": "OVC",
    "Ivy": "Ivy",
    "Patriot": "Patriot",
    "MEAC": "MEAC",
    "NEC": "NEC",
    "Pioneer": "Pioneer",
}

SEASONS_BACK = 4


def fetch_stats_by_conference(year, conference_api):
    logger.info(f"  API call: conference={conference_api} year={year}")
    resp = requests.get(
        f"{CFBD_URL}/stats/player/season",
        headers=HEADERS,
        params={"year": year, "conference": conference_api},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_stats_by_team(year, team):
    logger.info(f"  API call: team={team} year={year}")
    resp = requests.get(
        f"{CFBD_URL}/stats/player/season",
        headers=HEADERS,
        params={"year": year, "team": team},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def load_existing_index():
    """Build sets of already-fetched (conference, season) and (team, season)."""
    fetched_confs = set()
    fetched_teams = set()
    all_player_ids = set()

    for f in glob.glob(os.path.join(STATS_DIR, "*.csv")):
        df = pd.read_csv(f, usecols=["playerId", "team", "conference", "season"])
        all_player_ids.update(df["playerId"].unique())
        for _, row in df[["conference", "season"]].drop_duplicates().iterrows():
            fetched_confs.add((row["conference"], int(row["season"])))
        for _, row in df[["team", "season"]].drop_duplicates().iterrows():
            fetched_teams.add((row["team"], int(row["season"])))

    return fetched_confs, fetched_teams, all_player_ids


def find_missing_draftees():
    """Find draftees (2017-2025) whose collegeAthleteId is not in our stats."""
    missing = []
    for year in range(2017, 2026):
        df = pd.read_csv(os.path.join(RAW_DIR, f"draftees_{year}.csv"))
        for _, row in df.iterrows():
            pid = row.get("collegeAthleteId")
            if pd.notna(pid):
                missing.append({
                    "draft_year": year,
                    "pid": int(pid),
                    "team": row["collegeTeam"],
                    "conf": row.get("collegeConference"),
                })
    return missing


def save_data(data, label, year):
    """Save fetched data to college_stats directory."""
    if not data:
        return
    safe_label = label.lower().replace(" ", "_").replace("-", "_")
    filename = f"{safe_label}_{year}.csv"
    filepath = os.path.join(STATS_DIR, filename)
    df = pd.DataFrame(data)
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates()
    df.to_csv(filepath, index=False)
    logger.info(f"  Saved {len(df)} rows to {filename}")


def main():
    fetched_confs, fetched_teams, all_player_ids = load_existing_index()
    all_draftees = find_missing_draftees()

    # Filter to only truly missing players
    missing = [m for m in all_draftees if m["pid"] not in all_player_ids]
    logger.info(f"Found {len(missing)} draftees missing from stats")

    # Build needed fetches
    needed_conf_calls = set()   # (conf_display_name, conf_api_name, season)
    needed_team_calls = set()   # (team, season)

    for m in missing:
        conf = m["conf"]
        draft_year = m["draft_year"]
        for season in range(draft_year - SEASONS_BACK, draft_year):
            if pd.isna(conf) or conf == "FBS Independents":
                if (m["team"], season) not in fetched_teams:
                    needed_team_calls.add((m["team"], season))
            elif conf in CONF_API_NAMES:
                if (conf, season) not in fetched_confs:
                    needed_conf_calls.add((conf, CONF_API_NAMES[conf], season))
            else:
                # Small/unknown conference — fetch by team
                if (m["team"], season) not in fetched_teams:
                    needed_team_calls.add((m["team"], season))

    total_calls = len(needed_conf_calls) + len(needed_team_calls)
    logger.info(f"Need {len(needed_conf_calls)} conference calls + {len(needed_team_calls)} team calls = {total_calls} total")

    api_calls = 0

    # Conference-level fetches
    for conf_name, conf_api, season in sorted(needed_conf_calls):
        try:
            data = fetch_stats_by_conference(season, conf_api)
            api_calls += 1
            save_data(data, conf_name, season)
        except Exception as e:
            logger.error(f"  Failed: {conf_name} {season}: {e}")

    # Team-level fetches
    for team, season in sorted(needed_team_calls):
        try:
            data = fetch_stats_by_team(season, team)
            api_calls += 1
            save_data(data, team, season)
        except Exception as e:
            logger.error(f"  Failed: {team} {season}: {e}")

    logger.info(f"Done! Made {api_calls} API calls")


if __name__ == "__main__":
    main()
