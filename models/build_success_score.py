"""
Build NFL success scores for draftees (2017-2022) by position.

Composite score: 70% AV/game + 30% second contract value.
Arctan transform (s=5, center=0.35) to compress elite tier.

Manual contract fills for players without second contracts in data.
"""

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Arctan parameters
ARCTAN_STEEPNESS = 5
ARCTAN_CENTER = 0.35

# Composite weights
AV_WEIGHT = 0.7
CONTRACT_WEIGHT = 0.3

# Position abbreviation in career stats -> full name for file output
POSITION_FILE_NAMES = {
    "WR": "wide_receiver",
    "RB": "running_back",
    "TE": "tight_end",
    "QB": "quarterback",
}

# Manual second contract fills (pct_of_top) per position
MANUAL_CONTRACTS = {
    "WR": {
        "Drake London": 1.00,
        "Chris Olave": 0.90,
        "Michael Pittman Jr.": 0.486,
        "Henry Ruggs III": 0.40,
        "KJ Hamler": 0.12,
        "Ben Skowronek": 0.063,
    },
    "RB": {
        "Nyheim Hines": 0.18,
        "AJ Dillon": 0.05,
        "Benny Snell Jr.": 0.04,
    },
    "TE": {
        "Chris Herndon": 0.069,       # Listed as "Christopher Herndon IV" in contracts
        "Jaylen Samuels": 0.016,       # Listed under RB in contracts (position convert)
    },
    "QB": {
        "Gardner Minshew II": 0.07,    # Listed as "Gardner Minshew" in contracts
    },
}

# Players with no meaningful second contract per position
ZERO_CONTRACT = {
    "WR": [
        "Kadarius Toney", "Dyami Brown", "Tylan Wallace", "Nico Collins",
        "Terrace Marshall Jr.", "Tutu Atwell",
        "DJ Chark", "Laviska Shenault Jr.", "Olabisi Johnson",
        "Lynn Bowden Jr.", "John Metchie", "Riley Ridley",
        "Tremon Smith", "JJ Arcega-Whiteside", "Michael Woods II",
    ],
    "RB": [
        "Brandon Wilson", "Zamir White", "Pierre Strong", "Eno Benjamin",
        "Derrius Guice", "Chris Evans", "Hassan Haskins", "Larry Rountree",
        "Malcolm Perry", "Kerrith Whyte Jr", "Anthony McFarland Jr.",
        "T.J. Logan", "Matthew Dayes", "Nick Bawden", "Chandler Cox",
        "Cullen Gillaspia", "Kylin Hill", "Trestan Ebner",
    ],
    "TE": [
        "Brycen Hopkins",
    ],
    "QB": [
        "Ryan Finley",
    ],
}


def build_success(position: str) -> pd.DataFrame:
    """Build success scores for a position."""
    career = pd.read_csv(os.path.join(RAW_DIR, "draftees_nfl_career_stats.csv"))
    contracts = pd.read_csv(os.path.join(PROCESSED_DIR, "second_contracts.csv"))

    # Filter to position, 2017-2022, 5+ games
    df = career[
        (career["position"] == position)
        & (career["games"] >= 5)
        & (career["draft_year"] >= 2017)
        & (career["draft_year"] <= 2022)
    ].copy()
    df["av_per_game"] = df["w_av"] / df["games"]
    logger.info(f"{position}s with 5+ games (2017-2022): {len(df)}")

    # Normalize AV/game to 0-1
    av_min, av_max = df["av_per_game"].min(), df["av_per_game"].max()
    df["av_norm"] = (df["av_per_game"] - av_min) / (av_max - av_min)

    # Merge second contracts
    pos_contracts = contracts[contracts["position"] == position]
    df = df.merge(
        pos_contracts[["player", "pct_of_top"]],
        left_on="name", right_on="player", how="left",
    )
    df = df.drop(columns=["player"], errors="ignore")

    # Apply manual fills
    manual = MANUAL_CONTRACTS.get(position, {})
    for name, val in manual.items():
        df.loc[df["name"] == name, "pct_of_top"] = val

    zeros = ZERO_CONTRACT.get(position, [])
    for name in zeros:
        df.loc[df["name"] == name, "pct_of_top"] = 0.0

    # Check for remaining missing
    still_missing = df[df["pct_of_top"].isna()]["name"].tolist()
    if still_missing:
        logger.warning(f"Still missing contracts for: {still_missing}")

    # Normalize contract: cap at 1.1 (market resetters), scale to 0-1
    df["contract_norm"] = df["pct_of_top"].clip(upper=1.1) / 1.1
    df["contract_norm"] = df["contract_norm"].fillna(0)

    # 70/30 composite
    df["composite"] = AV_WEIGHT * df["av_norm"] + CONTRACT_WEIGHT * df["contract_norm"]

    # Arctan transform
    raw = np.arctan(ARCTAN_STEEPNESS * (df["composite"] - ARCTAN_CENTER))
    r_min = np.arctan(ARCTAN_STEEPNESS * (df["composite"].min() - ARCTAN_CENTER))
    r_max = np.arctan(ARCTAN_STEEPNESS * (df["composite"].max() - ARCTAN_CENTER))
    df["success_score"] = (raw - r_min) / (r_max - r_min)

    # Keep relevant columns
    output_cols = [
        "name", "position", "draft_year", "college", "games",
        "w_av", "av_per_game", "pct_of_top", "success_score",
    ]
    result = df[output_cols].sort_values("success_score", ascending=False).reset_index(drop=True)
    return result


if __name__ == "__main__":
    for pos, file_label in POSITION_FILE_NAMES.items():
        df = build_success(pos)
        logger.info(f"Scored {len(df)} {pos}s")

        outpath = os.path.join(PROCESSED_DIR, f"success_{file_label}.csv")
        df.to_csv(outpath, index=False)
        logger.info(f"Saved to {outpath}")

        print(f"\n=== {pos} Top 15 ===")
        print(df.head(15).to_string(index=False))
        print(f"\n=== {pos} Bottom 5 ===")
        print(df.tail(5).to_string(index=False))
