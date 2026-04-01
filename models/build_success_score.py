"""
Build NFL success scores for WRs (2017-2022 draftees).

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

# Manual second contract fills (pct_of_top)
MANUAL_CONTRACTS = {
    "Drake London": 1.00,
    "Chris Olave": 0.90,
    "Michael Pittman Jr.": 0.486,
    "Henry Ruggs III": 0.40,
    "KJ Hamler": 0.12,
    "Ben Skowronek": 0.063,
}

# Players with no meaningful second contract (busts / depth guys)
ZERO_CONTRACT = [
    "Kadarius Toney", "Dyami Brown", "Tylan Wallace", "Nico Collins",
    "Terrace Marshall Jr.", "Tutu Atwell",
    "DJ Chark", "Laviska Shenault Jr.", "Olabisi Johnson",
    "Lynn Bowden Jr.", "John Metchie", "Riley Ridley",
    "Tremon Smith", "JJ Arcega-Whiteside", "Michael Woods II",
]

# Arctan parameters
ARCTAN_STEEPNESS = 5
ARCTAN_CENTER = 0.35

# Composite weights
AV_WEIGHT = 0.7
CONTRACT_WEIGHT = 0.3


def build_wr_success() -> pd.DataFrame:
    """Build success scores for WRs."""
    career = pd.read_csv(os.path.join(RAW_DIR, "draftees_nfl_career_stats.csv"))
    contracts = pd.read_csv(os.path.join(PROCESSED_DIR, "second_contracts.csv"))

    # Filter to 2017-2022 WRs with 5+ games
    wr = career[
        (career["position"] == "WR")
        & (career["games"] >= 5)
        & (career["draft_year"] >= 2017)
        & (career["draft_year"] <= 2022)
    ].copy()
    wr["av_per_game"] = wr["w_av"] / wr["games"]
    logger.info(f"WRs with 5+ games (2017-2022): {len(wr)}")

    # Normalize AV/game to 0-1
    av_min, av_max = wr["av_per_game"].min(), wr["av_per_game"].max()
    wr["av_norm"] = (wr["av_per_game"] - av_min) / (av_max - av_min)

    # Merge second contracts
    wr_contracts = contracts[contracts["position"] == "WR"]
    wr = wr.merge(
        wr_contracts[["player", "pct_of_top"]],
        left_on="name", right_on="player", how="left",
    )
    wr = wr.drop(columns=["player"], errors="ignore")

    # Apply manual fills
    for name, val in MANUAL_CONTRACTS.items():
        wr.loc[wr["name"] == name, "pct_of_top"] = val
    for name in ZERO_CONTRACT:
        wr.loc[wr["name"] == name, "pct_of_top"] = 0.0

    # Normalize contract: cap at 1.1 (market resetters), scale to 0-1
    wr["contract_norm"] = wr["pct_of_top"].clip(upper=1.1) / 1.1
    wr["contract_norm"] = wr["contract_norm"].fillna(0)

    # 70/30 composite
    wr["composite"] = AV_WEIGHT * wr["av_norm"] + CONTRACT_WEIGHT * wr["contract_norm"]

    # Arctan transform
    raw = np.arctan(ARCTAN_STEEPNESS * (wr["composite"] - ARCTAN_CENTER))
    r_min = np.arctan(ARCTAN_STEEPNESS * (wr["composite"].min() - ARCTAN_CENTER))
    r_max = np.arctan(ARCTAN_STEEPNESS * (wr["composite"].max() - ARCTAN_CENTER))
    wr["success_score"] = (raw - r_min) / (r_max - r_min)

    # Keep relevant columns
    output_cols = [
        "name", "position", "draft_year", "college", "games",
        "w_av", "av_per_game", "pct_of_top", "success_score",
    ]
    result = wr[output_cols].sort_values("success_score", ascending=False).reset_index(drop=True)
    return result


if __name__ == "__main__":
    df = build_wr_success()
    logger.info(f"Scored {len(df)} WRs")

    outpath = os.path.join(PROCESSED_DIR, "success_wide_receiver.csv")
    df.to_csv(outpath, index=False)
    logger.info(f"Saved to {outpath}")

    print("\nTop 15:")
    print(df.head(15).to_string(index=False))
    print(f"\nBottom 5:")
    print(df.tail(5).to_string(index=False))
