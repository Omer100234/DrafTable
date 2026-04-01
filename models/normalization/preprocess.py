"""
Preprocessing steps applied before normalization.

1. COVID scaling: Scale 2020 season volume stats by conference game count.
2. Injury fill-forward: If all volume stats drop below 20% of previous season,
   carry forward the previous season's stats (volume + rate).
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

# 2020 COVID season game counts by conference
COVID_GAMES = {
    "SEC": 10,
    "Big Ten": 8,
    "Big 12": 9,
    "Pac-12": 6,
    "ACC": 11,
    "American Athletic": 10,
    "Mountain West": 8,
    "Sun Belt": 10,
    "Mid-American": 6,
    "Conference USA": 10,
    "FBS Independents": 10,
}
NORMAL_GAMES = 13

# Volume stats to scale (rate stats like YPR, YPC stay the same)
VOLUME_SUFFIXES = [
    "receiving_REC", "receiving_YDS", "receiving_TD",
    "rushing_CAR", "rushing_YDS", "rushing_TD",
    "defensive_TOT", "defensive_SOLO", "defensive_TFL",
    "defensive_SACKS", "defensive_QB_HUR",
    "interceptions_INT", "interceptions_YDS", "interceptions_TD",
    "passing_COMP", "passing_ATT", "passing_YDS", "passing_TD", "passing_INT",
    "kicking_FGM", "kicking_FGA", "kicking_XPM", "kicking_XPA",
    "punting_NO", "punting_YDS",
]

# Which season slots correspond to the 2020 season for each draft year
COVID_SLOTS = {
    2021: ["Y0"],
    2022: ["Y1"],
}

INJURY_THRESHOLD = 0.40


def apply_covid_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Scale 2020 season volume stats up by conference game ratio."""
    result = df.copy()
    scaled_count = 0

    for draft_year, slots in COVID_SLOTS.items():
        mask = result["draft_year"] == draft_year
        for slot in slots:
            for suffix in VOLUME_SUFFIXES:
                col = f"{slot}_{suffix}"
                if col not in result.columns:
                    continue
                for conf, games in COVID_GAMES.items():
                    conf_mask = mask & (result["conference"] == conf)
                    if conf_mask.sum() == 0:
                        continue
                    scale = NORMAL_GAMES / games
                    before = result.loc[conf_mask, col]
                    scaled_count += (before.notna() & (before != 0)).sum()
                    result.loc[conf_mask, col] = result.loc[conf_mask, col] * scale

    logger.info(f"COVID scaling: {scaled_count} values scaled across {sum(len(s) for s in COVID_SLOTS.values())} season slots")
    return result


def apply_injury_fill_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Fill forward stats when a season looks like an injury (all volume < 20% of prev)."""
    result = df.copy()
    fill_count = 0

    # Season pairs: (current, previous) — if current looks injured, fill from previous
    season_pairs = [("Y0", "Y1"), ("Y1", "Y2"), ("Y2", "Y3")]

    for idx, row in result.iterrows():
        for curr, prev in season_pairs:
            # Get all volume columns that exist for both seasons
            curr_cols = []
            prev_cols = []
            for suffix in VOLUME_SUFFIXES:
                cc = f"{curr}_{suffix}"
                pc = f"{prev}_{suffix}"
                if cc in result.columns and pc in result.columns:
                    curr_cols.append(cc)
                    prev_cols.append(pc)

            if not curr_cols:
                continue

            # Check if previous season has data
            prev_vals = row[prev_cols]
            if prev_vals.isna().all() or (prev_vals.fillna(0).eq(0)).all():
                continue

            # Check if current season has data
            curr_vals = row[curr_cols]
            if curr_vals.isna().all():
                continue

            # Check if ALL current volume stats are < 20% of previous
            all_below = True
            for cc, pc in zip(curr_cols, prev_cols):
                cv = row[cc] if pd.notna(row[cc]) else 0
                pv = row[pc] if pd.notna(row[pc]) else 0
                if pv > 0 and cv >= INJURY_THRESHOLD * pv:
                    all_below = False
                    break

            if all_below:
                # Fill ALL stat columns (volume + rate) from previous season
                all_suffixes = [s for s in result.columns if s.startswith(f"{curr}_")]
                for col in all_suffixes:
                    suffix = col[len(f"{curr}_"):]
                    prev_col = f"{prev}_{suffix}"
                    if prev_col in result.columns and pd.notna(row[prev_col]):
                        result.loc[idx, col] = row[prev_col]

                fill_count += 1
                logger.info(f"  Injury fill: {row['name']} {curr} <- {prev}")

    logger.info(f"Injury fill-forward: {fill_count} season slots filled")
    return result


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run all preprocessing steps."""
    df = apply_covid_scaling(df)
    df = apply_injury_fill_forward(df)
    return df
