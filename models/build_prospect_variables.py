"""
Build variables_class_2026_POSITION.csv files for draft prospects.
Converts prospect data (class-label seasons) to the same Y0/Y1/Y2/Y3 format
used by draftee variable files, then merges combine + age-derived features.
"""

import pandas as pd
import re
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
STATS_DIR = os.path.join(RAW_DIR, "college_stats")

DRAFT_YEAR = 2026

# Class -> ordered list of class labels, most recent season first (Y0, Y1, Y2, Y3)
CLASS_TO_Y = {
    "FR":  ["FR"],
    "SO":  ["SO", "FR"],
    "RSO": ["RSO", "FR"],
    "JR":  ["JR", "SO", "FR"],
    "RJR": ["RJR", "SO", "FR"],
    "SR":  ["SR", "JR", "SO", "FR"],
    "RSR": ["RSR", "JR", "SO", "FR"],
    "GR":  ["GR", "SR", "JR", "SO"],
}

# Prospect position abbreviations -> full position names (matching build_variables.py)
POSITION_MAP = {
    "QB": "Quarterback",
    "RB": "Running Back",
    "FB": "Fullback",
    "WR": "Wide Receiver",
    "WRS": "Wide Receiver",
    "TE": "Tight End",
    "OT": "Offensive Tackle",
    "OG": "Offensive Guard",
    "OC": "Center",
    "EDGE": "Defensive End",
    "DL5T": "Defensive End",
    "DL3T": "Defensive Tackle",
    "DL1T": "Defensive Tackle",
    "ILB": "Inside Linebacker",
    "OLB": "Outside Linebacker",
    "CB": "Cornerback",
    "CBN": "Cornerback",
    "S": "Safety",
    "PK": "Place Kicker",
    "P": "Punter",
    "LS": "Long Snapper",
}

# Stat categories relevant to each position group (same as build_variables.py)
POSITION_STATS = {
    "Quarterback": ["passing", "rushing", "fumbles"],
    "Running Back": ["rushing", "receiving", "fumbles", "kickReturns", "puntReturns"],
    "Fullback": ["rushing", "receiving", "fumbles"],
    "Wide Receiver": ["receiving", "rushing", "fumbles", "kickReturns", "puntReturns"],
    "Tight End": ["receiving", "rushing", "fumbles"],
    "Offensive Tackle": [],
    "Offensive Guard": [],
    "Center": [],
    "Defensive End": ["defensive", "fumbles"],
    "Defensive Tackle": ["defensive", "fumbles"],
    "Outside Linebacker": ["defensive", "fumbles", "interceptions"],
    "Inside Linebacker": ["defensive", "fumbles", "interceptions"],
    "Cornerback": ["defensive", "interceptions", "puntReturns", "kickReturns"],
    "Safety": ["defensive", "interceptions", "fumbles"],
    "Place Kicker": ["kicking"],
    "Punter": ["punting"],
}


def parse_height_inches(h):
    """Convert '6\\'2\"' or '6-2' format to inches."""
    if pd.isna(h):
        return None
    h = str(h).strip().replace('"', '').replace("'", "'")
    m = re.match(r"(\d+)'(\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    m = re.match(r"(\d+)-(\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return None


def build_team_conference_map():
    """Build team -> conference mapping from the most recent college stats."""
    team_conf = {}
    for f in os.listdir(STATS_DIR):
        if not f.endswith(".csv"):
            continue
        conf_name = f.rsplit("_", 1)[0]  # e.g. "sec" from "sec_2025.csv"
        try:
            df = pd.read_csv(os.path.join(STATS_DIR, f), usecols=["team", "conference"])
            for _, row in df.iterrows():
                if pd.notna(row["conference"]):
                    team_conf[row["team"]] = row["conference"]
        except Exception:
            continue
    return team_conf


def convert_class_to_y(df):
    """Convert class-label columns (JR_receiving_REC) to Y0/Y1/Y2/Y3 format."""
    all_class_labels = set()
    for labels in CLASS_TO_Y.values():
        all_class_labels.update(labels)

    rows = []
    for _, row in df.iterrows():
        player_class = row.get("class", "")
        y_labels = CLASS_TO_Y.get(player_class, [])

        new_row = {
            "name": row["name"],
            "position": row["position"],
            "college": row.get("current_college", ""),
        }

        # Map class labels to Y indices
        for y_idx, class_label in enumerate(y_labels):
            if y_idx >= 4:
                break
            y_prefix = f"Y{y_idx}"

            # Copy college/season meta
            college_col = f"{class_label}_college"
            season_col = f"{class_label}_season"
            if college_col in row.index and pd.notna(row[college_col]):
                new_row[f"{y_prefix}_college"] = row[college_col]
            if season_col in row.index and pd.notna(row[season_col]):
                new_row[f"{y_prefix}_season"] = row[season_col]

            # Copy all stat columns
            for col in row.index:
                if col.startswith(f"{class_label}_") and col not in [college_col, season_col]:
                    stat_suffix = col[len(class_label) + 1:]  # e.g. "receiving_REC"
                    new_row[f"{y_prefix}_{stat_suffix}"] = row[col]

        rows.append(new_row)

    return pd.DataFrame(rows)


def load_combine():
    """Load combine data for 2026."""
    filepath = os.path.join(RAW_DIR, "combine.csv")
    df = pd.read_csv(filepath)
    df = df[df["season"] == DRAFT_YEAR]
    logger.info(f"Loaded {len(df)} combine entries for {DRAFT_YEAR}")
    return df


def load_raw_prospects():
    """Load raw prospects for height/weight."""
    filepath = os.path.join(RAW_DIR, "prospects_2026.csv")
    df = pd.read_csv(filepath)
    df["height_inches"] = df["height"].apply(parse_height_inches)
    return df


def add_derived_features(df, position):
    """Add years_since_first_played and played_at_X features."""
    seasons = ["Y3", "Y2", "Y1", "Y0"]
    season_years_val = {"Y3": 4, "Y2": 3, "Y1": 2, "Y0": 1}

    all_season_cols = {}
    meta_suffixes = {"college", "season"}
    for season in seasons:
        all_season_cols[season] = [c for c in df.columns if c.startswith(f"{season}_")
                                   and c.split("_", 1)[1] not in meta_suffixes]

    # years_since_first_played
    df["years_since_first_played"] = 0
    for idx, row in df.iterrows():
        for season in seasons:
            cols = all_season_cols[season]
            if cols and row[cols].notna().any():
                df.loc[idx, "years_since_first_played"] = season_years_val[season]
                break

    # played_at_X columns (need age)
    age_cols = [f"played_at_{a}" for a in range(18, 24)]
    for col in age_cols:
        df[col] = 0

    # Compute position max for threshold
    pos_maxes = {}
    for season in seasons:
        for col in all_season_cols[season]:
            pos_maxes[col] = df[col].max()

    age_offsets = {"Y0": 1, "Y1": 2, "Y2": 3, "Y3": 4}

    for idx, row in df.iterrows():
        if pd.isna(row.get("age")):
            continue
        draft_age = int(row["age"])

        started = False
        first_started_age = None
        for season in seasons:
            season_age = draft_age - age_offsets[season]
            if season_age < 18 or season_age > 23:
                continue

            if not started:
                for col in all_season_cols[season]:
                    max_val = pos_maxes.get(col)
                    if pd.notna(max_val) and max_val > 0 and pd.notna(row[col]):
                        if row[col] >= 0.10 * max_val:
                            started = True
                            first_started_age = season_age
                            break

            if started:
                df.loc[idx, f"played_at_{season_age}"] = 1

        if started and first_started_age is not None:
            y0_age = draft_age - 1
            for a in range(first_started_age, y0_age + 1):
                if 18 <= a <= 23:
                    df.loc[idx, f"played_at_{a}"] = 1

    return df


def filter_position_cols(df, position):
    """Keep only stat columns relevant to the position."""
    stat_categories = POSITION_STATS.get(position, [])

    keep_cols = ["name", "position", "college", "conference", "height", "weight",
                 "forty", "bench", "vertical", "broad_jump", "cone", "shuttle",
                 "age", "years_since_first_played",
                 "played_at_18", "played_at_19", "played_at_20",
                 "played_at_21", "played_at_22", "played_at_23"]

    for col in df.columns:
        if col in keep_cols:
            continue
        if col[:3] in ["Y0_", "Y1_", "Y2_", "Y3_"]:
            if col.endswith("_college"):
                keep_cols.append(col)
                continue
            stat_part = col[3:]
            category = stat_part.split("_")[0]
            if category in stat_categories:
                keep_cols.append(col)

    return df[[c for c in keep_cols if c in df.columns]]


def main():
    # Load all data sources
    prospects_clean = pd.read_csv(os.path.join(PROCESSED_DIR, "prospects_clean_stats.csv"))
    raw_prospects = load_raw_prospects()
    combine_df = load_combine()
    team_conf = build_team_conference_map()

    logger.info(f"Loaded {len(prospects_clean)} prospects with stats")

    # Convert class labels to Y0/Y1/Y2/Y3
    converted = convert_class_to_y(prospects_clean)
    logger.info(f"Converted {len(converted)} prospects to Y format")

    # Map abbreviated positions to full names
    converted["position"] = converted["position"].map(POSITION_MAP).fillna(converted["position"])

    # Add conference from college stats team->conference mapping
    converted["conference"] = converted["college"].map(team_conf)
    matched_conf = converted["conference"].notna().sum()
    logger.info(f"Conference matched: {matched_conf}/{len(converted)}")

    # Add height/weight from raw prospects
    raw_hw = raw_prospects[["name", "height_inches", "weight"]].drop_duplicates(subset="name", keep="first")
    converted = converted.merge(raw_hw, on="name", how="left")
    converted["height"] = converted["height_inches"]
    converted = converted.drop(columns=["height_inches"])

    # Add combine measurables
    combine_cols = ["player_name", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
    combine_sub = combine_df[combine_cols].drop_duplicates(subset="player_name", keep="first")
    converted = converted.merge(combine_sub, left_on="name", right_on="player_name", how="left")
    converted = converted.drop(columns=["player_name"], errors="ignore")
    combine_matched = converted["forty"].notna().sum()
    logger.info(f"Combine matched: {combine_matched}/{len(converted)}")

    # Try to get age from combine data (ht column often has age info, or we estimate)
    # nflverse combine doesn't have age directly, so we'll leave it NaN for now
    # The model handles NaN via SimpleImputer
    if "age" not in converted.columns:
        converted["age"] = None

    # Process per position
    for position in converted["position"].unique():
        pos_df = converted[converted["position"] == position].copy()
        pos_df = add_derived_features(pos_df, position)
        pos_df = filter_position_cols(pos_df, position)

        # Drop columns that are entirely empty
        pos_df = pos_df.dropna(axis=1, how="all")

        pos_label = position.lower().replace(" ", "_")
        outpath = os.path.join(PROCESSED_DIR, f"variables_class_{DRAFT_YEAR}_{pos_label}.csv")
        pos_df.to_csv(outpath, index=False)
        logger.info(f"  {position}: {len(pos_df)} players, {len(pos_df.columns)} cols -> {outpath}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
