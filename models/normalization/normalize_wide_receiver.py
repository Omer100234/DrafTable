"""
Normalize wide receiver variables for ML.

Normalization strategy:
- Arctan (s=8): forty (center=4.45), cone (center=6.95), shuttle (center=4.28)
  Lower is better for all three.
- Linear min-max: height, weight, age, bench, vertical, broad_jump
- Linear min-max: college volume stats (REC, YDS, TD, CAR, rush YDS, rush TD)
- Linear min-max: college rate stats (YPR, YPC)
- Drop: LONG, returns, fumbles
- Untouched: name, position, college, conference, draft_year, binary flags

When normalizing new data (2023-2026), uses saved min/max from training data
so scales are consistent.
"""

import pandas as pd
import numpy as np
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "norm_params_wide_receiver.json")

# Columns to leave untouched
PASSTHROUGH = [
    "name", "position", "college", "conference", "draft_year",
    "years_since_first_played",
    "played_at_18", "played_at_19", "played_at_20",
    "played_at_21", "played_at_22", "played_at_23",
    "Y0_conf_prestige", "Y1_conf_prestige", "Y2_conf_prestige", "Y3_conf_prestige",
]

# Columns to drop
DROP_PATTERNS = ["_LONG", "kickReturns_", "puntReturns_", "fumbles_"]

# Arctan config: (center, steepness) — all lower-is-better
ARCTAN_COLS = {
    "forty": (4.45, 8),
    "cone": (6.95, 8),
    "shuttle": (4.28, 8),
}

# Linear min-max columns
LINEAR_COLS = ["height", "weight", "age", "bench", "vertical", "broad_jump"]

# Conference prestige tiers
CONFERENCE_SCORES = {
    "SEC": 1.0,
    "Big Ten": 1.0,
    "Big 12": 0.9,
    "Pac-12": 0.9,
    "ACC": 0.9,
    "American Athletic": 0.55,
    "Mountain West": 0.55,
    "Sun Belt": 0.35,
    "Mid-American": 0.35,
    "Conference USA": 0.35,
}

# Remap independent schools to their proper conference tier
COLLEGE_CONFERENCE_OVERRIDES = {
    "Notre Dame": "ACC",
    "BYU": "Big 12",
    "Liberty": "Sun Belt",
    "Massachusetts": "Sun Belt",
    "UConn": "American Athletic",
    "New Mexico State": "Conference USA",
}

DEFAULT_CONFERENCE_SCORE = 0.15  # FCS and unmapped independents

STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "college_stats")

_team_conf_cache = None


def get_team_conference_map():
    """Build team -> conference mapping from college stats files."""
    global _team_conf_cache
    if _team_conf_cache is not None:
        return _team_conf_cache
    team_conf = {}
    if os.path.isdir(STATS_DIR):
        for f in os.listdir(STATS_DIR):
            if not f.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(STATS_DIR, f), usecols=["team", "conference"])
                for _, row in df.iterrows():
                    if pd.notna(row["conference"]):
                        team_conf[row["team"]] = row["conference"]
            except Exception:
                continue
    _team_conf_cache = team_conf
    return team_conf


def college_to_prestige(college):
    """Map a college name to its conference prestige score."""
    if pd.isna(college):
        return None
    if college in COLLEGE_CONFERENCE_OVERRIDES:
        conf = COLLEGE_CONFERENCE_OVERRIDES[college]
        return CONFERENCE_SCORES.get(conf, DEFAULT_CONFERENCE_SCORE)
    team_conf = get_team_conference_map()
    conf = team_conf.get(college)
    if conf:
        return CONFERENCE_SCORES.get(conf, DEFAULT_CONFERENCE_SCORE)
    return DEFAULT_CONFERENCE_SCORE


def arctan_norm(series, center, steepness, xmin, xmax):
    """Arctan normalization using provided min/max for scaling."""
    raw = np.arctan(-steepness * (series - center))
    raw_min = np.arctan(-steepness * (xmax - center))
    raw_max = np.arctan(-steepness * (xmin - center))
    if raw_max == raw_min:
        return series * 0
    return (raw - raw_min) / (raw_max - raw_min)


def linear_norm(series, smin, smax):
    """Linear min-max normalization using provided min/max."""
    if smax == smin:
        return series * 0
    return (series - smin) / (smax - smin)


def should_drop(col):
    """Check if a column should be dropped."""
    return any(pattern in col for pattern in DROP_PATTERNS)


def get_conf_score(row):
    """Map conference to prestige score."""
    college = row["college"]
    conference = row["conference"]
    if college in COLLEGE_CONFERENCE_OVERRIDES:
        conference = COLLEGE_CONFERENCE_OVERRIDES[college]
    return CONFERENCE_SCORES.get(conference, DEFAULT_CONFERENCE_SCORE)


def fit_normalize(df):
    """Normalize training data and save scaling parameters."""
    result = df.copy()

    # Drop columns
    drop_cols = [c for c in result.columns if should_drop(c)]
    result = result.drop(columns=drop_cols)
    logger.info(f"Dropped {len(drop_cols)} columns (LONG, returns, fumbles)")

    # Per-season conference prestige
    for y in ["Y0", "Y1", "Y2", "Y3"]:
        col = f"{y}_college"
        if col in result.columns:
            result[f"{y}_conf_prestige"] = result[col].apply(college_to_prestige)
            result = result.drop(columns=[col])

    # Overall conference prestige (from current school)
    result["conference_prestige"] = result.apply(get_conf_score, axis=1)

    # Compute and save all scaling parameters
    params = {"arctan": {}, "linear": {}, "stat": {}}

    # Arctan normalization
    for col, (center, steepness) in ARCTAN_COLS.items():
        if col in result.columns:
            valid = result[col].dropna()
            if len(valid) == 0:
                continue
            xmin, xmax = float(valid.min()), float(valid.max())
            params["arctan"][col] = {"center": center, "steepness": steepness, "xmin": xmin, "xmax": xmax}
            result[col] = arctan_norm(result[col], center, steepness, xmin, xmax)
            logger.info(f"  Arctan: {col} (center={center}, s={steepness})")

    # Linear min-max for physical measurables
    for col in LINEAR_COLS:
        if col in result.columns:
            valid = result[col].dropna()
            if len(valid) == 0:
                continue
            smin, smax = float(valid.min()), float(valid.max())
            params["linear"][col] = {"min": smin, "max": smax}
            result[col] = linear_norm(result[col], smin, smax)
            logger.info(f"  Linear: {col} (min={smin}, max={smax})")

    # Linear min-max for college stats (exclude passthrough columns)
    stat_cols = [c for c in result.columns
                 if c[:3] in ["Y0_", "Y1_", "Y2_", "Y3_"] and c not in PASSTHROUGH]
    for col in stat_cols:
        valid = result[col].dropna()
        if len(valid) == 0:
            continue
        smin, smax = float(valid.min()), float(valid.max())
        params["stat"][col] = {"min": smin, "max": smax}
        result[col] = linear_norm(result[col], smin, smax)
    logger.info(f"  Linear: {len(stat_cols)} college stat columns")

    # Save params
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved normalization params to {PARAMS_PATH}")

    return result


def transform(df):
    """Normalize new data using saved scaling parameters from training."""
    with open(PARAMS_PATH) as f:
        params = json.load(f)

    result = df.copy()

    # Drop columns
    drop_cols = [c for c in result.columns if should_drop(c)]
    result = result.drop(columns=drop_cols)
    logger.info(f"Dropped {len(drop_cols)} columns (LONG, returns, fumbles)")

    # Per-season conference prestige
    for y in ["Y0", "Y1", "Y2", "Y3"]:
        col = f"{y}_college"
        if col in result.columns:
            result[f"{y}_conf_prestige"] = result[col].apply(college_to_prestige)
            result = result.drop(columns=[col])

    # Overall conference prestige (from current school)
    result["conference_prestige"] = result.apply(get_conf_score, axis=1)

    # Arctan using training params
    for col, p in params["arctan"].items():
        if col in result.columns:
            result[col] = arctan_norm(result[col], p["center"], p["steepness"], p["xmin"], p["xmax"])
            logger.info(f"  Arctan: {col}")

    # Linear using training params
    for col, p in params["linear"].items():
        if col in result.columns:
            result[col] = linear_norm(result[col], p["min"], p["max"])
            logger.info(f"  Linear: {col}")

    # Stat columns using training params
    stat_count = 0
    for col, p in params["stat"].items():
        if col in result.columns:
            result[col] = linear_norm(result[col], p["min"], p["max"])
            stat_count += 1
    logger.info(f"  Linear: {stat_count} college stat columns")

    return result


if __name__ == "__main__":
    from preprocess import preprocess

    # 1. Fit on training data (2017-2022)
    train_path = os.path.join(PROCESSED_DIR, "variables_wide_receiver.csv")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} training WRs, {len(train_df.columns)} columns")

    train_df = preprocess(train_df)
    normalized = fit_normalize(train_df)

    outpath = os.path.join(PROCESSED_DIR, "normalized_wide_receiver.csv")
    normalized.to_csv(outpath, index=False)
    logger.info(f"Saved training: {len(normalized)} rows -> {outpath}")

    # 2. Transform prediction classes (2023-2026) using same params
    for year in range(2023, 2027):
        filepath = os.path.join(PROCESSED_DIR, f"variables_class_{year}_wide_receiver.csv")
        if not os.path.exists(filepath):
            logger.info(f"No file for {year}, skipping")
            continue

        pred_df = pd.read_csv(filepath)
        pred_df["draft_year"] = year
        pred_df = preprocess(pred_df)
        pred_norm = transform(pred_df)

        pred_outpath = os.path.join(PROCESSED_DIR, f"normalized_class_{year}_wide_receiver.csv")
        pred_norm.to_csv(pred_outpath, index=False)
        logger.info(f"Saved {year}: {len(pred_norm)} WRs -> {pred_outpath}")

    print("\nDone!")
