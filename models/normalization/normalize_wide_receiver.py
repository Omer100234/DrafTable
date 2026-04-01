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
"""

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Columns to leave untouched
PASSTHROUGH = [
    "name", "position", "college", "conference", "draft_year",
    "years_since_first_played",
    "played_at_18", "played_at_19", "played_at_20",
    "played_at_21", "played_at_22", "played_at_23",
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


def arctan_norm(series: pd.Series, center: float, steepness: float) -> pd.Series:
    """Arctan normalization for lower-is-better metrics."""
    valid = series.dropna()
    if len(valid) == 0:
        return series
    xmin, xmax = valid.min(), valid.max()
    raw = np.arctan(-steepness * (series - center))
    raw_min = np.arctan(-steepness * (xmax - center))
    raw_max = np.arctan(-steepness * (xmin - center))
    return (raw - raw_min) / (raw_max - raw_min)


def linear_norm(series: pd.Series) -> pd.Series:
    """Linear min-max normalization to 0-1."""
    valid = series.dropna()
    if len(valid) == 0:
        return series
    smin, smax = valid.min(), valid.max()
    if smax == smin:
        return series * 0
    return (series - smin) / (smax - smin)


def should_drop(col: str) -> bool:
    """Check if a column should be dropped."""
    return any(pattern in col for pattern in DROP_PATTERNS)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize WR variables."""
    result = df.copy()

    # Drop columns with LONG, returns, fumbles
    drop_cols = [c for c in result.columns if should_drop(c)]
    result = result.drop(columns=drop_cols)
    logger.info(f"Dropped {len(drop_cols)} columns (LONG, returns, fumbles)")

    # Conference prestige score
    def get_conf_score(row):
        college = row["college"]
        conference = row["conference"]
        # Check if this college has an override (independents)
        if college in COLLEGE_CONFERENCE_OVERRIDES:
            conference = COLLEGE_CONFERENCE_OVERRIDES[college]
        return CONFERENCE_SCORES.get(conference, DEFAULT_CONFERENCE_SCORE)

    result["conference_prestige"] = result.apply(get_conf_score, axis=1)
    logger.info(f"  Conference prestige mapped")

    # Arctan normalization for speed/agility
    for col, (center, steepness) in ARCTAN_COLS.items():
        if col in result.columns:
            result[col] = arctan_norm(result[col], center, steepness)
            logger.info(f"  Arctan: {col} (center={center}, s={steepness})")

    # Linear min-max for physical measurables
    for col in LINEAR_COLS:
        if col in result.columns:
            result[col] = linear_norm(result[col])
            logger.info(f"  Linear: {col}")

    # Linear min-max for college stats (everything Y0-Y3 that's left)
    stat_cols = [c for c in result.columns if c[:3] in ["Y0_", "Y1_", "Y2_", "Y3_"]]
    for col in stat_cols:
        result[col] = linear_norm(result[col])
    logger.info(f"  Linear: {len(stat_cols)} college stat columns")

    return result


if __name__ == "__main__":
    from preprocess import preprocess

    filepath = os.path.join(PROCESSED_DIR, "variables_wide_receiver.csv")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} wide receivers, {len(df.columns)} columns")

    df = preprocess(df)
    normalized = normalize(df)
    logger.info(f"Output: {len(normalized)} rows, {len(normalized.columns)} columns")

    outpath = os.path.join(PROCESSED_DIR, "normalized_wide_receiver.csv")
    normalized.to_csv(outpath, index=False)
    logger.info(f"Saved to {outpath}")

    # Sanity check
    print("\nSample:")
    sample_cols = ["name", "draft_year", "height", "weight", "forty", "Y0_receiving_REC", "Y0_receiving_YDS", "Y0_receiving_TD"]
    print(normalized[sample_cols].dropna().head(10).to_string(index=False))
