"""
Normalize running back variables for ML.

Same strategy as WR normalizer:
- Arctan (s=8): forty (center=4.45), cone (center=6.95), shuttle (center=4.28)
- Linear min-max: height, weight, age, bench, vertical, broad_jump
- Linear min-max: college stats (rushing + receiving)
- Drop: LONG, returns, fumbles
- Conference prestige tiers

Uses fit_normalize/transform pattern with saved params for consistent scaling.
"""

import pandas as pd
import numpy as np
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "norm_params_running_back.json")

# Import shared normalization utilities from WR module
from normalize_wide_receiver import (
    PASSTHROUGH, ARCTAN_COLS, LINEAR_COLS,
    CONFERENCE_SCORES, COLLEGE_CONFERENCE_OVERRIDES, DEFAULT_CONFERENCE_SCORE,
    arctan_norm, linear_norm, get_conf_score, college_to_prestige,
)

# RB-specific drop patterns (same as WR — drop LONG, returns, fumbles)
DROP_PATTERNS = ["_LONG", "kickReturns_", "puntReturns_", "fumbles_"]


def should_drop(col):
    return any(pattern in col for pattern in DROP_PATTERNS)


def fit_normalize(df):
    """Normalize training data and save scaling parameters."""
    result = df.copy()

    drop_cols = [c for c in result.columns if should_drop(c)]
    result = result.drop(columns=drop_cols)
    logger.info(f"Dropped {len(drop_cols)} columns (LONG, returns, fumbles)")

    # Per-season conference prestige
    for y in ["Y0", "Y1", "Y2", "Y3"]:
        col = f"{y}_college"
        if col in result.columns:
            result[f"{y}_conf_prestige"] = result[col].apply(college_to_prestige)
            result = result.drop(columns=[col])

    result["conference_prestige"] = result.apply(get_conf_score, axis=1)

    params = {"arctan": {}, "linear": {}, "stat": {}}

    for col, (center, steepness) in ARCTAN_COLS.items():
        if col in result.columns:
            valid = result[col].dropna()
            if len(valid) == 0:
                continue
            xmin, xmax = float(valid.min()), float(valid.max())
            params["arctan"][col] = {"center": center, "steepness": steepness, "xmin": xmin, "xmax": xmax}
            result[col] = arctan_norm(result[col], center, steepness, xmin, xmax)
            logger.info(f"  Arctan: {col}")

    for col in LINEAR_COLS:
        if col in result.columns:
            valid = result[col].dropna()
            if len(valid) == 0:
                continue
            smin, smax = float(valid.min()), float(valid.max())
            params["linear"][col] = {"min": smin, "max": smax}
            result[col] = linear_norm(result[col], smin, smax)
            logger.info(f"  Linear: {col}")

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

    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved normalization params to {PARAMS_PATH}")

    return result


def transform(df):
    """Normalize new data using saved scaling parameters from training."""
    with open(PARAMS_PATH) as f:
        params = json.load(f)

    result = df.copy()

    drop_cols = [c for c in result.columns if should_drop(c)]
    result = result.drop(columns=drop_cols)
    logger.info(f"Dropped {len(drop_cols)} columns (LONG, returns, fumbles)")

    # Per-season conference prestige
    for y in ["Y0", "Y1", "Y2", "Y3"]:
        col = f"{y}_college"
        if col in result.columns:
            result[f"{y}_conf_prestige"] = result[col].apply(college_to_prestige)
            result = result.drop(columns=[col])

    result["conference_prestige"] = result.apply(get_conf_score, axis=1)

    for col, p in params["arctan"].items():
        if col in result.columns:
            result[col] = arctan_norm(result[col], p["center"], p["steepness"], p["xmin"], p["xmax"])
            logger.info(f"  Arctan: {col}")

    for col, p in params["linear"].items():
        if col in result.columns:
            result[col] = linear_norm(result[col], p["min"], p["max"])
            logger.info(f"  Linear: {col}")

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
    train_path = os.path.join(PROCESSED_DIR, "variables_running_back.csv")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} training RBs, {len(train_df.columns)} columns")

    train_df = preprocess(train_df)
    normalized = fit_normalize(train_df)

    outpath = os.path.join(PROCESSED_DIR, "normalized_running_back.csv")
    normalized.to_csv(outpath, index=False)
    logger.info(f"Saved training: {len(normalized)} rows -> {outpath}")

    # 2. Transform prediction classes (2023-2026) using same params
    for year in range(2023, 2027):
        filepath = os.path.join(PROCESSED_DIR, f"variables_class_{year}_running_back.csv")
        if not os.path.exists(filepath):
            logger.info(f"No file for {year}, skipping")
            continue

        pred_df = pd.read_csv(filepath)
        pred_df["draft_year"] = year
        pred_df = preprocess(pred_df)
        pred_norm = transform(pred_df)

        pred_outpath = os.path.join(PROCESSED_DIR, f"normalized_class_{year}_running_back.csv")
        pred_norm.to_csv(pred_outpath, index=False)
        logger.info(f"Saved {year}: {len(pred_norm)} RBs -> {pred_outpath}")

    print("\nDone!")
