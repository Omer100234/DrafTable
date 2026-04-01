"""
Clean Draftee Stats
Removes irrelevant stat categories based on position group.
Draftees use full position names, so we map them to stat categories.
"""

import pandas as pd
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Full position name -> set of stat categories to KEEP
POSITION_STATS = {
    "Quarterback":      {"passing", "rushing", "fumbles"},
    "Running Back":     {"rushing", "receiving", "fumbles", "kickReturns", "puntReturns"},
    "Fullback":         {"rushing", "receiving", "fumbles", "kickReturns", "puntReturns"},
    "Wide Receiver":    {"receiving", "rushing", "fumbles", "kickReturns", "puntReturns"},
    "Tight End":        {"receiving", "rushing", "fumbles", "kickReturns", "puntReturns"},
    "Offensive Tackle": {"fumbles"},
    "Offensive Guard":  {"fumbles"},
    "Center":           {"fumbles"},
    "Defensive End":    {"defensive", "fumbles", "interceptions"},
    "Defensive Edge":   {"defensive", "fumbles", "interceptions"},
    "Defensive Tackle": {"defensive", "fumbles", "interceptions"},
    "Inside Linebacker": {"defensive", "fumbles", "interceptions"},
    "Outside Linebacker": {"defensive", "fumbles", "interceptions"},
    "Linebacker":       {"defensive", "fumbles", "interceptions"},
    "Cornerback":       {"defensive", "fumbles", "interceptions", "kickReturns", "puntReturns"},
    "Safety":           {"defensive", "fumbles", "interceptions", "kickReturns", "puntReturns"},
    "Place Kicker":     {"kicking"},
    "Punter":           {"punting"},
    "Long Snapper":     {"fumbles"},
}

SEASON_LABELS = ["Y0", "Y1", "Y2", "Y3"]
META_SUFFIXES = {"college", "season"}


def get_stat_category(col):
    """Extract (season_label, category) from a column name, or None."""
    for label in SEASON_LABELS:
        prefix = f"{label}_"
        if col.startswith(prefix):
            remainder = col[len(prefix):]
            if remainder in META_SUFFIXES:
                return label, "__meta__"
            cat = remainder.split("_")[0]
            return label, cat
    return None, None


def clean_row(row):
    """Null out stats that are irrelevant for this player's position."""
    position = row["position"]
    keep_cats = POSITION_STATS.get(position, set())

    for col in row.index:
        label, cat = get_stat_category(col)
        if cat is None or cat == "__meta__":
            continue
        if cat not in keep_cats:
            row[col] = None
    return row


def drop_empty_season_meta(df):
    """Remove season/college columns for seasons where all stats were cleared."""
    for label in SEASON_LABELS:
        prefix = f"{label}_"
        stat_cols = [c for c in df.columns if c.startswith(prefix)
                     and c not in [f"{label}_college", f"{label}_season"]]
        meta_cols = [f"{label}_college", f"{label}_season"]
        existing_meta = [c for c in meta_cols if c in df.columns]

        for idx in df.index:
            if stat_cols and df.loc[idx, stat_cols].isna().all():
                for mc in existing_meta:
                    df.loc[idx, mc] = None
    return df


def main():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "draftees_stats.csv"))
    print(f"Loaded {len(df)} draftees, {len(df.columns)} columns")

    df = df.apply(clean_row, axis=1)
    df = drop_empty_season_meta(df)

    before = len(df.columns)
    df = df.dropna(axis=1, how="all")
    print(f"Dropped {before - len(df.columns)} fully empty columns, {len(df.columns)} remaining")

    df.to_csv(os.path.join(PROCESSED_DIR, "draftees_clean_stats.csv"), index=False)
    print(f"Saved to draftees_clean_stats.csv")


if __name__ == "__main__":
    main()
