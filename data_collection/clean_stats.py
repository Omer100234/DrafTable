"""
Clean Prospect Stats
Removes irrelevant stat categories based on position group.
"""

import pandas as pd
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Position group -> set of stat categories to KEEP
POSITION_STATS = {
    "QB":   {"passing", "rushing", "fumbles"},
    "RB":   {"rushing", "receiving", "fumbles", "kickReturns", "puntReturns"},
    "FB":   {"rushing", "receiving", "fumbles", "kickReturns", "puntReturns"},
    "WR":   {"receiving", "rushing", "fumbles", "kickReturns", "puntReturns"},
    "WRS":  {"receiving", "rushing", "fumbles", "kickReturns", "puntReturns"},
    "TE":   {"receiving", "rushing", "fumbles", "kickReturns", "puntReturns"},
    "OT":   {"fumbles"},
    "OG":   {"fumbles"},
    "OC":   {"fumbles"},
    "EDGE": {"defensive", "fumbles", "interceptions"},
    "DL1T": {"defensive", "fumbles", "interceptions"},
    "DL3T": {"defensive", "fumbles", "interceptions"},
    "DL5T": {"defensive", "fumbles", "interceptions"},
    "ILB":  {"defensive", "fumbles", "interceptions"},
    "OLB":  {"defensive", "fumbles", "interceptions"},
    "CB":   {"defensive", "fumbles", "interceptions", "kickReturns", "puntReturns"},
    "CBN":  {"defensive", "fumbles", "interceptions", "kickReturns", "puntReturns"},
    "S":    {"defensive", "fumbles", "interceptions", "kickReturns", "puntReturns"},
    "PK":   {"kicking"},
    "P":    {"punting"},
    "LS":   {"fumbles"},
}

SEASON_LABELS = ["FR", "SO", "JR", "SR", "RSO", "RJR", "RSR", "GR"]
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

        # For each row, if all stat cols are null, also null out the meta
        for idx in df.index:
            if stat_cols and df.loc[idx, stat_cols].isna().all():
                for mc in existing_meta:
                    df.loc[idx, mc] = None
    return df


def main():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "prospects_stats.csv"))
    print(f"Loaded {len(df)} prospects, {len(df.columns)} columns")

    df = df.apply(clean_row, axis=1)
    df = drop_empty_season_meta(df)

    # Drop columns that are entirely empty
    before = len(df.columns)
    df = df.dropna(axis=1, how="all")
    print(f"Dropped {before - len(df.columns)} fully empty columns, {len(df.columns)} remaining")

    df.to_csv(os.path.join(PROCESSED_DIR, "prospects_clean_stats.csv"), index=False)
    print(f"Saved to prospects_clean_stats.csv")


if __name__ == "__main__":
    main()
