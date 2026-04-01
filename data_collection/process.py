"""
Prospect Stats Processor
Matches 2026 draft prospects to their college stats across seasons.
Labels each season with the class year (FR, SO, JR, SR, etc.).
Tracks which college they played for each year.
"""

import pandas as pd
import os
import json

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
STATS_DIR = os.path.join(RAW_DIR, "college_stats")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# Class -> list of (season_year, label) going backwards from 2025
# 2025 is the most recent season for 2026 draft prospects
CLASS_SEASON_MAP = {
    "FR":  [(2025, "FR")],
    "SO":  [(2025, "SO"), (2024, "FR")],
    "RSO": [(2025, "RSO"), (2024, "FR")],
    "JR":  [(2025, "JR"), (2024, "SO"), (2023, "FR")],
    "RJR": [(2025, "RJR"), (2024, "SO"), (2023, "FR")],
    "SR":  [(2025, "SR"), (2024, "JR"), (2023, "SO"), (2022, "FR")],
    "RSR": [(2025, "RSR"), (2024, "JR"), (2023, "SO"), (2022, "FR")],
    "GR":  [(2025, "GR"), (2024, "SR"), (2023, "JR"), (2022, "SO"), (2021, "FR")],
}


def load_all_stats():
    """Load all college stats CSVs into a single DataFrame."""
    frames = []
    for f in os.listdir(STATS_DIR):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(STATS_DIR, f))
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def find_player_stats(name, season, all_stats):
    """Find all stat rows for a player name in a given season."""
    mask = (all_stats["player"] == name) & (all_stats["season"] == season)
    return all_stats[mask]


def main():
    prospects = pd.read_csv(os.path.join(RAW_DIR, "prospects_2026.csv"))
    all_stats = load_all_stats()

    results = []
    ambiguous = []

    for _, prospect in prospects.iterrows():
        name = prospect["name"]
        player_class = prospect["class"]
        seasons = CLASS_SEASON_MAP.get(player_class, [])

        for season_year, class_label in seasons:
            matches = find_player_stats(name, season_year, all_stats)

            if matches.empty:
                continue

            # Check for ambiguity: multiple different playerIds for the same name+season
            unique_ids = matches["playerId"].unique()
            if len(unique_ids) > 1:
                ambiguous.append({
                    "prospect_name": name,
                    "prospect_college": prospect["college"],
                    "season": season_year,
                    "matched_players": [
                        {"playerId": int(pid), "team": matches[matches["playerId"] == pid]["team"].iloc[0]}
                        for pid in unique_ids
                    ],
                })
                continue

            team = matches["team"].iloc[0]

            # Pivot stat rows into columns: category_statType -> value
            for _, row in matches.iterrows():
                stat_key = f"{class_label}_{row['category']}_{row['statType']}"
                # Find or create the result row for this prospect
                existing = next((r for r in results if r["name"] == name and r["rank"] == prospect["rank"]), None)
                if existing is None:
                    existing = {
                        "rank": prospect["rank"],
                        "name": name,
                        "position": prospect["position"],
                        "current_college": prospect["college"],
                        "class": player_class,
                    }
                    results.append(existing)

                existing[stat_key] = row["stat"]
                existing[f"{class_label}_college"] = team
                existing[f"{class_label}_season"] = season_year

    # Save results
    results_df = pd.DataFrame(results)
    results_df.sort_values("rank", inplace=True)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "prospects_stats.csv"), index=False)
    print(f"Saved {len(results_df)} prospects with stats to prospects_stats.csv")

    # Save ambiguous names
    if ambiguous:
        with open(os.path.join(OUTPUT_DIR, "ambiguous_names.json"), "w") as f:
            json.dump(ambiguous, f, indent=2)
        print(f"Found {len(ambiguous)} ambiguous matches - saved to ambiguous_names.json")
    else:
        print("No ambiguous names found")


if __name__ == "__main__":
    main()
