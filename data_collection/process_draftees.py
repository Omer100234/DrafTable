"""
Draftee Stats Processor
Matches draftees (2017-2025) to their college stats using collegeAthleteId.
Labels seasons as Y0 (draft_year - 1), Y1 (draft_year - 2), Y2, Y3.
Y0 is always the season before the draft, even if the player opted out.
Gaps from injuries/opt-outs show as empty stats for that year.
Tracks which college they played for each year.
"""

import pandas as pd
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
STATS_DIR = os.path.join(RAW_DIR, "college_stats")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

SEASONS_BACK = 4


def load_all_stats():
    """Load all college stats CSVs into a single DataFrame."""
    frames = []
    for f in os.listdir(STATS_DIR):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(STATS_DIR, f))
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    all_stats = load_all_stats()
    print(f"Loaded {len(all_stats)} total stat rows")

    stats_by_player = all_stats.groupby("playerId")

    results = []
    missing_count = 0

    for draft_year in range(2017, 2026):
        df = pd.read_csv(os.path.join(RAW_DIR, f"draftees_{draft_year}.csv"))
        print(f"Processing {draft_year} draft: {len(df)} picks")

        for _, draftee in df.iterrows():
            pid = draftee.get("collegeAthleteId")
            if pd.isna(pid):
                missing_count += 1
                continue
            pid = int(pid)

            if pid not in stats_by_player.groups:
                missing_count += 1
                continue

            player_stats = stats_by_player.get_group(pid)
            row_data = {
                "draft_year": draft_year,
                "overall_pick": draftee["overall"],
                "round": draftee["round"],
                "pick": draftee["pick"],
                "name": draftee["name"],
                "position": draftee["position"],
                "college": draftee["collegeTeam"],
                "conference": draftee.get("collegeConference"),
                "nfl_team": draftee["nflTeam"],
                "height": draftee.get("height"),
                "weight": draftee.get("weight"),
                "pre_draft_ranking": draftee.get("preDraftRanking"),
                "pre_draft_grade": draftee.get("preDraftGrade"),
            }

            for i in range(SEASONS_BACK):
                season = (draft_year - 1) - i
                label = f"Y{i}"
                season_stats = player_stats[player_stats["season"] == season]

                if season_stats.empty:
                    continue

                team = season_stats["team"].iloc[0]
                row_data[f"{label}_college"] = team
                row_data[f"{label}_season"] = season

                for _, stat_row in season_stats.iterrows():
                    stat_key = f"{label}_{stat_row['category']}_{stat_row['statType']}"
                    row_data[stat_key] = stat_row["stat"]

            results.append(row_data)

    results_df = pd.DataFrame(results)
    results_df.sort_values(["draft_year", "overall_pick"], inplace=True)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "draftees_stats.csv"), index=False)
    print(f"\nSaved {len(results_df)} draftees with stats to draftees_stats.csv")
    print(f"{missing_count} draftees had no matching stats")


if __name__ == "__main__":
    main()
