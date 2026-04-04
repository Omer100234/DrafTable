
import streamlit as st
import pandas as pd
import glob
import os
import json

st.set_page_config(page_title="DrafTables", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

page = st.sidebar.radio("View", ["Eligible Players", "Past Drafts", "Prospect Stats", "NFL AV Leaders", "Second Contracts", "Draft Class Variables", "Training Data", "Normalized Data", "Success Scores", "Predictions"])

if page == "Eligible Players":
    st.title("2026 Eligible Players")

    df = pd.read_csv(os.path.join(DATA_DIR, "eligible_players.csv"))

    col1, col2, col3 = st.columns(3)
    with col1:
        positions = ["All"] + sorted(df["position"].dropna().unique().tolist())
        selected_pos = st.selectbox("Position", positions)
    with col2:
        colleges = ["All"] + sorted(df["college"].dropna().unique().tolist())
        selected_college = st.selectbox("College", colleges)
    with col3:
        classes = ["All"] + sorted(df["experience"].dropna().unique().tolist())
        selected_class = st.selectbox("Class", classes)

    filtered = df.copy()
    if selected_pos != "All":
        filtered = filtered[filtered["position"] == selected_pos]
    if selected_college != "All":
        filtered = filtered[filtered["college"] == selected_college]
    if selected_class != "All":
        filtered = filtered[filtered["experience"] == selected_class]

    st.subheader(f"Showing {len(filtered)} players")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

elif page == "Past Drafts":
    st.title("Past NFL Drafts")

    draft_files = sorted(glob.glob(os.path.join(DATA_DIR, "draftees_*.csv")))
    years = [int(f.split("draftees_")[1].split(".csv")[0]) for f in draft_files]

    selected_year = st.selectbox("Draft Year", sorted(years, reverse=True))
    df = pd.read_csv(os.path.join(DATA_DIR, f"draftees_{selected_year}.csv"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rounds = ["All"] + sorted(df["round"].dropna().unique().tolist())
        selected_round = st.selectbox("Round", rounds)
    with col2:
        positions = ["All"] + sorted(df["position"].dropna().unique().tolist())
        selected_pos = st.selectbox("Position", positions)
    with col3:
        nfl_teams = ["All"] + sorted(df["nflTeam"].dropna().unique().tolist())
        selected_nfl = st.selectbox("NFL Team", nfl_teams)
    with col4:
        conferences = ["All"] + sorted(df["collegeConference"].dropna().unique().tolist())
        selected_conf = st.selectbox("Conference", conferences)

    filtered = df.copy()
    if selected_round != "All":
        filtered = filtered[filtered["round"] == selected_round]
    if selected_pos != "All":
        filtered = filtered[filtered["position"] == selected_pos]
    if selected_nfl != "All":
        filtered = filtered[filtered["nflTeam"] == selected_nfl]
    if selected_conf != "All":
        filtered = filtered[filtered["collegeConference"] == selected_conf]

    hidden_cols = ["collegeAthleteId", "nflAthleteId", "collegeId", "nflTeamId"]
    display_df = filtered.drop(columns=[c for c in hidden_cols if c in filtered.columns])

    st.subheader(f"{selected_year} NFL Draft — {len(filtered)} picks")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif page == "Prospect Stats":
    st.title("2026 Prospect Stats")

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "prospects_clean_stats.csv"))

    col1, col2 = st.columns(2)
    with col1:
        positions = ["All"] + sorted(df["position"].dropna().unique().tolist())
        selected_pos = st.selectbox("Position", positions)
    with col2:
        search = st.text_input("Search by name")

    filtered = df.copy()
    if selected_pos != "All":
        filtered = filtered[filtered["position"] == selected_pos]
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]

    st.subheader(f"{len(filtered)} prospects")

    selected_player = st.selectbox("Select a prospect", filtered["name"].tolist())

    if selected_player:
        player = df[df["name"] == selected_player].iloc[0]
        st.markdown(f"**{player['name']}** — {player['position']} — {player['current_college']} — {player['class']}")

        # Determine which season labels this player has
        class_labels = {
            "FR": "Freshman", "SO": "Sophomore", "JR": "Junior", "SR": "Senior",
            "RSO": "Redshirt Sophomore", "RJR": "Redshirt Junior", "RSR": "Redshirt Senior", "GR": "Graduate",
        }
        # Find all season labels present for this player
        season_prefixes = []
        for label in class_labels:
            if f"{label}_season" in player.index and pd.notna(player.get(f"{label}_season")):
                season_prefixes.append(label)

        for label in season_prefixes:
            season_year = int(player[f"{label}_season"])
            college = player.get(f"{label}_college", "Unknown")
            st.subheader(f"{class_labels.get(label, label)} ({season_year}) — {college}")

            # Collect stats for this season label
            prefix = f"{label}_"
            stat_cols = [c for c in df.columns if c.startswith(prefix)
                         and c not in [f"{label}_college", f"{label}_season"]]
            stats = {}
            for col in stat_cols:
                val = player[col]
                if pd.notna(val):
                    stat_name = col[len(prefix):]
                    category, stat_type = stat_name.split("_", 1) if "_" in stat_name else (stat_name, "")
                    if category not in stats:
                        stats[category] = {}
                    stats[category][stat_type] = val

            if stats:
                for category, stat_dict in stats.items():
                    st.markdown(f"**{category.title()}**")
                    st.dataframe(
                        pd.DataFrame([stat_dict]),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.write("No stats recorded for this season.")

elif page == "NFL AV Leaders":
    st.title("NFL AV Per Game Leaders by Draft Class")

    df = pd.read_csv(os.path.join(DATA_DIR, "draftees_nfl_career_stats.csv"))

    # Filter to players with at least 5 games and valid AV
    df = df[df["games"] >= 5].copy()
    df = df[df["w_av"].notna() & df["games"].notna()]
    df["av_per_game"] = (df["w_av"] / df["games"]).round(3)

    years = sorted(df["draft_year"].unique().tolist(), reverse=True)
    selected_year = st.selectbox("Draft Class", ["All"] + years)

    filtered = df.copy()
    if selected_year != "All":
        filtered = filtered[filtered["draft_year"] == selected_year]

    filtered = filtered.sort_values("av_per_game", ascending=False)



    st.subheader(f"{len(display_df)} players ranked by AV/Game (min 5 games)")
    st.dataframe(display_df, use_container_width=True)

elif page == "Second Contracts":
    st.title("Second Contract Value by Draft Class")

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "second_contracts.csv"))

    col1, col2, col3 = st.columns(3)
    with col1:
        years = sorted(df["draft_year"].dropna().unique().tolist(), reverse=True)
        selected_year = st.selectbox("Draft Class", ["All"] + [int(y) for y in years])
    with col2:
        positions = ["All"] + sorted(df["position"].dropna().unique().tolist())
        selected_pos = st.selectbox("Position", positions)
    with col3:
        market_filter = st.selectbox("Market Reset", ["All", "Yes", "No"])

    filtered = df.copy()
    if selected_year != "All":
        filtered = filtered[filtered["draft_year"] == selected_year]
    if selected_pos != "All":
        filtered = filtered[filtered["position"] == selected_pos]
    if market_filter == "Yes":
        filtered = filtered[filtered["market_reset"] == True]
    elif market_filter == "No":
        filtered = filtered[filtered["market_reset"] == False]

    filtered = filtered.sort_values("pct_of_top", ascending=False)

    display_cols = [
        "player", "position", "draft_year", "draft_round", "draft_overall",
        "college", "team", "year_signed", "apy", "pos_top_apy",
        "pct_of_top", "market_reset",
    ]
    display_df = filtered[display_cols].reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df["pct_of_top"] = (display_df["pct_of_top"] * 100).round(1).astype(str) + "%"

    st.subheader(f"{len(display_df)} players with second contracts")
    st.dataframe(display_df, use_container_width=True)

elif page == "Draft Class Variables":
    st.title("Draft Class Variables by Position")

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("Draft Class", list(range(2025, 2016, -1)))
    with col2:
        # Find available position files for this year
        var_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, f"variables_class_{selected_year}_*.csv")))
        positions = []
        for f in var_files:
            pos = os.path.basename(f).replace(f"variables_class_{selected_year}_", "").replace(".csv", "").replace("_", " ").title()
            positions.append(pos)
        selected_pos = st.selectbox("Position", positions)

    if selected_pos:
        pos_label = selected_pos.lower().replace(" ", "_")
        filepath = os.path.join(PROCESSED_DIR, f"variables_class_{selected_year}_{pos_label}.csv")
        df = pd.read_csv(filepath)

        search = st.text_input("Search by name")
        if search:
            df = df[df["name"].str.contains(search, case=False, na=False)]

        st.subheader(f"{selected_year} {selected_pos} — {len(df)} players, {len(df.columns)} variables")
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Training Data":
    st.title("Training Data by Position (2017-2022)")

    var_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "variables_[!c]*.csv")))
    positions = []
    for f in var_files:
        pos = os.path.basename(f).replace("variables_", "").replace(".csv", "").replace("_", " ").title()
        positions.append(pos)

    selected_pos = st.selectbox("Position", positions)

    if selected_pos:
        pos_label = selected_pos.lower().replace(" ", "_")
        filepath = os.path.join(PROCESSED_DIR, f"variables_{pos_label}.csv")
        df = pd.read_csv(filepath)

        col1, col2 = st.columns(2)
        with col1:
            years = ["All"] + sorted(df["draft_year"].unique().tolist(), reverse=True)
            selected_year = st.selectbox("Draft Class", years)
        with col2:
            search = st.text_input("Search by name")

        if selected_year != "All":
            df = df[df["draft_year"] == selected_year]
        if search:
            df = df[df["name"].str.contains(search, case=False, na=False)]

        st.subheader(f"{selected_pos} — {len(df)} players, {len(df.columns)} variables")
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Normalized Data":
    st.title("Normalized Training Data")

    norm_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "normalized_*.csv")))
    positions = []
    for f in norm_files:
        pos = os.path.basename(f).replace("normalized_", "").replace(".csv", "").replace("_", " ").title()
        positions.append(pos)

    if not positions:
        st.warning("No normalized data files found. Run the normalizers first.")
    else:
        selected_pos = st.selectbox("Position", positions)

        if selected_pos:
            pos_label = selected_pos.lower().replace(" ", "_")
            filepath = os.path.join(PROCESSED_DIR, f"normalized_{pos_label}.csv")
            df = pd.read_csv(filepath)

            col1, col2 = st.columns(2)
            with col1:
                years = ["All"] + sorted(df["draft_year"].unique().tolist(), reverse=True)
                selected_year = st.selectbox("Draft Class", years)
            with col2:
                search = st.text_input("Search by name")

            if selected_year != "All":
                df = df[df["draft_year"] == selected_year]
            if search:
                df = df[df["name"].str.contains(search, case=False, na=False)]

            st.subheader(f"{selected_pos} — {len(df)} players, {len(df.columns)} columns (normalized)")
            st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Success Scores":
    st.title("NFL Success Scores (2017-2022 Draftees)")
    st.markdown("**Composite:** 70% AV/game + 30% second contract value, arctan transformed (s=5)")

    success_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "success_*.csv")))
    positions = []
    for f in success_files:
        pos = os.path.basename(f).replace("success_", "").replace(".csv", "").replace("_", " ").title()
        positions.append(pos)

    if not positions:
        st.warning("No success score files found. Run build_success_score.py first.")
    else:
        selected_pos = st.selectbox("Position", positions)

        if selected_pos:
            pos_label = selected_pos.lower().replace(" ", "_")
            filepath = os.path.join(PROCESSED_DIR, f"success_{pos_label}.csv")
            df = pd.read_csv(filepath)

            col1, col2 = st.columns(2)
            with col1:
                years = ["All"] + sorted(df["draft_year"].unique().tolist(), reverse=True)
                selected_year = st.selectbox("Draft Class", years)
            with col2:
                search = st.text_input("Search by name")

            if selected_year != "All":
                df = df[df["draft_year"] == selected_year]
            if search:
                df = df[df["name"].str.contains(search, case=False, na=False)]

            df = df.sort_values("success_score", ascending=False).reset_index(drop=True)
            df.index = df.index + 1
            df["success_score"] = df["success_score"].round(3)
            df["av_per_game"] = df["av_per_game"].round(3)

            st.subheader(f"{selected_pos} — {len(df)} players ranked by success score")
            st.dataframe(df, use_container_width=True)

elif page == "Predictions":
    st.title("ML Predictions by Draft Class")

    import pickle
    import numpy as np
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from train_model import SmartImputer  # needed for unpickling

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "trained")

    # Find available models
    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "ensemble_*.pkl")))
    positions = []
    for f in model_files:
        pos = os.path.basename(f).replace("ensemble_", "").replace(".pkl", "").replace("_", " ").title()
        positions.append(pos)

    if not positions:
        st.warning("No trained models found. Run training first.")
    else:
        selected_pos = st.selectbox("Position", positions)

        if selected_pos:
            pos_label = selected_pos.lower().replace(" ", "_")

            # Load model
            with open(os.path.join(MODEL_DIR, f"ensemble_{pos_label}.pkl"), "rb") as f:
                data = pickle.load(f)
            models = data["models"]
            feature_cols = data["feature_cols"]

            # Show model stats
            meta_path = os.path.join(MODEL_DIR, f"ensemble_{pos_label}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as mf:
                    meta = json.load(mf)
                cols = st.columns(4)
                cols[0].metric("MAE", f"{meta['cv_mae']:.4f}")
                cols[1].metric("Correlation", f"{meta['cv_correlation']:.4f}")
                cols[2].metric("Spearman", f"{meta['cv_spearman']:.4f}")
                cols[3].metric("Training Size", meta['n_training'])

            # Find normalized class files for this position
            norm_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, f"normalized_class_*_{pos_label}.csv")))
            years = [int(os.path.basename(f).split("_")[2]) for f in norm_files]

            if not years:
                st.warning("No normalized prediction data found. Run the normalizer first.")
            else:
                selected_year = st.selectbox("Draft Class", sorted(years, reverse=True))

                df = pd.read_csv(os.path.join(PROCESSED_DIR, f"normalized_class_{selected_year}_{pos_label}.csv"))

                # Ensure all feature columns exist
                for col in feature_cols:
                    if col not in df.columns:
                        df[col] = np.nan

                # Predict
                preds = [m.predict(df[feature_cols]) for _, m in models]
                df["predicted_score"] = np.clip(np.mean(preds, axis=0), 0, 1).round(3)

                col1, col2 = st.columns(2)
                with col1:
                    search = st.text_input("Search by name")
                with col2:
                    pass

                if search:
                    df = df[df["name"].str.contains(search, case=False, na=False)]

                df = df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
                df.index = df.index + 1

                display_cols = ["name", "college", "predicted_score"]
                extra = ["height", "weight", "forty", "conference_prestige"]
                display_cols += [c for c in extra if c in df.columns]

                st.subheader(f"{selected_year} {selected_pos} — {len(df)} prospects ranked by predicted score")
                st.dataframe(df[display_cols], use_container_width=True)
