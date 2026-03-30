import streamlit as st
import pandas as pd
import glob
import os

st.set_page_config(page_title="DrafTables", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

page = st.sidebar.radio("View", ["Eligible Players", "Past Drafts"])

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
