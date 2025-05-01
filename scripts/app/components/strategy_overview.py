import streamlit as st
import os
from app_modules.agent import N06_rule_merging
import pandas as pd


def render_strategy_overview():

    st.markdown("---")
    st.subheader("F1 Strategy Analysis Overview")

    csv_path = "outputs/week6/spain_gp_recommendations.csv"

    # Button to run the analysis
    if st.button("Run Strategy Analysis for All Drivers"):
        with st.spinner("Running full strategy analysis..."):
            try:
                N06_rule_merging.run_all_drivers_analysis()
                st.session_state["strategy_csv_ready"] = True
                st.success(
                    "Strategy analysis completed! Recommendations CSV generated.")
            except Exception as e:
                st.session_state["strategy_csv_ready"] = False
                st.error(f"Error running strategy analysis: {e}")

    # Check if CSV is ready
    if st.session_state.get("strategy_csv_ready", False) or os.path.exists(csv_path):
        st.info("Strategy recommendations are ready and available for all components.")
        # Show action summary table
        try:
            df = pd.read_csv(csv_path)
            if "action" in df.columns:
                action_counts = df['action'].value_counts().reset_index()
                action_counts.columns = ["Action", "Count"]
                st.markdown("#### Action Summary")
                st.table(action_counts)
        except Exception as e:
            st.warning(f"Could not load recommendations CSV: {e}")

    else:
        st.warning(
            "No strategy recommendations available yet. Please run the analysis.")
