# components/team_radio_view.py

import streamlit as st
import pandas as pd


def render_radio_analysis(recommendations):
    """
    Renders the team radio analysis view.

    Parameters:
        recommendations (pd.DataFrame): Strategy recommendations with radio data
    """
    st.header("Team Radio Analysis")
    st.write("This section analyzes team radio communications for strategic insights.")

    # Placeholder for radio analysis
    if recommendations is None or recommendations.empty:
        st.warning("No radio analysis data available for the selected driver.")
        return

    # Display any radio-related recommendations
    radio_recs = None
    try:
        if 'action' in recommendations.columns and 'explanation' in recommendations.columns:
            radio_recs = recommendations[recommendations['action'].isin(
                ['prepare_rain_tires', 'reevaluate_pit_window', 'prioritize_pit']
            )]
    except Exception as e:
        st.error(f"Error processing radio recommendations: {e}")
        return

    if radio_recs is not None and not radio_recs.empty:
        st.subheader("Radio-Based Strategic Recommendations")
        for i, rec in radio_recs.iterrows():
            with st.expander(f"{rec['action']} (Lap {rec.get('LapNumber', 'N/A')})"):
                st.write(f"**Explanation:** {rec['explanation']}")
                st.write(f"**Confidence:** {rec.get('confidence', 'N/A')}")
    else:
        st.info("No radio-based recommendations found for this driver.")
