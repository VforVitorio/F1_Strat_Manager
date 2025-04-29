# components/recommendations_view.py

import streamlit as st
import pandas as pd


def render_recommendations_view(recommendations):
    """
    Renders the strategy recommendations section.

    Parameters:
        recommendations (pd.DataFrame): DataFrame containing strategy recommendations
    """

    st.markdown("---")
    st.header("Strategy Recommendations")
    st.write("Strategic recommendations for race optimization.")

    if recommendations is None or recommendations.empty:
        st.warning("No recommendations available for the selected driver.")
        return

    # Group recommendations by action type
    try:
        action_types = recommendations['action'].unique()

        # Create tabs for different recommendation types
        rec_tabs = st.tabs([action.replace('_', ' ').title()
                           for action in action_types])

        for i, action in enumerate(action_types):
            with rec_tabs[i]:
                action_recs = recommendations[recommendations['action'] == action]

                for j, rec in action_recs.iterrows():
                    with st.expander(f"Lap {rec.get('LapNumber', 'N/A')} - Confidence: {rec.get('confidence', 'N/A'):.2f}"):
                        st.write(f"**Explanation:** {rec['explanation']}")
                        st.write(f"**Priority:** {rec.get('priority', 'N/A')}")
                        if 'rule_fired' in rec:
                            st.write(f"**Rule:** {rec['rule_fired']}")
    except Exception as e:
        st.error(f"Error displaying recommendations: {e}")

        # Fallback - simple table display
        st.dataframe(recommendations)
