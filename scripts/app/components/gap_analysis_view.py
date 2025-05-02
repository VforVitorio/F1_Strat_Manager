import streamlit as st
import pandas as pd
import numpy as np

from utils.visualization import (
    st_plot_gap_evolution,
    st_plot_undercut_opportunities,
    st_plot_gap_consistency
)
from utils.processing import (
    calculate_strategic_windows
)


def render_gap_analysis(gap_data, selected_driver):
    """
    Renders the gap analysis view.

    Parameters:
        gap_data (pd.DataFrame): Processed gap data (all drivers)
        selected_driver (int): Driver number
    """

    st.markdown("---")
    st.header("Gap Analysis")
    st.write("This section analyzes the gaps between cars throughout the race.")

    if gap_data is None or gap_data.empty:
        st.warning("No gap data available.")
        return

    # Limit to valid laps (1 to 66)
    MAX_LAPS = 66
    if 'LapNumber' in gap_data.columns:
        gap_data = gap_data[(gap_data['LapNumber'] >= 1) &
                            (gap_data['LapNumber'] <= MAX_LAPS)]

    # Calculate strategic windows from the data
    try:
        strategic_data = calculate_strategic_windows(gap_data)
    except Exception as e:
        st.error(f"Error calculating strategic windows: {e}")
        strategic_data = gap_data

    # Create tabs for different gap analyses
    gap_tabs = st.tabs(["Gap Evolution", "Undercut Opportunities",
                       "Gap Consistency", "Strategic Insights"])

    with gap_tabs[0]:
        st.subheader("Gap Evolution")
        st.write(
            "This visualization shows how gaps to cars ahead and behind evolved throughout the race.")

        try:
            gap_fig = st_plot_gap_evolution(gap_data, selected_driver)
            if gap_fig:
                st.plotly_chart(gap_fig, use_container_width=True)
            else:
                st.info("Not enough data to create gap evolution visualization.")
        except Exception as e:
            st.error(f"Error creating gap evolution visualization: {e}")

    with gap_tabs[1]:
        st.subheader("Undercut Opportunities")
        st.write("Windows where undercut or overcut strategies were possible.")

        try:
            undercut_fig = st_plot_undercut_opportunities(
                gap_data, selected_driver)
            if undercut_fig:
                st.plotly_chart(undercut_fig, use_container_width=True)
            else:
                st.info(
                    "Not enough data to create undercut opportunity visualization.")
        except Exception as e:
            st.error(f"Error creating undercut visualization: {e}")

    with gap_tabs[2]:
        st.subheader("Gap Consistency Analysis")
        st.write("How consistently gaps were maintained over multiple laps.")

        try:
            consistency_fig = st_plot_gap_consistency(
                gap_data, selected_driver)
            if consistency_fig:
                st.plotly_chart(consistency_fig, use_container_width=True)
            else:
                st.info("Not enough data to create gap consistency visualization.")
        except Exception as e:
            st.error(f"Error creating gap consistency visualization: {e}")

    with gap_tabs[3]:
        st.subheader("Strategic Insights")
        st.write("Summary of strategic opportunities identified from gap analysis.")

        try:
            # Count strategic opportunities
            if 'undercut_opportunity' in strategic_data.columns:
                undercut_count = strategic_data['undercut_opportunity'].sum()
                overcut_count = strategic_data['overcut_opportunity'].sum()
                defensive_count = strategic_data['defensive_needed'].sum()

                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Undercut Windows",
                              f"{int(undercut_count)} laps")
                with col2:
                    st.metric("Overcut Windows", f"{int(overcut_count)} laps")
                with col3:
                    st.metric("Defensive Windows",
                              f"{int(defensive_count)} laps")

                # Show detailed table of opportunities
                if st.checkbox("Show detailed opportunity data"):
                    opportunity_data = strategic_data[
                        strategic_data['undercut_opportunity'] |
                        strategic_data['overcut_opportunity'] |
                        strategic_data['defensive_needed']
                    ]

                    if not opportunity_data.empty:
                        # Select columns to display
                        display_cols = ['LapNumber', 'GapToCarAhead', 'GapToCarBehind',
                                        'consistent_gap_ahead_laps', 'consistent_gap_behind_laps',
                                        'undercut_opportunity', 'overcut_opportunity', 'defensive_needed']

                        # Filter to only display selected columns that exist
                        display_cols = [
                            col for col in display_cols if col in opportunity_data.columns]

                        st.dataframe(opportunity_data[display_cols])
                    else:
                        st.info("No specific opportunities found in the data.")
            else:
                st.info("Strategic opportunity data not available.")

        except Exception as e:
            st.error(f"Error analyzing strategic insights: {e}")
