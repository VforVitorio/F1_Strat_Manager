# components/overview_view.py

import streamlit as st
import pandas as pd
import numpy as np


def render_overview(race_data, selected_driver, selected_race):
    """
    Renders the race overview section.

    Parameters:
        race_data (pd.DataFrame): Processed race data
        selected_driver (int): Driver number
        selected_race (str): Race name
    """
    st.header("Race Overview")

    if race_data is None or race_data.empty:
        st.warning("No race data available for the selected driver.")
        return

    # Display driver information
    st.subheader(f"Driver #{selected_driver} - {selected_race}")

    # Calculate key metrics from actual data
    avg_degradation = "N/A"
    pit_stops = "N/A"
    final_position = "N/A"

    # Try to calculate actual metrics
    try:
        if 'DegradationRate' in race_data.columns:
            avg_degradation = f"{race_data['DegradationRate'].mean():.3f} s/lap"

        if 'Stint' in race_data.columns:
            pit_stops = str(race_data['Stint'].nunique() - 1)

        if 'Position' in race_data.columns:
            final_position = str(race_data.iloc[-1]['Position'])
    except Exception as e:
        st.warning(f"Could not calculate some metrics: {e}")

    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Avg. Degradation", value=avg_degradation)
    with col2:
        st.metric(label="Pit Stops", value=pit_stops)
    with col3:
        st.metric(label="Final Position", value=final_position)

    # Create overview charts
    st.subheader("Race Performance Overview")

    # Add basic lap time visualization if available
    try:
        if 'LapTime' in race_data.columns and 'LapNumber' in race_data.columns:
            chart_data = race_data[['LapNumber',
                                    'LapTime']].sort_values('LapNumber')
            st.line_chart(chart_data.set_index('LapNumber'))
        else:
            st.info("Lap time data not available for visualization.")
    except Exception as e:
        st.error(f"Error creating overview visualization: {e}")
