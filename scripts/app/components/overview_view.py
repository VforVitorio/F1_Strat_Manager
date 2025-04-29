# components/overview_view.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render_overview(race_data, selected_driver, selected_race):
    """
    Renders the race overview section.

    Parameters:
        race_data (pd.DataFrame): Processed race data
        selected_driver (int): Driver number
        selected_race (str): Race name
    """
    st.markdown("---")

    st.header("Race Overview")

    if race_data is None or race_data.empty:
        st.warning("No race data available for the selected driver.")
        return

    # Clean column names in case there are spaces
    race_data.columns = race_data.columns.str.strip()

    # Filter data for the selected driver
    race_data['DriverNumber'] = race_data['DriverNumber'].astype(int)
    driver_num = int(selected_driver)
    filtered_data = race_data[race_data['DriverNumber'] == driver_num]

    # Display driver information
    st.markdown(
        f"<h3 style='text-align: center;'>{'Driver #' + str(selected_driver)} - {selected_race}</h3>",
        unsafe_allow_html=True
    )

    # Calculate key metrics from actual data for the selected driver only
    avg_degradation = "N/A"
    pit_stops = "N/A"
    final_position = "N/A"

    try:
        if 'DegradationRate' in filtered_data.columns and not filtered_data.empty:
            avg_degradation = f"{filtered_data['DegradationRate'].mean():.3f} s/lap"

        if 'Stint' in filtered_data.columns and not filtered_data.empty:
            pit_stops = str(filtered_data['Stint'].nunique() - 1)

        if 'Position' in filtered_data.columns and not filtered_data.empty:
            final_position = str(filtered_data.iloc[-1]['Position'])
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

    # Add a horizontal divider
    st.markdown("---")

    # Create overview charts
    st.subheader("Race Performance Overview")

    # Add interactive lap time visualization with Plotly
    try:
        required_cols = {'LapTime', 'LapNumber',
                         'DriverNumber', 'CompoundName'}
        if required_cols.issubset(set(race_data.columns)):
            race_data['LapNumber'] = race_data['LapNumber'].astype(int)
            if not filtered_data.empty:
                chart_data = filtered_data[[
                    'LapNumber', 'LapTime', 'CompoundName']].sort_values('LapNumber')

                # Define color mapping for compounds
                compound_colors = {
                    'SOFT': 'red',
                    'MEDIUM': 'yellow',
                    'HARD': 'gray',
                    # Add more compounds if needed
                }

                fig = go.Figure()
                stint_change_laps = []

                # Group by compound stints (where compound changes)
                for i, (_, stint_data) in enumerate(chart_data.groupby((chart_data['CompoundName'] != chart_data['CompoundName'].shift()).cumsum())):
                    compound_name = stint_data['CompoundName'].iloc[0]
                    color = compound_colors.get(
                        str(compound_name).upper(), 'white')
                    fig.add_trace(go.Scatter(
                        x=stint_data['LapNumber'],
                        y=stint_data['LapTime'],
                        mode='lines+markers',
                        line=dict(color=color, width=2),
                        marker=dict(size=4, color=color),
                        name=compound_name,
                        hovertemplate='Lap %{x}<br>Lap Time: %{y:.3f}s<br>Compound: ' + str(
                            compound_name),
                        connectgaps=True
                    ))
                    # Save the first lap of each stint except the first one
                    if i > 0:
                        stint_change_laps.append(
                            stint_data['LapNumber'].iloc[0])

                # Add vertical lines for stint changes
                for lap in stint_change_laps:
                    fig.add_shape(
                        type="line",
                        x0=lap, x1=lap,
                        y0=chart_data['LapTime'].min() - 1, y1=chart_data['LapTime'].max() + 1,
                        line=dict(color="white", width=1, dash="dash"),
                        layer="below"
                    )

                fig.update_layout(
                    xaxis_title="Lap Number",
                    yaxis_title="Lap Time (s)",
                    title=f"Lap Time Evolution for #Driver {driver_num}",
                    template="plotly_dark",
                    height=350,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    f"No lap time data available for driver {selected_driver}.")
        else:
            st.info("Lap time data not available for visualization.")
    except Exception as e:
        st.error(f"Error creating overview visualization: {e}")
