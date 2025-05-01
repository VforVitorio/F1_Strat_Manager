import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def plot_single_driver_position(race_data, selected_driver):
    """
    Plot position evolution for a single driver.
    Y axis: Position (1 = leader, 20 = last)
    X axis: Lap number
    """
    st.subheader("Position Evolution (Single Driver)")
    driver_data = race_data[race_data['DriverNumber']
                            == selected_driver].sort_values('LapNumber')
    if driver_data.empty:
        st.info("No data available for the selected driver.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=driver_data['LapNumber'],
        y=driver_data['Position'],
        mode='lines+markers',
        name=f"Driver {selected_driver}",
        line=dict(width=3, color='royalblue')
    ))
    fig.update_layout(
        xaxis_title="Lap",
        yaxis_title="Position (1 = leader)",
        yaxis=dict(autorange='reversed', dtick=1, range=[20, 1]),
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"single_driver_{selected_driver}")


def plot_multi_driver_position(race_data, selected_drivers):
    """
    Plot position evolution for up to 3 drivers.
    """
    st.subheader("Position Evolution Comparison (Up to 3 Drivers)")
    fig = go.Figure()
    colors = ['royalblue', 'crimson', 'orange']
    for idx, driver in enumerate(selected_drivers):
        driver_data = race_data[race_data['DriverNumber']
                                == driver].sort_values('LapNumber')
        if driver_data.empty:
            continue
        fig.add_trace(go.Scatter(
            x=driver_data['LapNumber'],
            y=driver_data['Position'],
            mode='lines+markers',
            name=f"Driver {driver}",
            line=dict(width=3, color=colors[idx % len(colors)])
        ))
    fig.update_layout(
        xaxis_title="Lap",
        yaxis_title="Position (1 = leader)",
        yaxis=dict(autorange='reversed', dtick=1, range=[20, 1]),
        template="plotly_dark"
    )
    key_str = "_".join(str(d) for d in selected_drivers)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"multi_driver_{key_str}")


def render_competitive_analysis_view(race_data, selected_driver):
    """
    Render the competitive analysis section (position evolution).
    """
    # Filter for valid race laps (e.g., Spain GP: 1-66)
    MAX_LAPS = 66
    race_data = race_data[(race_data['LapNumber'] >= 1)
                          & (race_data['LapNumber'] <= MAX_LAPS)]

    st.markdown("---")

    st.header("Competitive Analysis: Position Evolution")
    st.markdown("---")
    plot_single_driver_position(race_data, selected_driver)

    # Multi-driver comparison (up to 3 drivers)
    all_drivers = sorted(race_data['DriverNumber'].unique())
    driver_map = {f"Driver {int(d)}": int(d) for d in all_drivers}
    driver_options = list(driver_map.keys())

    st.markdown("---")
    selected_labels = st.multiselect(
        "Select up to 3 drivers to compare",
        options=driver_options,
        default=[f"Driver {selected_driver}"],
        max_selections=3
    )
    selected_drivers = [driver_map[label] for label in selected_labels]

    st.markdown("---")
    if selected_drivers:
        plot_multi_driver_position(race_data, selected_drivers)
