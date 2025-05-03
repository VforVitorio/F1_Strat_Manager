import streamlit as st
import plotly.graph_objects as go
import numpy as np


def plot_single_driver_position_figure(race_data, selected_driver):
    """
    Returns a Plotly figure of position evolution for a single driver.
    """
    driver_data = race_data[race_data['DriverNumber']
                            == selected_driver].sort_values('LapNumber')
    if driver_data.empty:
        return None
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
    return fig


def plot_multi_driver_position_figure(race_data, selected_drivers):
    """
    Returns a Plotly figure of position evolution for up to 3 drivers.
    """
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
    return fig


def extract_stints(race_data, driver_number):
    """
    Returns a list of (stint_start_lap, stint_end_lap, compound) for the given driver.
    """
    driver_data = race_data[race_data['DriverNumber']
                            == driver_number].sort_values('LapNumber')
    driver_data = driver_data.dropna(subset=['LapNumber', 'Compound'])
    stints = []
    last_compound = None
    stint_start = None
    for _, row in driver_data.iterrows():
        compound = row.get("Compound")
        lap = row["LapNumber"]
        if last_compound is None:
            last_compound = compound
            stint_start = lap
        elif compound != last_compound:
            stints.append((stint_start, lap-1, last_compound))
            stint_start = lap
            last_compound = compound
    if stint_start is not None and last_compound is not None:
        stints.append(
            (stint_start, driver_data["LapNumber"].max(), last_compound))
    return stints


def estimate_pit_window(stints):
    """
    Returns the typical stint length (mean, std) for a list of stints.
    """
    lengths = [end - start + 1 for start, end, _ in stints]
    if lengths:
        return int(np.mean(lengths)), int(np.std(lengths))
    return None, None


def get_competitive_analysis_figures(race_data, selected_driver):
    """
    Returns a list of Plotly figures for competitive analysis (for export).
    Does NOT render anything to Streamlit.
    """
    # Ensure LapNumber exists
    if 'LapNumber' not in race_data.columns:
        try:
            from utils.processing import add_race_lap_column
            race_data = add_race_lap_column(race_data)
        except ImportError:
            return []

    # Ensure Compound exists
    COMPOUND_NAMES = {1: "Soft", 2: "Medium",
                      3: "Hard", 4: "Intermediate", 5: "Wet"}
    if 'Compound' not in race_data.columns and 'CompoundID' in race_data.columns:
        race_data['Compound'] = race_data['CompoundID'].map(COMPOUND_NAMES)

    race_data = race_data.dropna(subset=['LapNumber', 'Compound'])
    MAX_LAPS = 66
    race_data = race_data[(race_data['LapNumber'] >= 1)
                          & (race_data['LapNumber'] <= MAX_LAPS)]

    figs = []
    fig1 = plot_single_driver_position_figure(race_data, selected_driver)
    if fig1 is not None:
        figs.append(fig1)

    # Multi-driver comparison (default: selected_driver + 2 closest rivals by number)
    all_drivers = sorted(race_data['DriverNumber'].unique())
    rivals = [d for d in all_drivers if d != selected_driver]
    selected_drivers = [selected_driver] + rivals[:2]
    fig2 = plot_multi_driver_position_figure(race_data, selected_drivers)
    if fig2 is not None:
        figs.append(fig2)

    return figs


def render_opponent_strategy_estimation(race_data, selected_driver):
    """
    Show a table with rivals, their current stint info, and pit/undercut/overcut alerts.
    """
    st.markdown("---")
    st.header("Opponent Strategy Estimation")
    rivals = sorted(
        [d for d in race_data['DriverNumber'].unique() if d != selected_driver])
    rows = []
    for rival in rivals:
        stints = extract_stints(race_data, rival)
        mean_stint, std_stint = estimate_pit_window(
            stints[:-1]) if len(stints) > 1 else (None, None)
        current_stint = stints[-1] if stints else (None, None, None)
        stint_start, stint_end, compound = current_stint
        laps_on_tyre = race_data[(race_data['DriverNumber'] == rival) & (
            race_data['LapNumber'] >= stint_start)].shape[0] if stint_start else None
        likely_pit = False
        if mean_stint and laps_on_tyre and laps_on_tyre >= (mean_stint - 2):
            likely_pit = True
        rows.append({
            "Rival": rival,
            "Tyre": compound,
            "Current Stint Laps": laps_on_tyre,
            "Typical Stint (mean)": mean_stint,
            "Pit Soon?": "Yes" if likely_pit else "No"
        })
    st.dataframe(rows, use_container_width=True)
    # Grouped alert box for all rivals likely to pit soon, sorted by stint laps ascending
    rows_sorted = sorted(
        [row for row in rows if row["Pit Soon?"] ==
            "Yes" and row["Current Stint Laps"] is not None],
        key=lambda x: x["Current Stint Laps"]
    )
    alerts = [
        f"- Rival {row['Rival']} likely to pit soon (on {row['Tyre']}, {row['Current Stint Laps']} laps)."
        for row in rows_sorted
    ]
    if alerts:
        st.warning("**Rivals likely to pit soon:**\n\n" + "\n".join(alerts))


def render_competitive_analysis_view(race_data, selected_driver):
    """
    Render the competitive analysis section (position evolution) in Streamlit.
    """
    # Ensure LapNumber exists
    if 'LapNumber' not in race_data.columns:
        try:
            from utils.processing import add_race_lap_column
            race_data = add_race_lap_column(race_data)
        except ImportError:
            st.error("LapNumber column missing and add_race_lap_column not found.")
            return

    # Ensure Compound exists
    COMPOUND_NAMES = {1: "Soft", 2: "Medium",
                      3: "Hard", 4: "Intermediate", 5: "Wet"}
    if 'Compound' not in race_data.columns and 'CompoundID' in race_data.columns:
        race_data['Compound'] = race_data['CompoundID'].map(COMPOUND_NAMES)

    race_data = race_data.dropna(subset=['LapNumber', 'Compound'])
    MAX_LAPS = 66
    race_data = race_data[(race_data['LapNumber'] >= 1)
                          & (race_data['LapNumber'] <= MAX_LAPS)]

    st.markdown("---")
    st.header("Competitive Analysis: Position Evolution")
    st.markdown("---")
    fig1 = plot_single_driver_position_figure(race_data, selected_driver)
    if fig1 is not None:
        st.plotly_chart(fig1, use_container_width=True,
                        key=f"single_driver_{selected_driver}")

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
        fig2 = plot_multi_driver_position_figure(race_data, selected_drivers)
        if fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True,
                            key=f"multi_driver_{'_'.join(map(str, selected_drivers))}")

    st.markdown("---")
    render_opponent_strategy_estimation(race_data, selected_driver)
