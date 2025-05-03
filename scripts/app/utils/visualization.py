import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ML_tyre_pred.ML_utils.N01_tire_prediction import (
        compound_colors,
        compound_names,
        LAP_TIME_IMPROVEMENT_PER_LAP
    )
    COMPOUND_COLORS = compound_colors
    COMPOUND_NAMES = compound_names
except ImportError as e:
    print(f"Warning: Could not import from N01_tire_prediction: {e}")
    COMPOUND_COLORS = {1: 'red', 2: 'yellow', 3: 'gray', 4: 'green', 5: 'blue'}
    COMPOUND_NAMES = {1: "Soft", 2: "Medium",
                      3: "Hard", 4: "Intermediate", 5: "Wet"}
    LAP_TIME_IMPROVEMENT_PER_LAP = 0.055

MAX_LAPS = 66  # Limit for valid laps


def st_plot_speed_vs_tire_age(processed_race_data, driver_number=None, compound_id=None):
    filtered_data = processed_race_data.copy()
    # Limit to valid laps (1 to 66)
    if 'LapNumber' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['LapNumber'] >= 1) & (
            filtered_data['LapNumber'] <= MAX_LAPS)]
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    if compound_id is None:
        compound_counts = filtered_data['CompoundID'].value_counts()
        compound_id = compound_counts.index[0] if not compound_counts.empty else 2
    filtered_data = filtered_data[filtered_data['CompoundID'] == compound_id]
    if filtered_data.empty:
        return None

    speed_cols = ['SpeedI1', 'SpeedI2', 'SpeedFL']
    compound_name = COMPOUND_NAMES.get(compound_id, f"Unknown ({compound_id})")
    fig = go.Figure()

    found_any = False
    for speed_col in speed_cols:
        if speed_col in filtered_data.columns:
            fig.add_trace(go.Scatter(
                x=filtered_data['TyreAge'],
                y=filtered_data[speed_col],
                mode='lines+markers',
                name=f"{compound_name} Tires ({speed_col})",
                line=dict(width=3)
            ))
            found_any = True

    if not found_any:
        st.warning("No speed columns found (SpeedI1, SpeedI2, SpeedFL).")
        return None

    fig.update_layout(
        title=f"Speed vs Tire Age for Driver {driver_number} - {compound_name} Tires" if driver_number else f"Speed vs Tire Age - {compound_name} Tires",
        xaxis_title="Tire Age (laps)",
        yaxis_title="Speed (km/h)",
        template="plotly_dark"
    )
    return fig


def st_plot_regular_vs_adjusted_degradation(tire_deg_data, compound_names=None, compound_colors=None, lap_time_improvement_per_lap=0.055):
    """
    Streamlit-friendly version: shows subplots for each compound,
    comparing regular vs fuel-adjusted degradation (absolute).
    """
    if compound_names is None:
        compound_names = {1: 'Soft', 2: 'Medium', 3: 'Hard'}
    if compound_colors is None:
        compound_colors = {1: 'red', 2: 'yellow', 3: 'gray'}

    compound_ids = tire_deg_data['CompoundID'].unique()
    n_compounds = len(compound_ids)
    fig, axes = plt.subplots(n_compounds, 1, figsize=(
        16, 4 * n_compounds), sharex=False)
    if n_compounds == 1:
        axes = [axes]

    for i, compound_id in enumerate(compound_ids):
        ax = axes[i]
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]
        color = compound_colors.get(compound_id, 'black')
        compound_name = compound_names.get(
            compound_id, f'Unknown ({compound_id})')

        reg_agg = compound_subset.groupby('TyreAge')['TireDegAbsolute'].mean()
        adj_agg = compound_subset.groupby(
            'TyreAge')['FuelAdjustedDegAbsolute'].mean()

        ax.plot(reg_agg.index, reg_agg.values, 'o--', color=color,
                alpha=0.5, label=f'{compound_name} (Regular)')
        ax.plot(adj_agg.index, adj_agg.values, 'o-', color=color,
                linewidth=2, label=f'{compound_name} (Fuel Adjusted)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylabel('Degradation (s)')
        ax.set_title(
            f'{compound_name} Tire Degradation: Regular vs. Fuel-Adjusted')
        ax.legend()
        ax.grid(True, alpha=0.3)

        min_lap = reg_agg.index.min() if not reg_agg.empty else 0
        max_lap = reg_agg.index.max() if not reg_agg.empty else 0
        total_laps = max_lap - min_lap
        total_fuel_effect = total_laps * lap_time_improvement_per_lap
        ax.annotate(f"Est. total fuel effect: ~{total_fuel_effect:.2f}s",
                    xy=(0.02, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        if i == n_compounds - 1:
            ax.set_xlabel('Tire Age (laps)')

    plt.tight_layout()
    return fig


def st_plot_fuel_adjusted_degradation(processed_race_data, driver_number=None):
    filtered_data = processed_race_data.copy()
    # Limit to valid laps (1 to 66)
    if 'LapNumber' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['LapNumber'] >= 1) & (
            filtered_data['LapNumber'] <= MAX_LAPS)]
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    if filtered_data.empty:
        return None
    fig = go.Figure()
    for compound_id in filtered_data['CompoundID'].unique():
        compound_name = COMPOUND_NAMES.get(
            compound_id, f"Unknown ({compound_id})")
        data = filtered_data[filtered_data['CompoundID'] == compound_id]
        if 'FuelAdjustedDegradation' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['TyreAge'],
                y=data['FuelAdjustedDegradation'],
                mode='lines+markers',
                name=f"{compound_name} Tires",
                line=dict(color=COMPOUND_COLORS.get(compound_id, 'gray'))
            ))
    fig.update_layout(
        title=f"Fuel-Adjusted Degradation for Driver {driver_number}" if driver_number else "Fuel-Adjusted Degradation for All Drivers",
        xaxis_title="Tire Age (laps)",
        yaxis_title="Fuel-Adjusted Degradation Rate (s/lap)",
        template="plotly_dark"
    )
    return fig


def st_plot_fuel_adjusted_percentage_degradation(processed_race_data, driver_number=None):
    filtered_data = processed_race_data.copy()
    # Limit to valid laps (1 to 66)
    if 'LapNumber' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['LapNumber'] >= 1) & (
            filtered_data['LapNumber'] <= MAX_LAPS)]
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    if filtered_data.empty:
        return None
    fig = go.Figure()
    for compound_id in filtered_data['CompoundID'].unique():
        compound_name = COMPOUND_NAMES.get(
            compound_id, f"Unknown ({compound_id})")
        data = filtered_data[filtered_data['CompoundID'] == compound_id]
        if 'FuelAdjustedPercentageDegradation' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['TyreAge'],
                y=data['FuelAdjustedPercentageDegradation'],
                mode='lines+markers',
                name=f"{compound_name} Tires",
                line=dict(color=COMPOUND_COLORS.get(compound_id, 'gray'))
            ))
    fig.update_layout(
        title=f"Fuel-Adjusted Percentage Degradation for Driver {driver_number}" if driver_number else "Fuel-Adjusted Percentage Degradation for All Drivers",
        xaxis_title="Tire Age (laps)",
        yaxis_title="Fuel-Adjusted Percentage Degradation (%)",
        template="plotly_dark"
    )
    return fig


def st_plot_degradation_rate(processed_race_data, driver_number=None):
    filtered_data = processed_race_data.copy()
    # Limit to valid laps (1 to 66)
    if 'LapNumber' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['LapNumber'] >= 1) & (
            filtered_data['LapNumber'] <= MAX_LAPS)]
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    if filtered_data.empty:
        return None
    fig = go.Figure()
    for compound_id in filtered_data['CompoundID'].unique():
        compound_name = COMPOUND_NAMES.get(
            compound_id, f"Unknown ({compound_id})")
        data = filtered_data[filtered_data['CompoundID'] == compound_id]
        if 'DegradationRate' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['TyreAge'],
                y=data['DegradationRate'],
                mode='lines+markers',
                name=f"{compound_name} Tires",
                line=dict(color=COMPOUND_COLORS.get(compound_id, 'gray'))
            ))
    fig.update_layout(
        title=f"Degradation Rate for Driver {driver_number}" if driver_number else "Degradation Rate for All Drivers",
        xaxis_title="Tire Age (laps)",
        yaxis_title="Degradation Rate (s/lap)",
        template="plotly_dark"
    )
    return fig


def st_plot_gap_evolution(gap_data, driver_number=None):
    filtered_data = gap_data.copy()
    # Limit to valid laps (1 to 66)
    if 'LapNumber' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['LapNumber'] >= 1) & (
            filtered_data['LapNumber'] <= MAX_LAPS)]
    if driver_number is not None and 'DriverNumber' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    if filtered_data.empty:
        return None
    gap_ahead_col = 'GapToCarAhead' if 'GapToCarAhead' in filtered_data.columns else None
    gap_behind_col = 'GapToCarBehind' if 'GapToCarBehind' in filtered_data.columns else None
    lap_col = 'LapNumber' if 'LapNumber' in filtered_data.columns else None
    if not all([gap_ahead_col, gap_behind_col, lap_col]):
        return None
    filtered_data = filtered_data.sort_values(lap_col)
    fig = go.Figure()
    # Gap to car ahead
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col], y=filtered_data[gap_ahead_col],
        mode='lines+markers', name='Gap to Car Ahead', line=dict(color='royalblue')
    ))
    # Gap to car behind
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col], y=filtered_data[gap_behind_col],
        mode='lines+markers', name='Gap to Car Behind', line=dict(color='firebrick')
    ))
    # Undercut window (2.0s)
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col],
        y=[2.0]*len(filtered_data),
        mode='lines',
        name='Undercut Window (2.0s)',
        line=dict(color='green', dash='dash')
    ))
    # DRS window (1.0s)
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col],
        y=[1.0]*len(filtered_data),
        mode='lines',
        name='DRS Window (1.0s)',
        line=dict(color='yellow', dash='dash')
    ))
    fig.update_layout(
        xaxis_title='Lap Number',
        yaxis_title='Gap (seconds)',
        title=f'Gap Evolution for Driver #{driver_number}' if driver_number else 'Gap Evolution',
        template="plotly_dark"
    )
    return fig


def st_plot_undercut_opportunities(gap_data, driver_number=None):
    filtered_data = gap_data.copy()
    # Limit to valid laps (1 to 66)
    if 'LapNumber' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['LapNumber'] >= 1) & (
            filtered_data['LapNumber'] <= MAX_LAPS)]
    if driver_number is not None and 'DriverNumber' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    if filtered_data.empty:
        return None
    gap_ahead_col = 'GapToCarAhead' if 'GapToCarAhead' in filtered_data.columns else None
    lap_col = 'LapNumber' if 'LapNumber' in filtered_data.columns else None
    if not all([gap_ahead_col, lap_col]):
        return None
    filtered_data = filtered_data.sort_values(lap_col)
    y_max = filtered_data[gap_ahead_col].max() + 1
    fig = go.Figure()
    # Undercut zone
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col], y=[2.0]*len(filtered_data),
        fill=None, mode='lines', line=dict(color='green', dash='dash'), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col], y=[3.5]*len(filtered_data),
        fill=None, mode='lines', line=dict(color='red', dash='dash'), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col], y=filtered_data[gap_ahead_col],
        mode='lines+markers', name='Gap to Car Ahead', line=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col], y=[0]*len(filtered_data),
        fill='tonexty', mode='none', fillcolor='rgba(0,255,0,0.2)', name='Undercut Zone (<2.0s)', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col], y=[2.0]*len(filtered_data),
        fill='tonexty', mode='none', fillcolor='rgba(255,165,0,0.2)', name='Overcut Zone (2.0-3.5s)', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=filtered_data[lap_col], y=[y_max]*len(filtered_data),
        fill='tonexty', mode='none', fillcolor='rgba(255,0,0,0.2)', name='No Strategy Zone (>3.5s)', showlegend=True
    ))
    fig.update_layout(
        xaxis_title='Lap Number',
        yaxis_title='Gap to Car Ahead (seconds)',
        title=f'Undercut/Overcut Opportunities for Driver #{driver_number}' if driver_number else 'Undercut/Overcut Opportunities',
        template="plotly_dark"
    )
    return fig


def st_plot_gap_consistency(gap_data, driver_number=None):
    filtered_data = gap_data.copy()
    # Limit to valid laps (1 to 66)
    if 'LapNumber' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['LapNumber'] >= 1) & (
            filtered_data['LapNumber'] <= MAX_LAPS)]
    if driver_number is not None and 'DriverNumber' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    if filtered_data.empty:
        return None
    lap_col = 'LapNumber' if 'LapNumber' in filtered_data.columns else None
    consistent_ahead = 'consistent_gap_ahead_laps' if 'consistent_gap_ahead_laps' in filtered_data.columns else None
    consistent_behind = 'consistent_gap_behind_laps' if 'consistent_gap_behind_laps' in filtered_data.columns else None
    if not all([lap_col, consistent_ahead, consistent_behind]):
        return None
    filtered_data = filtered_data.sort_values(lap_col)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered_data[lap_col], y=filtered_data[consistent_ahead],
        name='Consistent Laps Ahead', marker_color='royalblue'
    ))
    fig.add_trace(go.Bar(
        x=filtered_data[lap_col], y=filtered_data[consistent_behind],
        name='Consistent Laps Behind', marker_color='firebrick'
    ))
    fig.add_hline(y=3, line_dash="dash", line_color="green",
                  annotation_text="Strategic Threshold (3 laps)")
    fig.update_layout(
        xaxis_title='Lap Number',
        yaxis_title='Consistent Laps',
        barmode='group',
        title=f'Gap Consistency for Driver #{driver_number}' if driver_number else 'Gap Consistency',
        template="plotly_dark"
    )
    return fig
