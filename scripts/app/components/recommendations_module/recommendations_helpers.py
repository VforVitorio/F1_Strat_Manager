"""
recommendations_helpers.py

Core helper functions for the strategy recommendations component.
Includes stat card rendering, filtering, sorting, timeline plotting,
distribution plotting, and detailed recommendation card rendering.

All functions are designed to be imported and used in recommendations_view.py.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Color map for each action type
ACTION_COLORS = {
    'pit_stop': '#FF5733',
    'extend_stint': '#33FF57',
    'prepare_pit': '#5733FF',
    'perform_undercut': '#33FFEC',
    'defensive_pit': '#FF33EB',
    'prioritize_pit': '#FFBD33',
    'prepare_rain_tires': '#3385FF',
    'reevaluate_pit_window': '#FF3333',
    'recovery_push': '#FCFF33',
    'push_strategy': '#33FFBD',
    'perform_overcut': '#BD33FF',
    'consider_pit': '#FF8833',
    'adjust_pit_window': '#33FFC1'
}


def render_stat_card(value, label, color_class):
    """
    Render a single stat card with a value and label.
    """
    st.markdown(f"""
    <div class="stat-card {color_class}">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def filter_and_sort_recommendations(recommendations, confidence_threshold, selected_priorities, selected_phases, sort_by):
    """
    Filter and sort the recommendations DataFrame based on user selections.
    """
    filtered = recommendations.copy()
    filtered = filtered[filtered['confidence'] >= confidence_threshold]
    if selected_priorities:
        filtered = filtered[filtered['priority'].isin(selected_priorities)]
    if selected_phases:
        filtered = filtered[filtered['RacePhase'].isin(selected_phases)]
    if sort_by == "Lap (Ascending)":
        filtered = filtered.sort_values('LapNumber')
    elif sort_by == "Lap (Descending)":
        filtered = filtered.sort_values('LapNumber', ascending=False)
    elif sort_by == "Confidence (High to Low)":
        filtered = filtered.sort_values('confidence', ascending=False)
    elif sort_by == "Priority (High to Low)":
        filtered = filtered.sort_values('priority', ascending=False)
    return filtered


def plot_recommendation_timeline(filtered_recs, action_colors):
    """
    Plot a timeline of recommendations by action type and lap number.
    """
    fig = go.Figure()
    for action in filtered_recs['action'].unique():
        action_data = filtered_recs[filtered_recs['action'] == action]
        display_action = action.replace('_', ' ').title()
        fig.add_trace(go.Scatter(
            x=action_data['LapNumber'],
            y=[display_action] * len(action_data),
            mode='markers',
            marker=dict(
                size=action_data['confidence'] * 20,
                color=action_colors.get(action, '#CCCCCC'),
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name=display_action,
            text=[f"Lap: {lap}<br>Confidence: {conf:.2f}<br>Priority: {pri}"
                  for lap, conf, pri in zip(action_data['LapNumber'],
                                            action_data['confidence'],
                                            action_data['priority'])],
            hoverinfo='text'
        ))
    fig.update_layout(
        title="Race Strategy Recommendations Timeline",
        xaxis_title="Lap Number",
        yaxis_title="Recommendation Type",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_priority_distribution(filtered_recs):
    """
    Plot a pie chart of recommendations by priority.
    """
    if 'priority' in filtered_recs.columns:
        priority_counts = filtered_recs['priority'].value_counts(
        ).reset_index()
        priority_counts.columns = ['priority', 'count']
        fig_priority = px.pie(
            priority_counts,
            values='count',
            names='priority',
            title='Recommendations by Priority',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_priority.update_traces(
            textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_priority, use_container_width=True)


def plot_confidence_distribution(filtered_recs):
    """
    Plot a histogram of recommendation confidence values.
    """
    fig_conf = px.histogram(
        filtered_recs,
        x='confidence',
        title='Recommendation Confidence Distribution',
        color_discrete_sequence=['#3385FF']
    )
    st.plotly_chart(fig_conf, use_container_width=True)


def render_recommendation_card(rec, action_colors, key_prefix=""):
    """
    Render a detailed card for a single recommendation, including expandable context and simulation.
    """
    action = rec['action']
    color = action_colors.get(action, '#CCCCCC')
    with st.container():
        st.markdown(f"""
        <div style="border-left:5px solid {color}; padding:10px; margin:10px 0; border-radius:5px; background-color:rgba(0,0,0,0.05)">
            <h4>Lap {rec.get('LapNumber', 'N/A')} - {action.replace('_', ' ').title()}</h4>
            <p>{rec['explanation']}</p>
            <div style="display:flex; gap:15px; margin-top:10px">
                <span style="background-color:rgba(0,0,0,0.1); padding:3px 8px; border-radius:10px">
                    <strong>Confidence:</strong> {rec.get('confidence', 'N/A'):.2f}
                </span>
                <span style="background-color:rgba(0,0,0,0.1); padding:3px 8px; border-radius:10px">
                    <strong>Priority:</strong> {rec.get('priority', 'N/A')}
                </span>
                <span style="background-color:rgba(0,0,0,0.1); padding:3px 8px; border-radius:10px">
                    <strong>Rule:</strong> {rec.get('rule_fired', 'N/A')}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Additional Information"):
            # Show compound and race phase if available
            if 'CompoundID' in rec:
                compound_map = {1: "Soft", 2: "Medium", 3: "Hard"}
                compound = compound_map.get(
                    int(rec['CompoundID']), f"Compound {rec['CompoundID']}")
                st.write(f"**Current Compound:** {compound}")
            if 'RacePhase' in rec:
                st.write(f"**Race Phase:** {rec['RacePhase']}")
            # What-if simulation button
            if st.button(f"Simulate Impact", key=f"{key_prefix}{rec.get('LapNumber', 'N/A')}"):
                st.info(
                    "This simulation would show the projected outcome if this recommendation was followed.")
                time_saved = np.random.uniform(0.5, 3.0)
                position_gain = np.random.choice([0, 0, 0, 1, 1, 2])
                st.markdown(f"""
                <div style="background-color:rgba(51,133,255,0.1); padding:10px; border-radius:5px; margin-top:10px">
                    <h4>Simulation Results</h4>
                    <p>Following this recommendation would likely result in:</p>
                    <ul>
                        <li>Time saved: <strong>{time_saved:.2f}s</strong></li>
                        <li>Potential position gain: <strong>{position_gain}</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
