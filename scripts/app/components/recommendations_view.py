import streamlit as st
import pandas as pd
from components.recommendations_module.optimal_strategy_generator import (
    generate_optimal_strategy,
    plot_optimal_strategy_step_chart,
    plot_optimal_strategy_swimlane,
    render_optimal_strategy_summary_table
)
from components.recommendations_module.recommendations_helpers import (
    ACTION_COLORS,
    render_stat_card,
    filter_and_sort_recommendations,
    plot_recommendation_timeline,
    plot_priority_distribution,
    plot_confidence_distribution,
    render_recommendation_card
)

from components.recommendations_module.strategy_comparator import (
    render_strategy_comparison,
    generate_aggressive_strategy,
    generate_conservative_strategy
)
from utils.processing import get_processed_race_data


def render_recommendations_view(recommendations, selected_driver=None):
    """
    Main entry point for the strategy recommendations section.
    Uses helper functions for stats, filtering, timeline, distributions, and cards.
    """

    race_data = get_processed_race_data(selected_driver)
    st.header("Strategy Recommendations")
    st.write(
        "Interactive analysis of AI-generated strategic recommendations for race optimization.")

    if recommendations is None or recommendations.empty:
        st.warning("No recommendations available for the selected driver.")
        return

    # Stat cards at the top
    col1, col2, col3, col4 = st.columns(4)
    total_recs = len(recommendations)
    unique_actions = recommendations['action'].nunique()
    avg_confidence = recommendations['confidence'].mean()
    max_priority = recommendations['priority'].max(
    ) if 'priority' in recommendations.columns else "N/A"

    with col1:
        render_stat_card(total_recs, "Total Recommendations", "blue")
    with col2:
        render_stat_card(unique_actions, "Types of Actions", "green")
    with col3:
        render_stat_card(f"{avg_confidence:.2f}", "Avg Confidence", "orange")
    with col4:
        render_stat_card(max_priority, "Max Priority", "purple")

    # Filter controls
    with st.expander("Filter and Sort Recommendations", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_confidence = float(recommendations['confidence'].min())
            max_confidence = float(recommendations['confidence'].max())
            if min_confidence < max_confidence:
                confidence_threshold = st.slider(
                    "Minimum Confidence",
                    min_value=min_confidence,
                    max_value=max_confidence,
                    value=min_confidence,
                    step=0.05
                )
            else:
                st.write(f"**Confidence Value:** {min_confidence:.2f}")
                confidence_threshold = min_confidence
        with col2:
            if 'priority' in recommendations.columns:
                priorities = sorted(recommendations['priority'].unique())
                selected_priorities = st.multiselect(
                    "Priority Levels",
                    options=priorities,
                    default=priorities
                )
            else:
                selected_priorities = None
                st.write("**Priority:** Not available in data")
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=["Lap (Ascending)", "Lap (Descending)",
                         "Confidence (High to Low)", "Priority (High to Low)"]
            )
        if 'RacePhase' in recommendations.columns:
            race_phases = recommendations['RacePhase'].unique()
            selected_phases = st.multiselect(
                "Race Phases",
                options=race_phases,
                default=race_phases
            )
        else:
            selected_phases = None

    # Filtering and sorting
    filtered_recs = filter_and_sort_recommendations(
        recommendations,
        confidence_threshold,
        selected_priorities,
        selected_phases,
        sort_by
    )

    # Timeline visualization
    st.markdown("---")
    st.subheader("Recommendation Timeline")
    plot_recommendation_timeline(
        filtered_recs, ACTION_COLORS, key="main_timeline")

    # Distribution charts

    st.markdown("---")
    st.subheader("Recommendation Distribution")
    col1, col2 = st.columns(2)
    with col1:
        plot_priority_distribution(filtered_recs)
    with col2:
        plot_confidence_distribution(filtered_recs)

    # Detailed recommendation cards

    st.markdown("---")
    st.subheader("Detailed Recommendations")
    view_mode = st.radio(
        "View Mode",
        options=["Group by Action Type", "Chronological View"],
        horizontal=True
    )

    if view_mode == "Group by Action Type":
        action_types = filtered_recs['action'].unique()
        rec_tabs = st.tabs([action.replace('_', ' ').title()
                           for action in action_types])
        for i, action in enumerate(action_types):
            with rec_tabs[i]:
                action_recs = filtered_recs[filtered_recs['action'] == action]
                for j, rec in action_recs.iterrows():
                    render_recommendation_card(
                        rec, ACTION_COLORS, key_prefix=f"{action}_{j}_")
    else:
        chrono_recs = filtered_recs.sort_values('LapNumber')
        for j, rec in chrono_recs.iterrows():
            render_recommendation_card(
                rec, ACTION_COLORS, key_prefix=f"chrono_{j}_")

    # --- OPTIMAL STRATEGY SECTION ---
    st.markdown("---")
    st.subheader("Optimal Strategy")
    optimal_recs, summary, stints = generate_optimal_strategy(recommendations)

    aggressive_recs, aggressive_summary, aggressive_stints = generate_aggressive_strategy(
        recommendations)
    conservative_recs, conservative_summary, conservative_stints = generate_conservative_strategy(
        recommendations)

    all_strategies = {
        "Optimal": optimal_recs,
        "Aggressive": aggressive_recs,
        "Conservative": conservative_recs
    }

    st.markdown(f"**Strategy Narrative:**\n\n{summary}")

    st.markdown("---")
    st.subheader("Step Chart (Action Timeline)")
    plot_optimal_strategy_step_chart(optimal_recs)

    st.markdown("---")

    st.subheader("Swimlane Chart (Action Lanes)")
    plot_optimal_strategy_swimlane(optimal_recs)

    st.markdown("---")

    st.subheader("Summary Table")
    render_optimal_strategy_summary_table(optimal_recs)

    # --- STRATEGY COMPARATOR SECTION ---
    st.markdown("---")
    st.subheader("Strategy Comparator")
    render_strategy_comparison(all_strategies, race_data)
