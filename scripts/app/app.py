import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import load_race_data, load_recommendation_data, get_available_drivers
from utils.processing import get_processed_race_data, get_processed_recommendations, prepare_visualization_data, get_processed_gap_data
from components.degradation_view import render_degradation_view
from components.recommendations_view import render_recommendations_view
from components.team_radio_view import render_radio_analysis
from components.overview_view import render_overview
from components.radio_analysis_view import render_radio_analysis_view
from components.gap_analysis_view import render_gap_analysis
from components.time_predictions_view import render_time_predictions_view
from utils.processing import get_lap_time_predictions
from components.competitive_analysis_view import render_competitive_analysis_view, get_competitive_analysis_figures
from components.vision_view import render_vision_view
from components.strategy_overview import render_strategy_overview
from components.strategy_chat import render_strategy_chat, open_chat_with_image
from components.report_export import render_report_export_ui

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_css(css_file):
    """Load custom CSS into Streamlit app."""
    with open(css_file, 'r', encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Initialize session state for caching
if 'race_data' not in st.session_state:
    st.session_state.race_data = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_driver' not in st.session_state:
    st.session_state.selected_driver = None
if 'gap_data' not in st.session_state:
    st.session_state.gap_data = None

# Track the active page for navigation and AI chat redirection
if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"

# Page configuration
st.set_page_config(
    page_title="F1 Strategy Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css('scripts/app/assets/style.css')

# Title and description
st.title("🏎️ Formula 1 Strategy Dashboard")
st.markdown("""
This dashboard provides strategic insights and recommendations for Formula 1 races,
combining tire degradation analysis, gap calculations, and NLP from team radios.
""")

# Sidebar navigation
st.sidebar.markdown(
    '<div class="sidebar-title">Navigation</div>'
    '<div class="sidebar-nav">', unsafe_allow_html=True)
pages = [
    "Overview",
    "Tire Analysis",
    "Gap Analysis",
    "Lap Time Predictions",
    "Team Radio Analysis",
    "Strategy Recommendations",
    "Competitive Analysis",
    "Vision Gap Extraction",
    "Strategy Chat",
    "Export Strategy Report"
]
page = st.sidebar.radio(
    "",  # hide default label
    pages,
    index=pages.index(st.session_state.active_page)
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Update active page if changed
if page != st.session_state.active_page:
    st.session_state.active_page = page
    st.rerun()

# Data selection section
st.sidebar.markdown(
    '<div class="sidebar-title">Data Selection</div>'
    '<div class="sidebar-nav">', unsafe_allow_html=True)

# Fixed race selection
selected_race = "Spain 2023"
st.sidebar.markdown(f"**Race:** {selected_race}")

# Driver selection with error handling
try:
    available_drivers = get_available_drivers()
    if not available_drivers:
        available_drivers = list(range(1, 21))  # fallback list
        st.sidebar.warning(
            "Could not load driver data. Using placeholder values.")
except Exception as e:
    st.sidebar.error(f"Error loading driver data: {str(e)}")
    available_drivers = list(range(1, 21))

selected_driver = st.sidebar.selectbox("Choose a Driver", available_drivers)

# Cache driver selection and reset data if changed
if st.session_state.selected_driver != selected_driver:
    st.session_state.selected_driver = selected_driver
    st.session_state.race_data = None
    st.session_state.recommendations = None
    # Do NOT reset gap_data here, so we keep the full DataFrame loaded

# End of data selection container
st.sidebar.markdown('</div>', unsafe_allow_html=True)


def load_data():
    """Load and cache race data and strategy recommendations."""
    if st.session_state.race_data is None:
        with st.spinner("Loading race data..."):
            try:
                st.session_state.race_data = get_processed_race_data()
            except Exception as e:
                st.error(f"Error loading race data: {str(e)}")
                return None, None

    if st.session_state.recommendations is None:
        with st.spinner("Loading strategy recommendations..."):
            try:
                st.session_state.recommendations = get_processed_recommendations(
                    st.session_state.selected_driver)
            except Exception as e:
                st.error(f"Error loading recommendations: {str(e)}")
                st.session_state.recommendations = pd.DataFrame()

    # Always load the FULL gap data (all drivers), only once
    if st.session_state.gap_data is None:
        with st.spinner("Loading gap data..."):
            try:
                st.session_state.gap_data = get_processed_gap_data(
                    driver_number=None)
            except Exception as e:
                st.error(f"Error loading gap data: {str(e)}")
                st.session_state.gap_data = pd.DataFrame()

    return st.session_state.race_data, st.session_state.recommendations, st.session_state.gap_data


# Load data for rendering views
race_data, recommendations, gap_data = load_data()

csv_path = "outputs/week6/spain_gp_recommendations.csv"


def strategy_ready():
    return st.session_state.get("strategy_csv_ready", False) or os.path.exists(csv_path)


# Routing for each page WITHOUT Ask AI integration
if st.session_state.active_page == "Overview":
    render_strategy_overview()
    if strategy_ready():
        render_overview(race_data, selected_driver, selected_race)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif st.session_state.active_page == "Tire Analysis":
    if strategy_ready():
        render_degradation_view(race_data, selected_driver)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif st.session_state.active_page == "Gap Analysis":
    if strategy_ready():
        render_gap_analysis(gap_data, selected_driver)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif st.session_state.active_page == "Lap Time Predictions":
    if strategy_ready():
        predictions_df = get_lap_time_predictions(
            race_data, model_path="outputs/week3/xgb_sequential_model.pkl"
        )
        if predictions_df is not None:
            render_time_predictions_view(predictions_df, selected_driver)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif st.session_state.active_page == "Team Radio Analysis":
    if strategy_ready():
        render_radio_analysis(recommendations)
        render_radio_analysis_view()
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif st.session_state.active_page == "Strategy Recommendations":
    if strategy_ready():
        render_recommendations_view(recommendations)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif st.session_state.active_page == "Competitive Analysis":
    if strategy_ready():
        render_competitive_analysis_view(race_data, selected_driver)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif st.session_state.active_page == "Vision Gap Extraction":
    render_vision_view()

elif st.session_state.active_page == "Strategy Chat":
    render_strategy_chat()

elif st.session_state.active_page == "Export Strategy Report":
    if strategy_ready():
        # Load predictions (lap time and degradation)
        predictions_df = get_lap_time_predictions(
            race_data, model_path="outputs/week3/xgb_sequential_model.pkl"
        )

        # Competitive analysis figures (do NOT render, just get figures)
        competitive_charts = get_competitive_analysis_figures(
            race_data, selected_driver)

        # Degradation analysis figures
        from utils.visualization import (
            st_plot_degradation_rate,
            st_plot_regular_vs_adjusted_degradation,
            st_plot_speed_vs_tire_age
        )
        degradation_figs = []
        try:
            fig1 = st_plot_degradation_rate(race_data, selected_driver)
            if fig1:
                degradation_figs.append(fig1)
            fig2 = st_plot_regular_vs_adjusted_degradation(
                race_data, selected_driver)
            if fig2:
                degradation_figs.append(fig2)
            # For speed vs tire age, try all compounds
            if 'CompoundID' in race_data.columns:
                for compound_id in race_data['CompoundID'].unique():
                    fig3 = st_plot_speed_vs_tire_age(
                        race_data, selected_driver, compound_id)
                    if fig3:
                        degradation_figs.append(fig3)
        except Exception:
            pass

        # Gap analysis figures
        from utils.visualization import (
            st_plot_gap_evolution,
            st_plot_undercut_opportunities,
            st_plot_gap_consistency
        )
        gap_figs = []
        try:
            fig1 = st_plot_gap_evolution(gap_data, selected_driver)
            if fig1:
                gap_figs.append(fig1)
            fig2 = st_plot_undercut_opportunities(gap_data, selected_driver)
            if fig2:
                gap_figs.append(fig2)
            fig3 = st_plot_gap_consistency(gap_data, selected_driver)
            if fig3:
                gap_figs.append(fig3)
        except Exception:
            pass

        # Lap time prediction figures
        from components.time_predictions_view import render_time_predictions_view
        from utils.visualization import st_plot_fuel_adjusted_degradation
        prediction_figs = []
        try:
            fig1 = st_plot_fuel_adjusted_degradation(
                race_data, selected_driver)
            if fig1:
                prediction_figs.append(fig1)
        except Exception:
            pass

        # Overview figures (lap time evolution, comparison)
        from components.overview_view import render_overview
        # Not all overview charts are easily exportable, but you can add more here if needed

        # Prepare all data for export
        render_report_export_ui(
            selected_driver,
            race_data,
            recommendations,
            gap_data,
            predictions_df,
            None,  # radio_data
            race_data,  # competitive_data
            gap_charts=gap_figs,
            prediction_charts=prediction_figs,
            degradation_charts=degradation_figs,
            competitive_charts=competitive_charts
        )
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard."
        )

# Footer
st.markdown("---")
st.markdown("Developed for Second Semester Third Year Project")
