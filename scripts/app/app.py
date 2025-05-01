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
from components.competitive_analysis_view import render_competitive_analysis_view
from components.vision_view import render_vision_view
from components.strategy_overview import render_strategy_overview

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

# Modern sidebar navigation
st.sidebar.markdown(
    '<div class="sidebar-title">Navigation</div>'
    '<div class="sidebar-nav">', unsafe_allow_html=True)
page = st.sidebar.radio(
    "",  # hide default label
    ["Overview", "Tire Analysis", "Gap Analysis", "Lap Time Predictions",
        "Team Radio Analysis", "Strategy Recommendations", "Competitive Analysis", "Vision Gap Extraction"],
    index=0
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

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


# Routing for each page
if page == "Overview":
    render_strategy_overview()
    if strategy_ready():
        render_overview(race_data, selected_driver, selected_race)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif page == "Tire Analysis":
    if strategy_ready():
        render_degradation_view(race_data, selected_driver)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif page == "Gap Analysis":
    if strategy_ready():
        render_gap_analysis(gap_data, selected_driver)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif page == "Lap Time Predictions":
    if strategy_ready():
        predictions_df = get_lap_time_predictions(
            race_data, model_path="outputs/week3/xgb_sequential_model.pkl"
        )
        if predictions_df is not None:
            render_time_predictions_view(predictions_df, selected_driver)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif page == "Team Radio Analysis":
    if strategy_ready():
        render_radio_analysis(recommendations)
        render_radio_analysis_view()
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif page == "Strategy Recommendations":
    if strategy_ready():
        render_recommendations_view(recommendations)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif page == "Competitive Analysis":
    if strategy_ready():
        render_competitive_analysis_view(race_data, selected_driver)
    else:
        st.warning(
            "Please run the strategy analysis to generate recommendations before using the dashboard.")

elif page == "Vision Gap Extraction":
    render_vision_view()

# Footer
st.markdown("---")
st.markdown("Developed for Second Semester Third Year Project")
