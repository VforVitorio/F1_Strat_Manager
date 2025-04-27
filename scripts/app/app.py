# app/app.py

from utils.data_loader import load_race_data, load_recommendation_data, get_available_drivers
from utils.processing import get_processed_race_data, get_processed_recommendations, prepare_visualization_data
from utils.visualization import (
    # st_plot_lap_time_deltas,
    st_plot_speed_vs_tire_age,
    st_plot_regular_vs_adjusted_degradation,
    st_plot_fuel_adjusted_degradation,
    st_plot_degradation_rate
)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path so we can import from our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Function to load CSS from a file


def load_css(css_file):
    with open(css_file, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Initialize session state for data caching
if 'race_data' not in st.session_state:
    st.session_state.race_data = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_driver' not in st.session_state:
    st.session_state.selected_driver = None

# PAGE CONFIG
st.set_page_config(
    page_title="F1 Strategy Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css('app/assets/style.css')

# Title and description
st.title("🏎️ Formula 1 Strategy Dashboard")
st.markdown("""
This dashboard provides strategic insights and recommendations for Formula 1 races,
combining tire degradation analysis, gap calculations, and NLP from team radios.
""")

# Sidebar for navigation and filters
st.sidebar.title("Navigation")

# Create navigation options
page = st.sidebar.radio(
    "Select a Page",
    ["Overview", "Tire Analysis", "Gap Analysis",
        "Team Radio Analysis", "Strategy Recommendations"]
)

# Data loading section in sidebar
st.sidebar.title("Data Selection")

# Race selection (currently fixed to Spain 2023)
selected_race = "Spain 2023"
st.sidebar.text(f"Race: {selected_race}")

# Load available drivers dynamically
try:
    available_drivers = get_available_drivers()
    if not available_drivers:
        available_drivers = list(range(1, 21))  # Fallback
        st.sidebar.warning(
            "Could not load driver data. Using placeholder values.")
except Exception as e:
    st.sidebar.error(f"Error loading driver data: {str(e)}")
    available_drivers = list(range(1, 21))  # Fallback

# Driver selection with dynamic list
selected_driver = st.sidebar.selectbox("Choose a Driver", available_drivers)

# Cache the selected driver in session state to maintain across page changes
if st.session_state.selected_driver != selected_driver:
    st.session_state.selected_driver = selected_driver
    # Reset cached data when driver changes
    st.session_state.race_data = None
    st.session_state.recommendations = None

# Load data with loading indicator


def load_data():
    if st.session_state.race_data is None:
        with st.spinner("Loading race data..."):
            try:
                # Load processed race data for the selected driver
                race_data = get_processed_race_data(
                    st.session_state.selected_driver)
                st.session_state.race_data = race_data
            except Exception as e:
                st.error(f"Error loading race data: {str(e)}")
                return None

    if st.session_state.recommendations is None:
        with st.spinner("Loading strategy recommendations..."):
            try:
                # Load recommendations for the selected driver
                recommendations = get_processed_recommendations(
                    st.session_state.selected_driver)
                st.session_state.recommendations = recommendations
            except Exception as e:
                st.error(f"Error loading recommendations: {str(e)}")
                st.session_state.recommendations = pd.DataFrame()  # Empty DataFrame as fallback

    return st.session_state.race_data, st.session_state.recommendations

# Get min and max laps from the data for the lap range slider


def get_lap_range():
    if st.session_state.race_data is not None and 'LapNumber' in st.session_state.race_data.columns:
        min_lap = int(st.session_state.race_data['LapNumber'].min())
        max_lap = int(st.session_state.race_data['LapNumber'].max())
        return min_lap, max_lap
    else:
        return 1, 66  # Default fallback


# Get lap range based on actual data
min_lap, max_lap = get_lap_range()
lap_range = st.sidebar.slider(
    "Lap Range", min_lap, max_lap, (min_lap, max_lap))

# Load data for the current page
race_data, recommendations = load_data()

# Main content based on navigation
if page == "Overview":
    st.header("Race Overview")

    if race_data is None or race_data.empty:
        st.warning("No race data available for the selected driver.")
    else:
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

        # # Use actual visualization functions if data is available
        try:
            #     if 'LapTime' in race_data.columns:
            #         lap_time_fig = st_plot_lap_time_deltas(
            #             race_data, selected_driver, lap_range)
            #         st.pyplot(lap_time_fig)
            #     else:
            #         st.info("Lap time data not available for visualization.")
            pass
        except Exception as e:
            #     st.error(f"Error creating lap time visualization: {e}")
            #     # Fallback to a simple chart if visualization fails
            #     st.line_chart(race_data[['TyreAge', 'LapTime']] if 'TyreAge' in race_data.columns and 'LapTime' in race_data.columns else pd.DataFrame(
            #         np.random.randn(20, 3)))
            pass

elif page == "Tire Analysis":
    st.header("Tire Degradation Analysis")

    if race_data is None or race_data.empty:
        st.warning("No tire data available for the selected driver.")
    else:
        # Create tabs for different tire analyses
        tire_tabs = st.tabs(
            ["Degradation Rate", "Fuel-Adjusted Analysis", "Speed Comparison"])

        with tire_tabs[0]:
            st.subheader("Tire Degradation Rate Analysis")
            st.write(
                "This visualization shows how quickly tire performance degrades over time.")

            try:
                degradation_fig = st_plot_degradation_rate(
                    race_data, selected_driver)
                st.pyplot(degradation_fig)
            except Exception as e:
                st.error(f"Error creating degradation rate visualization: {e}")

        with tire_tabs[1]:
            st.subheader("Fuel-Adjusted Degradation Analysis")
            st.write(
                "Comparing raw vs fuel-adjusted tire degradation to isolate pure tire wear effects.")

            try:
                fuel_adj_fig = st_plot_regular_vs_adjusted_degradation(
                    race_data, selected_driver)
                st.pyplot(fuel_adj_fig)
            except Exception as e:
                st.error(f"Error creating fuel-adjusted visualization: {e}")

        with tire_tabs[2]:
            st.subheader("Speed vs Tire Age")
            st.write("How sector speeds evolve as tires age.")

            # Get compound options from the data
            try:
                compounds = race_data['CompoundID'].unique()
                compound_id = st.selectbox("Select Tire Compound", compounds,
                                           format_func=lambda x: {1: "Soft", 2: "Medium", 3: "Hard"}.get(x, f"Compound {x}"))

                speed_fig = st_plot_speed_vs_tire_age(
                    race_data, selected_driver, compound_id)
                st.pyplot(speed_fig)
            except Exception as e:
                st.error(
                    f"Error creating speed vs tire age visualization: {e}")

elif page == "Gap Analysis":
    st.header("Gap Analysis")
    st.write("This section analyzes the gaps between cars throughout the race.")

    # Placeholder for gap analysis with error handling
    if race_data is None or race_data.empty:
        st.warning("No gap data available for the selected driver.")
    else:
        st.info("Gap analysis visualizations will be implemented here.")
        # Future: Add the gap analysis visualizations using your functions

elif page == "Team Radio Analysis":
    st.header("Team Radio Analysis")
    st.write("This section analyzes team radio communications for strategic insights.")

    # Placeholder for radio analysis
    if recommendations is None or recommendations.empty:
        st.warning("No radio analysis data available for the selected driver.")
    else:
        # Display any radio-related recommendations
        radio_recs = None
        try:
            if 'action' in recommendations.columns and 'explanation' in recommendations.columns:
                radio_recs = recommendations[recommendations['action'].isin(
                    ['prepare_rain_tires', 'reevaluate_pit_window', 'prioritize_pit']
                )]
        except Exception as e:
            st.error(f"Error processing radio recommendations: {e}")

        if radio_recs is not None and not radio_recs.empty:
            st.subheader("Radio-Based Strategic Recommendations")
            for i, rec in radio_recs.iterrows():
                with st.expander(f"{rec['action']} (Lap {rec.get('LapNumber', 'N/A')})"):
                    st.write(f"**Explanation:** {rec['explanation']}")
                    st.write(f"**Confidence:** {rec.get('confidence', 'N/A')}")
        else:
            st.info("No radio-based recommendations found for this driver.")

elif page == "Strategy Recommendations":
    st.header("Strategy Recommendations")
    st.write("Strategic recommendations for race optimization.")

    if recommendations is None or recommendations.empty:
        st.warning("No recommendations available for the selected driver.")
    else:
        # Group recommendations by action type
        try:
            action_types = recommendations['action'].unique()

            # Create tabs for different recommendation types
            rec_tabs = st.tabs([action.replace('_', ' ').title()
                               for action in action_types])

            for i, action in enumerate(action_types):
                with rec_tabs[i]:
                    action_recs = recommendations[recommendations['action'] == action]

                    for j, rec in action_recs.iterrows():
                        with st.expander(f"Lap {rec.get('LapNumber', 'N/A')} - Confidence: {rec.get('confidence', 'N/A'):.2f}"):
                            st.write(f"**Explanation:** {rec['explanation']}")
                            st.write(
                                f"**Priority:** {rec.get('priority', 'N/A')}")
                            if 'rule_fired' in rec:
                                st.write(f"**Rule:** {rec['rule_fired']}")
        except Exception as e:
            st.error(f"Error displaying recommendations: {e}")

            # Fallback - simple table display
            st.dataframe(recommendations)

# Footer
st.markdown("---")
st.markdown("Developed for Second Semester Third Year Project")
