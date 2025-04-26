# app/app.py


from utils.data_loader import load_race_data, load_recommendation_data
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path so we can import from our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our utility modules


# PAGE CONFIG

st.set_page_config(
    page_title="F1 Strategy Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom css

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .sidebar .sidebar-content {
        background-color: #1e2130;
    }
    h1, h2, h3 {
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)


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

# Race selection

selected_race = "Spain 2023"
st.sidebar.text(f"Race: {selected_race}")


# Driver selection

drivers = list(range(1, 21))  # Driver numbers from 1 to 20

selected_driver = st.sidebar.selectbox("Choose a Driver", drivers)

# Lap range selection

lap_range = st.sidebar.slider("Lap Range", 1, 66, (1, 66))

# Main content based on navigation
if page == "Overview":
    st.header("Race Overview")
    st.info(f"Select options from the sidebar to analyze race data.")

    # Placeholder for race information
    st.subheader(f"Race: {selected_race}")
    st.subheader(f"Selected Driver: {selected_driver}")

    # Placeholder for key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Avg. Degradation", value="0.12 s/lap")
    with col2:
        st.metric(label="Pit Stops", value="2")
    with col3:
        st.metric(label="Final Position", value="5")

    # Placeholder for a chart
    st.subheader("Race Timeline")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Position', 'Gap to Leader', 'Lap Time']
    )
    st.line_chart(chart_data)

elif page == "Tire Analysis":
    st.header("Tire Degradation Analysis")
    st.write("This section will show detailed tire degradation data.")

elif page == "Gap Analysis":
    st.header("Gap Analysis")
    st.write("This section will show detailed gap analysis between cars.")

elif page == "Team Radio Analysis":
    st.header("Team Radio Analysis")
    st.write("This section will show insights from team radio communications.")

elif page == "Strategy Recommendations":
    st.header("Strategy Recommendations")
    st.write("This section will display strategic recommendations for the race.")

# Footer
st.markdown("---")
st.markdown("Developed for Second Semester Third Year Project")
