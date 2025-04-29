# components/degradation_view.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Corregir la importación para evitar el ciclo
from utils.visualization import (
    st_plot_degradation_rate,
    st_plot_regular_vs_adjusted_degradation,
    st_plot_speed_vs_tire_age
)


def render_degradation_view(race_data, selected_driver):
    """
    Renders the tire degradation analysis view.

    Parameters:
        race_data (pd.DataFrame): Processed race data
        selected_driver (int): Driver number
    """
    st.markdown("---")

    st.header("Tire Degradation Analysis")

    if race_data is None or race_data.empty:
        st.warning("No tire data available for the selected driver.")
        return

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
            st.error(f"Error creating speed vs tire age visualization: {e}")
