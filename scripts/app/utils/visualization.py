# -----------------------------------------------------------------------------
# MODULE: visualization.py
# PURPOSE: Contains all visualization functions for the Streamlit app.
#          Assumes all data is already processed and ready for plotting.
#          No data loading or processing logic should be present here.
# -----------------------------------------------------------------------------

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
# Add the project root to sys.path so 'app' can be imported
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Import plotting functions and constants from tire prediction utilities
try:
    from ML_tyre_pred.ML_utils.N01_tire_prediction import (
        compound_colors,
        compound_names,
        plot_lap_time_deltas,
        plot_speed_vs_tire_age,
        plot_regular_vs_adjusted_degradation,
        plot_fuel_adjusted_degradation,
        plot_fuel_adjusted_percentage_degradation,
        plot_degradation_rate,
        LAP_TIME_IMPROVEMENT_PER_LAP  # Import the default value
    )
    COMPOUND_COLORS = compound_colors
    COMPOUND_NAMES = compound_names
except ImportError as e:
    print(f"Warning: Could not import from N01_tire_prediction: {e}")
    COMPOUND_COLORS = {1: 'red', 2: 'yellow',
                       3: 'white', 4: 'green', 5: 'blue'}
    COMPOUND_NAMES = {1: "Soft", 2: "Medium",
                      3: "Hard", 4: "Intermediate", 5: "Wet"}
    LAP_TIME_IMPROVEMENT_PER_LAP = 0.055  # Use the known default

# -----------------------------------------------------------------------------
# Visualization functions (all assume data is already processed)
# -----------------------------------------------------------------------------


def st_plot_lap_time_deltas(processed_race_data, driver_number=None, lap_range=None):
    """
    Plot lap time deltas for a driver or all drivers.
    Assumes processed_race_data already contains all necessary columns.
    """
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    if lap_range is not None and 'LapNumber' in filtered_data.columns:
        filtered_data = filtered_data[
            (filtered_data['LapNumber'] >= lap_range[0]) &
            (filtered_data['LapNumber'] <= lap_range[1])
        ]
    plt.figure(figsize=(10, 6))
    fig = plot_lap_time_deltas(filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)
    plt.title(
        f"Lap Time Deltas for Driver {driver_number}" if driver_number else "Lap Time Deltas for All Drivers")
    return fig if fig else plt.gcf()


def st_plot_speed_vs_tire_age(processed_race_data, driver_number=None, compound_id=None):
    """
    Plot speed vs tire age for a driver and compound.
    Assumes processed_race_data already contains all necessary columns.
    """
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    if compound_id is None:
        compound_counts = filtered_data['CompoundID'].value_counts()
        compound_id = compound_counts.index[0] if not compound_counts.empty else 2
    plt.figure(figsize=(10, 6))
    plot_speed_vs_tire_age(filtered_data, compound_id,
                           COMPOUND_COLORS, COMPOUND_NAMES)
    compound_name = COMPOUND_NAMES.get(compound_id, f"Unknown ({compound_id})")
    plt.title(
        f"Speed vs Tire Age for Driver {driver_number} - {compound_name} Tires" if driver_number else f"Speed vs Tire Age - {compound_name} Tires")
    return plt.gcf()


def st_plot_regular_vs_adjusted_degradation(processed_race_data, driver_number=None, lap_time_improvement_per_lap=None):
    """
    Plot regular vs fuel-adjusted degradation for a driver or all drivers.
    Assumes processed_race_data already contains all necessary columns.
    """
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    # Use default if not provided
    if lap_time_improvement_per_lap is None:
        lap_time_improvement_per_lap = LAP_TIME_IMPROVEMENT_PER_LAP
    plt.figure(figsize=(12, 8))
    plot_regular_vs_adjusted_degradation(
        filtered_data, COMPOUND_COLORS, COMPOUND_NAMES, lap_time_improvement_per_lap)
    plt.suptitle(
        f"Regular vs Fuel-Adjusted Degradation for Driver {driver_number}" if driver_number else "Regular vs Fuel-Adjusted Degradation for All Drivers", fontsize=14)
    return plt.gcf()


def st_plot_fuel_adjusted_degradation(processed_race_data, driver_number=None):
    """
    Plot fuel-adjusted absolute degradation for a driver or all drivers.
    Assumes processed_race_data already contains all necessary columns.
    """
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    plt.figure(figsize=(10, 6))
    plot_fuel_adjusted_degradation(
        filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)
    plt.title(
        f"Fuel-Adjusted Degradation for Driver {driver_number}" if driver_number else "Fuel-Adjusted Degradation for All Drivers")
    return plt.gcf()


def st_plot_fuel_adjusted_percentage_degradation(processed_race_data, driver_number=None):
    """
    Plot fuel-adjusted percentage degradation for a driver or all drivers.
    Assumes processed_race_data already contains all necessary columns.
    """
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    plt.figure(figsize=(10, 6))
    plot_fuel_adjusted_percentage_degradation(
        filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)
    plt.title(
        f"Fuel-Adjusted Percentage Degradation for Driver {driver_number}" if driver_number else "Fuel-Adjusted Percentage Degradation for All Drivers")
    return plt.gcf()


def st_plot_degradation_rate(processed_race_data, driver_number=None):
    """
    Plot degradation rate for a driver or all drivers.
    Assumes processed_race_data already contains all necessary columns.
    """
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
    plt.figure(figsize=(10, 6))
    plot_degradation_rate(filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)
    plt.title(
        f"Degradation Rate for Driver {driver_number}" if driver_number else "Degradation Rate for All Drivers")
    return plt.gcf()

# -----------------------------------------------------------------------------
# Test function for visualization (uses processed data, does not process or load)
# -----------------------------------------------------------------------------


def test_visualization():
    """
    Test function to verify visualization functions work correctly.
    This function expects processed race data from processing.py.
    """
    try:
        from app.utils.data_loader import load_race_data
        from app.utils.processing import get_processed_race_data

        print("=" * 60)
        print("TESTING VISUALIZATION FUNCTIONS")
        print("=" * 60)

        # Get processed race data (all metrics already calculated)
        race_data = get_processed_race_data()
        print(f"Loaded processed race data with {len(race_data)} rows")

        # Print overall compound distribution
        compound_counts = race_data['CompoundID'].value_counts()
        print(f"Overall compound distribution:")
        for compound_id, count in compound_counts.items():
            compound_name = COMPOUND_NAMES.get(
                compound_id, f"Unknown ({compound_id})")
            print(f"  {compound_name} (ID: {compound_id}): {count} laps")

        # Get all drivers for testing
        drivers = sorted(race_data['DriverNumber'].unique())
        test_drivers = drivers[:3] if len(drivers) >= 3 else drivers

        for test_driver in test_drivers:
            print(f"\n==== Using driver {test_driver} for tests ====")
            driver_data = race_data[race_data['DriverNumber'] == test_driver]
            driver_compounds = driver_data['CompoundID'].unique()
            print(f"Compounds used by driver {test_driver}:")
            for comp_id in driver_compounds:
                comp_name = COMPOUND_NAMES.get(comp_id, f"Unknown ({comp_id})")
                comp_count = len(
                    driver_data[driver_data['CompoundID'] == comp_id])
                print(f"  {comp_name} (ID: {comp_id}): {comp_count} laps")

            print("\nTesting lap time deltas plot...")
            fig1 = st_plot_lap_time_deltas(
                race_data, driver_number=test_driver)
            print("✓ Lap time deltas plot created")
            plt.close(fig1)

            for compound_id in driver_compounds:
                compound_name = COMPOUND_NAMES.get(
                    compound_id, f"Unknown ({compound_id})")
                print(
                    f"\nTesting speed vs tire age plot for {compound_name} tires...")
                fig2 = st_plot_speed_vs_tire_age(
                    race_data, driver_number=test_driver, compound_id=compound_id)
                print(
                    f"✓ Speed vs tire age plot created for {compound_name} tires")
                plt.close(fig2)

            print("\nTesting regular vs adjusted degradation plot...")
            fig3 = st_plot_regular_vs_adjusted_degradation(
                race_data, driver_number=test_driver)
            print("✓ Regular vs adjusted degradation plot created")
            plt.close(fig3)

            print("\nTesting fuel adjusted degradation plot...")
            fig4 = st_plot_fuel_adjusted_degradation(
                race_data, driver_number=test_driver)
            print("✓ Fuel adjusted degradation plot created")
            plt.close(fig4)

            print("\nTesting fuel adjusted percentage degradation plot...")
            fig5 = st_plot_fuel_adjusted_percentage_degradation(
                race_data, driver_number=test_driver)
            print("✓ Fuel adjusted percentage degradation plot created")
            plt.close(fig5)

            print("\nTesting degradation rate plot...")
            fig6 = st_plot_degradation_rate(
                race_data, driver_number=test_driver)
            print("✓ Degradation rate plot created")
            plt.close(fig6)

        print("\nAll visualization tests completed!")

    except Exception as e:
        print(f"Error during visualization testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_visualization()
