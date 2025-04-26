# -----------------------------------------------------------------------------
# MODULE: visualization.py
# PURPOSE: Adapts all existing visualization functions from the project for use
#          in the Streamlit app. Ensures consistent plotting across the dashboard.
# -----------------------------------------------------------------------------

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Resolve current file path and determine project structure
FILE_PATH = Path(__file__).resolve()
# PROJECT_ROOT: scripts directory (two levels up: utils -> app -> scripts)
PROJECT_ROOT = FILE_PATH.parents[2]
# PARENT_DIR: one level above scripts (where `outputs` actually lives)
PARENT_DIR = PROJECT_ROOT.parent

# Ensure the project root is on sys.path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import all visualization functions and data processing functions from N01_tire_prediction
try:
    from ML_tyre_pred.ML_utils.N01_tire_prediction import (
        # Constants
        compound_colors,
        compound_names,
        LAP_TIME_IMPROVEMENT_PER_LAP,

        # Data processing functions
        calculate_fuel_adjusted_metrics,
        calculate_degradation_rate,

        # Plotting functions
        plot_lap_time_deltas,
        plot_speed_vs_tire_age,
        plot_regular_vs_adjusted_degradation,
        plot_fuel_adjusted_degradation,
        plot_fuel_adjusted_percentage_degradation,
        plot_degradation_rate
    )

    # Use imported constants for visualization
    COMPOUND_COLORS = compound_colors
    COMPOUND_NAMES = compound_names

    print("Successfully imported tire plot functions from N01_tire_prediction")

except ImportError as e:
    print(f"Warning: Could not import from N01_tire_prediction: {e}")

    # Fallback definitions if import fails
    COMPOUND_COLORS = {
        1: 'red',     # Soft
        2: 'yellow',  # Medium
        3: 'white',   # Hard
        4: 'green',   # Intermediate
        5: 'blue'     # Wet
    }

    COMPOUND_NAMES = {
        1: "Soft",
        2: "Medium",
        3: "Hard",
        4: "Intermediate",
        5: "Wet"
    }


def check_available_compounds(race_data, driver_number=None):
    """
    Check which tire compounds are available in the data

    Args:
        race_data (DataFrame): Race data 
        driver_number (int, optional): Specific driver to check

    Returns:
        dict: Dictionary mapping driver numbers to their available compounds
    """
    if driver_number is not None:
        filtered_data = race_data[race_data['DriverNumber'] == driver_number]
        drivers = [driver_number]
    else:
        filtered_data = race_data
        drivers = sorted(filtered_data['DriverNumber'].unique())

    result = {}

    for driver in drivers:
        driver_data = filtered_data[filtered_data['DriverNumber'] == driver]
        compounds = sorted(driver_data['CompoundID'].unique())
        compound_names = [COMPOUND_NAMES.get(
            c, f"Unknown ({c})") for c in compounds]
        result[driver] = {
            'compound_ids': compounds,
            'compound_names': compound_names
        }

    return result


def ensure_degradation_metrics(race_data):
    """
    Ensure all necessary degradation metrics are calculated in the dataframe,
    following the same pipeline as in the original notebook by using both
    calculate_fuel_adjusted_metrics and calculate_degradation_rate in sequence.

    Args:
        race_data (DataFrame): Race data

    Returns:
        DataFrame: Race data with added degradation metrics if needed
    """
    # Check if key degradation metrics are missing
    required_cols = [
        'FuelAdjustedLapTime', 'TireDegAbsolute', 'TireDegPercent',
        'FuelAdjustedDegAbsolute', 'FuelAdjustedDegPercent', 'DegradationRate'
    ]

    missing_cols = [
        col for col in required_cols if col not in race_data.columns]

    if missing_cols:
        print(f"Calculating missing degradation metrics: {missing_cols}")
        # Follow the pipeline from the notebook - calculate fuel metrics first, then degradation rate
        try:
            # Step 1: Calculate fuel-adjusted metrics
            processed_data = calculate_fuel_adjusted_metrics(race_data)

            # Step 2: Calculate degradation rate on the processed data
            processed_data = calculate_degradation_rate(processed_data)

            print("Successfully calculated all degradation metrics")
            return processed_data
        except Exception as e:
            print(f"Error calculating degradation metrics: {e}")
            return race_data

    return race_data


def st_plot_lap_time_deltas(race_data, driver_number=None, lap_range=None):
    """
    Plot lap time deltas using the existing function, adapted for Streamlit

    Args:
        race_data (DataFrame): Race data containing lap times
        driver_number (int, optional): Specific driver to filter for
        lap_range (tuple, optional): Range of laps to display (start, end)

    Returns:
        matplotlib.figure.Figure: Figure object for Streamlit
    """
    # Filter data if needed
    filtered_data = race_data.copy()

    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]

    if lap_range is not None and 'LapNumber' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['LapNumber'] >= lap_range[0]) &
                                      (filtered_data['LapNumber'] <= lap_range[1])]

    # Check what compounds are available
    compounds = check_available_compounds(filtered_data)
    print(f"Available compounds: {compounds}")

    # Ensure LapTime_Delta column exists
    if 'LapTime_Delta' not in filtered_data.columns:
        # Group by driver and calculate lap time deltas from best lap
        for driver in filtered_data['DriverNumber'].unique():
            driver_data = filtered_data[filtered_data['DriverNumber'] == driver]

            # Process each stint separately
            for stint in driver_data['Stint'].unique():
                stint_data = driver_data[driver_data['Stint'] == stint].sort_values(
                    'TyreAge')

                if len(stint_data) > 0:
                    # Use first lap of stint as baseline
                    first_lap_time = stint_data.iloc[0]['LapTime']
                    indices = stint_data.index
                    filtered_data.loc[indices, 'LapTime_Delta'] = filtered_data.loc[indices,
                                                                                    'LapTime'] - first_lap_time

    # Create figure
    plt.figure(figsize=(10, 6))

    # Use the imported function
    fig = plot_lap_time_deltas(filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)

    # Add a title with driver info
    if driver_number is not None:
        plt.title(f"Lap Time Deltas for Driver {driver_number}")
    else:
        plt.title("Lap Time Deltas for All Drivers")

    return fig if fig else plt.gcf()


def st_plot_speed_vs_tire_age(race_data, driver_number=None, compound_id=None):
    """
    Plot speed vs tire age using the existing function, adapted for Streamlit

    Args:
        race_data (DataFrame): Race data containing speed information
        driver_number (int, optional): Specific driver to filter for
        compound_id (int, optional): Tire compound to analyze (default: use most common compound)

    Returns:
        matplotlib.figure.Figure: Figure object for Streamlit
    """
    # Filter data if needed
    filtered_data = race_data.copy()

    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]

    # Check what compounds are available
    compounds_info = check_available_compounds(filtered_data)
    print(f"Available compounds for speed plot: {compounds_info}")

    # If compound_id not specified, use the most common one in the data
    if compound_id is None:
        # Get all compound counts
        compound_counts = filtered_data['CompoundID'].value_counts()
        if not compound_counts.empty:
            compound_id = compound_counts.index[0]
            print(
                f"Using most common compound: {compound_id} ({COMPOUND_NAMES.get(compound_id, 'Unknown')})")
        else:
            compound_id = 2  # Default to medium if no compounds found

    # Check if required speed columns exist
    required_columns = ['SpeedI1', 'SpeedI2', 'SpeedFL']
    missing_columns = [
        col for col in required_columns if col not in filtered_data.columns]

    if missing_columns:
        # Create a message about missing columns
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Missing required speed columns: {missing_columns}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        return plt.gcf()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Use the imported function
    plot_speed_vs_tire_age(filtered_data, compound_id,
                           COMPOUND_COLORS, COMPOUND_NAMES)

    # Add more detailed title
    compound_name = COMPOUND_NAMES.get(compound_id, f"Unknown ({compound_id})")
    if driver_number is not None:
        plt.title(
            f"Speed vs Tire Age for Driver {driver_number} - {compound_name} Tires")
    else:
        plt.title(f"Speed vs Tire Age - {compound_name} Tires")

    return plt.gcf()


def st_plot_regular_vs_adjusted_degradation(race_data, driver_number=None, lap_time_improvement_per_lap=LAP_TIME_IMPROVEMENT_PER_LAP):
    """
    Plot regular vs fuel-adjusted degradation, adapted for Streamlit

    Args:
        race_data (DataFrame): Race data containing degradation information
        driver_number (int, optional): Specific driver to filter for
        lap_time_improvement_per_lap (float): Estimated lap time improvement per lap due to fuel burn

    Returns:
        matplotlib.figure.Figure: Figure object for Streamlit
    """
    # Filter data if needed
    filtered_data = race_data.copy()

    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]

    # Check what compounds are available
    compounds_info = check_available_compounds(filtered_data)
    print(f"Available compounds for degradation plot: {compounds_info}")

    # Ensure degradation metrics exist
    filtered_data = ensure_degradation_metrics(filtered_data)

    # Check required columns
    required_columns = ['TireDegAbsolute', 'FuelAdjustedDegAbsolute']
    missing_columns = [
        col for col in required_columns if col not in filtered_data.columns]

    if missing_columns:
        # Create a message about missing columns
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f"Missing required degradation columns: {missing_columns}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        return plt.gcf()

    # Create figure
    plt.figure(figsize=(12, 8))

    # Use the imported function
    plot_regular_vs_adjusted_degradation(
        filtered_data, COMPOUND_COLORS, COMPOUND_NAMES, lap_time_improvement_per_lap)

    # Add more detailed title
    if driver_number is not None:
        plt.suptitle(
            f"Regular vs Fuel-Adjusted Degradation for Driver {driver_number}", fontsize=14)
    else:
        plt.suptitle(
            "Regular vs Fuel-Adjusted Degradation for All Drivers", fontsize=14)

    return plt.gcf()


def st_plot_fuel_adjusted_degradation(race_data, driver_number=None):
    """
    Plot fuel-adjusted absolute degradation, adapted for Streamlit

    Args:
        race_data (DataFrame): Race data containing degradation information
        driver_number (int, optional): Specific driver to filter for

    Returns:
        matplotlib.figure.Figure: Figure object for Streamlit
    """
    # Filter data if needed
    filtered_data = race_data.copy()

    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]

    # Check what compounds are available
    compounds_info = check_available_compounds(filtered_data)
    print(f"Available compounds for fuel-adjusted plot: {compounds_info}")

    # Ensure degradation metrics exist
    filtered_data = ensure_degradation_metrics(filtered_data)

    # Check if required column exists
    if 'FuelAdjustedDegAbsolute' not in filtered_data.columns:
        # Create a message about missing column
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Missing required column: FuelAdjustedDegAbsolute",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        return plt.gcf()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Use the imported function
    plot_fuel_adjusted_degradation(
        filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)

    # Add more detailed title
    if driver_number is not None:
        plt.title(f"Fuel-Adjusted Degradation for Driver {driver_number}")
    else:
        plt.title("Fuel-Adjusted Degradation for All Drivers")

    return plt.gcf()


def st_plot_fuel_adjusted_percentage_degradation(race_data, driver_number=None):
    """
    Plot fuel-adjusted percentage degradation, adapted for Streamlit

    Args:
        race_data (DataFrame): Race data containing degradation information
        driver_number (int, optional): Specific driver to filter for

    Returns:
        matplotlib.figure.Figure: Figure object for Streamlit
    """
    # Filter data if needed
    filtered_data = race_data.copy()

    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]

    # Check what compounds are available
    compounds_info = check_available_compounds(filtered_data)
    print(f"Available compounds for percentage plot: {compounds_info}")

    # Ensure degradation metrics exist
    filtered_data = ensure_degradation_metrics(filtered_data)

    # Check if required column exists
    if 'FuelAdjustedDegPercent' not in filtered_data.columns:
        # Create a message about missing column
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Missing required column: FuelAdjustedDegPercent",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        return plt.gcf()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Use the imported function
    plot_fuel_adjusted_percentage_degradation(
        filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)

    # Add more detailed title
    if driver_number is not None:
        plt.title(
            f"Fuel-Adjusted Percentage Degradation for Driver {driver_number}")
    else:
        plt.title("Fuel-Adjusted Percentage Degradation for All Drivers")

    return plt.gcf()


def st_plot_degradation_rate(race_data, driver_number=None):
    """
    Plot degradation rate, adapted for Streamlit

    Args:
        race_data (DataFrame): Race data containing degradation information
        driver_number (int, optional): Specific driver to filter for

    Returns:
        matplotlib.figure.Figure: Figure object for Streamlit
    """
    # Filter data if needed
    filtered_data = race_data.copy()

    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]

    # Check what compounds are available
    compounds_info = check_available_compounds(filtered_data)
    print(f"Available compounds for rate plot: {compounds_info}")

    # Ensure degradation metrics exist
    filtered_data = ensure_degradation_metrics(filtered_data)

    # Check if required column exists
    if 'DegradationRate' not in filtered_data.columns:
        # Create a message about missing column
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Missing required column: DegradationRate",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        return plt.gcf()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Use the imported function
    plot_degradation_rate(filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)

    # Add more detailed title
    if driver_number is not None:
        plt.title(f"Degradation Rate for Driver {driver_number}")
    else:
        plt.title("Degradation Rate for All Drivers")

    return plt.gcf()


def test_visualization():
    """Test function to verify visualization functions work correctly"""
    # Try importing data_loader from relative path
    try:
        from app.utils.data_loader import load_race_data

        print("=" * 60)
        print("TESTING VISUALIZATION FUNCTIONS")
        print("=" * 60)

        # Load race data
        try:
            race_data = load_race_data()
            print(f"Loaded race data with {len(race_data)} rows")

            # Print overall compound distribution
            compound_counts = race_data['CompoundID'].value_counts()
            print(f"Overall compound distribution:")
            for compound_id, count in compound_counts.items():
                compound_name = COMPOUND_NAMES.get(
                    compound_id, f"Unknown ({compound_id})")
                print(f"  {compound_name} (ID: {compound_id}): {count} laps")

            # Get all drivers for testing
            drivers = sorted(race_data['DriverNumber'].unique())

            # Test first 3 drivers
            test_drivers = drivers[:3] if len(drivers) >= 3 else drivers

            for test_driver in test_drivers:
                print(f"\n==== Using driver {test_driver} for tests ====")
                driver_data = race_data[race_data['DriverNumber']
                                        == test_driver]

                # Print available compounds for this driver
                driver_compounds = driver_data['CompoundID'].unique()
                print(f"Compounds used by driver {test_driver}:")
                for comp_id in driver_compounds:
                    comp_name = COMPOUND_NAMES.get(
                        comp_id, f"Unknown ({comp_id})")
                    comp_count = len(
                        driver_data[driver_data['CompoundID'] == comp_id])
                    print(f"  {comp_name} (ID: {comp_id}): {comp_count} laps")

                # Ensure degradation metrics are calculated
                driver_data = ensure_degradation_metrics(driver_data)

                # Test all visualization functions
                print("\nTesting lap time deltas plot...")
                fig1 = st_plot_lap_time_deltas(
                    race_data, driver_number=test_driver)
                print("✓ Lap time deltas plot created")

                for compound_id in driver_compounds:
                    compound_name = COMPOUND_NAMES.get(
                        compound_id, f"Unknown ({compound_id})")
                    print(
                        f"\nTesting speed vs tire age plot for {compound_name} tires...")
                    fig2 = st_plot_speed_vs_tire_age(
                        race_data, driver_number=test_driver, compound_id=compound_id)
                    print(
                        f"✓ Speed vs tire age plot created for {compound_name} tires")

                # Continue with other plots
                print("\nTesting regular vs adjusted degradation plot...")
                fig3 = st_plot_regular_vs_adjusted_degradation(
                    race_data, driver_number=test_driver)
                print("✓ Regular vs adjusted degradation plot created")

                print("\nTesting fuel adjusted degradation plot...")
                fig4 = st_plot_fuel_adjusted_degradation(
                    race_data, driver_number=test_driver)
                print("✓ Fuel adjusted degradation plot created")

                print("\nTesting fuel adjusted percentage degradation plot...")
                fig5 = st_plot_fuel_adjusted_percentage_degradation(
                    race_data, driver_number=test_driver)
                print("✓ Fuel adjusted percentage degradation plot created")

                print("\nTesting degradation rate plot...")
                fig6 = st_plot_degradation_rate(
                    race_data, driver_number=test_driver)
                print("✓ Degradation rate plot created")

            print("\nAll visualization tests completed!")

        except Exception as e:
            print(f"Error during visualization testing: {e}")
            import traceback
            traceback.print_exc()

    except ImportError as e:
        print(f"Error importing data_loader: {e}")


if __name__ == "__main__":
    test_visualization()
