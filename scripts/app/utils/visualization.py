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
                       3: 'gray', 4: 'green', 5: 'blue'}
    COMPOUND_NAMES = {1: "Soft", 2: "Medium",
                      3: "Hard", 4: "Intermediate", 5: "Wet"}
    LAP_TIME_IMPROVEMENT_PER_LAP = 0.055  # Use the known default

# -----------------------------------------------------------------------------
# Visualization functions (all assume data is already processed)
# -----------------------------------------------------------------------------


# def st_plot_lap_time_deltas(processed_race_data, driver_number=None, lap_range=None):
#     print("DEBUG: Entering st_plot_lap_time_deltas")
#     print(f"DEBUG: DataFrame shape: {processed_race_data.shape}")
#     print(f"DEBUG: Columns: {processed_race_data.columns.tolist()}")
#     if 'CompoundID' in processed_race_data.columns:
#         print(
#             f"DEBUG: Unique CompoundID values: {processed_race_data['CompoundID'].unique()}")
#     if 'DriverNumber' in processed_race_data.columns:
#         print(
#             f"DEBUG: Unique DriverNumber values: {processed_race_data['DriverNumber'].unique()}")
#     filtered_data = processed_race_data.copy()
#     if driver_number is not None:
#         filtered_data = filtered_data[filtered_data['DriverNumber']
#                                       == driver_number]
#         print(
#             f"DEBUG: After filtering by driver {driver_number}, shape: {filtered_data.shape}")
#     if lap_range is not None and 'LapNumber' in filtered_data.columns:
#         filtered_data = filtered_data[
#             (filtered_data['LapNumber'] >= lap_range[0]) &
#             (filtered_data['LapNumber'] <= lap_range[1])
#         ]
#         print(
#             f"DEBUG: After filtering by lap_range {lap_range}, shape: {filtered_data.shape}")
#     if filtered_data.empty:
#         print("WARNING: Filtered DataFrame is empty in st_plot_lap_time_deltas.")
#         return None
#     plt.figure(figsize=(10, 6))
#     fig = plot_lap_time_deltas(filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)
#     plt.title(
#         f"Lap Time Deltas for Driver {driver_number}" if driver_number else "Lap Time Deltas for All Drivers")
#     return fig if fig else plt.gcf()


def st_plot_speed_vs_tire_age(processed_race_data, driver_number=None, compound_id=None):
    print("DEBUG: Entering st_plot_speed_vs_tire_age")
    print(f"DEBUG: DataFrame shape: {processed_race_data.shape}")
    print(f"DEBUG: Columns: {processed_race_data.columns.tolist()}")
    if 'CompoundID' in processed_race_data.columns:
        print(
            f"DEBUG: Unique CompoundID values: {processed_race_data['CompoundID'].unique()}")
    if 'DriverNumber' in processed_race_data.columns:
        print(
            f"DEBUG: Unique DriverNumber values: {processed_race_data['DriverNumber'].unique()}")
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
        print(
            f"DEBUG: After filtering by driver {driver_number}, shape: {filtered_data.shape}")
    if compound_id is None:
        compound_counts = filtered_data['CompoundID'].value_counts()
        compound_id = compound_counts.index[0] if not compound_counts.empty else 2
        print(
            f"DEBUG: No compound_id provided, using most common: {compound_id}")
    else:
        print(f"DEBUG: Using provided compound_id: {compound_id}")
    filtered_data = filtered_data[filtered_data['CompoundID'] == compound_id]
    print(
        f"DEBUG: After filtering by compound_id {compound_id}, shape: {filtered_data.shape}")
    if filtered_data.empty:
        print("WARNING: Filtered DataFrame is empty in st_plot_speed_vs_tire_age.")
        return None
    plt.figure(figsize=(10, 6))
    plot_speed_vs_tire_age(filtered_data, compound_id,
                           COMPOUND_COLORS, COMPOUND_NAMES)
    compound_name = COMPOUND_NAMES.get(compound_id, f"Unknown ({compound_id})")
    plt.title(
        f"Speed vs Tire Age for Driver {driver_number} - {compound_name} Tires" if driver_number else f"Speed vs Tire Age - {compound_name} Tires")
    return plt.gcf()


def st_plot_regular_vs_adjusted_degradation(processed_race_data, driver_number=None, lap_time_improvement_per_lap=None):
    print("DEBUG: Entering st_plot_regular_vs_adjusted_degradation")
    print(f"DEBUG: DataFrame shape: {processed_race_data.shape}")
    print(f"DEBUG: Columns: {processed_race_data.columns.tolist()}")
    if 'CompoundID' in processed_race_data.columns:
        print(
            f"DEBUG: Unique CompoundID values: {processed_race_data['CompoundID'].unique()}")
    if 'DriverNumber' in processed_race_data.columns:
        print(
            f"DEBUG: Unique DriverNumber values: {processed_race_data['DriverNumber'].unique()}")
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
        print(
            f"DEBUG: After filtering by driver {driver_number}, shape: {filtered_data.shape}")
    # Use default if not provided
    if lap_time_improvement_per_lap is None:
        lap_time_improvement_per_lap = LAP_TIME_IMPROVEMENT_PER_LAP
        print(
            f"DEBUG: Using default lap_time_improvement_per_lap: {lap_time_improvement_per_lap}")
    if filtered_data.empty:
        print("WARNING: Filtered DataFrame is empty in st_plot_regular_vs_adjusted_degradation.")
        return None
    plt.figure(figsize=(12, 8))
    plot_regular_vs_adjusted_degradation(
        filtered_data, COMPOUND_COLORS, COMPOUND_NAMES, lap_time_improvement_per_lap)
    plt.suptitle(
        f"Regular vs Fuel-Adjusted Degradation for Driver {driver_number}" if driver_number else "Regular vs Fuel-Adjusted Degradation for All Drivers", fontsize=14)
    return plt.gcf()


def st_plot_fuel_adjusted_degradation(processed_race_data, driver_number=None):
    print("DEBUG: Entering st_plot_fuel_adjusted_degradation")
    print(f"DEBUG: DataFrame shape: {processed_race_data.shape}")
    print(f"DEBUG: Columns: {processed_race_data.columns.tolist()}")
    if 'CompoundID' in processed_race_data.columns:
        print(
            f"DEBUG: Unique CompoundID values: {processed_race_data['CompoundID'].unique()}")
    if 'DriverNumber' in processed_race_data.columns:
        print(
            f"DEBUG: Unique DriverNumber values: {processed_race_data['DriverNumber'].unique()}")
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
        print(
            f"DEBUG: After filtering by driver {driver_number}, shape: {filtered_data.shape}")
    if filtered_data.empty:
        print("WARNING: Filtered DataFrame is empty in st_plot_fuel_adjusted_degradation.")
        return None
    plt.figure(figsize=(10, 6))
    plot_fuel_adjusted_degradation(
        filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)
    plt.title(
        f"Fuel-Adjusted Degradation for Driver {driver_number}" if driver_number else "Fuel-Adjusted Degradation for All Drivers")
    return plt.gcf()


def st_plot_fuel_adjusted_percentage_degradation(processed_race_data, driver_number=None):
    print("DEBUG: Entering st_plot_fuel_adjusted_percentage_degradation")
    print(f"DEBUG: DataFrame shape: {processed_race_data.shape}")
    print(f"DEBUG: Columns: {processed_race_data.columns.tolist()}")
    if 'CompoundID' in processed_race_data.columns:
        print(
            f"DEBUG: Unique CompoundID values: {processed_race_data['CompoundID'].unique()}")
    if 'DriverNumber' in processed_race_data.columns:
        print(
            f"DEBUG: Unique DriverNumber values: {processed_race_data['DriverNumber'].unique()}")
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
        print(
            f"DEBUG: After filtering by driver {driver_number}, shape: {filtered_data.shape}")
    if filtered_data.empty:
        print("WARNING: Filtered DataFrame is empty in st_plot_fuel_adjusted_percentage_degradation.")
        return None
    plt.figure(figsize=(10, 6))
    plot_fuel_adjusted_percentage_degradation(
        filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)
    plt.title(
        f"Fuel-Adjusted Percentage Degradation for Driver {driver_number}" if driver_number else "Fuel-Adjusted Percentage Degradation for All Drivers")
    return plt.gcf()


def st_plot_degradation_rate(processed_race_data, driver_number=None):
    print("DEBUG: Entering st_plot_degradation_rate")
    print(f"DEBUG: DataFrame shape: {processed_race_data.shape}")
    print(f"DEBUG: Columns: {processed_race_data.columns.tolist()}")
    if 'CompoundID' in processed_race_data.columns:
        print(
            f"DEBUG: Unique CompoundID values: {processed_race_data['CompoundID'].unique()}")
    if 'DriverNumber' in processed_race_data.columns:
        print(
            f"DEBUG: Unique DriverNumber values: {processed_race_data['DriverNumber'].unique()}")
    filtered_data = processed_race_data.copy()
    if driver_number is not None:
        filtered_data = filtered_data[filtered_data['DriverNumber']
                                      == driver_number]
        print(
            f"DEBUG: After filtering by driver {driver_number}, shape: {filtered_data.shape}")
    if filtered_data.empty:
        print("WARNING: Filtered DataFrame is empty in st_plot_degradation_rate.")
        return None
    plt.figure(figsize=(10, 6))
    plot_degradation_rate(filtered_data, COMPOUND_COLORS, COMPOUND_NAMES)
    plt.title(
        f"Degradation Rate for Driver {driver_number}" if driver_number else "Degradation Rate for All Drivers")
    return plt.gcf()


# Gap visualization functions for the Gap Analysis view

def st_plot_gap_evolution(gap_data, driver_number=None):
    """
    Plot the evolution of gap to cars ahead and behind throughout the race.

    Args:
        gap_data (pd.DataFrame): Processed gap data with gap measurements
        driver_number (int, optional): Driver to focus analysis on

    Returns:
        matplotlib.figure.Figure: Figure with gap evolution plot
    """
    print("DEBUG: Entering st_plot_gap_evolution")
    print(f"DEBUG: DataFrame shape: {gap_data.shape}")
    print(f"DEBUG: Columns: {gap_data.columns.tolist()}")

    filtered_data = gap_data.copy()

    # Filter for specific driver if provided
    if driver_number is not None:
        if 'DriverNumber' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['DriverNumber']
                                          == driver_number]
            print(
                f"DEBUG: After filtering by driver {driver_number}, shape: {filtered_data.shape}")

    if filtered_data.empty:
        print("WARNING: Filtered DataFrame is empty in st_plot_gap_evolution.")
        return None

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Check for required columns
    gap_ahead_col = 'GapToCarAhead' if 'GapToCarAhead' in filtered_data.columns else None
    gap_behind_col = 'GapToCarBehind' if 'GapToCarBehind' in filtered_data.columns else None
    lap_col = 'LapNumber' if 'LapNumber' in filtered_data.columns else None

    if all([gap_ahead_col, gap_behind_col, lap_col]):
        # Sort data by lap
        filtered_data = filtered_data.sort_values(lap_col)

        # Plot gap to car ahead
        plt.plot(filtered_data[lap_col], filtered_data[gap_ahead_col],
                 'b-', linewidth=2, label='Gap to Car Ahead')

        # Plot gap to car behind
        plt.plot(filtered_data[lap_col], filtered_data[gap_behind_col],
                 'r-', linewidth=2, label='Gap to Car Behind')

        # Add undercut window reference line
        plt.axhline(y=2.0, color='g', linestyle='--',
                    alpha=0.7, label='Undercut Window (2.0s)')

        # Add DRS window reference line
        plt.axhline(y=1.0, color='y', linestyle='--',
                    alpha=0.7, label='DRS Window (1.0s)')

        plt.xlabel('Lap Number')
        plt.ylabel('Gap (seconds)')
        plt.title(
            f'Gap Evolution for Driver #{driver_number}' if driver_number else 'Gap Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Required gap data columns not available',
                 horizontalalignment='center', fontsize=12)

    return plt.gcf()


def st_plot_undercut_opportunities(gap_data, driver_number=None):
    """
    Visualize potential undercut and overcut windows throughout the race.

    Args:
        gap_data (pd.DataFrame): Processed gap data with gap measurements
        driver_number (int, optional): Driver to focus analysis on

    Returns:
        matplotlib.figure.Figure: Figure with undercut opportunity plot
    """
    print("DEBUG: Entering st_plot_undercut_opportunities")
    print(f"DEBUG: DataFrame shape: {gap_data.shape}")
    print(f"DEBUG: Columns: {gap_data.columns.tolist()}")

    filtered_data = gap_data.copy()

    # Filter for specific driver if provided
    if driver_number is not None:
        if 'DriverNumber' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['DriverNumber']
                                          == driver_number]
            print(
                f"DEBUG: After filtering by driver {driver_number}, shape: {filtered_data.shape}")

    if filtered_data.empty:
        print("WARNING: Filtered DataFrame is empty in st_plot_undercut_opportunities.")
        return None

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Check for required columns
    gap_ahead_col = 'GapToCarAhead' if 'GapToCarAhead' in filtered_data.columns else None
    lap_col = 'LapNumber' if 'LapNumber' in filtered_data.columns else None

    if all([gap_ahead_col, lap_col]):
        # Sort data by lap
        filtered_data = filtered_data.sort_values(lap_col)

        # Define zones for different strategies
        y_max = filtered_data[gap_ahead_col].max() + 1

        # Create filled areas for different zones
        plt.fill_between(filtered_data[lap_col], 0, 2.0,
                         color='green', alpha=0.2, label='Undercut Zone (<2.0s)')
        plt.fill_between(filtered_data[lap_col], 2.0, 3.5,
                         color='orange', alpha=0.2, label='Overcut Zone (2.0-3.5s)')
        plt.fill_between(filtered_data[lap_col], 3.5, y_max,
                         color='red', alpha=0.2, label='No Strategy Zone (>3.5s)')

        # Plot the actual gap line
        plt.plot(filtered_data[lap_col], filtered_data[gap_ahead_col],
                 'b-', linewidth=2, label='Gap to Car Ahead')

        # Add reference lines
        plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.7)
        plt.axhline(y=3.5, color='red', linestyle='--', alpha=0.7)

        plt.xlabel('Lap Number')
        plt.ylabel('Gap to Car Ahead (seconds)')
        plt.title(
            f'Undercut/Overcut Opportunities for Driver #{driver_number}' if driver_number else 'Undercut/Overcut Opportunities')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Required gap data columns not available',
                 horizontalalignment='center', fontsize=12)

    return plt.gcf()


def st_plot_gap_consistency(gap_data, driver_number=None):
    """
    Plot the consistency of gaps over multiple laps to identify stable racing conditions.

    Args:
        gap_data (pd.DataFrame): Processed gap data with consistency metrics
        driver_number (int, optional): Driver to focus analysis on

    Returns:
        matplotlib.figure.Figure: Figure with gap consistency plot
    """
    print("DEBUG: Entering st_plot_gap_consistency")
    print(f"DEBUG: DataFrame shape: {gap_data.shape}")
    print(f"DEBUG: Columns: {gap_data.columns.tolist()}")

    filtered_data = gap_data.copy()

    # Filter for specific driver if provided
    if driver_number is not None:
        if 'DriverNumber' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['DriverNumber']
                                          == driver_number]
            print(
                f"DEBUG: After filtering by driver {driver_number}, shape: {filtered_data.shape}")

    if filtered_data.empty:
        print("WARNING: Filtered DataFrame is empty in st_plot_gap_consistency.")
        return None

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Check for required columns
    lap_col = 'LapNumber' if 'LapNumber' in filtered_data.columns else None
    consistent_ahead = 'consistent_gap_ahead_laps' if 'consistent_gap_ahead_laps' in filtered_data.columns else None
    consistent_behind = 'consistent_gap_behind_laps' if 'consistent_gap_behind_laps' in filtered_data.columns else None

    if all([lap_col, consistent_ahead, consistent_behind]):
        # Sort data by lap
        filtered_data = filtered_data.sort_values(lap_col)

        # Plot consistency for gap ahead
        ax1.bar(filtered_data[lap_col], filtered_data[consistent_ahead],
                color='blue', alpha=0.7, label='Consistent Laps Ahead')
        ax1.set_ylabel('Consistent Laps (Gap Ahead)')
        ax1.set_title(
            f'Gap Ahead Consistency for Driver #{driver_number}' if driver_number else 'Gap Ahead Consistency')
        ax1.axhline(y=3, color='green', linestyle='--', alpha=0.7,
                    label='Strategic Threshold (3 laps)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot consistency for gap behind
        ax2.bar(filtered_data[lap_col], filtered_data[consistent_behind],
                color='red', alpha=0.7, label='Consistent Laps Behind')
        ax2.set_xlabel('Lap Number')
        ax2.set_ylabel('Consistent Laps (Gap Behind)')
        ax2.set_title(
            f'Gap Behind Consistency for Driver #{driver_number}' if driver_number else 'Gap Behind Consistency')
        ax2.axhline(y=3, color='green', linestyle='--', alpha=0.7,
                    label='Strategic Threshold (3 laps)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'Required consistency metrics not available',
                 horizontalalignment='center', fontsize=12)

    return fig


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

            # print("\nTesting lap time deltas plot...")
            # fig1 = st_plot_lap_time_deltas(
            #     race_data, driver_number=test_driver)
            # print("✓ Lap time deltas plot created")
            # plt.close(fig1)

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
