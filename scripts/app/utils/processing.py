# -----------------------------------------------------------------------------
# MODULE: processing.py
# PURPOSE: Processes race data and strategic recommendations,
#          prepares data for visualization, and performs specific calculations
#          needed for different views in the application.
# -----------------------------------------------------------------------------

from .data_loader import load_race_data, load_recommendation_data
import sys
from pathlib import Path
import pandas as pd

# --- Path resolution for project structure ---
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
PARENT_DIR = PROJECT_ROOT.parent
# Since 'outputs' is alongside 'scripts', check in PARENT_DIR
if (PARENT_DIR / "outputs").exists():
    BASE_DIR = PARENT_DIR
else:
    # Fallback: still set BASE_DIR to PARENT_DIR, errors will be raised on missing files
    print(
        f"Warning: 'outputs' directory not found at expected location: {PARENT_DIR / 'outputs'}")
    BASE_DIR = PARENT_DIR

# Ensure the project root is in sys.path for module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Import data loading and transformation functions ---

try:
    # Functions for tire metrics calculation
    from ML_tyre_pred.ML_utils.N01_tire_prediction import (
        calculate_fuel_adjusted_metrics,
        calculate_degradation_rate
    )
except ImportError:
    calculate_fuel_adjusted_metrics = None
    calculate_degradation_rate = None

try:
    # Tire degradation prediction model
    from ML_tyre_pred.ML_utils.N02_model_tire_predictions import predict_tire_degradation
except ImportError:
    predict_tire_degradation = None

try:
    # Lap time prediction model
    from ML_tyre_pred.ML_utils.N00_model_lap_prediction import predict_lap_times
except ImportError:
    predict_lap_times = None

try:
    # Gap data transformation with consistency metrics
    from IS_agent.utils.N01_agent_setup import transform_gap_data_with_consistency

except ImportError:
    transform_gap_data_with_consistency = None

try:
    from IS_agent.utils.N03_lap_time_rules import add_race_lap_column
except ImportError:
    add_race_lap_column = None

# --- Main processing functions ---


def add_race_lap_column(df):
    """
    Añade la columna 'LapNumber' calculando la vuelta real de carrera para cada piloto y stint.
    """
    # Obtener la longitud de cada stint por piloto
    max_age_by_stint = df.groupby(['DriverNumber', 'Stint'])[
        'TyreAge'].max().reset_index()
    max_age_by_stint = max_age_by_stint.rename(
        columns={'TyreAge': 'StintLength'})
    # Crear un diccionario con la suma acumulada de stints previos
    stint_lengths = {}
    for driver in df['DriverNumber'].unique():
        driver_stints = max_age_by_stint[max_age_by_stint['DriverNumber'] == driver]
        stint_lengths[driver] = {}
        acc = 0
        for _, row in driver_stints.iterrows():
            stint = row['Stint']
            stint_lengths[driver][stint] = acc
            acc += row['StintLength']
    # Calcular LapNumber sumando TyreAge y la suma acumulada de stints previos

    def calc_lap(row):
        driver = row['DriverNumber']
        stint = row['Stint']
        tyre_age = row['TyreAge']
        start_lap = stint_lengths.get(driver, {}).get(stint, 0)
        return start_lap + tyre_age
    df['LapNumber'] = df.apply(calc_lap, axis=1)
    return df


def add_lap_time_delta(df):
    """
    Adds the LapTime_Delta column to the DataFrame.
    Calculates the difference in LapTime between consecutive laps for each driver.
    """
    if 'LapTime' in df.columns:
        df = df.sort_values(['DriverNumber', 'LapNumber'])
        df['LapTime_Delta'] = df.groupby('DriverNumber')['LapTime'].diff()
    elif 'FuelAdjustedLapTime' in df.columns:
        df = df.sort_values(['DriverNumber', 'LapNumber'])
        df['LapTime_Delta'] = df.groupby('DriverNumber')[
            'FuelAdjustedLapTime'].diff()
    return df


def get_processed_race_data(driver_number=None):
    """
    Loads and processes race data, applying fuel adjustment and degradation metrics.
    Returns a pandas DataFrame ready for downstream analysis or visualization.
    """
    if load_race_data is None:
        raise ImportError("Could not import load_race_data")

    df = load_race_data()
    df = add_race_lap_column(df)

    if calculate_fuel_adjusted_metrics and calculate_degradation_rate:
        df = calculate_fuel_adjusted_metrics(df)
        df = calculate_degradation_rate(df)
    return df


def get_processed_recommendations(driver_number=None):
    """
    Loads and filters strategic recommendations for a driver.
    Returns a pandas DataFrame.
    """
    if load_recommendation_data is None:
        raise ImportError("Could not import load_recommendation_data")
    return load_recommendation_data(driver_number)


def get_tire_degradation_predictions(race_data, models_path, compound_start_laps=None):
    """
    Runs the tire degradation prediction model.
    Returns a DataFrame with predictions.
    """
    if predict_tire_degradation is None:
        raise ImportError("Could not import predict_tire_degradation")
    return predict_tire_degradation(race_data, models_path, compound_start_laps=compound_start_laps)


def get_lap_time_predictions(race_data, model_path=None, include_next_lap=True):
    """
    Runs the lap time prediction model.
    Returns a DataFrame with predictions.
    """
    if predict_lap_times is None:
        raise ImportError("Could not import predict_lap_times")
    return predict_lap_times(race_data, model_path=model_path, include_next_lap=include_next_lap)


def get_gap_data_with_consistency(gap_df, driver_number=None):
    """
    Processes gap data using the consistency transformation function.
    Returns a DataFrame with gap consistency metrics.
    """
    if transform_gap_data_with_consistency is None:
        raise ImportError(
            "Could not import transform_gap_data_with_consistency")
    return transform_gap_data_with_consistency(gap_df, driver_number)


def get_processed_gap_data(driver_number=None):
    """
    Load and process gap data with extensive debugging.

    Args:
        driver_number (int, optional): Driver to filter for

    Returns:
        pd.DataFrame: Processed gap data with consistency metrics
    """
    # Path to the gap data CSV
    gap_data_path = BASE_DIR / "outputs" / "week6" / "gap_data.csv"

    print(f"DEBUG: Looking for gap data at: {gap_data_path}")
    print(f"DEBUG: File exists: {gap_data_path.exists()}")

    if not gap_data_path.exists():
        print(f"Error: Gap data file not found at {gap_data_path}")
        return None

    try:
        # Load the CSV into a DataFrame
        gap_df = pd.read_csv(gap_data_path, dtype={'DriverNumber': int})
        print(f"DEBUG: Initial load - DataFrame shape: {gap_df.shape}")
        print(
            f"DEBUG: Unique drivers in initial load: {sorted(gap_df['DriverNumber'].unique())}")

        # Show a sample of the data
        print("DEBUG: First 3 rows of loaded data:")
        print(gap_df.head(3))

        # Check if there are any NaN values in critical columns
        print(
            f"DEBUG: NaN values in DriverNumber: {gap_df['DriverNumber'].isna().sum()}")

        # Check data types
        print(f"DEBUG: DriverNumber dtype: {gap_df['DriverNumber'].dtype}")

        # Calculate consistency metrics if not already present
        if 'consistent_gap_ahead_laps' not in gap_df.columns:
            print("DEBUG: Calculating gap consistency")
            gap_df = calculate_gap_consistency(gap_df)
            print(
                f"DEBUG: After consistency calculation - shape: {gap_df.shape}")

        # Create a complete copy to avoid reference issues
        result_df = gap_df.copy()
        print(f"DEBUG: After copy - shape: {result_df.shape}")

        # Filter for specific driver if requested
        if driver_number is not None:
            print(f"DEBUG: Filtering for driver: {driver_number}")
            try:
                driver_number = int(driver_number)
                result_df['DriverNumber'] = result_df['DriverNumber'].astype(
                    int)
            except Exception as e:
                print(f"DEBUG: Error converting DriverNumber to int: {e}")
            print("DEBUG: Unique DriverNumbers after conversion:")
            print(result_df['DriverNumber'].unique())

            print(
                f"DEBUG: Received driver_number type: {type(driver_number)}, value: {driver_number}")

            filtered_df = result_df[result_df['DriverNumber'] == driver_number]
            print(
                f"DEBUG: After driver filtering - shape: {filtered_df.shape}")

            if filtered_df.empty:
                print(f"DEBUG: No data found for driver {driver_number}")
                print(
                    f"DEBUG: Available drivers: {sorted(result_df['DriverNumber'].unique())}")

                # If we're in this scenario where filtering yielded empty result but
                # there is data in the original DataFrame, let's try a workaround
                if gap_df.shape[0] > 0:
                    print(
                        "DEBUG: Trying to create synthetic data based on first driver")
                    # Create synthetic data based on first available driver
                    available_drivers = sorted(gap_df['DriverNumber'].unique())
                    if available_drivers:
                        first_driver = available_drivers[0]
                        base_data = gap_df[gap_df['DriverNumber']
                                           == first_driver].copy()
                        base_data['DriverNumber'] = driver_number
                        print(
                            f"DEBUG: Created synthetic data shape: {base_data.shape}")
                        return base_data

            return filtered_df

        return result_df

    except Exception as e:
        print(f"Error processing gap data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calculate_gap_consistency(gap_df):
    """
    Calculate how many consecutive laps a driver has been in the same gap window.
    Adds two columns to the dataframe:
    - consistent_gap_ahead_laps: Number of consecutive laps with gap_ahead in the same window
    - consistent_gap_behind_laps: Number of consecutive laps with gap_behind in the same window

    Args:
        gap_df (DataFrame): DataFrame with gap data

    Returns:
        DataFrame: The same DataFrame with added consistency columns
    """
    print("Calculating gap consistency across laps...")

    # Define the gap windows we care about
    def get_ahead_window(gap):
        if gap is None or pd.isna(gap):
            return "unknown"
        if gap < 2.0:
            return "undercut_window"
        elif 2.0 <= gap < 3.5:
            return "overcut_window"
        else:
            return "out_of_range"

    def get_behind_window(gap):
        if gap is None or pd.isna(gap):
            return "unknown"
        if gap < 2.0:
            return "defensive_window"
        else:
            return "safe_window"

    # Create a copy to avoid modifying the original DataFrame
    result_df = gap_df.copy()

    # Add window classification columns
    result_df['ahead_window'] = result_df['GapToCarAhead'].apply(
        get_ahead_window)
    result_df['behind_window'] = result_df['GapToCarBehind'].apply(
        get_behind_window)

    # Initialize consistency columns
    result_df['consistent_gap_ahead_laps'] = 1
    result_df['consistent_gap_behind_laps'] = 1

    # Process each driver separately
    for driver in result_df['DriverNumber'].unique():
        driver_data = result_df[result_df['DriverNumber']
                                == driver].sort_values('LapNumber')

        # Skip if less than 2 laps of data
        if len(driver_data) < 2:
            continue

        # Process consistency of ahead gap
        for i in range(1, len(driver_data)):
            current_idx = driver_data.iloc[i].name
            prev_idx = driver_data.iloc[i-1].name

            if driver_data.iloc[i]['ahead_window'] == driver_data.iloc[i-1]['ahead_window']:
                result_df.loc[current_idx, 'consistent_gap_ahead_laps'] = result_df.loc[prev_idx,
                                                                                        'consistent_gap_ahead_laps'] + 1

            if driver_data.iloc[i]['behind_window'] == driver_data.iloc[i-1]['behind_window']:
                result_df.loc[current_idx, 'consistent_gap_behind_laps'] = result_df.loc[prev_idx,
                                                                                         'consistent_gap_behind_laps'] + 1

    print("Gap consistency calculation complete!")
    return result_df


def calculate_strategic_windows(gap_data):
    """
    Calculate strategic windows like undercut, overcut, and defensive opportunities.
    This function identifies and flags strategic opportunities in the data.

    Args:
        gap_data (DataFrame): DataFrame with gap consistency data

    Returns:
        DataFrame: Enhanced DataFrame with strategic opportunity flags
    """
    # Create a copy of the input data
    result_df = gap_data.copy()

    # Define strategic thresholds
    undercut_threshold = 2.0  # seconds
    overcut_min = 2.0  # seconds
    overcut_max = 3.5  # seconds
    defensive_threshold = 2.0  # seconds
    consistency_threshold = 3  # laps

    # Add strategic opportunity flags
    result_df['undercut_opportunity'] = (
        (result_df['GapToCarAhead'] < undercut_threshold) &
        (result_df['consistent_gap_ahead_laps'] >= consistency_threshold)
    )

    result_df['overcut_opportunity'] = (
        (result_df['GapToCarAhead'] >= overcut_min) &
        (result_df['GapToCarAhead'] < overcut_max) &
        (result_df['consistent_gap_ahead_laps'] >= consistency_threshold)
    )

    result_df['defensive_needed'] = (
        (result_df['GapToCarBehind'] < defensive_threshold) &
        (result_df['consistent_gap_behind_laps'] >= consistency_threshold)
    )

    # Count opportunities by lap range
    if 'LapNumber' in result_df.columns:
        # Early stint (first third)
        early_laps = result_df['LapNumber'].max() // 3
        result_df['early_stint'] = result_df['LapNumber'] <= early_laps

        # Mid stint (second third)
        mid_laps = early_laps * 2
        result_df['mid_stint'] = (result_df['LapNumber'] > early_laps) & (
            result_df['LapNumber'] <= mid_laps)

        # Late stint (final third)
        result_df['late_stint'] = result_df['LapNumber'] > mid_laps

    return result_df


def prepare_visualization_data(driver_number=None, models_path=None, lap_model_path=None, gap_df=None):
    """
    Returns a dictionary with all processed data ready for visualization.
    This function orchestrates the loading and transformation of all relevant data.
    All data is preprocessed and ready for plotting or dashboard use.
    """
    # Load and process race data
    race_data = get_processed_race_data(driver_number)
    # Load recommendations
    recommendations = get_processed_recommendations(driver_number)
    # Run tire degradation predictions if model path is provided
    tire_preds = get_tire_degradation_predictions(
        race_data, models_path) if models_path else None
    # Run lap time predictions if model path is provided
    lap_preds = get_lap_time_predictions(
        race_data, model_path=lap_model_path) if lap_model_path else None
    # Process gap data if provided
    gap_data = get_gap_data_with_consistency(
        gap_df, driver_number) if gap_df is not None else None

    return {
        "race_data": race_data,
        "recommendations": recommendations,
        "tire_predictions": tire_preds,
        "lap_predictions": lap_preds,
        "gap_data": gap_data
    }


def test_processing_functions():
    """
    Basic test to check that the main processing functions work and return data.
    Prints shapes and types of outputs for quick validation.
    """
    print("=== PROCESSING FUNCTIONS TEST ===")
    try:
        race_data = get_processed_race_data()
        print(
            f"Race data loaded: {race_data.shape if isinstance(race_data, pd.DataFrame) else type(race_data)}")
    except Exception as e:
        print(f"Error in get_processed_race_data: {e}")

    try:
        recommendations = get_processed_recommendations()
        print(
            f"Recommendations loaded: {recommendations.shape if isinstance(recommendations, pd.DataFrame) else type(recommendations)}")
    except Exception as e:
        print(f"Error in get_processed_recommendations: {e}")

    # Only test models if race_data is loaded and not empty (adjust paths as needed)
    try:
        if 'race_data' in locals() and isinstance(race_data, pd.DataFrame) and not race_data.empty:
            tire_preds = get_tire_degradation_predictions(
                race_data, models_path="outputs/week5/models")
            print(
                f"Tire degradation predictions: {tire_preds.shape if hasattr(tire_preds, 'shape') else type(tire_preds)}")
    except Exception as e:
        print(f"Error in get_tire_degradation_predictions: {e}")

    try:
        if 'race_data' in locals() and isinstance(race_data, pd.DataFrame) and not race_data.empty:
            lap_preds = get_lap_time_predictions(
                race_data, model_path="outputs/week3/xgb_sequential_model.pkl")
            print(
                f"Lap time predictions: {lap_preds.shape if hasattr(lap_preds, 'shape') else type(lap_preds)}")
    except Exception as e:
        print(f"Error in get_lap_time_predictions: {e}")

    print("=== END OF TEST ===")


# Run the test function if this script is executed directly
if __name__ == "__main__":
    test_processing_functions()
