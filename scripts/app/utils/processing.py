# -----------------------------------------------------------------------------
# MODULE: processing.py
# PURPOSE: Processes race data and strategic recommendations,
#          prepares data for visualization, and performs specific calculations
#          needed for different views in the application.
# -----------------------------------------------------------------------------

import sys
from pathlib import Path
import pandas as pd

# --- Path resolution for project structure ---
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
PARENT_DIR = PROJECT_ROOT.parent

# Ensure the project root is in sys.path for module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Import data loading and transformation functions ---
try:
    # Functions to load race data and recommendations
    from app.utils.data_loader import load_race_data, load_recommendation_data
except ImportError:
    load_race_data = None
    load_recommendation_data = None

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

# --- Main processing functions ---


def get_processed_race_data(driver_number=None):
    """
    Loads and processes race data, applying fuel adjustment and degradation metrics.
    Optionally filters by driver_number.
    Returns a pandas DataFrame ready for downstream analysis or visualization.
    """
    if load_race_data is None:
        raise ImportError("Could not import load_race_data")
    df = load_race_data(driver_number)
    # Apply fuel adjustment and degradation calculations if available
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
