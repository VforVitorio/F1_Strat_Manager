import sys
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# MODULE: data_loader.py
# PURPOSE: Load and filter race data and strategy recommendations
#          Handles cases where `outputs` directory sits alongside `scripts`.
# -----------------------------------------------------------------------------

# Resolve current file path and determine project structure
FILE_PATH = Path(__file__).resolve()
# PROJECT_ROOT: scripts directory (two levels up: utils -> app -> scripts)
PROJECT_ROOT = FILE_PATH.parents[2]
# PARENT_DIR: one level above scripts (where `outputs` actually lives)
PARENT_DIR = PROJECT_ROOT.parent

# Determine base directory containing 'outputs':
# Since 'outputs' is alongside 'scripts', check in PARENT_DIR
if (PARENT_DIR / "outputs").exists():
    BASE_DIR = PARENT_DIR
else:
    # Fallback: still set BASE_DIR to PARENT_DIR, errors will be raised on missing files
    print(
        f"Warning: 'outputs' directory not found at expected location: {PARENT_DIR / 'outputs'}")
    BASE_DIR = PARENT_DIR

# Ensure the project root is on sys.path for module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def load_race_data(driver_number: int = None) -> pd.DataFrame:
    """
    Load race lap prediction data and optionally filter by driver number.

    Args:
        driver_number (int, optional): specific driver to filter

    Returns:
        pandas.DataFrame: race data (filtered if driver_number is set)
    """
    # Build path to the race data CSV
    race_data_path = BASE_DIR / "outputs" / "week3" / "lap_prediction_data.csv"

    # Debug info: print where we are looking
    # print(f"Looking for race data at: {race_data_path}")

    # Validate existence of the data file
    if not race_data_path.exists():
        print(f"Error: Race data file not found at {race_data_path}")
        print(f"Base directory: {BASE_DIR}")
        raise FileNotFoundError(
            f"Could not locate race data at {race_data_path}")

    # Load the CSV into a DataFrame
    race_df = pd.read_csv(race_data_path)

    # If driver_number is provided, filter the DataFrame
    if driver_number is not None:
        filtered_df = race_df[race_df['DriverNumber'] == driver_number]
        if filtered_df.empty:
            print(f"Warning: No race data entries for driver {driver_number}")
        return filtered_df

    return race_df


def load_recommendation_data(driver_number: int = None) -> pd.DataFrame:
    """
    Load strategy recommendation data and optionally filter by driver number.

    Args:
        driver_number (int, optional): driver to filter recommendations for

    Returns:
        pandas.DataFrame: strategy recommendations
    """
    # Path to the strategy recommendations CSV
    rec_path = BASE_DIR / "outputs" / "week6" / "spain_gp_recommendations.csv"

    # Debug info: print where we are looking
    # print(f"Looking for recommendation data at: {rec_path}")

    if not rec_path.exists():
        print(f"Error: Recommendations file not found at {rec_path}")
        print(f"Base directory: {BASE_DIR}")
        raise FileNotFoundError(
            f"Could not locate recommendation data at {rec_path}")

    # Load the CSV
    rec_df = pd.read_csv(rec_path)

    # Filter by driver if requested
    if driver_number is not None:
        filtered_rec = rec_df[rec_df['DriverNumber'] == driver_number]
        if filtered_rec.empty:
            print(f"Warning: No recommendations for driver {driver_number}")
        return filtered_rec

    return rec_df


def get_available_drivers() -> list:
    """
    Retrieve a sorted list of unique driver numbers in the race data.

    Returns:
        list[int]: available driver numbers
    """
    try:
        df = load_race_data()
        drivers = sorted(df['DriverNumber'].unique().tolist())
        return drivers
    except Exception as e:
        print(f"Error retrieving drivers: {e}")
        return []


def test_data_loader() -> None:
    """
    Test harness for data loader functions. Prints status and sample outputs.
    """
    print("=" * 60)
    print("RUNNING DATA LOADER TESTS")
    print("=" * 60)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"BASE_DIR for outputs: {BASE_DIR}")
    print(f"Current Working Directory: {Path().resolve()}\n")

    # Test loading race data
    print("- Loading race data...")
    try:
        race_df = load_race_data()
        print(f"  ✓ Race data loaded: {len(race_df)} rows")
    except Exception as e:
        print(f"  ✗ Failed to load race data: {e}")

    # Test driver listing
    print("- Retrieving available drivers...")
    drivers = get_available_drivers()
    print(f"  ✓ Drivers found: {drivers}\n")

    # Test loading recommendations
    print("- Loading recommendation data...")
    try:
        rec_df = load_recommendation_data()
        print(f"  ✓ Recommendations loaded: {len(rec_df)} rows")
    except Exception as e:
        print(f"  ✗ Failed to load recommendations: {e}")

    # Test driver-specific loads if any drivers exist
    if drivers:
        test_driver = drivers[0]
        print(f"- Testing filter for driver {test_driver}...")
        rd = load_race_data(test_driver)
        print(f"  ● Race rows for driver: {len(rd)}")
        rr = load_recommendation_data(test_driver)
        print(f"  ● Recommendation rows for driver: {len(rr)}")


if __name__ == "__main__":
    test_data_loader()
