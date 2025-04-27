import tempfile
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from experta import Fact, Field, KnowledgeEngine

# Add parent directory to path if needed
sys.path.append(os.path.abspath('../'))


class TelemetryFact(Fact):
    """
    Facts about car telemetry and performance
    """
    lap_time = Field(float, mandatory=False)           # Current lap time
    # Predicted lap time by the model
    predicted_lap_time = Field(float, mandatory=False)
    # Age of the current tire set in laps
    tire_age = Field(int, mandatory=False)
    compound_id = Field(int, mandatory=False)           # Tire type with ID
    position = Field(int, mandatory=False)             # Current race position
    driver_number = Field(int, mandatory=False)         # Driver number


class DegradationFact(Fact):
    """
    Facts about tire degradation including future predictions
    """
    degradation_rate = Field(
        float, mandatory=False)           # Current seconds lost per lap
    # Historical degradation rates
    previous_rates = Field(list, mandatory=False)
    # Percentage degradation adjusted for fuel
    fuel_adjusted_deg_percent = Field(float, mandatory=False)
    # Array of predicted future degradation rates
    predicted_rates = Field(list, mandatory=False)


class GapFact(Fact):
    """
    Facts about gaps to other cars
    """
    driver_number = Field(
        int, mandatory=True)          # Driver this gap data is for
    # Time to car ahead (seconds)
    gap_ahead = Field(float, mandatory=False)
    # Time to car behind (seconds)
    gap_behind = Field(float, mandatory=False)
    # Change in gap ahead over last laps
    gap_ahead_trend = Field(float, mandatory=False)
    # Change in gap behind over last laps
    gap_behind_trend = Field(float, mandatory=False)
    # Driver number of car ahead
    car_ahead = Field(int, mandatory=False)
    # Driver number of car behind
    car_behind = Field(int, mandatory=False)
    # Whether in undercut window (gap < 1.5s)
    in_undercut_window = Field(bool, mandatory=False)
    # Whether in DRS window (gap < 1.0s)
    in_drs_window = Field(bool, mandatory=False)


class RadioFact(Fact):
    """
    Facts from radio communications analysis
    """
    sentiment = Field(str, mandatory=False)  # positive, negative, neutral
    intent = Field(str, mandatory=False)     # WARNING, QUESTION, etc.
    # Detected entities categorized (SITUATION, INCIDENT, PIT_CALL, etc)
    entities = Field(dict, mandatory=False)
    timestamp = Field(float, mandatory=False)  # When the message was received


class RaceStatusFact(Fact):
    """
    Facts about current race status
    """
    lap = Field(int, mandatory=True)               # Current lap
    total_laps = Field(int, mandatory=True)        # Total race laps
    race_phase = Field(str, mandatory=False)       # start, mid, end
    # clear, yellow, safety car, red flag
    track_status = Field(str, mandatory=False)


class StrategyRecommendation(Fact):
    """
    Reccommendation produced by the Expert System
    """
    action = Field(
        str, mandatory=True)                        # Specific action to take
    # Confidende level (0-1)
    confidence = Field(float, mandatory=True)
    # Natural Language Explanation
    explanation = Field(str, mandatory=True)
    # Priority level (higher = more urgent)
    priority = Field(int, mandatory=False, default=0)
    # Lap when reccomendation was made
    lap_issued = Field(int, mandatory=True)


class F1StrategyEngine(KnowledgeEngine):
    """
    Formula 1 strategy expert system engine
    """

    def __init__(self):
        super().__init__()
        self.rules_fired = []  # Tracking the rules that have been activated

    def get_recommendations(self):
        """
        Retrieve all current recommendations, sorted by priority and confidence
        """

        recommendations = []
        for fact in self.facts.values():
            if isinstance(fact, StrategyRecommendation):
                recommendations.append(
                    {
                        "action": fact["action"],
                        "confidence": fact["confidence"],
                        "explanation": fact["explanation"],
                        "priority": fact.get("priority", 0),
                        "lap_issued": fact["lap_issued"]
                    }
                )
        return sorted(
            recommendations,
            key=lambda x: (x["priority"], x["confidence"]),
            reverse=True
        )

    def record_rule_fired(self, rule_name):
        """
        Record when a rule is fired for explanation and debugging
        """

        current_lap = None
        for fact in self.facts.values():
            if isinstance(fact, RaceStatusFact):
                current_lap = fact.get("lap")
                break

        self.rules_fired.append(
            {
                "rule": rule_name,
                "lap": current_lap,
                "timestamp": pd.Timestamp.now()
            }
        )


def transform_tire_predictions(predictions_df, driver_number):
    """
    Transform the output from predict_tire_degradation into facts for the rule engine.

    Args:
        predictions_df (DataFrame): Output from predict_tire_degradation function
        driver_number (int): The driver number to extract data for

    Returns:
        dict: Dictionary with facts to declare
    """
    # Filter data for the specific driver
    driver_data = predictions_df[predictions_df['DriverNumber']
                                 == driver_number]

    if driver_data.empty:
        print(f"Warning: No prediction data found for driver {driver_number}")
        return None

    # Get the most recent stint
    latest_stint = driver_data['Stint'].max()
    stint_data = driver_data[driver_data['Stint'] == latest_stint]

    # Group predictions by current tire age (we want the most recent window)
    latest_age = stint_data['CurrentTyreAge'].max()
    latest_data = stint_data[stint_data['CurrentTyreAge'] == latest_age]

    # Sort predictions by how far in the future they are
    future_data = latest_data.sort_values('LapsAheadPred')

    # Extract future degradation rates
    predicted_rates = future_data['PredictedDegradationRate'].tolist()

    # Get basic info about current state
    current_info = future_data.iloc[0]

    # CORRECTION: Use the first predicted degradation rate as the current rate
    # instead of using 0.0 as a placeholder
    current_degradation_rate = predicted_rates[0] if predicted_rates else 0.0
    print(
        f"Using first predicted rate as current degradation: {current_degradation_rate}")

    # Create degradation fact with current and predicted data
    degradation_fact = DegradationFact(
        # Now using the actual predicted rate
        degradation_rate=current_degradation_rate,
        predicted_rates=predicted_rates  # Array of future predictions
    )

    # Create corresponding telemetry fact
    telemetry_fact = TelemetryFact(
        tire_age=int(latest_age),
        compound_id=int(current_info['CompoundID']),
        driver_number=int(driver_number),
        # Add position if available
        position=int(current_info.get('Position', 0))
    )

    # Add current lap time if available
    if 'CurrentLapTime' in current_info:
        telemetry_fact['lap_time'] = float(current_info['CurrentLapTime'])

    return {
        'degradation': degradation_fact,
        'telemetry': telemetry_fact
    }


def load_tire_predictions(race_data, models_path, compound_thresholds=None):
    """
    Load tire predictions from the prediction module.

    Args:
        race_data (DataFrame): Race telemetry data
        models_path (str): Path to the directory containing model files
        compound_thresholds (dict): Dictionary mapping compound IDs to starting lap numbers
                                  (e.g., {1: 6, 2: 12, 3: 25})

    Returns:
        DataFrame: Tire degradation predictions
    """

    # Import the module
    from pred import N02_model_tire_predictions as tdp

    # Default thresholds based on F1 knowledge if none provided
    if compound_thresholds is None:
        compound_thresholds = {
            1: 6,   # Soft tires: monitor from lap 6 onwards
            2: 12,  # Medium tires: monitor from lap 12 onwards
            3: 25   # Hard tires: monitor from lap 25 onwards
        }

    # Get predictions
    predictions = tdp.predict_tire_degradation(
        race_data,
        models_path,
        compound_start_laps=compound_thresholds
    )

    return predictions


def get_current_degradation(telemetry_data, driver_number):
    """
    Extract current degradation rate from telemetry data.

    Args:
        telemetry_data (DataFrame): Processed telemetry data with DegradationRate
        driver_number (int): Driver number to get data for

    Returns:
        float: Current degradation rate or 0.0 if not available
    """
    # Filter for the specific driver
    driver_data = telemetry_data[telemetry_data['DriverNumber']
                                 == driver_number]

    if driver_data.empty:
        return 0.0

    # Get the most recent lap data
    latest_data = driver_data.sort_values('TyreAge', ascending=False).iloc[0]

    # Return degradation rate if available
    return float(latest_data.get('DegradationRate', 0.0))


def transform_lap_time_predictions(predictions_df, driver_number):
    """
    Transform the output from predict_lap_times into facts for the rule engine.

    Args:
        predictions_df (DataFrame): Output from predict_lap_times function
        driver_number (int): The driver number to extract data for

    Returns:
        dict: Dictionary with facts to declare
    """
    # Filter data for the specific driver
    driver_data = predictions_df[predictions_df['DriverNumber']
                                 == driver_number]

    if driver_data.empty:
        print(
            f"Warning: No lap time prediction data found for driver {driver_number}")
        return None

    # Get the most recent lap data
    latest_lap = driver_data.iloc[-1]

    # Check if this is a next lap prediction (future prediction)
    is_future = latest_lap.get('IsNextLapPrediction', False)

    # Create telemetry fact with current and predicted lap times
    telemetry_fact = TelemetryFact(
        driver_number=int(driver_number),
        # Current lap time if available
        lap_time=float(latest_lap['LapTime']) if not pd.isna(
            latest_lap['LapTime']) else None,
        # Future lap time prediction
        predicted_lap_time=float(latest_lap['PredictedLapTime']),
        # Include other available telemetry data
        compound_id=int(latest_lap.get('CompoundID', 0)),
        tire_age=int(latest_lap.get('TyreAge', 0)),
        position=int(latest_lap.get('Position', 0))
    )

    return {
        'telemetry': telemetry_fact
    }


def load_lap_time_predictions(race_data, model_path=None):
    """
    Load lap time predictions from the prediction module.

    Args:
        race_data (DataFrame): Race telemetry data
        model_path (str): Path to the model file

    Returns:
        DataFrame: Lap time predictions
    """

    # Use a dynamic import to avoid issues if module structure changes
    try:
        # Try importing the module separately
        from pred.N00_model_lap_prediction import predict_lap_times

        # Get predictions
        predictions = predict_lap_times(
            race_data,
            model_path=model_path,
            include_next_lap=True
        )

        return predictions
    except ImportError:
        print("Warning: Could not import lap prediction module.")
        print("Make sure 'lap_prediction_module.py' is in the specified path.")
        return None
    except Exception as e:
        print(f"Error predicting lap times: {str(e)}")
        return None


def transform_radio_analysis(radio_json_path):
    """
    Transform NLP radio analysis into facts.

    Args:
        radio_json_path (str): Path to the JSON file containing radio analysis

    Returns:
        RadioFact: Fact with radio analysis information
    """
    # Load the JSON file
    with open(radio_json_path, 'r') as file:
        radio_data = json.load(file)

    # Extract the analysis section
    analysis = radio_data['analysis']

    # Create the RadioFact
    return RadioFact(
        sentiment=analysis['sentiment'],
        intent=analysis['intent'],
        entities=analysis['entities'],
        timestamp=pd.Timestamp.now().timestamp()
    )


def process_radio_message(message, is_audio=False):
    """
    Process a radio message (text or audio) and get its analysis.

    Args:
        message (str): Text message or path to audio file
        is_audio (bool): Whether the input is an audio file

    Returns:
        str: Path to the JSON file with the analysis
    """
    # Import the radio processing module
    import sys
    import os
    sys.path.append(os.path.abspath('../'))

    try:
        from NLP_radio_processing.NLP_utils import N06_model_merging as radio_nlp

        # If it's audio, first transcribe it
        if is_audio:
            print(f"Transcribing audio from: {message}")
            message_text = radio_nlp.transcribe_audio(message)
            print(f"Transcription: '{message_text}'")
        else:
            message_text = message

        # Analyze the message
        print(f"Analyzing message: '{message_text}'")
        json_path = radio_nlp.analyze_radio_message(message_text)

        return json_path

    except ImportError:
        print("Error: Could not import NLP module. Make sure it's in the correct path.")
        return None
    except Exception as e:
        print(f"Error processing radio message: {str(e)}")
        return None


def analyze_and_transform_radio(message, is_audio=False):
    """
    Complete function to process a radio message and transform it into a fact.

    Args:
        message (str): Text message or path to audio file
        is_audio (bool): Whether the input is an audio file

    Returns:
        RadioFact: Fact with structured radio analysis
    """
    # Step 1: Process the message
    json_path = process_radio_message(message, is_audio)

    if json_path is None:
        print("Failed to process radio message.")
        return None

    # Step 2: Transform the analysis into a fact
    return transform_radio_analysis(json_path)


def calculate_gap_consistency(gaps_df):
    """
    Calculate how many consecutive laps a driver has been in the same gap window.

    This function adds two columns to the dataframe:
    - consistent_gap_ahead_laps: Number of consecutive laps with gap_ahead in the same window
    - consistent_gap_behind_laps: Number of consecutive laps with gap_behind in the same window

    Args:
        gaps_df (DataFrame): DataFrame with gap data

    Returns:
        DataFrame: The same DataFrame with added consistency columns
    """
    print("Calculating gap consistency across laps...")

    # Define the gap windows we care about
    def get_ahead_window(gap):
        if gap < 2.0:
            return "undercut_window"
        elif 2.0 <= gap < 3.5:
            return "overcut_window"
        else:
            return "out_of_range"

    def get_behind_window(gap):
        if gap < 2.0:
            return "defensive_window"
        else:
            return "safe_window"

    # Add window classification columns
    gaps_df['ahead_window'] = gaps_df['GapToCarAhead'].apply(get_ahead_window)
    gaps_df['behind_window'] = gaps_df['GapToCarBehind'].apply(
        get_behind_window)

    # Initialize consistency columns
    gaps_df['consistent_gap_ahead_laps'] = 1
    gaps_df['consistent_gap_behind_laps'] = 1

    # Process each driver separately
    for driver in gaps_df['DriverNumber'].unique():
        driver_data = gaps_df[gaps_df['DriverNumber']
                              == driver].sort_values('LapNumber')

        # Skip if less than 2 laps of data
        if len(driver_data) < 2:
            continue

        # Process consistency of ahead gap
        for i in range(1, len(driver_data)):
            current_idx = driver_data.iloc[i].name
            prev_idx = driver_data.iloc[i-1].name

            if driver_data.iloc[i]['ahead_window'] == driver_data.iloc[i-1]['ahead_window']:
                gaps_df.loc[current_idx, 'consistent_gap_ahead_laps'] = gaps_df.loc[prev_idx,
                                                                                    'consistent_gap_ahead_laps'] + 1

            if driver_data.iloc[i]['behind_window'] == driver_data.iloc[i-1]['behind_window']:
                gaps_df.loc[current_idx, 'consistent_gap_behind_laps'] = gaps_df.loc[prev_idx,
                                                                                     'consistent_gap_behind_laps'] + 1

    print("Gap consistency calculation complete!")
    return gaps_df


def transform_gap_data_with_consistency(gaps_df, driver_number):
    """
    Enhanced version of transform_gap_data that includes consistency metrics

    Args:
        gaps_df (DataFrame): DataFrame with gap data including consistency columns
        driver_number (int): The driver number to extract data for

    Returns:
        GapFact: Fact with enhanced gap information
    """
    # Filter data for the specific driver
    driver_data = gaps_df[gaps_df['DriverNumber'] == driver_number]

    if driver_data.empty:
        print(f"Warning: No gap data found for driver {driver_number}")
        return None

    # Get the most recent gap data
    latest_data = driver_data.sort_values('LapNumber', ascending=False).iloc[0]

    # Handle car ahead number conversion safely
    try:
        car_ahead_val = latest_data.get('CarAheadNumber', 0)
        # If it's "Leader" or another non-numeric string, use a placeholder value
        if isinstance(car_ahead_val, str) and not car_ahead_val.isdigit():
            car_ahead = -1  # Use -1 to represent the leader
        else:
            car_ahead = int(car_ahead_val) if not pd.isna(
                car_ahead_val) else -1
    except (ValueError, TypeError):
        print(
            f"Warning: Could not convert CarAheadNumber '{car_ahead_val}' to int. Using -1.")
        car_ahead = -1

    # Handle car behind number conversion safely
    try:
        car_behind_val = latest_data.get('CarBehindNumber', 0)
        # If it's "Tail" or another non-numeric string, use a placeholder value
        if isinstance(car_behind_val, str) and not car_behind_val.isdigit():
            car_behind = -2  # Use -2 to represent the tail
        else:
            car_behind = int(car_behind_val) if not pd.isna(
                car_behind_val) else -2
    except (ValueError, TypeError):
        print(
            f"Warning: Could not convert CarBehindNumber '{car_behind_val}' to int. Using -2.")
        car_behind = -2

    # Check if consistency columns exist, if not, we need to calculate them
    if 'consistent_gap_ahead_laps' not in latest_data:
        print("Warning: consistency metrics not found in data. Ensure calculate_gap_consistency was called.")

    # Create gap fact with safe type conversions
    gap_fact = GapFact(
        driver_number=int(driver_number),
        gap_ahead=float(latest_data.get('GapToCarAhead', 0.0)),
        gap_behind=float(latest_data.get('GapToCarBehind', 0.0)),
        car_ahead=car_ahead,
        car_behind=car_behind,
        gap_to_leader=float(latest_data.get('GapToLeader', 0.0)),
        consistent_gap_ahead_laps=int(
            latest_data.get('consistent_gap_ahead_laps', 1)),
        consistent_gap_behind_laps=int(
            latest_data.get('consistent_gap_behind_laps', 1)),
        in_undercut_window=bool(latest_data.get('InUndercutWindow', False)),
        in_drs_window=bool(latest_data.get('InDRSWindow', False))
    )

    return gap_fact


def load_gap_data(race_session, window_size=3):
    """
    Load and process gap data from FastF1 session.

    Args:
        race_session: FastF1 session object with loaded data
        window_size (int): Number of laps to calculate trend over

    Returns:
        DataFrame: Processed gap data with relevant metrics
    """
    # Make sure we have the necessary data
    if not hasattr(race_session, 'laps'):
        print("Error: Session does not have lap data loaded")
        return pd.DataFrame()

    # Get all laps data
    laps_df = race_session.laps

    # Initialize our result dataframe
    gaps_data = []

    # Get unique drivers
    drivers = laps_df['DriverNumber'].unique()

    # Process each lap for each driver
    for lap_number in sorted(laps_df['LapNumber'].unique()):
        # Get this lap's data for all drivers
        lap_data = laps_df[laps_df['LapNumber'] == lap_number]

        # Sort by position to find cars ahead and behind
        lap_data = lap_data.sort_values('Position')

        # For each driver, calculate gaps to cars ahead and behind
        for i, driver_lap in lap_data.iterrows():
            driver_number = driver_lap['DriverNumber']
            position = driver_lap['Position']

            # Find car ahead
            car_ahead_data = lap_data[lap_data['Position'] == position - 1]
            car_behind_data = lap_data[lap_data['Position'] == position + 1]

            # Calculate gaps
            gap_to_car_ahead = None
            car_ahead_number = None
            if not car_ahead_data.empty:
                car_ahead_number = car_ahead_data.iloc[0]['DriverNumber']
                # Calculate gap using time difference
                # This is simplified - in real code you might need more complex calculation
                gap_to_car_ahead = driver_lap['LapTime'] - \
                    car_ahead_data.iloc[0]['LapTime']

            gap_to_car_behind = None
            car_behind_number = None
            if not car_behind_data.empty:
                car_behind_number = car_behind_data.iloc[0]['DriverNumber']
                # Similar calculation for car behind
                gap_to_car_behind = car_behind_data.iloc[0]['LapTime'] - \
                    driver_lap['LapTime']

            # Store this data point
            gaps_data.append({
                'DriverNumber': driver_number,
                'LapNumber': lap_number,
                'Position': position,
                'CarAheadNumber': car_ahead_number,
                'CarBehindNumber': car_behind_number,
                'GapToCarAhead': gap_to_car_ahead,
                'GapToCarBehind': gap_to_car_behind,
                'InUndercutWindow': gap_to_car_ahead is not None and gap_to_car_ahead < 1.5,
                'InDRSWindow': gap_to_car_ahead is not None and gap_to_car_ahead < 1.0
            })

    # Convert to DataFrame
    gaps_df = pd.DataFrame(gaps_data)

    # Calculate trends over the specified window
    if not gaps_df.empty:
        # Group by driver
        for driver in drivers:
            driver_data = gaps_df[gaps_df['DriverNumber']
                                  == driver].sort_values('LapNumber')

            # Calculate rolling difference for gap ahead
            if 'GapToCarAhead' in driver_data.columns:
                gaps_df.loc[driver_data.index, 'GapToCarAheadTrend'] = \
                    driver_data['GapToCarAhead'].diff(periods=window_size)

            # Calculate rolling difference for gap behind
            if 'GapToCarBehind' in driver_data.columns:
                gaps_df.loc[driver_data.index, 'GapToCarBehindTrend'] = \
                    driver_data['GapToCarBehind'].diff(periods=window_size)

    return gaps_df


def calculate_race_phase(current_lap, total_laps):
    """Calculate the current phase of the race."""
    percentage = (current_lap / total_laps) * 100
    if percentage < 25:
        return "start"
    elif percentage > 75:
        return "end"
    else:
        return "mid"


if __name__ == "__main__":
    # Create an engine instance
    engine = F1StrategyEngine()
    engine.reset()

    # Example declaring some initial facts
    engine.declare(RaceStatusFact(lap=1, total_laps=60,
                   race_phase="start", track_status="clear"))

    # Print the engine state to verify initialization
    print(f"Engine initialized with {len(engine.facts)} facts")
    facts_list = [f for f in engine.facts.values()]
    print(f"Initial facts: {facts_list}")

    # 1. TIRE DEGRADATION EXAMPLE
    # --------------------------
    print("\n=== TIRE DEGRADATION ANALYSIS ===")

    # Example of transforming model predictions into facts
    mock_degradation_data = pd.DataFrame({
        'DriverNumber': [44, 44, 44],  # Same driver
        'Stint': [1, 1, 1],  # Same stint
        'CurrentTyreAge': [4, 4, 4],  # Same current tire age
        'LapsAheadPred': [1, 2, 3],  # Predictions for 1, 2, and 3 laps ahead
        # Increasing degradation
        'PredictedDegradationRate': [0.07, 0.09, 0.12],
        'CompoundID': [2, 2, 2],  # Medium tires
        'Position': [1, 1, 1],  # Position
        'FuelAdjustedDegPercent': [5.0, 6.0, 7.0]  # Optional
    })

    # Transform degradation data into facts
    tire_facts = transform_tire_predictions(mock_degradation_data, 44)
    if tire_facts:
        engine.declare(tire_facts['degradation'])
        engine.declare(tire_facts['telemetry'])
        print(f"Tire facts declared: {tire_facts}")
    else:
        print("Failed to create tire facts")

    # Count facts after tire data
    print(f"Engine now has {len(engine.facts)} facts")

    # 2. LAP TIME PREDICTION EXAMPLE
    # -----------------------------
    print("\n=== LAP TIME PREDICTION ===")

    # Example lap time data
    mock_lap_time_data = pd.DataFrame({
        'DriverNumber': [44, 44],
        'LapNumber': [3, 4],
        'LapTime': [80.5, 80.3],
        'PredictedLapTime': [80.1, 79.9],
        'CompoundID': [2, 2],
        'TyreAge': [3, 4],
        'Position': [1, 1],
        'IsNextLapPrediction': [False, False]
    })

    # Transform lap time predictions into facts
    lap_facts = transform_lap_time_predictions(mock_lap_time_data, 44)
    if lap_facts:
        engine.declare(lap_facts['telemetry'])
        print(f"Lap time facts declared: {lap_facts}")
    else:
        print("Failed to create lap time facts")

    # Count facts after lap time data
    print(f"Engine now has {len(engine.facts)} facts")

    # 3. RADIO ANALYSIS EXAMPLE
    # -----------------------
    print("\n=== RADIO ANALYSIS ===")

    # Mock radio analysis result (simulating the JSON output)
    mock_radio_json = {
        "message": "Box this lap for softs, there's rain expected in 10 minutes",
        "analysis": {
            "sentiment": "neutral",
            "intent": "ORDER",
            "entities": {
                "ACTION": [],
                "SITUATION": ["rain expected"],
                "INCIDENT": [],
                "STRATEGY_INSTRUCTION": [],
                "POSITION_CHANGE": [],
                "PIT_CALL": ["Box this lap"],
                "TRACK_CONDITION": [],
                "TECHNICAL_ISSUE": [],
                "WEATHER": ["rain"]
            }
        }
    }

    # Save mock JSON to temporary file for processing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(mock_radio_json, tmp)
        tmp_path = tmp.name

    # Transform radio analysis into fact
    radio_fact = transform_radio_analysis(tmp_path)
    if radio_fact:
        engine.declare(radio_fact)
        print(f"Radio fact declared: {radio_fact}")
    else:
        print("Failed to create radio fact")

    # Final fact count
    print(f"Engine now has {len(engine.facts)} facts")

    # Display all facts in engine
    print("\n=== ALL ENGINE FACTS ===")
    for i, fact in enumerate(engine.facts.values()):
        print(f"Fact {i+1}: {type(fact).__name__} - {fact}")
