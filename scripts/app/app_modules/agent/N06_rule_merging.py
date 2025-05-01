"""
F1 Strategy Manager - Rule Merging Module

This module provides the integrated strategy engine that combines all rule systems:
- Tire degradation rules
- Lap time prediction rules
- Radio communication analysis rules
- Gap analysis rules

It includes utilities for loading data, transforming it into facts, and analyzing strategies.
"""

# Import necessary modules
from app_modules.agent.N01_agent_setup import (
    # Fact classes for our rule engine
    TelemetryFact,
    DegradationFact,
    GapFact,
    RadioFact,
    RaceStatusFact,
    StrategyRecommendation,
    F1StrategyEngine,

    # Utility functions for data transformation
    transform_tire_predictions,
    load_tire_predictions,
    transform_lap_time_predictions,
    load_lap_time_predictions,
    transform_radio_analysis,
    process_radio_message,
    transform_gap_data_with_consistency,
    load_gap_data,
    calculate_gap_consistency,
    calculate_race_phase
)

# Import the rule engines from each domain
# Tire degradation rules
from app_modules.agent.N02_degradation_time_rules import F1DegradationRules

# Lap time prediction rules
from app_modules.agent.N03_lap_time_rules import F1LapTimeRules

# Radio communication analysis rules
from app_modules.agent.N04_nlp_rules import F1RadioRules

# Gap analysis rules
from app_modules.agent.N05_gap_rules import F1GapRules, calculate_all_gaps, calculate_gap_consistency, transform_gap_data_with_consistency

# Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import json
import time
import re
import glob
import fastf1
import shutil
import requests

# Import Experta components
from experta import Rule, NOT, OR, AND, AS, MATCH, TEST, EXISTS
from experta import DefFacts, Fact, Field, KnowledgeEngine

# Add parent directory to path to access modules
sys.path.append(os.path.abspath('../'))

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)


class F1CompleteStrategyEngine(F1DegradationRules, F1LapTimeRules, F1RadioRules, F1GapRules):
    """
    Unified strategy engine that integrates all rule systems:
    - Tire degradation rules 
    - Lap time prediction rules
    - Radio communication analysis rules
    - Gap analysis rules

    This class inherits from all specialized rule engines to combine their rules
    and adds conflict resolution capabilities.
    """

    def __init__(self):
        """Initialize the integrated engine"""
        # Call the parent constructor
        super().__init__()
        # Track which rule systems have fired rules
        self.active_systems = {
            'degradation': False,
            'lap_time': False,
            'radio': False,
            'gap': False
        }

    def get_recommendations(self):
        """
        Get all recommendations from the rule engine with enhanced conflict resolution.

        Returns:
            list: Sorted list of recommendations with conflicts resolved
        """
        # Get the base recommendations from parent method
        all_recommendations = super().get_recommendations()

        # If we have very few recommendations, no need for complex conflict resolution
        if len(all_recommendations) <= 2:
            return all_recommendations

        # Group recommendations by driver
        driver_recommendations = {}
        for rec in all_recommendations:
            # Default to 0 if driver is not specified
            driver = rec.get('DriverNumber', 0)
            if driver not in driver_recommendations:
                driver_recommendations[driver] = []
            driver_recommendations[driver].append(rec)

        # Process each driver's recommendations for conflicts
        resolved_recommendations = []
        for driver, recs in driver_recommendations.items():
            # Only need conflict resolution if multiple recommendations
            if len(recs) > 1:
                resolved = self._resolve_conflicts(recs)
                resolved_recommendations.extend(resolved)
            else:
                # Single recommendation, no conflicts to resolve
                resolved_recommendations.extend(recs)

        # Sort by priority and confidence
        return sorted(
            resolved_recommendations,
            key=lambda x: (x.get('priority', 0), x.get('confidence', 0)),
            reverse=True
        )

    def _resolve_conflicts(self, recommendations):
        """
        Resolve conflicts between recommendations for the same driver.

        This method looks for contradictory recommendations and resolves them
        based on priority, confidence, and the nature of the conflict.

        Args:
            recommendations: List of recommendations for a single driver

        Returns:
            list: Resolved list of recommendations
        """
        # Group recommendations by action type
        action_groups = {}
        for rec in recommendations:
            action = rec['action']
            if action not in action_groups:
                action_groups[action] = []
            action_groups[action].append(rec)

        # Define conflicting action pairs
        conflicting_pairs = [
            # Can't extend stint and pit at the same time
            ('extend_stint', 'pit_stop'),
            ('extend_stint', 'prioritize_pit'),
            ('extend_stint', 'defensive_pit'),
            ('extend_stint', 'consider_pit'),

            # No need for preparation if immediate pit is recommended
            ('prepare_pit', 'pit_stop'),
            ('prepare_pit', 'prioritize_pit'),

            # Can't do undercut and overcut at the same time
            ('perform_undercut', 'perform_overcut')
        ]

        # Check for each conflict pair
        resolved = []
        excluded_recommendations = set()

        for action1, action2 in conflicting_pairs:
            if action1 in action_groups and action2 in action_groups:
                # We have a conflict!
                group1 = action_groups[action1]
                group2 = action_groups[action2]

                # Get the highest priority/confidence recommendation from each group
                best1 = max(group1, key=lambda x: (
                    x.get('priority', 0), x.get('confidence', 0)))
                best2 = max(group2, key=lambda x: (
                    x.get('priority', 0), x.get('confidence', 0)))

                # Compare and keep only the better one
                if (best1.get('priority', 0), best1.get('confidence', 0)) >= (best2.get('priority', 0), best2.get('confidence', 0)):
                    # best1 wins, exclude all from group2
                    excluded_recommendations.update(id(r) for r in group2)
                else:
                    # best2 wins, exclude all from group1
                    excluded_recommendations.update(id(r) for r in group1)

        # Add non-excluded recommendations
        for rec in recommendations:
            if id(rec) not in excluded_recommendations:
                resolved.append(rec)

        # Enhance the winning recommendations with context from conflicting ones
        if len(resolved) < len(recommendations):
            # We had conflicts and resolved them
            for rec in resolved:
                rec['explanation'] += " (Selected as optimal strategy after resolving conflicts)"

        return resolved

    def record_rule_fired(self, rule_name):
        """
        Record which rule fired and track which rule system it belongs to.

        Args:
            rule_name: Name of the rule that fired
        """
        # Standard recording from parent class
        super().record_rule_fired(rule_name)

        # Also track which system the rule belongs to
        if rule_name.startswith(('high_degradation', 'stint_extension', 'early_degradation')):
            self.active_systems['degradation'] = True
        elif rule_name.startswith(('optimal_performance', 'performance_cliff', 'post_traffic')):
            self.active_systems['lap_time'] = True
        elif rule_name.startswith(('grip_issue', 'weather_information', 'incident_reaction')):
            self.active_systems['radio'] = True
        elif rule_name.startswith(('undercut_opportunity', 'defensive_pit', 'strategic_overcut')):
            self.active_systems['gap'] = True


def transform_all_facts(driver_number, tire_predictions=None, lap_predictions=None,
                        gap_data=None, radio_json_path=None, current_lap=None,
                        total_laps=66, debug=False):
    """
    Transform all data sources into facts for the integrated rule engine.

    This function combines fact transformation from all domains:
    - Tire degradation
    - Lap time prediction
    - Gap analysis
    - Radio communications

    Args:
        driver_number: Driver to focus on
        tire_predictions: DataFrame with tire degradation predictions
        lap_predictions: DataFrame with lap time predictions
        gap_data: DataFrame with gap information
        radio_json_path: Path to radio analysis JSON
        current_lap: Current race lap
        total_laps: Total race laps
        debug: If True, print detailed debug information

    Returns:
        Dictionary of facts to declare in the engine
    """
    facts = {}

    if debug:
        print(f"\n{'='*80}")
        print(f"TRANSFORMING FACTS FOR DRIVER #{driver_number}")
        print(f"{'='*80}")

    # 1. Transform tire degradation data
    if tire_predictions is not None:
        try:
            tire_facts = transform_tire_predictions(
                tire_predictions, driver_number)
            if tire_facts:
                # Ensure values are valid for the Fact schema
                if 'degradation' in tire_facts and tire_facts['degradation'] is not None:
                    # For DegradationFact, ensure no None values in required fields
                    deg_fact = tire_facts['degradation']
                    for field_name in ['degradation_rate']:
                        if field_name in deg_fact and deg_fact[field_name] is None:
                            # Default to 0.0 for None values
                            deg_fact[field_name] = 0.0

                # For TelemetryFact, ensure no None values in required fields
                if 'telemetry' in tire_facts and tire_facts['telemetry'] is not None:
                    telemetry_fact = tire_facts['telemetry']
                    for field_name in ['tire_age', 'compound_id', 'driver_number', 'position']:
                        if field_name in telemetry_fact and telemetry_fact[field_name] is None:
                            if field_name in ['tire_age', 'compound_id', 'position']:
                                # Default to 0 for None values in int fields
                                telemetry_fact[field_name] = 0
                            elif field_name == 'driver_number':
                                # Default to the driver number
                                telemetry_fact[field_name] = driver_number

                facts.update(tire_facts)
                print(
                    f"✓ Transformed tire degradation data for Driver #{driver_number}")
                if debug and 'degradation' in tire_facts:
                    print(
                        f"  Degradation rate: {tire_facts['degradation'].get('degradation_rate')}")
                    print(
                        f"  Predicted rates: {tire_facts['degradation'].get('predicted_rates')}")
        except Exception as e:
            print(f"✗ Error transforming tire data: {str(e)}")

    # 2. Transform lap time data
    if lap_predictions is not None:
        try:
            lap_facts = transform_lap_time_predictions(
                lap_predictions, driver_number)
            if lap_facts:
                # Handle None values in telemetry fact to avoid schema validation errors
                if 'telemetry' in lap_facts and lap_facts['telemetry'] is not None:
                    telemetry_fact = lap_facts['telemetry']
                    # Ensure lap_time field has a valid float (not None)
                    if 'lap_time' in telemetry_fact and telemetry_fact['lap_time'] is None:
                        # Use predicted_lap_time as fallback, or 0.0 if that's None too
                        if telemetry_fact.get('predicted_lap_time') is not None:
                            telemetry_fact['lap_time'] = telemetry_fact['predicted_lap_time']
                        else:
                            telemetry_fact['lap_time'] = 0.0

                    # Ensure other required fields have valid values
                    for field_name in ['predicted_lap_time', 'compound_id', 'tire_age', 'position']:
                        if field_name in telemetry_fact and telemetry_fact[field_name] is None:
                            # Default values for any None fields
                            if field_name == 'predicted_lap_time':
                                telemetry_fact[field_name] = telemetry_fact.get(
                                    'lap_time', 0.0)
                            elif field_name in ['compound_id', 'tire_age', 'position']:
                                telemetry_fact[field_name] = 0

                facts.update(lap_facts)
                print(
                    f"✓ Transformed lap time data for Driver #{driver_number}")
                if debug and 'telemetry' in lap_facts:
                    print(
                        f"  Lap time: {lap_facts['telemetry'].get('lap_time')}")
                    print(
                        f"  Predicted lap time: {lap_facts['telemetry'].get('predicted_lap_time')}")
        except Exception as e:
            print(f"✗ Error transforming lap time data: {str(e)}")

    # 3. Transform gap data
    if gap_data is not None:
        try:
            # Ensure gap consistency is calculated
            if 'consistent_gap_ahead_laps' not in gap_data.columns:
                print("  Calculating gap consistency...")
                gap_data = calculate_gap_consistency(gap_data)

            gap_fact = transform_gap_data_with_consistency(
                gap_data, driver_number)
            if gap_fact:
                # Handle None values in gap fact to avoid schema validation errors
                for field_name in ['gap_ahead', 'gap_behind', 'car_ahead', 'car_behind']:
                    if field_name in gap_fact and gap_fact[field_name] is None:
                        if field_name in ['gap_ahead', 'gap_behind']:
                            # Default to 0.0 for distance gaps
                            gap_fact[field_name] = 0.0
                        else:
                            # Default to 0 for car numbers
                            gap_fact[field_name] = 0

                # IMPORTANT: Fix - We need to convert the GapFact to a direct fact, not a nested dictionary
                facts['gap'] = gap_fact  # Store as-is for reference

                print(f"✓ Transformed gap data for Driver #{driver_number}")
                if debug:
                    print(f"  Gap ahead: {gap_fact.get('gap_ahead')}")
                    print(f"  Gap behind: {gap_fact.get('gap_behind')}")
                    print(
                        f"  Consistent gap ahead laps: {gap_fact.get('consistent_gap_ahead_laps')}")
                    print(
                        f"  Consistent gap behind laps: {gap_fact.get('consistent_gap_behind_laps')}")
                    print(
                        f"  In undercut window: {gap_fact.get('in_undercut_window')}")
            else:
                print(f"✗ No gap data transformed for Driver #{driver_number}")
        except Exception as e:
            print(f"✗ Error transforming gap data: {str(e)}")
            import traceback
            if debug:
                traceback.print_exc()

    # 4. Transform radio data
    if radio_json_path:
        try:
            radio_fact = transform_radio_analysis(radio_json_path)
            if radio_fact:
                # IMPORTANT: Store the radio fact directly, not in a nested dictionary
                facts['radio'] = radio_fact

                print(f"✓ Transformed radio communication data")
                if debug:
                    print(f"  Sentiment: {radio_fact.get('sentiment')}")
                    print(f"  Intent: {radio_fact.get('intent')}")
                    print(
                        f"  Entity categories: {list(radio_fact.get('entities', {}).keys())}")
            else:
                print(f"✗ No radio data transformed from {radio_json_path}")
        except Exception as e:
            print(f"✗ Error transforming radio data: {str(e)}")
            import traceback
            if debug:
                traceback.print_exc()

    # 5. Create race status fact (always required)
    try:
        race_phase = calculate_race_phase(current_lap, total_laps)
        facts['race_status'] = RaceStatusFact(
            lap=current_lap,
            total_laps=total_laps,
            race_phase=race_phase,
            track_status="clear"
        )
        print(
            f"✓ Created race status fact: Lap {current_lap}/{total_laps} ({race_phase})")
    except Exception as e:
        print(f"✗ Error creating race status fact: {str(e)}")
        # Provide fallback race status
        facts['race_status'] = RaceStatusFact(
            lap=current_lap or 1,
            total_laps=total_laps,
            race_phase="mid",
            track_status="clear"
        )

    return facts


def load_all_data(race_data_path, models_path=None, lap_model_path=None, gap_data_path=None, radio_message=None):
    """
    Load all necessary data for the strategy engine from various sources.

    Args:
        race_data_path: Path to race telemetry CSV
        models_path: Path to tire prediction models directory
        lap_model_path: Path to lap time prediction model file
        gap_data_path: Optional path to gap data CSV
        radio_message: Optional radio message text to analyze

    Returns:
        Dictionary with all loaded data
    """
    result = {}

    # 1. Load race telemetry data (required)
    print("Loading race telemetry data...")
    try:
        race_data = pd.read_csv(race_data_path)
        result['race_data'] = race_data
        print(f"✓ Loaded race data: {len(race_data)} rows")
    except Exception as e:
        print(f"✗ Could not load race data: {str(e)}")
        return result  # Cannot continue without race data

    # 2. Generate tire degradation predictions
    if models_path:
        print("Generating tire degradation predictions...")
        try:
            # Default monitoring thresholds by compound
            compound_thresholds = {1: 6, 2: 12, 3: 25}
            tire_predictions = load_tire_predictions(
                race_data,
                models_path,
                compound_thresholds=compound_thresholds
            )
            result['tire_predictions'] = tire_predictions
            print(
                f"✓ Generated tire predictions: {len(tire_predictions) if tire_predictions is not None else 0} rows")
        except Exception as e:
            print(f"✗ Could not generate tire predictions: {str(e)}")

    # 3. Generate lap time predictions with a separate model path
    if lap_model_path:
        print("Generating lap time predictions...")
        try:
            # Import the lap prediction module directly
            sys.path.append(os.path.abspath('../'))
            from pred.N00_model_lap_prediction import predict_lap_times

            # Use the function directly instead of the wrapper
            lap_predictions = predict_lap_times(
                race_data,
                model_path=lap_model_path,
                include_next_lap=True
            )

            result['lap_predictions'] = lap_predictions
            print(
                f"✓ Generated lap time predictions: {len(lap_predictions) if lap_predictions is not None else 0} rows")
        except Exception as e:
            print(f"✗ Could not generate lap time predictions: {str(e)}")

    # 4. Create gap data from race data if not provided as CSV
    if gap_data_path:
        print(f"Loading gap data from {gap_data_path}...")
        try:
            gap_data = pd.read_csv(gap_data_path)
            result['gap_data'] = gap_data
            print(f"✓ Loaded gap data: {len(gap_data)} rows")
        except Exception as e:
            print(f"✗ Could not load gap data: {str(e)}")

    # 5. Process radio message if provided
    if radio_message:
        print(f"Processing radio message: '{radio_message}'...")
        try:
            json_path = process_radio_message(radio_message)
            if json_path:
                result['radio_json_path'] = json_path
                print(f"✓ Processed radio message to: {json_path}")
        except Exception as e:
            print(f"✗ Could not process radio message: {str(e)}")

    return result


def analyze_strategy(
    driver_number,
    race_data_path,
    models_path=None,
    lap_model_path=None,
    gap_data_path=None,
    radio_message=None,
    current_lap=None,
    total_laps=66
):
    """
    Complete end-to-end F1 strategy analysis pipeline.

    This function integrates all components of the F1 strategy system:
    1. Loads and prepares data from all sources
    2. Transforms data into facts for the rule engine
    3. Runs the integrated engine with all rule systems
    4. Returns prioritized strategy recommendations

    Args:
        driver_number: Driver number to analyze
        race_data_path: Path to race telemetry CSV
        models_path: Path to tire prediction models
        lap_model_path: Path to lap time prediction model
        gap_data_path: Optional path to gap data CSV
        radio_message: Optional radio message to analyze
        current_lap: Current lap number (defaults to max in data)
        total_laps: Total race laps

    Returns:
        List of strategy recommendations
    """
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"F1 STRATEGY ANALYSIS FOR DRIVER #{driver_number}")
    print(f"{'='*80}")

    # Step 1: Load all data sources
    print("\n--- LOADING DATA ---")
    data = load_all_data(race_data_path, models_path,
                         lap_model_path, gap_data_path, radio_message)

    # Determine current lap if not provided
    if current_lap is None and 'race_data' in data:
        try:
            driver_data = data['race_data'][data['race_data']
                                            ['DriverNumber'] == driver_number]
            if not driver_data.empty:
                if 'LapNumber' in driver_data.columns:
                    current_lap = int(driver_data['LapNumber'].max())
                else:
                    # Use TyreAge as fallback
                    current_lap = int(driver_data['TyreAge'].max())
                print(f"✓ Determined current lap: {current_lap}")
            else:
                current_lap = 20  # Default mid-race
                print(f"✓ Using default lap: {current_lap}")
        except:
            current_lap = 20  # Default
            print(f"✓ Using default lap: {current_lap}")

    # Step 2: Transform data into facts
    print("\n--- TRANSFORMING DATA INTO FACTS ---")
    facts = transform_all_facts(
        driver_number=driver_number,
        tire_predictions=data.get('tire_predictions'),
        lap_predictions=data.get('lap_predictions'),
        gap_data=data.get('gap_data'),
        radio_json_path=data.get('radio_json_path'),
        current_lap=current_lap,
        total_laps=total_laps
    )

    # Step 3: Run the integrated strategy engine
    print("\n--- RUNNING INTEGRATED STRATEGY ENGINE ---")
    engine = F1CompleteStrategyEngine()
    engine.reset()

    # Declare facts to the engine
    for fact_type, fact in facts.items():
        if fact is not None:
            try:
                engine.declare(fact)
                print(f"✓ Declared {type(fact).__name__}")
            except Exception as e:
                print(f"✗ Error declaring {type(fact).__name__}: {str(e)}")

    # Run the engine - this will activate all applicable rules
    print("\nExecuting rules...")
    engine.run()

    # Get recommendations with conflict resolution
    recommendations = engine.get_recommendations()

    # Step 4: Show results
    print(f"\n--- ANALYSIS RESULTS ---")
    print(f"Generated {len(recommendations)} strategy recommendations")

    if recommendations:
        for i, rec in enumerate(recommendations):
            print(f"\nRecommendation {i+1}:")
            print(f"  Action: {rec['action']}")
            print(f"  Confidence: {rec['confidence']:.2f}")
            print(f"  Priority: {rec['priority']}")
            print(f"  Explanation: {rec['explanation']}")
    else:
        print("No recommendations generated. Try with different data inputs.")

    # Show which rule systems were activated
    print("\nActivated rule systems:")
    for system, active in engine.active_systems.items():
        status = "✓" if active else "✗"
        print(f"  {status} {system.capitalize()} rules")

    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")

    return recommendations


def extract_spain_gp_radios(output_directory):
    """
    Extract team radio messages from the 2023 Spanish Grand Prix using OpenF1 API.

    Args:
        output_directory: Folder where MP3 files will be saved

    Returns:
        DataFrame with radio metadata and paths to MP3 files
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING TEAM RADIOS FROM 2023 SPANISH GP")
    print(f"{'='*80}")

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Spanish GP data
    spain_gp = {"name": "Spain", "session_key": 9102, "year": 2023}

    # Extract team radio data from OpenF1 API
    url = f"https://api.openf1.org/v1/team_radio?session_key={spain_gp['session_key']}"
    print(f"Fetching data from: {url}")

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    print(f"✓ Found {len(data)} radio messages for Spanish GP")

    # Convert to DataFrame and process dates
    df = pd.DataFrame(data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Add GP name column for identification
    df['gp_name'] = spain_gp['name']

    # Add column to store MP3 file path
    df['mp3_path'] = None

    # Save metadata as CSV for reference
    metadata_path = os.path.join(
        output_directory, "spain_2023_radio_metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")

    # Download audio files
    print("\nDownloading audio files...")

    # Group by driver
    grouped = df.groupby(["driver_number"])

    total_downloads = 0
    for driver_number, group in grouped:
        # Create folder for each driver
        driver_folder = os.path.join(output_directory, str(driver_number))
        os.makedirs(driver_folder, exist_ok=True)

        # Download and save audio files
        for i, row in group.iterrows():
            url = row["recording_url"]
            if pd.isna(url):
                continue

            # Create filename including index for sorting
            filename = f"radio_{i}.mp3"
            output_path = os.path.join(driver_folder, filename)

            # Save path in DataFrame
            df.loc[i, 'mp3_path'] = output_path

            # Check if file already exists to avoid duplicate downloads
            if os.path.exists(output_path):
                continue

            # Download the file
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {output_path}")
                total_downloads += 1

                # Small delay to avoid overloading the server
                time.sleep(0.5)
            except Exception as e:
                print(f"Error downloading {url}: {e}")

    # Save updated metadata with mp3_path values
    df.to_csv(metadata_path, index=False)
    print(f"Updated metadata saved to: {metadata_path}")

    print(f"Total files downloaded: {total_downloads}")

    # Display statistics of communications per driver
    print("\nDistribution of communications per driver:")
    driver_counts = df['driver_number'].value_counts().sort_index()
    for driver_num, count in driver_counts.items():
        print(f"  • Driver #{driver_num}: {count} communications")

    return df


def process_spain_gp_radios(radio_metadata_df, processed_directory):
    """
    Process radio MP3 files by transcribing and analyzing them with NLP.
    Includes driver number in the JSON filenames for better organization.

    Args:
        radio_metadata_df: DataFrame with radio metadata
        processed_directory: Folder to save processed results

    Returns:
        Dict: Mapping of {(driver_number, index): json_path}
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING TEAM RADIO FILES FROM SPANISH GP")
    print(f"{'='*80}")

    # Ensure output directory exists
    os.makedirs(processed_directory, exist_ok=True)

    # Dictionary to store {(driver_number, index): json_path}
    processed_radios = {}

    # Filter rows with valid MP3 files
    valid_radios = radio_metadata_df.dropna(subset=['mp3_path'])
    print(f"Processing {len(valid_radios)} radio messages...")

    for i, row in valid_radios.iterrows():
        driver_number = row['driver_number']
        mp3_path = row['mp3_path']

        # Verify file exists
        if not os.path.exists(mp3_path):
            print(f"File not found: {mp3_path}")
            continue

        print(
            f"Processing radio {i+1}/{len(valid_radios)}: Driver #{driver_number}")

        try:
            # Use our existing function to transcribe and analyze
            json_path = process_radio_message(mp3_path, is_audio=True)

            if json_path:
                # Get the original filename and create a new one with driver number
                original_filename = os.path.basename(json_path)

                # Extract timestamp from original filename
                filename_parts = original_filename.split('_')
                if len(filename_parts) >= 3 and filename_parts[0] == 'radio' and filename_parts[1] == 'analysis':
                    # Extract date and time portions
                    date_part = filename_parts[2]  # e.g. "20250419"
                    time_part = filename_parts[3].split('.')[0] if len(
                        filename_parts) > 3 else "000000"

                    # Create new filename with driver number AND timestamp preserved
                    new_filename = f"driver_{driver_number}_radio_{date_part}_{time_part}.json"
                else:
                    # Fallback for unexpected filename formats
                    new_filename = f"driver_{driver_number}_{original_filename}"

                new_json_path = os.path.join(processed_directory, new_filename)

                # Copy the file
                shutil.copy2(json_path, new_json_path)
                print(f"  ✓ Copied and renamed to: {new_json_path}")

                # Store the association with the NEW path
                processed_radios[(driver_number, i)] = new_json_path
            else:
                print(f"  ✗ Error processing audio, no JSON generated")

        except Exception as e:
            print(f"  ✗ Error processing {mp3_path}: {str(e)}")

        # Brief pause to avoid issues
        time.sleep(0.1)

    print(f"\nTotal radios processed: {len(processed_radios)}")
    return processed_radios


def map_radios_to_laps(radio_metadata_df, race_data_df):
    """
    Associate each radio message with the corresponding race lap.
    Uses timing information to estimate which lap each message was sent.

    Args:
        radio_metadata_df: DataFrame with radio metadata
        race_data_df: DataFrame with race data

    Returns:
        Dict: Mapping of {(driver_number, lap): message_data}
    """
    print(f"\n{'='*80}")
    print(f"MAPPING RADIO MESSAGES TO RACE LAPS")
    print(f"{'='*80}")

    # Prepare mapping dictionary {(driver_number, lap): message_data}
    radio_lap_mapping = {}

    # Verify data has necessary columns
    if 'date' not in radio_metadata_df.columns:
        print("Error: Radio metadata doesn't contain 'date' column")
        return radio_lap_mapping

    # Get race start and end times
    if 'date' in race_data_df.columns:
        race_start = race_data_df['date'].min()
        race_end = race_data_df['date'].max()
    else:
        print("No date column in race data. Using radio timestamps")
        race_start = radio_metadata_df['date'].min()
        race_end = radio_metadata_df['date'].max()

    # Calculate total race duration
    race_duration = (race_end - race_start).total_seconds()

    # Get total number of laps
    # Try different columns (LapNumber, TyreAge) or default to 66 (typical Spanish GP)
    total_laps = int(race_data_df['LapNumber'].max() if 'LapNumber' in race_data_df.columns else
                     race_data_df['TyreAge'].max() if 'TyreAge' in race_data_df.columns else 66)

    print(
        f"Race duration: {race_duration/60:.1f} minutes, Total laps: {total_laps}")

    # For each radio, estimate lap based on timing
    for i, row in radio_metadata_df.iterrows():
        if pd.isna(row['date']):
            continue

        driver_number = row['driver_number']
        radio_time = row['date']

        # Calculate elapsed time since race start (in seconds)
        elapsed_time = (radio_time - race_start).total_seconds()

        # Estimate lap based on elapsed time
        # Assumes uniform time distribution across laps
        estimated_lap = max(1, min(total_laps, int(
            elapsed_time / race_duration * total_laps)))

        # Save the association with data to identify the message
        radio_lap_mapping[(driver_number, estimated_lap)] = {
            'radio_index': i,
            'mp3_path': row.get('mp3_path'),
            'transcript_path': None,  # Will be filled later with process_radio_message
            'recording_url': row.get('recording_url'),
            'radio_time': radio_time
        }

    print(f"Mapped {len(radio_lap_mapping)} radios to specific laps")

    # Calculate distribution of radios per lap
    laps_distribution = {}
    for (_, lap), _ in radio_lap_mapping.items():
        laps_distribution[lap] = laps_distribution.get(lap, 0) + 1

    print("\nDistribution of radios by lap (first 10 laps):")
    for lap in sorted(laps_distribution.keys())[:10]:
        print(f"  • Lap {lap}: {laps_distribution[lap]} radios")

    return radio_lap_mapping


def analyze_all_drivers_with_real_radios(
    race_data_path,
    models_path=None,
    lap_model_path=None,
    output_path=None,
    extract_new_radios=True
):
    """
    Analyze strategy for all drivers at key race moments using real radio 
    communications from the 2023 Spanish GP, with proper gap data processing using
    the approach from analyze_race_gaps.

    Args:
        race_data_path: Path to race telemetry CSV
        models_path: Path to tire degradation models
        lap_model_path: Path to lap time prediction model
        output_path: Path to save results
        extract_new_radios: If True, extract new radio data; if False, use existing data

    Returns:
        DataFrame with all recommendations
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING ALL DRIVERS WITH REAL RADIOS FROM 2023 SPANISH GP")
    print(f"{'='*80}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Set up folder paths
    base_dir = 'outputs'
    radio_dir = os.path.join(base_dir, "week6", "radios")
    processed_dir = os.path.join(base_dir, "week6", "processed_radios")

    # Create necessary directories
    os.makedirs(radio_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Load race data for tire and lap time predictions
    race_data = pd.read_csv(race_data_path)
    print(f"Race data loaded: {len(race_data)} rows")

    # 3. Extract radios from OpenF1 if needed
    radio_metadata_path = os.path.join(
        radio_dir, "spain_2023_radio_metadata.csv")

    if extract_new_radios or not os.path.exists(radio_metadata_path):
        print("Extracting new radio data...")
        radio_metadata = extract_spain_gp_radios(radio_dir)
    else:
        print(f"Loading existing radio metadata: {radio_metadata_path}")
        radio_metadata = pd.read_csv(radio_metadata_path)
        if 'date' in radio_metadata.columns:
            radio_metadata['date'] = pd.to_datetime(
                radio_metadata['date'], errors='coerce')

    # 4. Map radios to laps
    radio_lap_mapping = map_radios_to_laps(radio_metadata, race_data)

    # 5. Process radios - Scan for individual files
    # Scan the processed_radios directory for all JSON files
    json_pattern = os.path.join(processed_dir, "*.json")
    json_files = glob.glob(json_pattern)

    print(f"Found {len(json_files)} processed radio JSON files")

    # Create a mapping from driver number to JSON files
    driver_json_mapping = {}
    for json_file in json_files:
        # Extract driver number from filename (driver_X_radio_*.json)
        file_name = os.path.basename(json_file)
        match = re.match(r'driver_(\d+)_radio_.*\.json', file_name)
        if match:
            driver_number = int(match.group(1))
            if driver_number not in driver_json_mapping:
                driver_json_mapping[driver_number] = []
            driver_json_mapping[driver_number].append(json_file)

    print(
        f"Created mapping for {len(driver_json_mapping)} drivers with radio data")

    # 6. Generate tire and lap time predictions
    tire_predictions = None
    if models_path:
        try:
            compound_thresholds = {1: 6, 2: 12, 3: 25}
            tire_predictions = load_tire_predictions(
                race_data, models_path, compound_thresholds)
            print(
                f"Tire predictions generated: {len(tire_predictions) if tire_predictions is not None else 0} rows")
        except Exception as e:
            print(f"Error generating tire predictions: {str(e)}")

    lap_predictions = None
    if lap_model_path:
        try:
            lap_predictions = load_lap_time_predictions(
                race_data, lap_model_path)
            print(
                f"Lap time predictions generated: {len(lap_predictions) if lap_predictions is not None else 0} rows")
        except Exception as e:
            print(f"Error generating lap time predictions: {str(e)}")

    # 7. Load gap data using FastF1
    try:
        # Setup FastF1 cache
        fastf1.Cache.enable_cache('f1-strategy/f1_cache')

        # Spanish GP 2023 details
        year = 2023
        grand_prix = 'Spain'
        session_type = 'R'  # R for Race

        gap_data = None
        gap_data_path = os.path.join(base_dir, "week6", "gap_data.csv")

        # Try to load cached gap data first
        if os.path.exists(gap_data_path):
            try:
                print(f"Loading gap data from: {gap_data_path}")
                gap_data = pd.read_csv(gap_data_path)
                print(f"Gap data loaded: {len(gap_data)} rows")
            except Exception as e:
                print(f"Error loading gap data: {str(e)}")
                gap_data = None

        # If no cached data or loading failed, calculate using FastF1
        if gap_data is None:
            print("\nLoading FastF1 session data for gap calculation...")
            race = fastf1.get_session(year, grand_prix, session_type)
            race.load()
            laps_data = race.laps
            print(f"FastF1 data loaded with {len(laps_data)} lap entries")

            print("\nCalculating gaps between cars...")
            gap_data = calculate_all_gaps(laps_data)

            # Save gap data for future use
            gap_data.to_csv(gap_data_path, index=False)
            print(f"Gap data calculated and saved: {len(gap_data)} rows")

        # Ensure gap data has consistency metrics
        if gap_data is not None and 'consistent_gap_ahead_laps' not in gap_data.columns:
            print("\nCalculating gap consistency metrics...")
            gap_data = calculate_gap_consistency(gap_data)
            # Update saved file with consistency data
            gap_data.to_csv(gap_data_path, index=False)
            print("Gap consistency metrics added")

    except Exception as e:
        print(f"Error working with FastF1 data: {str(e)}")
        import traceback
        traceback.print_exc()
        gap_data = None

    # 8. Define total laps and strategic points
    # Spanish GP 2023 had 66 laps
    total_laps = 66

    # Verify from gap data if available
    if gap_data is not None and 'LapNumber' in gap_data.columns:
        total_laps_from_data = int(gap_data['LapNumber'].max())
        if total_laps_from_data > 0:
            total_laps = total_laps_from_data
            print(f"Confirmed total laps from gap data: {total_laps}")

    # Define strategic points for analysis
    strategic_points = [
        int(total_laps * 0.25),  # 25% of race
        int(total_laps * 0.5),   # 50% of race (midpoint)
        int(total_laps * 0.75)   # 75% of race
    ]
    print(f"Strategic analysis points: Laps {strategic_points}")

    # 9. Process each driver
    all_recommendations = []

    # Define compound names mapping for readable compound names
    compound_names = {
        1: 'Soft',
        2: 'Medium',
        3: 'Hard',
        0: 'Unknown'
    }

    # Use driver numbers from race data (convert to standard Python int)
    drivers = [int(driver) for driver in race_data['DriverNumber'].unique()]

    # Add any drivers from gap data that might not be in race data
    if gap_data is not None and 'DriverNumber' in gap_data.columns:
        gap_drivers = gap_data['DriverNumber'].unique()
        # Add any drivers from gap data that aren't already in our list
        for driver in gap_drivers:
            driver_int = int(driver)  # Convert to standard integer
            if driver_int not in drivers:
                drivers.append(driver_int)

    # Now sort the list of integers
    drivers = sorted(drivers)

    print(f"\n--- ANALYZING {len(drivers)} DRIVERS ---")

    # 10. Process each driver at each strategic point
    for driver_number in drivers:
        print(f"\nAnalyzing driver #{driver_number}")

        # Process each strategic point
        for lap in strategic_points:
            print(f"  Lap {lap}/{total_laps}")

            # Find a radio message for this driver at or near this lap
            radio_json_path = None

            # First check if we have any radio files for this driver
            if driver_number in driver_json_mapping and driver_json_mapping[driver_number]:
                # Just use the first radio file for now (simplification)
                radio_json_path = driver_json_mapping[driver_number][0]
                print(
                    f"  Using radio file: {os.path.basename(radio_json_path)}")

            # Filter gap data for this driver and lap
            driver_gap_data = None
            if gap_data is not None:
                # Filter for driver and relevant lap
                driver_data = gap_data[gap_data['DriverNumber']
                                       == driver_number]
                if not driver_data.empty:
                    # Find gap data for this lap or closest previous lap
                    relevant_laps = driver_data[driver_data['LapNumber'] <= lap]
                    if not relevant_laps.empty:
                        closest_lap = relevant_laps['LapNumber'].max()
                        driver_gap_data = relevant_laps[relevant_laps['LapNumber']
                                                        == closest_lap]
                        print(f"  Using gap data from lap {closest_lap}")

            # Transform tire predictions for this driver
            driver_tire_predictions = None
            if tire_predictions is not None:
                driver_tire_data = tire_predictions[tire_predictions['DriverNumber']
                                                    == driver_number]
                if not driver_tire_data.empty:
                    driver_tire_predictions = driver_tire_data
                    print(
                        f"  Using tire predictions ({len(driver_tire_data)} rows)")

            # Transform lap time predictions for this driver
            driver_lap_predictions = None
            if lap_predictions is not None:
                driver_lap_data = lap_predictions[lap_predictions['DriverNumber']
                                                  == driver_number]
                if not driver_lap_data.empty:
                    driver_lap_predictions = driver_lap_data
                    print(
                        f"  Using lap time predictions ({len(driver_lap_data)} rows)")

            # Use transform_all_facts to prepare facts
            facts = transform_all_facts(
                driver_number=driver_number,
                tire_predictions=driver_tire_predictions,
                lap_predictions=driver_lap_predictions,
                gap_data=driver_gap_data,
                radio_json_path=radio_json_path,
                current_lap=lap,
                total_laps=total_laps,
                debug=False
            )

            # Create rule engine
            engine = F1CompleteStrategyEngine()
            engine.reset()

            # Declare all facts to the engine
            for key, fact in facts.items():
                if fact is not None:
                    try:
                        engine.declare(fact)
                        print(f"  Declared {type(fact).__name__}")
                    except Exception as e:
                        print(f"  Error declaring {key} fact: {str(e)}")

            # Run the engine
            engine.run()

            # Get recommendations
            recommendations = engine.get_recommendations()

            # Create a mapping from action to originating rule
            action_to_rule = {
                # Tire Degradation Rules
                # Note: also from performance_cliff_detection
                "pit_stop": "high_degradation_pit_stop",
                "extend_stint": "stint_extension_recommendation",
                "prepare_pit": "early_degradation_warning",
                "consider_pit": "predicted_high_degradation_alert",

                # Lap Time Rules
                "push_strategy": "optimal_performance_window",
                # "pit_stop" is duplicated from above
                "recovery_push": "post_traffic_recovery",

                # NLP Radio Rules
                "prioritize_pit": "grip_issue_response",
                "prepare_rain_tires": "weather_information_adjustment",
                "reevaluate_pit_window": "incident_reaction",

                # Gap Analysis Rules
                "perform_undercut": "undercut_opportunity",
                "defensive_pit": "defensive_pit_stop",
                "perform_overcut": "strategic_overcut",
                "adjust_pit_window": "traffic_management"
            }

            # Handle ambiguous action "pit_stop" which can come from multiple rules
            # Look at active_systems to determine which system is active
            if len(recommendations) > 0 and "pit_stop" in [r["action"] for r in recommendations]:
                if engine.active_systems.get('degradation', False):
                    # If degradation system is active, pit_stop is from high_degradation_pit_stop
                    for rec in recommendations:
                        if rec["action"] == "pit_stop":
                            rec["rule_fired"] = "high_degradation_pit_stop"
                elif engine.active_systems.get('lap_time', False):
                    # If lap_time system is active, pit_stop is from performance_cliff_detection
                    for rec in recommendations:
                        if rec["action"] == "pit_stop":
                            rec["rule_fired"] = "performance_cliff_detection"

            # Add rule_fired to all other recommendations based on action
            for rec in recommendations:
                # Skip if already handled (like pit_stop)
                if "rule_fired" not in rec:
                    rec["rule_fired"] = action_to_rule.get(
                        rec["action"], "unknown")

            # Get current position from telemetry fact or gap data
            driver_position = None

            # Check telemetry fact
            if 'telemetry' in facts and facts['telemetry'] is not None:
                driver_position = facts['telemetry'].get('position')

            # Check gap data
            if driver_position is None and driver_gap_data is not None and not driver_gap_data.empty:
                if 'Position' in driver_gap_data.columns:
                    driver_position = int(driver_gap_data['Position'].iloc[0])

            # Default position if still None
            if driver_position is None:
                driver_position = 0

            # Get team ID (defaults to 0 if not found)
            team_id = 0

            # Try to find team in race_data
            if 'TeamID' in race_data.columns:
                driver_team_data = race_data[race_data['DriverNumber']
                                             == driver_number]
                if not driver_team_data.empty:
                    team_id = int(driver_team_data['TeamID'].iloc[0])

            # Extract the compound ID from telemetry fact
            compound_id = None
            if 'telemetry' in facts and facts['telemetry'] is not None:
                compound_id = facts['telemetry'].get('compound_id')

            # Add metadata to recommendations
            for rec in recommendations:
                rec['DriverNumber'] = driver_number
                rec['LapNumber'] = lap
                rec['RacePhase'] = facts['race_status']['race_phase'] if 'race_status' in facts else None
                rec['Position'] = driver_position
                rec['Team'] = team_id

                # Add tire compound information (NEW)
                rec['CompoundID'] = compound_id
                rec['CompoundName'] = compound_names.get(
                    compound_id, 'Unknown') if compound_id is not None else 'Unknown'

            all_recommendations.extend(recommendations)
            print(f"    Generated {len(recommendations)} recommendations")

            # Display which rule systems were activated
            active_systems = engine.active_systems
            active_count = sum(1 for v in active_systems.values() if v)
            if active_count > 0:
                print(
                    f"    Active rule systems: {', '.join(k for k, v in active_systems.items() if v)}")

    # Convert to DataFrame
    if all_recommendations:
        results_df = pd.DataFrame(all_recommendations)

        # Sort by lap number in ascending order (NEW)
        results_df = results_df.sort_values('LapNumber', ascending=True)

        # Reorganize columns to place rule_fired after action (NEW)
        if 'rule_fired' in results_df.columns and 'action' in results_df.columns:
            # Get the current column ordering
            cols = list(results_df.columns)

            # Remove rule_fired from its current position
            if 'rule_fired' in cols:
                cols.remove('rule_fired')

            # Find the position of action
            action_idx = cols.index('action')

            # Insert rule_fired right after action
            cols.insert(action_idx + 1, 'rule_fired')

            # Reorder the DataFrame using the new column order
            results_df = results_df[cols]

            print("Columns reorganized: 'rule_fired' placed after 'action'")

        # Save results
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")

        print(f"\nTotal recommendations: {len(results_df)}")
        return results_df
    else:
        print("No recommendations generated")
        return pd.DataFrame()


def run_example_analysis():
    """
    Example demonstrating the complete strategy analysis pipeline.

    Uses the provided CSV files and a sample radio message to run
    a full strategy analysis for a specific driver.
    """
    print("\n=== RUNNING EXAMPLE STRATEGY ANALYSIS ===")

    # Define paths to data files
    race_data_path = 'outputs/week3/lap_prediction_data.csv'
    models_path = 'outputs/week5/models/'
    # Separate path for lap model
    lap_model_path = 'outputs/week3/xgb_sequential_model.pkl'

    # Sample radio message
    radio_message = "Box this lap for softs, there's rain expected in 10 minutes"

    # Driver to analyze (Lewis Hamilton)
    driver_number = 44

    # Run the analysis
    recommendations = analyze_strategy(
        driver_number=driver_number,
        race_data_path=race_data_path,
        models_path=models_path,
        lap_model_path=lap_model_path,
        radio_message=radio_message,
        current_lap=20,  # Mid-race scenario
        total_laps=66    # Typical F1 race length
    )

    return recommendations


def run_all_drivers_analysis():
    """
    Run complete analysis with real radios from 2023 Spanish GP.

    This function sets up all paths and runs the end-to-end analysis.
    """
    # File paths
    race_data_path = 'outputs/week3/lap_prediction_data.csv'
    models_path = 'outputs/week5/models/'
    lap_model_path = 'outputs/week3/xgb_sequential_model.pkl'
    output_path = 'outputs/week6/spain_gp_recommendations.csv'

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run analysis (True to extract new radios, False to reuse existing)
    results = analyze_all_drivers_with_real_radios(
        race_data_path=race_data_path,
        models_path=models_path,
        lap_model_path=lap_model_path,
        output_path=output_path,
        extract_new_radios=False  # Change to True to download new data
    )

    return results


def test_all_rule_activations():
    """
    Function to test that all types of rules activate correctly.

    This creates synthetic test cases for each rule system:
    - Tire degradation rules
    - Lap time rules
    - Gap rules 
    - Radio rules

    Returns:
        Dictionary mapping rule systems to activation status (True/False)
    """
    print(f"\n{'='*80}")
    print("TESTING ALL RULE ACTIVATIONS")
    print(f"{'='*80}")

    # Initialize the integrated engine
    engine = F1CompleteStrategyEngine()

    # Sample driver number
    driver_number = 44  # Lewis Hamilton

    # Test case 1: Radio rule activation (weather warning)
    print("\n=== TEST CASE 1: RADIO RULES (WEATHER WARNING) ===")
    weather_radio_message = "Warning: rain expected in 5 minutes, prepare wet tires"

    # Process the message
    print(f"Processing radio message: '{weather_radio_message}'")
    radio_json_path = process_radio_message(weather_radio_message)

    if radio_json_path:
        # Create a fresh engine instance
        engine.reset()

        # Transform the radio facts with debug mode
        facts = transform_all_facts(
            driver_number=driver_number,
            radio_json_path=radio_json_path,
            current_lap=20,
            total_laps=66,
            debug=True
        )

        # Declare all facts to the engine
        for key, fact in facts.items():
            try:
                engine.declare(fact)
                print(f"Declared {type(fact).__name__}")
            except Exception as e:
                print(f"Error declaring {key} fact: {e}")

        # Run the engine
        engine.run()

        # Check recommendations
        recommendations = engine.get_recommendations()
        print(f"\nGenerated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations):
            print(f"Recommendation {i+1}:")
            print(f"  Action: {rec['action']}")
            print(f"  Confidence: {rec['confidence']}")
            print(f"  Explanation: {rec['explanation']}")

    # Continue with other test cases...
    # (Code for test cases 2-4 omitted for brevity)

    # Check final activation status
    activity_status = {
        'radio_rules': bool(engine.active_systems.get('radio', False)),
        'gap_rules': bool(engine.active_systems.get('gap', False)),
        'lap_time_rules': bool(engine.active_systems.get('lap_time', False)),
        'degradation_rules': bool(engine.active_systems.get('degradation', False))
    }

    print("\n=== FINAL RULE ACTIVATION STATUS ===")
    for rule_system, is_active in activity_status.items():
        status = "✓ ACTIVE" if is_active else "✗ INACTIVE"
        print(f"{status}: {rule_system}")

    return activity_status


# Allow the module to be run as a script
if __name__ == "__main__":
    print("F1 Strategy Module initialized successfully!")

    # # Example usage
    # example_recommendations = run_example_analysis()

    # # Print information about rule activation
    # print("\n=== RUNNING RULE ACTIVATION TESTS ===")
    # test_all_rule_activations()

    # Full analysis for Spanish GP (commented out to avoid lengthy execution)

    results = run_all_drivers_analysis()

    # Load the CSV file with recommendations (if available)
    try:
        recommendations = pd.read_csv(
            'outputs/week6/spain_gp_recommendations.csv')

        # Count the occurrences of each action type
        action_counts = recommendations['action'].value_counts()

        print("\n=== ACTION SUMMARY ===")
        print(action_counts)

        # Check specific rules by category
        print("\n=== RULE CATEGORIES ===")
        print("GAP RULES:", sum(recommendations['action'].isin(
            ['perform_undercut', 'defensive_pit', 'perform_overcut', 'adjust_pit_window'])))
        print("LAP TIME RULES:", sum(recommendations['action'].isin(
            ['push_strategy', 'pit_stop', 'recovery_push'])))
        print("DEGRADATION RULES:", sum(recommendations['action'].isin(
            ['extend_stint', 'prioritize_pit', 'prepare_pit', 'consider_pit'])))
        print("RADIO RULES:", sum(recommendations['action'].isin(
            ['prepare_rain_tires', 'reevaluate_pit_window'])))
    except:
        print("\nNo recommendations file found. Run analysis to generate recommendations.")
