from utils.N01_agent_setup import (
    TelemetryFact,
    DegradationFact,
    GapFact,
    RaceStatusFact,
    StrategyRecommendation,
    F1StrategyEngine,
    transform_gap_data_with_consistency,
    load_gap_data,
    calculate_gap_consistency
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import requests
import fastf1

from experta import Rule, NOT, OR, AND, AS, MATCH, TEST, EXISTS
from experta import DefFacts, Fact, Field, KnowledgeEngine

sys.path.append(os.path.abspath('../'))
fastf1.Cache.enable_cache('../../f1-strategy/f1_cache')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)


def get_gap_at_lap_completion(driver, lap_number, laps_data):
    driver_lap = laps_data[(laps_data['Driver'] == driver) & (
        laps_data['LapNumber'] == lap_number)]
    if driver_lap.empty:
        return None
    driver_finish_time = driver_lap.iloc[0]['Time']
    lap_group = laps_data[laps_data['LapNumber'] == lap_number]
    leader_time = lap_group['Time'].min()
    gap_to_leader = driver_finish_time - leader_time
    if hasattr(gap_to_leader, 'total_seconds'):
        gap_to_leader = gap_to_leader.total_seconds()
    return gap_to_leader


def calculate_all_gaps(laps_data):
    """
    Calculates the gaps for all drivers at the completion of each lap.

    Args:
        laps_data (DataFrame): Lap data from FastF1 or another source

    Returns:
        DataFrame: Processed gap data with all calculated metrics
    """
    gap_results = []

    # Check if we need to use 'DriverNumber' instead of 'Driver'
    driver_column = 'Driver' if 'Driver' in laps_data.columns else 'DriverNumber'

    print(f"Using '{driver_column}' as driver identifier column")

    # Get unique drivers and sorted lap numbers from the lap data
    drivers = laps_data[driver_column].unique()
    lap_numbers = sorted(laps_data['LapNumber'].unique(
    )) if 'LapNumber' in laps_data.columns else sorted(laps_data['TyreAge'].unique())
    lap_column = 'LapNumber' if 'LapNumber' in laps_data.columns else 'TyreAge'

    print(
        f"Processing gaps for {len(drivers)} drivers across {len(lap_numbers)} laps using '{lap_column}' column...")

    # Function to get the gap at lap completion
    def get_gap_at_lap_completion(driver_id, lap_num):
        # Filter for specific driver and lap
        driver_lap = laps_data[(laps_data[driver_column] == driver_id) & (
            laps_data[lap_column] == lap_num)]
        if driver_lap.empty:
            return None

        # Get reference time - either 'Time' column or calculated from 'LapTime'
        if 'Time' in driver_lap.columns:
            driver_finish_time = driver_lap.iloc[0]['Time']
        elif 'LapTime' in driver_lap.columns:
            # If we only have lap time, use that as a proxy
            driver_finish_time = driver_lap.iloc[0]['LapTime']
        else:
            return None

        # Get all laps at this lap number
        lap_group = laps_data[laps_data[lap_column] == lap_num]

        # Get minimum time for this lap (leader's time)
        if 'Time' in lap_group.columns:
            leader_time = lap_group['Time'].min()
        elif 'LapTime' in lap_group.columns:
            leader_time = lap_group['LapTime'].min()
        else:
            return None

        # Calculate gap to leader
        gap_to_leader = driver_finish_time - leader_time

        # Handle timedelta objects if needed
        if hasattr(gap_to_leader, 'total_seconds'):
            gap_to_leader = gap_to_leader.total_seconds()

        return gap_to_leader

    # Process each lap
    for lap_num in lap_numbers:
        print(f"Processing lap {lap_num}...", end='\r')

        # Dictionary to store gap to leader for each driver
        lap_positions = {}

        # Calculate gap to leader for each driver
        for driver in drivers:
            gap_to_leader = get_gap_at_lap_completion(driver, lap_num)
            if gap_to_leader is not None:
                lap_positions[driver] = gap_to_leader

        # Sort drivers by gap to leader
        sorted_drivers = sorted(lap_positions.items(), key=lambda x: x[1])

        # For each driver, calculate additional gap metrics
        for i, (driver, gap_to_leader) in enumerate(sorted_drivers):
            # Get driver info (driver number and team)
            driver_info = laps_data[laps_data[driver_column] == driver].iloc[0]

            # Initialize default values
            driver_number = driver
            if driver_column == 'Driver' and 'DriverNumber' in driver_info:
                driver_number = driver_info['DriverNumber']

            # Initialize gap values
            gap_ahead = None
            car_ahead = None
            gap_behind = None
            car_behind = None
            car_ahead_number = None
            car_behind_number = None

            # Calculate gap to car ahead
            if i > 0:
                car_ahead = sorted_drivers[i-1][0]
                gap_ahead = gap_to_leader - sorted_drivers[i-1][1]

                # Get car ahead number
                if car_ahead in laps_data[driver_column].values:
                    car_ahead_info = laps_data[laps_data[driver_column]
                                               == car_ahead].iloc[0]
                    if driver_column == 'Driver' and 'DriverNumber' in car_ahead_info:
                        car_ahead_number = car_ahead_info['DriverNumber']
                    else:
                        car_ahead_number = car_ahead

            # Calculate gap to car behind
            if i < len(sorted_drivers) - 1:
                car_behind = sorted_drivers[i+1][0]
                gap_behind = sorted_drivers[i+1][1] - gap_to_leader

                # Get car behind number
                if car_behind in laps_data[driver_column].values:
                    car_behind_info = laps_data[laps_data[driver_column]
                                                == car_behind].iloc[0]
                    if driver_column == 'Driver' and 'DriverNumber' in car_behind_info:
                        car_behind_number = car_behind_info['DriverNumber']
                    else:
                        car_behind_number = car_behind

            # Create result entry
            gap_results.append({
                'LapNumber': lap_num,
                'Driver': driver,
                'DriverNumber': driver_number,
                'Position': i + 1,  # Position is 1-indexed
                'GapToLeader': gap_to_leader,
                'CarAhead': car_ahead,
                'CarAheadNumber': car_ahead_number,
                'GapToCarAhead': gap_ahead,
                'CarBehind': car_behind,
                'CarBehindNumber': car_behind_number,
                'GapToCarBehind': gap_behind,
                'InUndercutWindow': gap_ahead is not None and gap_ahead < 2.5,
                'InDRSWindow': gap_ahead is not None and gap_ahead < 1.0
            })

    print("\nProcessing complete!")
    return pd.DataFrame(gap_results)


class F1GapRules(F1StrategyEngine):
    @Rule(
        GapFact(driver_number=MATCH.driver_number),
        GapFact(gap_ahead=MATCH.gap_ahead),
        TEST(lambda gap_ahead: gap_ahead is not None and gap_ahead < 2.0),
        GapFact(consistent_gap_ahead_laps=MATCH.consistent_laps),
        TEST(lambda consistent_laps: consistent_laps is not None and consistent_laps >= 3),
        RaceStatusFact(lap=MATCH.lap),
        TEST(lambda lap: (6 <= lap <= 26) or (26 < lap <= 48))
    )
    def undercut_opportunity(self, driver_number, gap_ahead, consistent_laps, lap):
        stint_phase = "early" if lap <= 26 else "mid"
        print(f"Rule activated: undercut_opportunity")
        print(f"  - Driver: {driver_number}")
        print(f"  - Current lap: {lap} ({stint_phase} stint)")
        print(f"  - Gap ahead: {gap_ahead:.2f}s")
        print(f"  - Consistent laps in this window: {consistent_laps}")
        self.declare(
            StrategyRecommendation(
                action="perform_undercut",
                confidence=0.85,
                explanation=f"Strong undercut opportunity detected in {stint_phase} stint (lap {lap}). Gap to car ahead ({gap_ahead:.2f}s) has been consistently within undercut window for {consistent_laps} laps, suggesting on-track overtaking is difficult.",
                priority=2,
                lap_issued=lap
            )
        )
        self.record_rule_fired("undercut_opportunity")

    @Rule(
        GapFact(driver_number=MATCH.driver_number),
        GapFact(gap_behind=MATCH.gap_behind),
        TEST(lambda gap_behind: gap_behind is not None and gap_behind < 2.0),
        GapFact(consistent_gap_behind_laps=MATCH.consistent_laps),
        TEST(lambda consistent_laps: consistent_laps is not None and consistent_laps >= 3),
        RaceStatusFact(lap=MATCH.lap),
        TEST(lambda lap: (6 <= lap <= 26) or (26 < lap <= 48))
    )
    def defensive_pit_stop(self, driver_number, gap_behind, consistent_laps, lap):
        stint_phase = "early" if lap <= 26 else "mid"
        print(f"Rule activated: defensive_pit_stop")
        print(f"  - Driver: {driver_number}")
        print(f"  - Current lap: {lap} ({stint_phase} stint)")
        print(f"  - Gap behind: {gap_behind:.2f}s")
        print(f"  - Consistent laps in this window: {consistent_laps}")
        self.declare(
            StrategyRecommendation(
                action="defensive_pit",
                confidence=0.8,
                explanation=f"Defensive pit stop strongly recommended in {stint_phase} stint (lap {lap}). Car behind has been consistently close ({gap_behind:.2f}s) for {consistent_laps} laps, indicating high probability of undercut attempt.",
                priority=2,
                lap_issued=lap
            )
        )
        self.record_rule_fired("defensive_pit_stop")

    @Rule(
        GapFact(driver_number=MATCH.driver_number),
        GapFact(gap_ahead=MATCH.gap_ahead),
        TEST(lambda gap_ahead: gap_ahead is not None and 2.0 < gap_ahead < 3.5),
        GapFact(consistent_gap_ahead_laps=MATCH.consistent_laps),
        TEST(lambda consistent_laps: consistent_laps is not None and consistent_laps >= 4),
        RaceStatusFact(lap=MATCH.lap),
        TEST(lambda lap: (6 <= lap <= 26) or (26 < lap <= 48))
    )
    def strategic_overcut(self, driver_number, gap_ahead, consistent_laps, lap):
        stint_phase = "early" if lap <= 26 else "mid"
        print(f"Rule activated: strategic_overcut")
        print(f"  - Driver: {driver_number}")
        print(f"  - Current lap: {lap} ({stint_phase} stint)")
        print(f"  - Gap ahead: {gap_ahead:.2f}s")
        print(f"  - Consistent laps in this window: {consistent_laps}")
        self.declare(
            StrategyRecommendation(
                action="perform_overcut",
                confidence=0.8,
                explanation=f"Strong overcut opportunity detected in {stint_phase} stint (lap {lap}). Gap to car ahead ({gap_ahead:.2f}s) has remained in optimal overcut range for {consistent_laps} laps. Consider staying out longer when they pit.",
                priority=2,
                lap_issued=lap
            )
        )
        self.record_rule_fired("strategic_overcut")

    @Rule(
        GapFact(driver_number=MATCH.driver_number),
        TelemetryFact(position=MATCH.position),
        GapFact(gap_to_leader=MATCH.gap_to_leader),
        TEST(lambda position, gap_to_leader: position >
             10 and gap_to_leader > 30.0),
        RaceStatusFact(lap=MATCH.lap),
        TEST(lambda lap: (6 <= lap <= 26) or (26 < lap <= 48))
    )
    def traffic_management(self, driver_number, position, gap_to_leader, lap):
        stint_phase = "early" if lap <= 26 else "mid"
        print(f"Rule activated: traffic_management")
        print(f"  - Driver: {driver_number}")
        print(f"  - Current lap: {lap} ({stint_phase} stint)")
        print(f"  - Position: {position}")
        print(f"  - Gap to leader: {gap_to_leader:.2f}s")
        self.declare(
            StrategyRecommendation(
                action="adjust_pit_window",
                confidence=0.7,
                explanation=f"Potential traffic detected in {stint_phase} stint (lap {lap}). Position {position} with {gap_to_leader:.2f}s gap to leader suggests traffic concerns. Consider adjusting pit window by 1-2 laps to avoid traffic.",
                priority=1,
                lap_issued=lap
            )
        )
        self.record_rule_fired("traffic_management")


def test_gap_rules(scenario_name, driver_number, gap_data, current_lap=20, total_laps=66, additional_facts=None):
    print(f"\n{'='*80}")
    print(f"TESTING SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    print(f"Driver: {driver_number}, Lap: {current_lap}/{total_laps}")

    print("\nCreating gap fact with provided data...")

    gap_values = {
        'gap_ahead': None,
        'gap_behind': None,
        'car_ahead': None,
        'car_behind': None,
        'gap_to_leader': 0.0,
        'consistent_gap_ahead_laps': 1,
        'consistent_gap_behind_laps': 1,
        'in_undercut_window': False,
        'in_drs_window': False
    }
    gap_values.update(gap_data)
    gap_fact = GapFact(
        driver_number=driver_number,
        gap_ahead=gap_values['gap_ahead'],
        gap_behind=gap_values['gap_behind'],
        car_ahead=gap_values['car_ahead'],
        car_behind=gap_values['car_behind'],
        gap_to_leader=gap_values['gap_to_leader'],
        consistent_gap_ahead_laps=gap_values['consistent_gap_ahead_laps'],
        consistent_gap_behind_laps=gap_values['consistent_gap_behind_laps'],
        in_undercut_window=gap_values['in_undercut_window'],
        in_drs_window=gap_values['in_drs_window']
    )
    print(f"Gap fact created:")
    print(f"  - gap_ahead: {gap_values['gap_ahead']}")
    print(f"  - gap_behind: {gap_values['gap_behind']}")
    print(
        f"  - consistent_gap_ahead_laps: {gap_values['consistent_gap_ahead_laps']}")
    print(
        f"  - consistent_gap_behind_laps: {gap_values['consistent_gap_behind_laps']}")
    print(f"  - gap_to_leader: {gap_values['gap_to_leader']}")

    print("\nInitializing rule engine...")
    engine = F1GapRules()
    engine.reset()

    race_phase = "start" if current_lap < total_laps * \
        0.25 else "mid" if current_lap < total_laps * 0.75 else "end"
    race_status = RaceStatusFact(
        lap=current_lap,
        total_laps=total_laps,
        race_phase=race_phase,
        track_status="clear"
    )
    engine.declare(race_status)
    print(f"Race status declared: Lap {current_lap}, Phase: {race_phase}")

    engine.declare(gap_fact)

    telemetry_fact = TelemetryFact(
        driver_number=driver_number,
        position=10
    )
    engine.declare(telemetry_fact)

    if additional_facts:
        print("Declaring additional facts:")
        for fact_name, fact in additional_facts.items():
            print(f"  - {fact_name}: {fact}")
            engine.declare(fact)

    print("\nRunning rule engine...")
    engine.run()

    recommendations = engine.get_recommendations()

    print(f"\nGenerated {len(recommendations)} recommendations:")
    if recommendations:
        for i, rec in enumerate(recommendations):
            print(f"\nRecommendation {i+1}:")
            print(f"Action: {rec['action']}")
            print(f"Confidence: {rec['confidence']}")
            print(f"Explanation: {rec['explanation']}")
            print(f"Priority: {rec['priority']}")
            print(f"Lap issued: {rec['lap_issued']}")
    else:
        print("No recommendations generated.")

    print("\nTriggered rules:")
    if engine.rules_fired:
        for rule in engine.rules_fired:
            print(f"- {rule['rule']} (Lap {rule['lap']})")
    else:
        print("No rules were triggered.")

    return recommendations


def analyze_race_gaps(year, grand_prix, session_type='R', save_path=None, test_drivers=None, debug=False):
    import fastf1
    import pandas as pd
    import os
    import time
    from datetime import datetime

    def time_operation(func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time

    print(f"=== Gap Analysis for {grand_prix} {year} ({session_type}) ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        fastf1.Cache.enable_cache('../../f1-strategy/f1_cache')
        print("\n1. Loading session data...")
        race, load_time = time_operation(
            fastf1.get_session, year, grand_prix, session_type)
        race.load()
        print(f"   - Session loaded in {load_time:.2f} seconds")
        laps_data = race.laps
        print(f"   - Loaded {len(laps_data)} lap entries")
        print("\n2. Calculating gaps between cars...")
        all_gaps_df, calc_time = time_operation(calculate_all_gaps, laps_data)
        print(f"   - Gaps calculated in {calc_time:.2f} seconds")
        print(f"   - Generated {len(all_gaps_df)} gap data points")
        driver_info = {}
        for _, lap in laps_data.iterrows():
            driver = lap['Driver']
            if driver not in driver_info:
                driver_info[driver] = {
                    'DriverNumber': lap['DriverNumber'],
                    'Team': lap['Team']
                }
        all_gaps_df['Team'] = all_gaps_df['Driver'].map(
            lambda x: driver_info.get(x, {}).get('Team', 'Unknown'))
        print("\n3. Processing and cleaning gap data...")
        all_gaps_df['CarAhead'] = all_gaps_df['CarAhead'].fillna('Leader')
        all_gaps_df['CarAheadNumber'] = all_gaps_df['CarAheadNumber'].fillna(
            -1)
        all_gaps_df['GapToCarAhead'] = all_gaps_df['GapToCarAhead'].fillna(0)
        all_gaps_df['CarBehind'] = all_gaps_df['CarBehind'].fillna('Tail')
        all_gaps_df['CarBehindNumber'] = all_gaps_df['CarBehindNumber'].fillna(
            -1)
        all_gaps_df['GapToCarBehind'] = all_gaps_df['GapToCarBehind'].fillna(0)
        all_gaps_df, consistency_time = time_operation(
            calculate_gap_consistency, all_gaps_df)
        print(
            f"   - Gap consistency calculated in {consistency_time:.2f} seconds")
        if save_path:
            print("\n4. Saving processed gap data...")
            os.makedirs(save_path, exist_ok=True)
            output_path = os.path.join(
                save_path, f"gaps_{grand_prix.lower()}_{year}_data.csv")
            all_gaps_df.to_csv(output_path, float_format="%.3f")
            print(f"   - Saved to: {output_path}")
        print("\n5. Generating strategic recommendations...")
        all_recommendations = []
        strategic_windows = [
            (6, 26),
            (27, 48)
        ]
        for window_name, (start_lap, end_lap) in zip(["Early stint", "Mid stint"], strategic_windows):
            laps_in_window = len(
                all_gaps_df[all_gaps_df['LapNumber'].between(start_lap, end_lap)])
            print(
                f"   - {window_name} window (laps {start_lap}-{end_lap}): {laps_in_window} entries")
        if test_drivers:
            driver_numbers = test_drivers
            print(f"   - Analyzing specific drivers: {driver_numbers}")
        else:
            driver_numbers = all_gaps_df['DriverNumber'].unique()
            print(f"   - Analyzing all {len(driver_numbers)} drivers")
        total_laps = laps_data['LapNumber'].max()
        print(f"   - Total race laps: {total_laps}")
        recommendation_counter = {}
        max_recommendations_per_action = {
            'perform_undercut': 3,
            'defensive_pit': 3,
            'perform_overcut': 3,
            'adjust_pit_window': 5
        }
        min_gap_change_threshold = 0.4
        last_gaps = {}
        for driver_number in driver_numbers:
            print(f"\n   - Processing driver #{driver_number}:")
            recommendation_counter[driver_number] = {
                'early': {'perform_undercut': 0, 'defensive_pit': 0, 'perform_overcut': 0, 'adjust_pit_window': 0},
                'mid': {'perform_undercut': 0, 'defensive_pit': 0, 'perform_overcut': 0, 'adjust_pit_window': 0}
            }
            last_gaps[driver_number] = {
                'early': {'gap_ahead': None, 'gap_behind': None},
                'mid': {'gap_ahead': None, 'gap_behind': None}
            }
            driver_data = all_gaps_df[all_gaps_df['DriverNumber']
                                      == driver_number]
            if driver_data.empty:
                print(f"     * No data found for this driver. Skipping...")
                continue
            for window_name, (start_lap, end_lap) in zip(["early", "mid"], strategic_windows):
                window_laps = driver_data[
                    (driver_data['LapNumber'] >= start_lap) &
                    (driver_data['LapNumber'] <= end_lap)
                ].sort_values('LapNumber')
                if window_laps.empty:
                    print(
                        f"     * No laps in {window_name} stint window (laps {start_lap}-{end_lap})")
                    continue
                print(
                    f"     * Found {len(window_laps)} laps in {window_name} stint window (laps {start_lap}-{end_lap})")
                sampling_interval = 4
                strategic_indices = [0, len(window_laps)-1]
                for i in range(sampling_interval, len(window_laps)-1, sampling_interval):
                    strategic_indices.append(i)
                strategic_indices = sorted(set(strategic_indices))
                strategic_laps = window_laps.iloc[strategic_indices]
                print(
                    f"       - Selected {len(strategic_laps)} strategic laps for analysis")
                for _, lap_data in strategic_laps.iterrows():
                    current_lap = int(lap_data['LapNumber'])
                    print(f"       - Analyzing lap {current_lap}...")
                    single_lap_df = pd.DataFrame([lap_data])
                    try:
                        gap_fact = transform_gap_data_with_consistency(
                            single_lap_df, driver_number)
                        if not gap_fact:
                            print(
                                f"         * Could not create gap fact. Skipping lap...")
                            continue
                        gap_ahead = gap_fact.get('gap_ahead', None)
                        gap_behind = gap_fact.get('gap_behind', None)
                        last_gap_ahead = last_gaps[driver_number][window_name]['gap_ahead']
                        last_gap_behind = last_gaps[driver_number][window_name]['gap_behind']
                        last_gaps[driver_number][window_name]['gap_ahead'] = gap_ahead
                        last_gaps[driver_number][window_name]['gap_behind'] = gap_behind
                        if last_gap_ahead is not None and last_gap_behind is not None:
                            gap_ahead_change = abs(
                                gap_ahead - last_gap_ahead) if gap_ahead and last_gap_ahead else 0
                            gap_behind_change = abs(
                                gap_behind - last_gap_behind) if gap_behind and last_gap_behind else 0
                            if gap_ahead_change < min_gap_change_threshold and gap_behind_change < min_gap_change_threshold:
                                print(
                                    f"         * No significant gap changes. Skipping detailed analysis.")
                                continue
                    except Exception as e:
                        print(
                            f"         * Error creating gap fact: {str(e)}. Skipping lap...")
                        continue
                    position = int(lap_data.get('Position', 0))
                    telemetry_fact = TelemetryFact(
                        driver_number=int(driver_number),
                        position=position
                    )
                    race_status_fact = RaceStatusFact(
                        lap=current_lap,
                        total_laps=int(total_laps),
                        race_phase=window_name,
                        track_status="clear"
                    )
                    engine = F1GapRules()
                    engine.reset()
                    engine.declare(gap_fact)
                    engine.declare(telemetry_fact)
                    engine.declare(race_status_fact)
                    engine.run()
                    lap_recommendations = engine.get_recommendations()
                    if lap_recommendations:
                        driver_name = lap_data[
                            'Driver'] if 'Driver' in lap_data.index else f"Driver-{driver_number}"
                        team = lap_data['Team'] if 'Team' in lap_data.index else "Unknown"
                        filtered_recommendations = []
                        for rec in lap_recommendations:
                            action = rec['action']
                            current_count = recommendation_counter[driver_number][window_name].get(
                                action, 0)
                            max_count = max_recommendations_per_action.get(
                                action, 2)
                            if current_count < max_count:
                                rec['DriverNumber'] = driver_number
                                rec['DriverName'] = driver_name
                                rec['LapNumber'] = current_lap
                                rec['RacePhase'] = window_name
                                rec['Position'] = position
                                rec['Team'] = team
                                filtered_recommendations.append(rec)
                                recommendation_counter[driver_number][window_name][action] = current_count + 1
                        all_recommendations.extend(filtered_recommendations)
                        print(
                            f"         * Generated {len(filtered_recommendations)} recommendations (filtered from {len(lap_recommendations)})")
                        if debug and filtered_recommendations:
                            for i, rec in enumerate(filtered_recommendations):
                                print(
                                    f"           * Recommendation {i+1}: {rec['action']} (Confidence: {rec['confidence']}, Priority: {rec['priority']})")
                    else:
                        print(f"         * No recommendations for this lap")
                    if engine.rules_fired:
                        print(
                            f"         * Rules triggered: {[rule['rule'] for rule in engine.rules_fired]}")
        if all_recommendations:
            recommendations_df = pd.DataFrame(all_recommendations)
            print(
                f"\nTotal recommendations generated: {len(recommendations_df)}")
            if not recommendations_df.empty:
                recommendations_df = recommendations_df.sort_values(
                    ['lap_issued', 'priority', 'confidence'],
                    ascending=[True, False, False]
                )
            print("\n=== Gap Analysis Completed Successfully ===")
            return recommendations_df
        else:
            print("\nNo recommendations were generated for any driver.")
            return pd.DataFrame()
    except Exception as e:
        print(f"\nError during gap analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def test_analyze_race_gaps(
    year=2023,
    grand_prix='Spain',
    session_type='R',
    save_path='../../f1-strategy/data/processed/',
    test_drivers=None
):
    print("Starting comprehensive gap analysis...")
    recommendations = analyze_race_gaps(
        year=year,
        grand_prix=grand_prix,
        session_type=session_type,
        save_path=save_path,
        test_drivers=test_drivers,
        debug=False
    )
    if not recommendations.empty:
        print("\nTop 5 strategic recommendations:")
        print(recommendations.head(5))
        output_path = os.path.join(
            save_path, f"recommendations_{grand_prix.lower()}_{year}.csv")
        recommendations.to_csv(output_path, index=False)
        print(f"Recommendations saved to: {output_path}")
    else:
        print("No recommendations were generated. This could be due to:")
        print("- Insufficient data")
        print("- No cars in strategic windows")
        print("- Rules not matching current race conditions")


if __name__ == "__main__":
    test_analyze_race_gaps()
