from utils.N01_agent_setup import (
    TelemetryFact,
    DegradationFact,
    RaceStatusFact,
    StrategyRecommendation,
    F1StrategyEngine,
    transform_lap_time_predictions,
    load_lap_time_predictions
)
import utils.N01_agent_setup as agent_setup
from experta import DefFacts, Fact, Field, KnowledgeEngine
from experta import Rule, TEST, MATCH
import pandas as pd              # For data manipulation and analysis
import numpy as np               # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns            # For enhanced visualizations
from datetime import datetime    # For timestamp handling
import os                        # For operating system interactions
import sys                       # For system-specific parameters and functions

# Add parent directory to system path to make custom modules accessible
sys.path.append(os.path.abspath('../'))

# Try importing the lap prediction module if available
try:
    from ML_tyre_pred.ML_utils import N00_model_lap_prediction as lp
except ImportError:
    lp = None

# Import Experta components for building the rule engine

# Import custom fact classes and functions from previous modules

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)


def load_lap_time_data(file_path='../../outputs/week3/lap_prediction_data.csv'):
    """
    Load lap time data from file and preprocess it for threshold analysis
    """
    df = pd.read_csv(file_path)
    integer_columns = ['Position', 'TyreAge',
                       'DriverNumber', 'CompoundID', 'TeamID']
    for col in integer_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    sort_columns = [col for col in ['DriverNumber',
                                    'LapNumber', 'TyreAge'] if col in df.columns]
    df = df.sort_values(sort_columns)
    if 'LapTime' in df.columns:
        df['PrevLapTime'] = df.groupby('DriverNumber')['LapTime'].shift(1)
        df['LapTimeDifference'] = df['LapTime'] - df['PrevLapTime']
        if 'Position' in df.columns:
            df['PrevPosition'] = df.groupby('DriverNumber')[
                'Position'].shift(1)
            df['PositionChange'] = df['PrevPosition'] - df['Position']
    return df


class F1LapTimeRules(F1StrategyEngine):
    """
    Engine implementing lap time related rules for F1 Strategy.
    """
    @Rule(
        TelemetryFact(lap_time=MATCH.lap_time),
        TelemetryFact(predicted_lap_time=MATCH.predicted_lap_time),
        TEST(lambda lap_time, predicted_lap_time:
             lap_time is not None and predicted_lap_time is not None and float(predicted_lap_time) < float(lap_time)),
        TelemetryFact(position=MATCH.position),
        TEST(lambda position: position > 3),
        TelemetryFact(tire_age=MATCH.tire_age),
        TEST(lambda tire_age: int(tire_age) < 8),
        RaceStatusFact(lap=MATCH.lap)
    )
    def optimal_performance_window(self, lap_time, predicted_lap_time, position, tire_age, lap):
        improvement = lap_time - predicted_lap_time
        self.declare(
            StrategyRecommendation(
                action="push_strategy",
                confidence=0.75,
                explanation=(f"Car is in optimal performance window. "
                             f"Lap times improving by {improvement:.2f}s with fresh tires ({tire_age} laps). "
                             f"Currently P{position}, opportunity to gain positions."),
                priority=1,
                lap_issued=lap
            )
        )
        self.record_rule_fired("optimal_performance_window")

    @Rule(
        TelemetryFact(lap_time=MATCH.lap_time),
        TelemetryFact(predicted_lap_time=MATCH.predicted_lap_time),
        TEST(lambda lap_time, predicted_lap_time:
             lap_time is not None and predicted_lap_time is not None and
             float(predicted_lap_time) > float(lap_time) + 0.7),
        TelemetryFact(tire_age=MATCH.tire_age),
        TEST(lambda tire_age: int(tire_age) > 15),
        RaceStatusFact(lap=MATCH.lap)
    )
    def performance_cliff_detection(self, lap_time, predicted_lap_time, tire_age, lap):
        slowdown = predicted_lap_time - lap_time
        self.declare(
            StrategyRecommendation(
                action="pit_stop",
                confidence=0.85,
                explanation=(f"Tire performance cliff detected. "
                             f"Model predicts {slowdown:.2f}s slowdown next lap with aged tires ({tire_age} laps). "
                             f"Immediate pit stop recommended."),
                priority=3,
                lap_issued=lap
            )
        )
        self.record_rule_fired("performance_cliff_detection")

    @Rule(
        TelemetryFact(lap_time=MATCH.lap_time),
        TelemetryFact(predicted_lap_time=MATCH.predicted_lap_time),
        TEST(lambda lap_time, predicted_lap_time:
             lap_time is not None and predicted_lap_time is not None and
             float(predicted_lap_time) < float(lap_time) - 0.5),
        RaceStatusFact(lap=MATCH.lap)
    )
    def post_traffic_recovery(self, lap_time, predicted_lap_time, lap):
        improvement = lap_time - predicted_lap_time
        self.declare(
            StrategyRecommendation(
                action="recovery_push",
                confidence=0.7,
                explanation=(f"Clear track ahead detected. "
                             f"Model predicts {improvement:.2f}s improvement in lap time. "
                             f"Recommend recovery push to make up for lost time."),
                priority=2,
                lap_issued=lap
            )
        )
        self.record_rule_fired("post_traffic_recovery")


def test_with_realistic_approach(race_data_path, model_path=None):
    """
    Test the lap time rules using real lap times and predictions.
    """
    race_data = pd.read_csv(race_data_path)
    predictions = load_lap_time_predictions(race_data, model_path)
    processed_data = {}
    driver_groups = race_data.groupby('DriverNumber')
    prediction_groups = predictions.groupby('DriverNumber')
    for driver, group in driver_groups:
        if driver not in prediction_groups.groups:
            continue
        driver_predictions = prediction_groups.get_group(driver)
        driver_data = []
        if 'LapNumber' in group.columns and 'LapNumber' in driver_predictions.columns:
            sorted_laps = group.sort_values('LapNumber')
            sorted_predictions = driver_predictions.sort_values('LapNumber')
            for _, lap in sorted_laps.iterrows():
                current_lap_num = lap['LapNumber']
                next_pred = sorted_predictions[sorted_predictions['LapNumber']
                                               > current_lap_num]
                if not next_pred.empty:
                    pred = next_pred.iloc[0]
                    record = {
                        'DriverNumber': driver,
                        'LapNumber': current_lap_num,
                        'lap_time': lap['LapTime'],
                        'predicted_lap_time': pred['PredictedLapTime'],
                        'TyreAge': lap.get('TyreAge', 0),
                        'Position': lap.get('Position', 0),
                        'CompoundID': lap.get('CompoundID', 0)
                    }
                    driver_data.append(record)
        if driver_data:
            processed_data[driver] = pd.DataFrame(driver_data)
    results = {}
    test_drivers = list(processed_data.keys())[:3]
    for driver in test_drivers:
        driver_data = processed_data[driver]
        driver_data['TimeDifference'] = driver_data['predicted_lap_time'] - \
            driver_data['lap_time']
        performance_cliff = driver_data[driver_data['TimeDifference'] > 0].nlargest(
            1, 'TimeDifference')
        recovery_lap = driver_data[driver_data['TimeDifference'] < 0].nsmallest(
            1, 'TimeDifference')
        test_laps = []
        if not performance_cliff.empty:
            test_laps.append(performance_cliff)
        if not recovery_lap.empty:
            test_laps.append(recovery_lap)
        if not test_laps:
            test_laps.append(driver_data.iloc[[-1]])
        driver_results = []
        for test_lap_df in test_laps:
            test_lap = test_lap_df.iloc[0]
            telemetry_fact = TelemetryFact(
                driver_number=int(driver),
                lap_time=float(test_lap['lap_time']),
                predicted_lap_time=float(test_lap['predicted_lap_time']),
                compound_id=int(test_lap['CompoundID']),
                tire_age=int(test_lap['TyreAge']),
                position=int(test_lap['Position'])
            )
            current_lap = int(test_lap['LapNumber'])
            race_status_fact = RaceStatusFact(
                lap=current_lap,
                total_laps=60,
                race_phase="mid" if 0.25 <= current_lap /
                60 <= 0.75 else "start" if current_lap/60 < 0.25 else "end",
                track_status="clear"
            )
            engine = F1LapTimeRules()
            engine.reset()
            engine.declare(telemetry_fact)
            engine.declare(race_status_fact)
            engine.run()
            recommendations = engine.get_recommendations()
            driver_results.append({
                'lap': current_lap,
                'recommendations': recommendations,
                'rules_fired': engine.rules_fired
            })
        results[driver] = driver_results
    return results


if __name__ == "__main__":
    print("Libraries and fact classes loaded successfully.")

    # Data analysis and threshold computation
    lap_time_data = load_lap_time_data()
    print("Loaded lap time data (first 5 rows):")
    print(lap_time_data.head())

    # Analyze lap time distribution
    if 'LapTimeDifference' in lap_time_data.columns:
        diffs = lap_time_data[(lap_time_data['LapTimeDifference'] > -3)
                              & (lap_time_data['LapTimeDifference'] < 3)]['LapTimeDifference']
        plt.figure(figsize=(10, 6))
        plt.hist(diffs, bins=30, alpha=0.7)
        plt.axvline(diffs.quantile(0.75), linestyle='--',
                    label=f'75th percentile: {diffs.quantile(0.75):.3f}')
        plt.axvline(diffs.quantile(0.25), linestyle='--',
                    label=f'25th percentile: {diffs.quantile(0.25):.3f}')
        plt.axvline(0.7, linestyle='-.',
                    label='Performance cliff threshold: 0.7')
        plt.axvline(-0.5, linestyle='-.',
                    label='Recovery opportunity threshold: -0.5')
        plt.xlabel('Lap Time Difference (s)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Lap Time Differences')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Analyze tire age vs lap time
    if 'TyreAge' in lap_time_data.columns and 'LapTime' in lap_time_data.columns:
        sample = sorted(lap_time_data['DriverNumber'].unique())[:3]
        plt.figure(figsize=(12, 8))
        for d in sample:
            data = lap_time_data[lap_time_data['DriverNumber'] == d]
            trend = data.groupby('TyreAge')['LapTime'].mean().reset_index()
            plt.plot(trend['TyreAge'], trend['LapTime'],
                     'o-', label=f'Driver {d}')
        plt.axvline(8, linestyle='--', label='Fresh tire threshold (8 laps)')
        plt.axvline(15, linestyle='--', label='Old tire threshold (15 laps)')
        plt.xlabel('Tire Age (laps)')
        plt.ylabel('Average Lap Time (s)')
        plt.title('Relationship Between Tire Age and Lap Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Run tests
    race_data_path = '../../outputs/week3/lap_prediction_data.csv'
    realistic_results = test_with_realistic_approach(race_data_path)
    print("Test results:", realistic_results)
