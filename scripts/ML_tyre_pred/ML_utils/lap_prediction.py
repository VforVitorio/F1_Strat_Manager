# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: f1_strat_manager
#     language: python
#     name: python3
# ---

# # Lap Time prediction for Formula 1 Spanish Grand Prix
#
# This notebook implements predictive models to estimate F1 car lap times based on various factors such as tire type, weather conditions, and track state.
#
# ## Objectives
# 1. Load and preprocess data from FastF1 and OpenF1
# 2. Perform feature engineering to enhance predictive capability
# 3. Include analysis of tire degradation and pit stops
# 4. Train predictive models (XGBoost and optionally a Neural Network)
# 5. Evaluate performance and visualize results
#

# ## 1. Modules Used

from sklearn.preprocessing import StandardScaler
import IPython.display as ipd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


compound_colors = {
    1: 'red',     # SOFT
    2: 'yellow',  # MEDIUM
    3: 'gray',    # HARD
    4: 'green',   # INTERMEDIATE
    5: 'blue'     # WET
}

# Compound names for better labels in the legend
compound_names = {
    1: 'SOFT',
    2: 'MEDIUM',
    3: 'HARD',
    4: 'INTERMEDIATE',
    5: 'WET'
}


# Load Sequential XgBoost
with open(r"C:\Users\victo\Desktop\Documents\Tercer año\Segundo Cuatrimestre\Finales\outputs\week3\best_xgb_model.pkl", 'rb') as f:
    model = pickle.load(f)

print("Model loaded succesfully")

# catch all features the model awaits.
feature_names = model.feature_names_in_
print(f"Model waits for {len(feature_names)} features")

print(feature_names)


def create_driver_data(driver_number, stint=1, num_laps=10,
                       base_lap_time=80.0, tyre_compound="Medium",
                       fuel_load_start=100, tyre_age_start=0,
                       tire_deg_per_lap=0.1, fuel_effect=0.03,
                       team_id=1):
    """
    Creates simulation data for a driver over a specified number of laps.

    Parameters:
    -----------
    driver_number : int
        Driver identification number
    stint : int, default=1
        Current stint number (increases after each pit stop)
    num_laps : int, default=10
        Number of laps to simulate
    base_lap_time : float, default=80.0
        Base lap time in seconds (without effects)
    tyre_compound : str, default="Medium"
        Tyre compound type ("Soft", "Medium", "Hard")
    fuel_load_start : float, default=100
        Starting fuel load in kg
    tyre_age_start : int, default=0
        Starting tyre age in laps
    tire_deg_per_lap : float, default=0.1
        Base tyre degradation effect per lap in seconds
    fuel_effect : float, default=0.03
        Effect of 1kg of fuel on lap time in seconds
    team_id : int, default=1
        Team identification number (1-10, where lower is faster)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing simulated lap data with all required features
    """
    data = []

    # Map tyre compounds to their corresponding IDs
    compound_to_id = {"Soft": 1, "Medium": 2, "Hard": 3}
    # Default to Medium if unknown
    compound_id = compound_to_id.get(tyre_compound, 2)

    # Determine next compound (typical F1 strategy patterns)
    if compound_id == 1:  # Soft
        next_compound_id = 2  # Teams typically go to Medium after Soft
    elif compound_id == 2:  # Medium
        next_compound_id = 3  # Teams typically go to Hard after Medium
    else:  # Hard
        next_compound_id = 1  # Back to Soft if started on Hard

    # Realistic degradation coefficients per compound type
    # (seconds lost per lap due to tyre age)
    deg_coefficients = {
        1: 0.15,  # Soft degrades faster
        2: 0.10,  # Medium degrades moderately
        3: 0.05   # Hard degrades slower
    }

    # Base speed by compound (kph) - affects sector speeds
    # Softer tyres provide more grip and higher speeds
    speed_base = {
        1: 220,  # Soft - fastest
        2: 217,  # Medium
        3: 214   # Hard - slowest
    }

    for lap in range(1, num_laps + 1):
        # Calculate tyre age and fuel load for current lap
        tyre_age = tyre_age_start + lap - 1
        fuel_load = max(0, fuel_load_start - (lap - 1)
                        * 2)  # Consume 2kg fuel per lap

        # DRS usage simulation
        # In real F1, DRS is used on straights when within 1 second of car ahead
        drs_used = 1 if np.random.random() < 0.7 else 0  # 70% probability of using DRS

        # DRS effect on lap time (0.4-0.7s gain when activated)
        drs_effect = drs_used * np.random.uniform(0.3, 0.4)

        # Calculate performance effects
        tire_deg_rate = deg_coefficients[compound_id]
        tyre_effect = tyre_age * tire_deg_rate  # Time lost due to tyre wear
        fuel_effect_total = fuel_load * fuel_effect  # Time lost due to fuel weight

        # Team performance effect (top teams are faster)
        # Teams 1-3 gain time, teams 7-10 lose time compared to midfield
        team_effect = (team_id - 5) * 0.1

        # Calculate realistic sector speeds (affected by tyre compound, wear, DRS)
        base_speed = speed_base[compound_id]

        # Sector 1 speed (typically includes DRS zone)
        speed_i1 = base_speed - tyre_effect + \
            np.random.normal(0, 2) + (drs_used * 5)

        # Sector 2 speed (typically more technical)
        speed_i2 = base_speed - 10 - tyre_effect + np.random.normal(0, 2)

        # Final sector speed
        speed_fl = base_speed - 20 - tyre_effect + np.random.normal(0, 3)

        # Speed trap reading (usually on the longest straight)
        speed_st = base_speed - 15 - tyre_effect + np.random.normal(0, 2)

        # Calculate final lap time with all effects
        # Small random variation simulates minor driving differences
        lap_time = (base_lap_time + tyre_effect + fuel_effect_total +
                    team_effect - drs_effect + np.random.normal(0, 0.1))

        # Additional race parameters
        position = np.random.randint(1, 20)  # Track position
        pit_next_lap = 1 if lap == num_laps else 0  # Flag for upcoming pit stop
        fresh_tyre = 0  # Not using fresh tyres during stint
        laps_since_pit = lap  # Laps completed since last pit stop

        # Create data row with all features required by the model
        row = {
            'DriverNumber': driver_number,
            'Stint': stint,
            'LapNumber': lap,
            'CompoundID': compound_id,
            'TyreAge': tyre_age,
            'FuelLoad': fuel_load,
            'SpeedI1': speed_i1,
            'SpeedI2': speed_i2,
            'SpeedFL': speed_fl,
            'SpeedST': speed_st,
            'Position': position,
            'PitNextLap': pit_next_lap,
            'FreshTyreAfterStop': fresh_tyre,
            'LapsSincePitStop': laps_since_pit,
            'DRSUsed': drs_used,
            'TeamID': team_id,
            'NextCompoundID': next_compound_id,
            'LapTime': lap_time
        }
        data.append(row)

    return pd.DataFrame(data)


def add_sequential_features(df):
    """
    Adds sequential features to the DataFrame to capture lap-by-lap evolution.

    This function processes raw lap data to create time-series features that help
    the model understand how performance changes over consecutive laps. These features
    are crucial for accurate lap time prediction in F1, where historical patterns
    strongly influence future performance.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing raw lap data with at least DriverNumber, Stint,
        LapNumber, and LapTime columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with added sequential features:
        - Previous lap values (Prev_*)
        - Delta changes between consecutive laps (*_Delta)
        - Trend calculations over multiple laps (LapTime_Trend)
    """
    # Initialize list to store processed rows
    new_df = []

    # Process data for each driver separately
    for driver in df['DriverNumber'].unique():
        driver_data = df[df['DriverNumber'] == driver]

        # Process each stint separately to avoid incorrect sequences across pit stops
        for stint in driver_data['Stint'].unique():
            stint_data = driver_data[driver_data['Stint'] == stint].copy()

            # Sort by lap number to ensure correct sequential order
            stint_data = stint_data.sort_values('LapNumber')

            # We can only create sequential features from the second lap onwards
            for i in range(1, len(stint_data)):
                # Current lap data
                row = stint_data.iloc[i].copy()
                # Previous lap data
                prev_lap = stint_data.iloc[i-1]

                # Add previous lap values for key performance metrics
                # These help the model understand the starting conditions for the current lap
                for col in ['LapTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'TyreAge']:
                    if col in stint_data.columns:
                        row[f'Prev_{col}'] = prev_lap[col]

                # Calculate lap time delta (improvement or slowdown from previous lap)
                # This is one of the most important indicators of performance trend
                if 'LapTime' in stint_data.columns:
                    row['LapTime_Delta'] = row['LapTime'] - prev_lap['LapTime']

                # Calculate speed deltas for each timing sector
                # These help identify where on track performance is changing
                for speed_col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
                    if speed_col in stint_data.columns:
                        row[f'{speed_col}_Delta'] = row[speed_col] - \
                            prev_lap[speed_col]

                # Calculate second-order trend (acceleration of lap time changes)
                # This helps detect if the car is progressively improving or deteriorating
                if i >= 2:
                    prev2_lap = stint_data.iloc[i-2]
                    if 'LapTime' in stint_data.columns:
                        # Second derivative: how the rate of change itself is changing
                        row['LapTime_Trend'] = (
                            row['LapTime'] - prev_lap['LapTime']) - (prev_lap['LapTime'] - prev2_lap['LapTime'])
                else:
                    # For the second lap, we can't calculate a trend, so use neutral value
                    row['LapTime_Trend'] = 0

                # Add the processed row to our result
                new_df.append(row)

    # Convert list of rows to DataFrame
    result_df = pd.DataFrame(new_df)

    # Fill any missing values in the new columns with zeros
    # This ensures the model doesn't encounter NaN values
    new_cols = [col for col in result_df.columns if col.startswith(
        'Prev_') or col.endswith('_Delta') or col.endswith('_Trend')]
    result_df[new_cols] = result_df[new_cols].fillna(0)

    return result_df


def live_race_simulation(n_laps=10, update_interval=5):
    """
    Simulates a Formula 1 race in real-time with periodic updates and lap time predictions.

    Parameters:
    -----------
    n_laps : int, default=10
        Number of laps to simulate
    update_interval : int, default=5
        Time in seconds between updates
    """
    # Initial configuration of drivers with details
    drivers = [
        {'number': 44, 'compound': 'Soft',
            'base_time': 79.0, 'team_id': 1},    # Mercedes
        {'number': 1, 'compound': 'Medium',
            'base_time': 79.5, 'team_id': 2},   # Red Bull
        {'number': 16, 'compound': 'Hard',
            'base_time': 80.0, 'team_id': 3}     # Ferrari
    ]

    # DataFrame to store all data
    all_data = pd.DataFrame()

    # Store historical data for plotting
    history = {driver['number']: {'laps': [], 'times': [],
                                  'preds': [], 'pred_laps': []} for driver in drivers}

    # Simulate lap by lap
    for lap in range(1, n_laps + 1):
        print(f"\n--- LAP {lap} ---")

        # Data for this lap for all drivers
        lap_data = []

        # Update each driver
        for driver in drivers:
            # Simulate data for this lap with all parameters
            d_num = driver['number']
            data = create_driver_data(
                driver_number=d_num,
                stint=1,
                num_laps=lap,
                base_lap_time=driver['base_time'],
                tyre_compound=driver['compound'],
                tyre_age_start=0,
                team_id=driver['team_id']
            )

            # Save only the last lap
            last_lap = data.iloc[-1:].copy()
            lap_data.append(last_lap)

            # Save for history
            laptime = last_lap['LapTime'].values[0]
            history[d_num]['laps'].append(lap)
            history[d_num]['times'].append(laptime)

            # Display lap time
            print(
                f"Driver {d_num} ({driver['compound']}, Team {driver['team_id']}): {laptime:.3f}s")

        # Concatenate data from this lap
        current_lap_df = pd.concat(lap_data, ignore_index=True)

        # Add to historical data
        all_data = pd.concat([all_data, current_lap_df], ignore_index=True)

        # If there are enough laps, predict the next one
        if lap >= 2:
            # Create sequential features
            seq_data = add_sequential_features(all_data)

            # Prepare for prediction
            X_pred = seq_data.drop('LapTime', axis=1)

            # Ensure we have the correct features
            missing_cols = set(feature_names) - set(X_pred.columns)
            for col in missing_cols:
                X_pred[col] = 0

            extra_cols = set(X_pred.columns) - set(feature_names)
            if extra_cols:
                X_pred = X_pred.drop(columns=extra_cols)

            # Ensure correct column order
            X_pred = X_pred[feature_names]

            # Predict
            predictions = model.predict(X_pred)

            # Show predictions for the next lap
            print("\nPredictions for the next lap:")
            for i, driver in enumerate(drivers):
                idx = seq_data['DriverNumber'] == driver['number']
                if any(idx):
                    pred_idx = np.where(idx)[0][-1]
                    pred = predictions[pred_idx]
                    d_num = driver['number']
                    print(f"Driver {d_num}: {pred:.3f}s (estimated)")

                    # Save prediction
                    if lap <= n_laps - 1:  # Only if there is a next lap
                        history[d_num]['preds'].append(pred)
                        history[d_num]['pred_laps'].append(lap + 1)

        # Visualize evolution in real time
        if lap >= 3:  # When there's enough data to show trends
            plt.figure(figsize=(14, 8))

            # Colors and markers by tire type
            styles = {
                'Soft': {'color': 'red', 'marker': 'o'},
                'Medium': {'color': 'yellow', 'marker': 's'},
                'Hard': {'color': 'gray', 'marker': '^'}
            }

            for driver in drivers:
                d_num = driver['number']
                style = styles[driver['compound']]

                # Real times
                plt.plot(history[d_num]['laps'], history[d_num]['times'],
                         color=style['color'], marker=style['marker'],
                         label=f"Driver {d_num} ({driver['compound']})")

                # Predictions
                if len(history[d_num]['preds']) > 0:
                    plt.plot(history[d_num]['pred_laps'], history[d_num]['preds'],
                             color=style['color'], linestyle='--', alpha=0.7,
                             label=f"Driver {d_num} - Predictions")

            plt.title("Real-Time Lap Time Evolution")
            plt.xlabel("Lap Number")
            plt.ylabel("Time (s)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ipd.display(plt.gcf())  # Cambiado a ipd.display
            plt.close()

        # Wait before the next lap
        if lap < n_laps:
            print(f"\nWaiting {update_interval} seconds...")
            time.sleep(update_interval)
            ipd.clear_output(wait=True)  # Cambiado a ipd.clear_output
