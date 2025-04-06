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

# # Lap Prediction Module
#
# This notebook is intended for creating a Python Module where a single function is able to make all the workflow used in the `scripts/lap_prediction.ipynb` notebook.

# ---

# ## 1. Importing Libraries

import pandas as pd
import numpy as np
import pickle
import os

# ---

# ## 2. Loading the prediction model


def load_lap_prediction_model(model_path=None):
    """
    Load the lap time prediction model.
    
    Parameters:
    -----------
    model_path : str, optional
        Path to the model file. If None, uses default path.
        
    Returns:
    --------
    tuple
        (model, feature_names) - The loaded model and its required features
    """
    if model_path is None:
        # Default path
        model_path = "../../outputs/week3/best_xgb_model.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Extract feature names
        feature_names = model.feature_names_in_
        print(f"Model loaded successfully with {len(feature_names)} features")
        return model, feature_names
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

# ---

# ## 3. Validating Lap Data


def validate_lap_data(input_data):
    """
    Validates input data for lap time prediction.
    
    Parameters:
    -----------
    input_data : str or DataFrame
        Path to CSV file or DataFrame containing lap data
        
    Returns:
    --------
    DataFrame
        Validated data
    """
    # Load data if it's a file path
    if isinstance(input_data, str):
        try:
            df = pd.read_csv(input_data)
            print(f"Loaded data from {input_data}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    else:
        # Make a copy to avoid modifying the original
        df = input_data.copy()
        print(f"Using provided DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for required columns
    required_columns = [
        'DriverNumber', 'Stint', 'CompoundID', 'TyreAge', 
        'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'Position'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None
    
    # Check for numerical data types in key columns
    for col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'TyreAge']:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column {col} should be numeric")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add LapNumber if not present (required for sequential features)
    if 'LapNumber' not in df.columns:
        print("Adding LapNumber column based on sequence")
        df['LapNumber'] = df.groupby(['DriverNumber', 'Stint']).cumcount() + 1
    
    # If LapTime is missing (prediction scenario), add placeholder
    if 'LapTime' not in df.columns:
        print("Adding placeholder LapTime column (for prediction only)")
        df['LapTime'] = np.nan
    
    return df


# ---

# ## 4. Adding Sequential Features

def add_sequential_features(df):
    """
    Adds sequential features needed for lap time prediction.
    
    Parameters:
    -----------
    df : DataFrame
        Lap data
        
    Returns:
    --------
    DataFrame
        Data with sequential features added
    """
    # Initialize list to store processed rows
    new_df = []
    
    # Process data for each driver
    for driver in df['DriverNumber'].unique():
        driver_data = df[df['DriverNumber'] == driver]
        
        # Process each stint
        for stint in driver_data['Stint'].unique():
            stint_data = driver_data[driver_data['Stint'] == stint].copy()
            
            # Sort by lap number
            stint_data = stint_data.sort_values('LapNumber')
            
            # We need at least 2 laps to create sequential features
            if len(stint_data) < 2:
                print(f"Skipping driver {driver}, stint {stint}: not enough laps")
                continue
            
            # For each lap starting from the second one
            for i in range(1, len(stint_data)):
                row = stint_data.iloc[i].copy()
                prev_lap = stint_data.iloc[i-1]
                
                # Add previous lap values
                for col in ['LapTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'TyreAge']:
                    if col in stint_data.columns:
                        row[f'Prev_{col}'] = prev_lap[col]
                
                # Calculate delta features
                if 'LapTime' in stint_data.columns and not pd.isna(row['LapTime']) and not pd.isna(prev_lap['LapTime']):
                    row['LapTime_Delta'] = row['LapTime'] - prev_lap['LapTime']
                else:
                    row['LapTime_Delta'] = 0
                
                # Speed deltas
                for speed_col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
                    if speed_col in stint_data.columns:
                        row[f'{speed_col}_Delta'] = row[speed_col] - prev_lap[speed_col]
                
                # Trend calculation (second derivative)
                if i >= 2 and 'LapTime' in stint_data.columns:
                    prev2_lap = stint_data.iloc[i-2]
                    if not pd.isna(row['LapTime']) and not pd.isna(prev_lap['LapTime']) and not pd.isna(prev2_lap['LapTime']):
                        row['LapTime_Trend'] = (row['LapTime'] - prev_lap['LapTime']) - (prev_lap['LapTime'] - prev2_lap['LapTime'])
                    else:
                        row['LapTime_Trend'] = 0
                else:
                    row['LapTime_Trend'] = 0
                
                new_df.append(row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(new_df)
    
    # Fill any missing values in new columns
    seq_cols = [col for col in result_df.columns if col.startswith('Prev_') or col.endswith('_Delta') or col.endswith('_Trend')]
    result_df[seq_cols] = result_df[seq_cols].fillna(0)
    
    print(f"Created sequential features: {len(seq_cols)} new columns")
    return result_df

# ---

# ## 5. Prepare the Features for Prediction


def prepare_features_for_prediction(df, feature_names):
    """
    Prepares features for prediction, ensuring correct format and columns.
    
    Parameters:
    -----------
    df : DataFrame
        Data with sequential features
    feature_names : list
        Required feature names for the model
        
    Returns:
    --------
    DataFrame
        Data prepared for prediction
    """
    # Drop LapTime for prediction input
    X = df.drop('LapTime', axis=1, errors='ignore')
    
    # Handle missing columns
    missing_cols = set(feature_names) - set(X.columns)
    for col in missing_cols:
        print(f"Adding missing column: {col}")
        X[col] = 0
    
    # Remove extra columns
    extra_cols = set(X.columns) - set(feature_names)
    if extra_cols:
        print(f"Removing extra columns: {extra_cols}")
        X = X.drop(columns=extra_cols)
    
    # Ensure correct column order
    X = X[feature_names]
    
    return X



# ---

# ## 6. Formatting the Lap Prediction

def format_lap_predictions(df, predictions):
    """
    Formats prediction results.
    
    Parameters:
    -----------
    df : DataFrame
        Original data with sequential features
    predictions : array
        Model predictions
        
    Returns:
    --------
    DataFrame
        Formatted predictions
    """
    # Create a DataFrame with predictions
    result_df = df.copy()
    result_df['PredictedLapTime'] = predictions
    
    # Calculate prediction metrics where actual lap times exist
    if 'LapTime' in result_df.columns and not result_df['LapTime'].isna().all():
        result_df['PredictionError'] = result_df['PredictedLapTime'] - result_df['LapTime']
        
        # Summary statistics
        rmse = np.sqrt(np.mean(result_df['PredictionError'].dropna() ** 2))
        mae = np.mean(np.abs(result_df['PredictionError'].dropna()))
        print(f"Prediction performance - RMSE: {rmse:.3f}s, MAE: {mae:.3f}s")
    
    # Get key information for a cleaner output
    output_columns = [
        'DriverNumber', 'Stint', 'LapNumber', 'CompoundID', 
        'TyreAge', 'Position', 'LapTime', 'PredictedLapTime'
    ]
    
    output_columns = [col for col in output_columns if col in result_df.columns]
    
    # For drivers, also add next lap prediction
    next_lap_predictions = []
    
    for driver in result_df['DriverNumber'].unique():
        for stint in result_df[result_df['DriverNumber'] == driver]['Stint'].unique():
            driver_stint_data = result_df[(result_df['DriverNumber'] == driver) & 
                                         (result_df['Stint'] == stint)].sort_values('LapNumber')
            
            if len(driver_stint_data) > 0:
                last_lap = driver_stint_data.iloc[-1]
                next_lap_num = last_lap['LapNumber'] + 1
                
                # Create a row for the next lap prediction
                next_lap = {
                    'DriverNumber': driver,
                    'Stint': stint,
                    'LapNumber': next_lap_num,
                    'CompoundID': last_lap['CompoundID'],
                    'TyreAge': last_lap['TyreAge'] + 1,
                    'Position': last_lap['Position'],
                    'LapTime': None,
                    'PredictedLapTime': last_lap['PredictedLapTime'],
                    'IsNextLapPrediction': True
                }
                
                next_lap_predictions.append(next_lap)
    
    # Add next lap predictions if available
    if next_lap_predictions:
        next_lap_df = pd.DataFrame(next_lap_predictions)
        result_df = pd.concat([result_df, next_lap_df], ignore_index=True)
        result_df['IsNextLapPrediction'] = result_df['IsNextLapPrediction'].fillna(False)
    
    # Sort results
    result_df = result_df.sort_values(['DriverNumber', 'Stint', 'LapNumber'])
    
    return result_df


# ---

# ## 7. Predict the lap times

def predict_lap_times(input_data, model_path=None, include_next_lap=True):
    """
    Complete function to predict lap times from telemetry data.
    
    Parameters:
    -----------
    input_data : str or DataFrame
        Path to CSV file or DataFrame containing lap data
    model_path : str, optional
        Path to the model file. If None, uses default.
    include_next_lap : bool, default=True
        Whether to include prediction for the next lap
        
    Returns:
    --------
    DataFrame
        Lap time predictions and relevant metrics
    """
    # Step 1: Load model
    model, feature_names = load_lap_prediction_model(model_path)
    if model is None:
        print("Failed to load model. Aborting prediction.")
        return None
    
    # Step 2: Validate input data
    df = validate_lap_data(input_data)
    if df is None:
        print("Data validation failed. Aborting prediction.")
        return None
    
    # Step 3: Add sequential features
    df_seq = add_sequential_features(df)
    if len(df_seq) == 0:
        print("Failed to create sequential features. Aborting prediction.")
        return None
    
    # Step 4: Prepare features for prediction
    X = prepare_features_for_prediction(df_seq, feature_names)
    
    # Step 5: Make predictions
    print("Making predictions...")
    predictions = model.predict(X)
    
    # Step 6: Format results
    result_df = format_lap_predictions(df_seq, predictions)
    
    print(f"Predictions complete: {len(result_df)} rows")
    return result_df


# ---

# telemetry_data = pd.read_csv('../../outputs/week3/lap_prediction_data.csv.')

# predictions_df = predict_lap_times(telemetry_data)
# predictions_df.head()

if __name__ == "main":
    predictions_df = predict_lap_times('../../outputs/week3/lap_prediction_data.csv.')
    telemetry_data = pd.read_csv('../../outputs/week3/lap_prediction_data.csv.')
    predictions_df = predict_lap_times(telemetry_data)
    predictions_df.head()
    

