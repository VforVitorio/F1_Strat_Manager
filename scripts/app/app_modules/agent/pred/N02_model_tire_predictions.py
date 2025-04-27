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

# # Notebook for making a Module for making predictions of any data
#
# This notebook is intented to implement, through all the code designed in `N01_tire_prediction.ipynb`, a function that we could use in any other notebook for making fast predictions with our tire models.

# ## 1. Importing neccesary Libraries

# ---

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt  # Optional, for visualization


# ## 2. Redefining the model´s class.
#
# This step is important for loading the models, as `.pth` files do not save the model instantiation. This can be good as we can have more options for making further changes in the code, but I will just re-implement the class.

# ---

# Define the TCN model architecture class
class EnhancedTCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, dropout=0.3):
        super(EnhancedTCN, self).__init__()

        # Input projection to higher-dimensional space
        self.input_proj = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.bn_input = nn.BatchNorm1d(hidden_size)

        # Multi-scale block with exponential dilations
        self.dilated_conv1 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding='same', dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dilated_conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding='same', dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dilated_conv4 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding='same', dilation=4)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.dilated_conv8 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding='same', dilation=8)
        self.bn8 = nn.BatchNorm1d(hidden_size)

        # Simple temporal attention mechanism
        self.attention = nn.Conv1d(hidden_size, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # Output layer with higher dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, features]
        # Transform to [batch, features, seq_len] for Conv1d
        x = x.transpose(1, 2)

        # Initial projection
        x = F.relu(self.bn_input(self.input_proj(x)))

        # Apply dilated convolutions with residual connections
        residual = x
        x = F.relu(self.bn1(self.dilated_conv1(x)))
        x = x + residual  # Residual connection

        residual = x
        x = F.relu(self.bn2(self.dilated_conv2(x)))
        x = x + residual  # Residual connection

        residual = x
        x = F.relu(self.bn4(self.dilated_conv4(x)))
        x = x + residual  # Residual connection

        residual = x
        x = F.relu(self.bn8(self.dilated_conv8(x)))
        x = x + residual  # Residual connection

        # Apply attention mechanism
        # Shape: [batch, 1, seq_len]
        attn_weights = self.softmax(self.attention(x))
        x = x * attn_weights

        # Global pooling: sum over time dimension
        x = torch.sum(x, dim=2)  # Shape: [batch, channels]
        x = self.dropout(x)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ---

# ## 3. Load and Validate Data
#
#
# In this function we ensure that the model is going to receive the data disposed in the way it is needed.

def load_and_validate_data(input_data):
    """
    Loads and validates input data, ensuring it contains all necessary columns.

    Parameters:
    - input_data: Can be a DataFrame or a path to a CSV file

    Returns:
    - DataFrame with validated data

    Raises:
    - ValueError: If required columns are missing or contain null values
    """
    # Load data - handle both DataFrame and file path inputs
    if isinstance(input_data, str):
        # If input is a string, assume it's a file path and read the CSV
        df = pd.read_csv(input_data)
    else:
        # If input is already a DataFrame, make a copy to avoid modifying the original
        df = input_data.copy()

    # Define required columns for the model
    # These are the essential columns needed for prediction
    required_columns = [
        'LapTime',       # Time taken to complete the lap
        'Stint',         # Current stint number (resets after pit stop)
        'CompoundID',    # ID of the tire compound (1=Soft, 2=Medium, 3=Hard)
        'TyreAge',       # Number of laps the current set of tires has completed
        'FuelLoad',      # Estimated fuel load in kg
        'DriverNumber',  # Driver's race number
        'Position',      # Current race position
        'SpeedI1',       # Speed at first intermediate point
        'SpeedI2',       # Speed at second intermediate point
        'SpeedFL'        # Speed at flying lap point
    ]

    # Verify all required columns are present in the DataFrame
    missing_columns = [
        col for col in required_columns if col not in df.columns]
    if missing_columns:
        # If any required columns are missing, raise an error with details
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for null values in required columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.sum() > 0:
        # If null values are found, print a warning with details
        print(f"Warning: Null values found in data:")
        print(null_counts[null_counts > 0])

        # For TyreAge and LapTime, we cannot have null values as they are critical
        if df['TyreAge'].isnull().any() or df['LapTime'].isnull().any():
            raise ValueError(
                "Cannot have null values in 'TyreAge' or 'LapTime'")

    # Ensure data types are correct for model processing
    # Convert categorical IDs to integers
    df['CompoundID'] = df['CompoundID'].astype(int)
    df['DriverNumber'] = df['DriverNumber'].astype(int)

    # Ensure numeric columns are properly typed
    numeric_columns = ['LapTime', 'TyreAge',
                       'FuelLoad', 'SpeedI1', 'SpeedI2', 'SpeedFL']
    for col in numeric_columns:
        # Convert to numeric, coercing errors (invalid values become NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Log successful validation
    print(
        f"Data loaded and validated: {df.shape[0]} rows, {df.shape[1]} columns")

    # Return the validated DataFrame
    return df


# ## 4. Calculate Degradation Metrics

# ---

def calculate_degradation_metrics(df):
    """
    Calculates degradation metrics adjusted for the fuel effect.

    This function processes lap data to compute tire degradation metrics after removing
    the effect of fuel burn, which naturally makes cars faster as the race progresses.

    Parameters:
    - df: DataFrame with lap data

    Returns:
    - DataFrame with added degradation metrics:
        - FuelAdjustedLapTime: Lap time with fuel effect removed
        - FuelAdjustedDegPercent: Percentage degradation compared to baseline
        - DegradationRate: Rate of degradation (seconds per lap)
    """
    # Constant for lap time improvement due to fuel reduction
    # Based on empirical F1 data: cars improve ~0.055s per lap due to fuel burning
    LAP_TIME_IMPROVEMENT_PER_LAP = 0.055  # seconds per lap

    # Create a copy of the DataFrame to store results
    result_df = df.copy()

    # Define mapping of compound names for logging
    compound_names = {
        0: 'Unknown',
        1: 'Soft',
        2: 'Medium',
        3: 'Hard'
    }

    # Process each tire compound separately
    for compound_id in df['CompoundID'].unique():
        compound_name = compound_names.get(
            compound_id, f"Unknown ({compound_id})")
        print(f"Processing {compound_name} tires (ID: {compound_id})...")

        # Filter data for this compound
        compound_data = df[df['CompoundID'] == compound_id].copy()

        # Check if we have enough data for meaningful analysis
        if len(compound_data) < 5:
            print(f"  Not enough data for {compound_name} tires, skipping")
            continue

        # Establish baseline information
        # Ideally use new tires (TyreAge=1) as baseline, or minimum available age
        if 1 in compound_data['TyreAge'].values:
            # Get baseline from new tires (TyreAge=1)
            baseline_data = compound_data[compound_data['TyreAge'] == 1]
            baseline_lap_time = baseline_data['LapTime'].mean()
            baseline_tire_age = 1
        else:
            # If no data with new tires, use minimum available age
            min_age = compound_data['TyreAge'].min()
            baseline_data = compound_data[compound_data['TyreAge'] == min_age]
            baseline_lap_time = baseline_data['LapTime'].mean()
            baseline_tire_age = min_age
            print(
                f"  No laps with new tires for {compound_name}, using TyreAge={min_age} as baseline")

        # Calculate fuel adjustment based on laps from baseline
        # Each lap burned fuel makes the car ~0.055s faster per lap
        compound_data['LapsFromBaseline'] = compound_data['TyreAge'] - \
            baseline_tire_age
        compound_data['FuelEffect'] = compound_data['LapsFromBaseline'] * \
            LAP_TIME_IMPROVEMENT_PER_LAP

        # Calculate fuel-adjusted lap time
        # Add fuel effect back to compensate for the artificial improvement
        compound_data['FuelAdjustedLapTime'] = compound_data['LapTime'] + \
            compound_data['FuelEffect']

        # Calculate degradation percentage (compared to baseline)
        # This shows how much slower (in %) the car is compared to baseline performance
        # For new tires, no adjustment needed
        baseline_adjusted_lap_time = baseline_lap_time
        compound_data['FuelAdjustedDegPercent'] = (
            compound_data['FuelAdjustedLapTime'] / baseline_adjusted_lap_time - 1) * 100

        # Add calculated metrics to the result DataFrame
        for col in ['LapsFromBaseline', 'FuelEffect', 'FuelAdjustedLapTime', 'FuelAdjustedDegPercent']:
            idx = compound_data.index
            result_df.loc[idx, col] = compound_data[col]

    # Calculate degradation rate (lap-to-lap changes in performance)
    # This requires grouping by driver, stint, and compound
    groupby_columns = ['DriverNumber', 'Stint', 'CompoundID']

    # Initialize degradation rate column
    result_df['DegradationRate'] = 0

    # Process each driver-stint-compound group separately
    for name, group in result_df.groupby(groupby_columns):
        # Sort by TyreAge to ensure correct calculation
        group = group.sort_values('TyreAge')

        # Calculate differences between consecutive laps
        if len(group) > 1:
            group_idx = group.index
            lap_times = group['FuelAdjustedLapTime'].values

            # Calculate lap time differences (how much slower each lap is than the previous)
            # Using np.diff with prepend to keep array length consistent
            diffs = np.diff(lap_times, prepend=lap_times[0])
            # First value is invalid (self-comparison), set to 0
            diffs[0] = 0

            # Assign to the result DataFrame
            result_df.loc[group_idx, 'DegradationRate'] = diffs

    # Fill any remaining null values in DegradationRate
    # New tires (first lap of a stint) have no degradation rate yet
    result_df['DegradationRate'] = result_df['DegradationRate'].fillna(0)

    print("Degradation metrics successfully calculated")
    return result_df


# ## 5. Variable Cleaning

# ---

def clean_redundant_variables(df):
    """
    Ensures data format matches exactly what the model was trained on.

    Our model expects exactly 16 features in a specific order.
    This function ensures we provide data in the expected format.

    Parameters:
    - df: DataFrame with degradation metrics calculated

    Returns:
    - DataFrame with exact 16 features needed for model
    """
    # Check which columns are available
    available_columns = set(df.columns)

    # Define required columns in the exact order expected by the model
    # These must match exactly what was used during training
    required_columns = [
        # Core metrics
        'FuelAdjustedLapTime',
        'FuelAdjustedDegPercent',
        'DegradationRate',
        'TyreAge',
        'CompoundID',

        # Speed metrics
        'SpeedI1',
        'SpeedI2',
        'SpeedFL',
        'SpeedST',  # This was missing in our original list

        # Race context
        'Position',
        'FuelLoad',
        'DriverNumber',
        'Stint',
        'LapsSincePitStop',  # This was missing
        'DRSUsed',           # This was missing
        'TeamID'             # This was missing
    ]

    # Ensure we have all required columns available
    missing_columns = [
        col for col in required_columns if col not in available_columns]
    if missing_columns:
        print(f"Warning: Missing columns required by model: {missing_columns}")
        print("Adding these columns with default values.")

        # Add missing columns with sensible defaults
        for col in missing_columns:
            if col == 'SpeedST':
                # Use average of other speeds
                df[col] = df[['SpeedI1', 'SpeedI2', 'SpeedFL']].mean(axis=1)
            elif col == 'DRSUsed':
                # Default to 0 (not used)
                df[col] = 0
            elif col == 'TeamID':
                # Default to 0
                df[col] = 0
            elif col == 'LapsSincePitStop':
                # Set to same as TyreAge
                df[col] = df['TyreAge']
            else:
                # Default to 0 for any other missing column
                df[col] = 0

    # Create DataFrame with exactly the required columns in the correct order
    cleaned_df = df[required_columns].copy()

    print(f"Processed data format: {cleaned_df.shape[1]} features")
    return cleaned_df


# ## 6. Create sequences for predictions

# ### Strategic Tire Degradation Monitoring
#
# #### Why Monitor Tire Degradation from Specific Lap Thresholds?
#
# Formula 1 tire performance follows a predictable pattern throughout its lifecycle:
#
# 1. **Initial Phase (First Few Laps)**:
#    - Tires are in their optimal window with minimal degradation
#    - Performance is relatively stable and predictable
#    - Strategic decisions rarely needed during this phase
#
# 2. **Critical Monitoring Phase**:
#    - Begins when tires start showing meaningful degradation patterns
#    - Different compounds reach this phase at different times:
#      - **Soft Compounds**: ~6 laps (faster degradation)
#      - **Medium Compounds**: ~12 laps (moderate degradation)
#      - **Hard Compounds**: ~25 laps (slower degradation)
#    - This is when predictive models become most valuable for strategy decisions
#
# 3. **End-of-Life Phase**:
#    - Severe performance drop-off ("cliff")
#    - Critical for pit stop timing decisions
#
# By focusing our monitoring on the compound-specific critical phases, we:
# - Reduce noise from initial break-in laps
# - Focus computational resources on strategically relevant predictions
# - Improve model accuracy by training on more consistent degradation patterns
# - Better align predictions with real-world strategic decision points
#
# This approach mirrors how F1 teams monitor tires during races, where they typically start considering tire strategy once compounds reach their respective monitoring thresholds.
#
#

# ---

def create_sequences_for_prediction(df, input_length=5, compound_start_laps=None):
    """
    Creates sequences for prediction starting from specific lap thresholds based on tire compound.

    Parameters:
    - df: Clean DataFrame with relevant variables
    - input_length: Number of consecutive laps to include in each sequence (default: 5)
    - compound_start_laps: Dictionary mapping compound IDs to starting lap numbers
                          Example: {1: 6, 2: 12, 3: 25} for monitoring soft tires from lap 6,
                          medium from lap 12, and hard from lap 25

    Returns:
    - sequences: List of DataFrames, each containing a sequence of consecutive laps
    - metadata: List of dictionaries with metadata for each sequence
    """
    # Initialize lists to store sequences and their metadata
    sequences = []
    metadata = []

    # Default thresholds if none provided
    if compound_start_laps is None:
        # Default: monitor all from lap 1
        compound_start_laps = {1: 1, 2: 1, 3: 1}

    # Group data by driver, stint, and compound
    groupby_columns = ['DriverNumber', 'Stint', 'CompoundID']

    # Process each group separately
    for name, group in df.groupby(groupby_columns):
        # Unpack the group identifier
        driver, stint, compound = name

        # Get the starting lap threshold for this compound
        min_tyre_age = compound_start_laps.get(compound, 1)

        # Sort by TyreAge to ensure chronological order
        sorted_group = group.sort_values('TyreAge').reset_index(drop=True)

        # Filter group to only include laps at or after the starting threshold
        filtered_group = sorted_group[sorted_group['TyreAge'] >= min_tyre_age]

        # Skip if we don't have enough laps for a complete sequence
        if len(filtered_group) < input_length:
            continue

        # Create sliding window sequences
        for i in range(len(filtered_group) - input_length + 1):
            # Extract sequence of 'input_length' consecutive laps
            seq = filtered_group.iloc[i:i+input_length]

            # Add sequence to the list
            sequences.append(seq)

            # Store metadata
            meta = {
                'DriverNumber': driver,
                'Stint': stint,
                'CompoundID': compound,
                'StartLap': seq['TyreAge'].iloc[0],
                'EndLap': seq['TyreAge'].iloc[-1],
                'LatestLapTime': seq['FuelAdjustedLapTime'].iloc[-1]
            }
            metadata.append(meta)

    # Log information about created sequences
    print(f"Created {len(sequences)} sequences of {input_length} laps each")
    print(
        f"Sequences by compound: {pd.Series([m['CompoundID'] for m in metadata]).value_counts().to_dict()}")

    return sequences, metadata


# ## 7. Loading the models

# ---

def load_models(models_path):
    """
    Loads pre-trained models (global and compound-specialized).

    This function loads the global TCN model that was trained on all data, as well as
    any available specialized models that were trained on specific tire compounds.

    Parameters:
    - models_path: Path where trained models are stored

    Returns:
    - global_model: Trained TCN model for all compounds
    - specialized_models: Dictionary mapping compound IDs to specialized models

    Raises:
    - FileNotFoundError: If global model cannot be found
    - ImportError: If PyTorch is not available
    """
    import torch
    import os
    from pathlib import Path

    # Check if the models directory exists
    if not os.path.exists(models_path):
        raise ValueError(f"Models path does not exist: {models_path}")

    # Define model architecture parameters (must match training configuration)
    input_size = 16  # Number of features in input sequence
    output_size = 3  # Number of future laps to predict

    # Determine computation device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize global model with correct architecture
    global_model = EnhancedTCN(input_size, output_size)

    # Path to global model weights
    global_model_path = os.path.join(models_path, 'tire_degradation_tcn.pth')

    # Load global model weights
    if os.path.exists(global_model_path):
        global_model.load_state_dict(torch.load(
            global_model_path, map_location=device))
        global_model.to(device)  # Move model to the correct device
        global_model.eval()      # Set model to evaluation mode
        print(f"Global model loaded from: {global_model_path}")
    else:
        raise FileNotFoundError(
            f"Global model not found at: {global_model_path}")

    # Initialize dictionary to store specialized models
    specialized_models = {}

    # Try to load specialized models for each compound
    # We only consider compounds 1-3 (Soft, Medium, Hard)
    compound_ids = [1, 2, 3]  # Soft, Medium, Hard

    for compound_id in compound_ids:
        # Path to specialized model for this compound
        model_path = os.path.join(
            models_path, f'tcn_compound_{compound_id}.pth')

        # If specialized model exists, load it
        if os.path.exists(model_path):
            # Initialize model with the same architecture
            specialized_model = EnhancedTCN(input_size, output_size)

            # Load weights and move to device
            specialized_model.load_state_dict(
                torch.load(model_path, map_location=device))
            specialized_model.to(device)
            specialized_model.eval()

            # Add to our dictionary of specialized models
            specialized_models[compound_id] = specialized_model
            print(f"Specialized model for compound {compound_id} loaded")

    # Summarize what was loaded
    print(
        f"Models loaded: 1 global model and {len(specialized_models)} specialized models")

    return global_model, specialized_models


# ## 8. Preparing sequences for the model

# ---

def prepare_sequences_for_model(sequences):
    """
    Converts DataFrame sequences into PyTorch tensors suitable for the TCN model.

    This function takes a list of DataFrame sequences and converts them into a
    single tensor with shape [num_sequences, sequence_length, num_features].

    Parameters:
    - sequences: List of DataFrames, each containing a sequence of consecutive laps

    Returns:
    - PyTorch tensor with shape [num_sequences, sequence_length, num_features]

    Raises:
    - ValueError: If the sequences list is empty
    """
    import numpy as np
    import torch

    # Check if we have sequences to process
    if not sequences:
        raise ValueError("No sequences to process")

    # Get dimensions from the first sequence
    n_features = len(sequences[0].columns)  # Number of features (columns)
    sequence_length = len(sequences[0])     # Length of each sequence (rows)

    # Initialize a numpy array to hold all sequences
    # Shape: [num_sequences, sequence_length, num_features]
    X = np.zeros((len(sequences), sequence_length, n_features))

    # Fill the array with data from each sequence
    for i, seq in enumerate(sequences):
        X[i] = seq.values

    # Convert numpy array to PyTorch tensor
    X_tensor = torch.FloatTensor(X)

    print(f"Prepared tensor for model: shape={X_tensor.shape}")
    return X_tensor


# ## 9. Making the ensemble predictions

# ---

def make_ensemble_predictions(sequences, metadata, global_model, specialized_models, device=None):
    """
    Makes predictions using an ensemble of global and compound-specialized models.

    This function:
    1. Converts sequences to tensor format
    2. Obtains predictions from the global model
    3. For sequences where a specialized model exists, gets specialized predictions
    4. Combines predictions using a weighted average based on model performance

    Parameters:
    - sequences: List of DataFrames, each containing a sequence of consecutive laps
    - metadata: List of dictionaries with metadata for each sequence
    - global_model: Trained TCN model for all compounds
    - specialized_models: Dictionary mapping compound IDs to specialized models
    - device: Computing device (CPU/GPU); if None, will be auto-detected

    Returns:
    - List of dictionaries containing predictions and metadata for each sequence
    """
    import torch
    import numpy as np

    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert sequences to tensor format suitable for the model
    X = prepare_sequences_for_model(sequences)
    X = X.to(device)

    # Reference RMSE values for each model
    # These values come from model evaluation in the notebook
    global_rmse = 0.355017  # RMSE of the global model
    compound_rmse = {
        1: 0.334325,  # Soft
        2: 0.392661,  # Medium
        3: 0.295417   # Hard
    }

    # Initialize list to store all prediction results
    all_predictions = []

    # Get predictions from the global model for all sequences at once
    global_model.eval()  # Set to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        global_preds = global_model(X).cpu().numpy()

    # Process each sequence individually to apply the appropriate ensemble weights
    for i, meta in enumerate(metadata):
        compound_id = meta['CompoundID']

        # Initialize result dictionary with global prediction
        result = {
            'metadata': meta,
            'global_prediction': global_preds[i],
            'global_weight': 1.0,
            'specialized_prediction': None,
            'specialized_weight': 0.0,
            # Default to global prediction
            'ensemble_prediction': global_preds[i]
        }

        # If a specialized model exists for this compound, get its prediction
        if compound_id in specialized_models:
            specialized_model = specialized_models[compound_id]
            specialized_model.eval()

            # Extract single sequence with batch dimension preserved
            sequence_tensor = X[i:i+1]

            # Get prediction from specialized model
            with torch.no_grad():
                specialized_pred = specialized_model(
                    sequence_tensor).cpu().numpy()[0]

            # Calculate ensemble weights based on inverse RMSE
            # Lower RMSE = higher weight (better model gets more influence)
            specialized_rmse = compound_rmse.get(compound_id, global_rmse)
            global_weight = 1 / global_rmse
            specialized_weight = 1 / specialized_rmse

            # Normalize weights to sum to 1
            total_weight = global_weight + specialized_weight
            global_weight /= total_weight
            specialized_weight /= total_weight

            # Weighted combination of predictions
            ensemble_pred = (global_weight * global_preds[i] +
                             specialized_weight * specialized_pred)

            # Update result with specialized model info
            result['specialized_prediction'] = specialized_pred
            result['specialized_weight'] = specialized_weight
            result['global_weight'] = global_weight
            result['ensemble_prediction'] = ensemble_pred

        # Add to complete results list
        all_predictions.append(result)

    print(
        f"Generated ensemble predictions for {len(all_predictions)} sequences")
    return all_predictions


# ## 10. Prediction formating

# ---

def format_predictions(predictions, original_df):
    """
    Formats raw ensemble predictions into a structured, analysis-ready DataFrame
    with simplified columns and controlled decimal precision.

    Parameters:
    - predictions: List of dictionaries with ensemble predictions and metadata
    - original_df: Original DataFrame for reference information

    Returns:
    - DataFrame with formatted predictions for each future lap
    """
    import pandas as pd

    # Define mapping of compound IDs to names for better readability
    compound_names = {
        1: 'Soft',
        2: 'Medium',
        3: 'Hard'
    }

    # Initialize list to store formatted results
    results = []

    # Process each prediction (one per sequence)
    for pred in predictions:
        # Extract metadata for this sequence
        meta = pred['metadata']
        # Get the ensemble prediction (array of 3 values for future laps)
        ensemble_prediction = pred['ensemble_prediction']
        # Get compound name for readability
        compound_name = compound_names.get(
            meta['CompoundID'], f"Unknown {meta['CompoundID']}")

        # Create base information common to all future laps
        base_info = {
            'DriverNumber': meta['DriverNumber'],         # Driver identifier
            'Stint': meta['Stint'],                       # Current stint
            'CompoundID': meta['CompoundID'],             # Numeric compound ID
            'CompoundName': compound_name,                # Human-readable compound name
            # Current age of tires
            'CurrentTyreAge': meta['EndLap'],
            # Most recent lap time (3 decimals)
            'CurrentLapTime': round(meta['LatestLapTime'], 3)
        }

        # Create a row for each future lap prediction
        for i, future_value in enumerate(ensemble_prediction):
            # Calculate the future lap number
            lap_number = meta['EndLap'] + i + 1

            # Create a result entry for this future lap
            result = base_info.copy()
            result.update({
                # Future lap number (TyreAge)
                'FutureLap': lap_number,
                'LapsAheadPred': i + 1,                 # How many laps ahead of current
                # Predicted degradation rate (3 decimals)
                'PredictedDegradationRate': round(future_value, 3)
            })

            # Add to results list
            results.append(result)

    # Convert list of dictionaries to DataFrame
    results_df = pd.DataFrame(results)

    # Sort for easier analysis
    # First by driver, then by stint, then by future lap
    results_df = results_df.sort_values(['DriverNumber', 'Stint', 'FutureLap'])

    # Log summary statistics
    print(
        f"Formatted results: {len(results_df)} predictions for {results_df['DriverNumber'].nunique()} drivers")

    return results_df


# # 11. The great Function: Predicting tire degradation

# ---

def predict_tire_degradation(input_data, models_path='../../outputs/week5/models/', compound_start_laps=None):
    """
    Complete function to predict tire degradation starting from specified tire age thresholds.

    Parameters:
    - input_data: DataFrame or path to CSV with lap data
    - models_path: Path where the saved models are located
    - compound_start_laps: Dictionary mapping compound IDs to starting lap numbers
                          Example: {1: 6, 2: 12, 3: 25}

    Returns:
    - DataFrame with degradation predictions for the next 3 laps
    """
    # Step 1: Load and validate data
    df = load_and_validate_data(input_data)

    # Step 2: Calculate metrics
    df = calculate_degradation_metrics(df)

    # Step 3: Clean redundant variables
    df = clean_redundant_variables(df)

    # Step 4-5: Create sequences
    sequences, metadata = create_sequences_for_prediction(
        df, compound_start_laps=compound_start_laps)

    # Step 6: Load models
    global_model, specialized_models = load_models(models_path)

    # Step 7: Ensemble prediction
    predictions = make_ensemble_predictions(
        sequences, metadata, global_model, specialized_models)

    # Step 8: Format results
    results = format_predictions(predictions, df)

    return results


# ## 12. Example for Calling it

# #### Disclaimer for compound start laps
#
#
# For the example usage, as the data is from the 2023 Spanish Grand Prix, I will base my starting predictions from the pitstops made in the Grand Prix.
# <p align="center">
#   <img src="../ML_tyre_pred/ML_utils/pitstops_bar.jpg" alt="Texto alternativo" width="800"/>
# </p>
#


def simulate_real_time_predictions(csv_path, models_path, interval=5, compound_start_laps=None, max_rows=None, prediction_horizon=3):
    """
    Simulates a real-time racing environment by processing a CSV row by row
    and running the complete prediction model at regular intervals.

    Parameters:
    - csv_path: Path to CSV with lap data
    - models_path: Path where models are stored
    - interval: Time interval between updates in seconds (default: 5)
    - compound_start_laps: Lap thresholds for each compound
    - max_rows: Maximum number of rows to process (None = all)
    - prediction_horizon: Number of laps ahead to highlight in predictions (default: 3)

    Returns:
    - DataFrame with complete prediction history
    """
    import pandas as pd
    import time
    from IPython.display import display, clear_output

    # Read the full CSV for incremental processing
    full_df = pd.read_csv(csv_path)

    # Limit the number of rows if specified
    if max_rows is not None:
        full_df = full_df.iloc[:max_rows]

    # Accumulative DataFrame to simulate real-time data arrival
    accumulated_df = pd.DataFrame(columns=full_df.columns)

    # DataFrame to store all historical predictions
    all_predictions = pd.DataFrame()

    print(
        f"Starting simulation with {len(full_df)} data rows, interval: {interval}s")

    # Process row by row
    for i, row in full_df.iterrows():
        # Add the new row to the accumulative DataFrame
        accumulated_df = pd.concat(
            [accumulated_df, pd.DataFrame([row])], ignore_index=True)

        # Execute complete prediction with accumulated data
        try:
            # Use existing function with all accumulated data so far
            latest_predictions = predict_tire_degradation(
                accumulated_df,  # Pass DataFrame instead of CSV path
                models_path,
                compound_start_laps
            )

            # Clear all previous output now that we have new predictions
            clear_output(wait=False)

            # If there are predictions, display and save them
            if not latest_predictions.empty:
                # Add timestamp for tracking (without DataRowsProcessed)
                latest_predictions['PredictionTimestamp'] = pd.Timestamp.now()

                # First show key degradation predictions
                print("=== KEY DEGRADATION PREDICTIONS ===")

                # Group by driver
                for driver, driver_group in latest_predictions.groupby('DriverNumber'):
                    # Get the latest stint for this driver
                    latest_stint = driver_group['Stint'].max()
                    stint_data = driver_group[driver_group['Stint']
                                              == latest_stint]

                    # Get predictions for the requested horizon
                    near_future = stint_data[stint_data['LapsAheadPred']
                                             <= prediction_horizon]

                    if not near_future.empty:
                        # Get only the unique LapsAheadPred values, sorted
                        lap_ahead_values = sorted(
                            near_future['LapsAheadPred'].unique())

                        # Get the most recent CurrentTyreAge (sort by CurrentTyreAge descending and take first)
                        most_recent_data = near_future.sort_values(
                            'CurrentTyreAge', ascending=False).iloc[0]
                        compound = most_recent_data['CompoundName']
                        current_age = most_recent_data['CurrentTyreAge']

                        print(
                            f"\nDriver #{driver} - {compound} tires (Current age: {current_age})")

                        # For each unique lap ahead value, get only the most recent prediction
                        for lap_ahead in lap_ahead_values:
                            # Get predictions for this specific lap ahead
                            lap_preds = near_future[near_future['LapsAheadPred']
                                                    == lap_ahead]

                            # Sort by current tyre age (most recent window)
                            lap_preds = lap_preds.sort_values(
                                'CurrentTyreAge', ascending=False)

                            # Take only the first (most recent) prediction
                            latest_pred = lap_preds.iloc[0]

                            print(
                                f"  Lap +{int(lap_ahead)}: Predicted degradation = {latest_pred['PredictedDegradationRate']:.3f} s/lap")

                # Then show processing status
                print(
                    f"\nProcessing row {i+1} of {len(full_df)} ({(i+1)/len(full_df)*100:.1f}%)")

                # Display DataFrame for reference
                print("\n=== PREDICTION DATAFRAME ===")
                display(latest_predictions.head(10))

                # Save to history
                all_predictions = pd.concat(
                    [all_predictions, latest_predictions], ignore_index=True)
                print(f"Total accumulated predictions: {len(all_predictions)}")
            else:
                # Clear output and show status even if no predictions
                clear_output(wait=False)
                print(
                    f"Processing row {i+1} of {len(full_df)} ({(i+1)/len(full_df)*100:.1f}%)")
                print(
                    "No predictions generated with current data (possibly insufficient sequences)")
        except Exception as e:
            # Clear output and show error
            clear_output(wait=False)
            print(
                f"Processing row {i+1} of {len(full_df)} ({(i+1)/len(full_df)*100:.1f}%)")
            print(f"Error during processing: {str(e)}")

        # Wait for the specified interval before next update
        if i < len(full_df) - 1:  # Don't wait after the last row
            print(f"Waiting {interval} seconds for next update...")
            time.sleep(interval)

    print("\n--- Simulation completed ---")
    print(f"Total rows processed: {len(full_df)}")
    print(f"Total predictions generated: {len(all_predictions)}")

    return all_predictions


if __name__ == "main":
    # Define path to the CSV file
    csv_path = '../../outputs/week3/lap_prediction_data.csv'

    # Define path to models
    models_path = '../../outputs/week5/models/'

    # Define monitoring thresholds by compound
    compound_start_laps = {
        1: 6,   # Soft tires: monitor from lap 6 onwards
        2: 12,  # Medium tires: monitor from lap 12 onwards
        3: 25   # Hard tires: monitor from lap 25 onwards
    }

    # Call the prediction function
    predictions = predict_tire_degradation(
        csv_path,
        models_path,
        compound_start_laps=compound_start_laps
    )
    # Display the predictions
    predictions.head()

    # Run simulation
    predictions_history = simulate_real_time_predictions(
        csv_path='../../outputs/week3/lap_prediction_data.csv',
        models_path='../../outputs/week5/models/',
        interval=0.1,  # 5 seconds between updates
        compound_start_laps=compound_start_laps,
        max_rows=200,  # Optional: limit number of rows for testing
        prediction_horizon=3  # Show predictions for next 3 laps
    )
