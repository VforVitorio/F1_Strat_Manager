# Import libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Import utility functions from our module
from .lap_prediction import compound_colors, compound_names


# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directories if they don't exist
os.makedirs('../../outputs/week5/models', exist_ok=True)


# EnhancedTCN model class
class EnhancedTCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, dropout=0.3):
        super(EnhancedTCN, self).__init__()

        # 1. Increase capacity and add regularization
        # Project the input to a higher-dimensional space using a 1D convolution.
        self.input_proj = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        # Batch normalization on the projected input.
        self.bn_input = nn.BatchNorm1d(hidden_size)

        # 2. Multi-Scale Block with Exponential Dilations
        # Residual layers with different dilation rates to capture multi-scale patterns.
        # First dilated convolution with dilation=1.
        self.dilated_conv1 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding='same', dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        # Second dilated convolution with dilation=2.
        self.dilated_conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding='same', dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        # Third dilated convolution with dilation=4.
        self.dilated_conv4 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding='same', dilation=4)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        # Fourth dilated convolution with dilation=8.
        self.dilated_conv8 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, padding='same', dilation=8)
        self.bn8 = nn.BatchNorm1d(hidden_size)

        # 3. Simple Temporal Attention Mechanism
        # Use a 1D convolution to compute attention scores over the sequence.
        self.attention = nn.Conv1d(hidden_size, 1, kernel_size=1)
        # Softmax is applied along the time dimension (seq_len).
        self.softmax = nn.Softmax(dim=2)

        # 4. Output Layer with Higher Dropout for Regularization
        self.dropout = nn.Dropout(dropout)
        # Fully connected layers for final prediction.
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, features]
        # Transpose the input to [batch, features, seq_len] for Conv1d operations.
        x = x.transpose(1, 2)

        # Initial projection: apply convolution, batch norm and ReLU activation.
        x = F.relu(self.bn_input(self.input_proj(x)))

        # Apply dilated convolutions with residual connections.
        # First residual block with dilation=1.
        residual = x
        x = F.relu(self.bn1(self.dilated_conv1(x)))
        x = x + residual  # Residual connection

        # Second residual block with dilation=2.
        residual = x
        x = F.relu(self.bn2(self.dilated_conv2(x)))
        x = x + residual  # Residual connection

        # Third residual block with dilation=4.
        residual = x
        x = F.relu(self.bn4(self.dilated_conv4(x)))
        x = x + residual  # Residual connection

        # Fourth residual block with dilation=8.
        residual = x
        x = F.relu(self.bn8(self.dilated_conv8(x)))
        x = x + residual  # Residual connection

        # Apply the attention mechanism.
        # Compute attention weights for each time step.
        # Shape: [batch, 1, seq_len]
        attn_weights = self.softmax(self.attention(x))
        # Multiply feature maps with the attention weights.
        x = x * attn_weights

        # Global pooling: sum over the time dimension.
        x = torch.sum(x, dim=2)  # Resulting shape: [batch, channels]
        x = self.dropout(x)

        # Pass through fully connected layers with dropout and ReLU activations.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Function definitions
def create_sequences(df, input_length=5, prediction_horizon=3, target_column='DegradationRate'):
    """
    Create sequences for any sequential model (LSTM, TCN) from the tire degradation data.
    Groups by driver, stint and compound to ensure proper sequencing.

    Args:
        df: DataFrame with tire degradation data
        input_length: Number of consecutive laps to include in input sequence
        prediction_horizon: Number of future laps to predict
        target_column: Column to predict

    Returns:
        sequences: List of DataFrame sequences
        targets: List of target arrays
    """
    sequences = []
    targets = []

    # Group by DriverNumber, Stint, and CompoundID
    groupby_columns = ['DriverNumber', 'Stint', 'CompoundID']

    # Process each driver-stint-compound group separately
    for name, group in df.groupby(groupby_columns):
        # Sort by TyreAge to ensure chronological order
        sorted_group = group.sort_values('TyreAge').reset_index(drop=True)

        # Skip if we don't have enough laps for a sequence
        if len(sorted_group) < input_length + prediction_horizon:
            continue

        # Create sliding window sequences
        for i in range(len(sorted_group) - input_length - prediction_horizon + 1):
            # Get input sequence (all features)
            seq = sorted_group.iloc[i:i+input_length]

            # Get target values (future values to predict)
            target = sorted_group.iloc[i+input_length:i +
                                       input_length+prediction_horizon][target_column].values

            sequences.append(seq)
            targets.append(target)

    print(f"Created {len(sequences)} sequences of {input_length} laps each")
    return sequences, targets


def split_by_compound(df, sequences, targets):
    """
    Divide sequences and targets per compound ID
    """
    # Identify unique tire id
    compounds = df['CompoundID'].unique()

    # Create dictionaries for storing sequences per compound
    compound_sequences = {c: [] for c in compounds}
    compound_targets = {c: [] for c in compounds}

    # Assign each sequence to the corresponend dictionary
    for i, seq in enumerate(sequences):
        compound = seq['CompoundID'].iloc[0]
        compound_sequences[compound].append(seq)
        compound_targets[compound].append(targets[i])

    # Informar sobre la distribución de secuencias
    for compound in compounds:
        compound_name = compound_names.get(compound, f"Compound {compound}")
        print(
            f"{compound_name}: {len(compound_sequences[compound])} sequences")

    return compound_sequences, compound_targets, compounds


def verify_sequences_with_targets(sequences, targets, num_to_check=3):
    print("COMPLETE SEQUENCE VERIFICATION (WITH TARGETS):")
    print("=============================================")

    # Check a few consecutive sequences from the same group
    driver_stint_compounds = []

    for i, seq in enumerate(sequences):
        # Get identifier for this sequence
        identifier = (seq['DriverNumber'].iloc[0],
                      seq['Stint'].iloc[0], seq['CompoundID'].iloc[0])
        driver_stint_compounds.append(identifier)

    # Find groups with consecutive sequences
    for i in range(len(sequences)-1):
        # Check if consecutive sequences are from same driver-stint-compound
        if driver_stint_compounds[i] == driver_stint_compounds[i+1]:
            seq1 = sequences[i]
            seq2 = sequences[i+1]

            # Get tire ages and targets
            ages1 = seq1['TyreAge'].values
            ages2 = seq2['TyreAge'].values
            target1 = targets[i]
            target2 = targets[i+1]

            # Calculate what the next tire ages should be (for targets)
            expected_target_ages1 = np.array(
                [ages1[-1] + j + 1 for j in range(len(target1))])
            expected_target_ages2 = np.array(
                [ages2[-1] + j + 1 for j in range(len(target2))])

            # Check if sliding window pattern is correct
            sliding_window_correct = np.array_equal(ages1[1:], ages2[:-1])

            # Print results
            print(f"\nSequences {i} and {i+1}:")
            print(
                f"Driver: {seq1['DriverNumber'].iloc[0]}, Stint: {seq1['Stint'].iloc[0]}, Compound: {seq1['CompoundID'].iloc[0]}")
            print(f"Tire ages seq {i}: {ages1}")
            print(f"TARGET values seq {i}: {target1}")
            print(f"Expected target ages seq {i}: {expected_target_ages1}")
            print(f"Tire ages seq {i+1}: {ages2}")
            print(f"TARGET values seq {i+1}: {target2}")
            print(f"Expected target ages seq {i+1}: {expected_target_ages2}")
            print(f"Sliding window pattern: {sliding_window_correct}")

            # Verify that target1 corresponds to the next values after seq1
            # We'd need the original dataframe to check this precisely

            # Only check a limited number
            num_to_check -= 1
            if num_to_check <= 0:
                break

    print("\nVERIFICATION SUMMARY:")
    print("1. Each sequence should advance by one lap (sliding window pattern)")
    print("2. Targets should contain the FuelAdjustedDegPercent values for the next 3 laps")
    print("3. Each target should start exactly where its sequence ends")


def prepare_for_lstm(sequences, targets):
    """
    Convert the list of DataFrames and targets into numpy arrays suitable for LSTM training
    """
    # Get the number of features (columns) in the sequence DataFrames
    n_features = len(sequences[0].columns)
    sequence_length = len(sequences[0])
    prediction_horizon = len(targets[0])

    # Initialize arrays
    X = np.zeros((len(sequences), sequence_length, n_features))
    y = np.zeros((len(sequences), prediction_horizon))

    # Fill the arrays
    for i, (seq, target) in enumerate(zip(sequences, targets)):
        X[i] = seq.values
        y[i] = target

    print(f"Prepared data for LSTM with shape: X: {X.shape}, y: {y.shape}")
    return X, y


def prepare_compound_data(compound_sequences, compound_targets):
    """
    Prepare data of each compound for the model
    """
    compound_data = {}

    for compound in unique_compounds:
        if len(compound_sequences[compound]) > 0:
            # Convert sequences to numpy format
            X, y = prepare_for_lstm(
                compound_sequences[compound], compound_targets[compound])
            compound_data[compound] = (X, y)

            print(
                f"Compound {compound_names.get(compound, compound)}: X shape {X.shape}, y shape {y.shape}")

    return compound_data


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backpropagation y optimización
        optimizer.zero_grad()
        loss.backward()
        # Añadir gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    total_loss = running_loss / len(data_loader.dataset)
    return total_loss


def train_compound_model(compound_id, X, y, device):
    """Trains a specific model for a given compound"""

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(
        X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    # Dataloaders
    compound_train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    compound_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    compound_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    compound_model = EnhancedTCN(input_size, output_size)
    compound_model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        compound_model.parameters(),
        lr=0.0005,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Initial restart period (5 epochs)
        T_mult=2,  # Multiplier for increasing restart periods
        eta_min=1e-6  # Minimum learning rate
    )

    criterion = nn.MSELoss()  # Mean Squared Error Loss function

    # Training with early stopping
    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    counter = 0  # Early stopping counter
    train_losses = []  # Store training losses per epoch
    val_losses = []  # Store validation losses per epoch

    for epoch in range(compound_epochs):
        # Train for one epoch and compute training loss
        train_loss = train_epoch(
            compound_model, compound_train_loader, criterion, optimizer, device)
        # Evaluate on validation set and compute validation loss
        val_loss = evaluate(
            compound_model, compound_val_loader, criterion, device)

        # Store loss values
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update learning rate scheduler
        scheduler.step()

        print(
            f'Epoch {epoch+1}/{compound_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Early stopping and model saving logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update best validation loss
            counter = 0  # Reset early stopping counter
            # Save the model with a unique filename per compound
            torch.save(compound_model.state_dict(
            ), f'../../outputs/week5/models/tcn_compound_{compound_id}.pth')
            print(f"Saving model (improved val_loss: {best_val_loss:.6f})")
        else:
            counter += 1  # Increment early stopping counter
            if counter >= compound_patience:  # If no improvement for 'compound_patience' epochs
                print(f"Early stopping at epoch {epoch+1}")
                break  # Stop training early

    # Load the best model version before testing
    compound_model.load_state_dict(torch.load(
        f'../../outputs/week5/models/tcn_compound_{compound_id}.pth'))

    # Evaluate on test set
    test_loss = evaluate(
        compound_model, compound_test_loader, criterion, device)

    # Model evaluation metrics
    compound_model.eval()  # Set model to evaluation mode
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in compound_test_loader:
            inputs = inputs.to(device)
            outputs = compound_model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.numpy())

    if len(predictions) > 0:
        # Convert lists into arrays for metric calculation
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)

        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        mae = mean_absolute_error(actuals, predictions)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    else:
        # Handle case where no predictions are generated
        metrics = {'mse': float('nan'), 'rmse': float(
            'nan'), 'mae': float('nan')}

    return compound_model, metrics, (train_losses, val_losses)


def visualize_compound_predictions(compound_id, compound_model, X_test, y_test, device):
    """
    Visualizes predictions versus actual values for a specific compound.
    """
    # Filter test data indices corresponding to this compound.
    compound_test_indices = []
    for i, sequence in enumerate(sequences):
        # Check if the current sequence belongs to the given compound using the 'CompoundID'
        if sequence['CompoundID'].iloc[0] == compound_id:
            compound_test_indices.append(i)

    # Convert the test data to a PyTorch tensor and send it to the specified device (CPU or GPU).
    X_compound = torch.FloatTensor(X_test).to(device)

    # Set the model to evaluation mode and compute predictions without tracking gradients.
    compound_model.eval()
    with torch.no_grad():
        y_pred = compound_model(X_compound).cpu().numpy()

    # y_actual remains the provided ground truth values for comparison.
    y_actual = y_test

    # Determine the number of samples to visualize (up to 5 samples, or less if fewer predictions exist)
    n_samples = min(5, len(y_pred))
    if n_samples > 0:
        # Randomly select sample indices without replacement.
        sample_indices = np.random.choice(
            len(y_pred), size=n_samples, replace=False)

        # Create a figure with a grid of subplots (2 rows and 3 columns) to display each sample.
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
        # Flatten the 2D array of axes to a 1D list for easier indexing.
        axes = axes.flatten()

        # Retrieve the compound name from the dictionary, or default to "Compound <compound_id>".
        compound_name = compound_names.get(
            compound_id, f"Compound {compound_id}")
        fig.suptitle(
            f'Predictions vs. Actual Values for {compound_name}', fontsize=16)

        # Plot each selected sample in a separate subplot.
        for i, idx in enumerate(sample_indices):
            if i < len(axes):
                ax = axes[i]

                # Define the forecast horizon; here, we assume predictions for 3 future laps.
                horizon = range(1, 4)  # Future laps: 1, 2, 3

                # Plot the actual values as a blue solid line with markers.
                ax.plot(horizon, y_actual[idx], 'o-',
                        label='Actual', color='blue')
                # Plot the predicted values as a red dashed line with markers.
                ax.plot(horizon, y_pred[idx], 'o--',
                        label='Predicted', color='red')

                # Set subplot title, labels, and enable grid lines.
                ax.set_title(f'Sample {idx}')
                ax.set_xlabel('Future Lap')
                ax.set_ylabel('Degradation (s/lap)')
                ax.grid(True)

                # Only add a legend to the first subplot.
                if i == 0:
                    ax.legend()

        # Turn off any unused subplots if less than the available axes.
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        # Adjust the layout to accommodate the figure title and subplots.
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # Save the figure to the specified output path, including the compound name in the filename.
        plt.savefig(f'../../outputs/week5/predictions_{compound_name}.png')
        plt.show()
    else:
        print(
            f"There are not enough samples to visualize for compound {compound_name}")


def ensemble_predict(X, compound_id, global_model, specialized_models, device,
                     weights=None):
    """
    Makes predictions using an ensemble of the global model and the specialized model.

    Args:
        X: Input data (tensor)
        compound_id: ID of the compound for which the prediction is made
        global_model: Global model trained on all data
        specialized_models: Dictionary containing specialized models for each compound
        device: Device for computation (CPU/GPU)
        weights: Weights to combine predictions [global_weight, specialized_weight]
                 If None, weights are calculated based on the inverse RMSE.

    Returns:
        Ensemble predictions and the weights used for combining
    """
    # Convert X to a tensor if it is not already one.
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)

    # Move the input tensor to the specified device (CPU/GPU)
    X = X.to(device)

    # Get predictions from the global model.
    # Set the model to evaluation mode and disable gradient computation.
    global_model.eval()
    with torch.no_grad():
        global_pred = global_model(X).cpu().numpy()

    # Check if there is a specialized model for the given compound.
    if compound_id in specialized_models:
        specialized_model = specialized_models[compound_id]
        specialized_model.eval()
        with torch.no_grad():
            specialized_pred = specialized_model(X).cpu().numpy()

        # If weights are not provided, calculate them based on the inverse of the RMSE.
        if weights is None:
            global_rmse = 0.355017  # RMSE of the global model
            specialized_rmse = compound_performance[compound_id]['rmse']

            # Use the inverse of the RMSE as the weighting factor (better performance = higher weight)
            global_weight = 1 / global_rmse
            specialized_weight = 1 / specialized_rmse

            # Normalize the weights so that their sum is equal to 1.
            total = global_weight + specialized_weight
            global_weight /= total
            specialized_weight /= total

            weights = [global_weight, specialized_weight]

        # Combine the predictions from the global and specialized models using the calculated weights.
        ensemble_pred = weights[0] * global_pred + \
            weights[1] * specialized_pred

        return ensemble_pred, weights
    else:
        # If no specialized model exists, return only the global prediction.
        return global_pred, [1.0, 0.0]


def visualize_ensemble_predictions(ensemble_results):
    """
    Visualizes the ensemble model's predictions against the actual values.

    Args:
        ensemble_results: List of tuples (compound_id, predictions, actuals)
    """
    # Iterate over each compound's ensemble results
    for compound_id, predictions, actuals in ensemble_results:
        # Retrieve the compound name from the dictionary, or default if not found
        compound_name = compound_names.get(
            compound_id, f"Compound {compound_id}")

        # Select up to 5 samples to visualize
        n_samples = min(5, len(predictions))
        if n_samples == 0:
            continue  # Skip if there are no samples

        # Randomly choose sample indices without replacement
        sample_indices = np.random.choice(
            len(predictions), size=n_samples, replace=False)

        # Create a figure with a grid of subplots (2 rows x 3 columns)
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
        axes = axes.flatten()  # Flatten to simplify indexing

        # Set the overall title for the figure
        fig.suptitle(
            f'Ensemble Predictions vs. Actual Values for {compound_name}', fontsize=16)

        # Loop over each selected sample index to plot
        for i, idx in enumerate(sample_indices):
            if i < len(axes):
                ax = axes[i]

                # Define the forecast horizon (here, 3 future laps)
                horizon = range(1, 4)

                # Prepare the input sample for prediction by reshaping and sending to device
                # Assuming each sample has a shape that should be reshaped to (1, 5, 16)
                x_tensor = torch.FloatTensor(
                    X_test[idx].reshape(1, 5, 16)).to(device)

                # Obtain the global model prediction for the sample
                with torch.no_grad():
                    global_pred = model(x_tensor).cpu().numpy()[0]

                # Check if there is a specialized model for this compound; if so, use it
                if compound_id in specialized_models:
                    specialized_model = specialized_models[compound_id]
                    with torch.no_grad():
                        specialized_pred = specialized_model(
                            x_tensor).cpu().numpy()[0]
                else:
                    specialized_pred = global_pred  # Fallback to global prediction

                # Plot the actual values as a blue line with markers
                ax.plot(horizon, actuals[idx], 'o-',
                        label='Actual', color='blue')
                # Plot the ensemble prediction as a purple dashed line with markers
                ax.plot(horizon, predictions[idx], 'o--',
                        label='Ensemble', color='purple')
                # Plot the global model prediction as a red dashed line with a different marker, semi-transparent
                ax.plot(horizon, global_pred, 'x--',
                        label='Global', color='red', alpha=0.4)
                # Plot the specialized model prediction as a green dashed line with a different marker, semi-transparent
                ax.plot(horizon, specialized_pred, '*--',
                        label='Specialized', color='green', alpha=0.4)

                # Set subplot title and axis labels
                ax.set_title(f'Sample {idx}')
                ax.set_xlabel('Future Lap')
                ax.set_ylabel('Degradation (s/lap)')
                ax.grid(True)

                # Only add a legend in the first subplot for clarity
                if i == 0:
                    ax.legend()

        # Hide any unused subplots in the grid
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        # Adjust layout to ensure the overall title does not overlap with subplots
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # Save the figure to the specified file path
        plt.savefig(
            f'../../outputs/week5/ensemble_predictions_{compound_name}.png')
        # Display the plot
        plt.show()


def predict_with_ensemble(input_tensor, compound_id):
    """
    Bridge function that uses ensemble_predict to make predictions.

    Args:
        input_tensor: Tensor with input data [1, seq_len, features]
        compound_id: ID of the compound for which to predict

    Returns:
        Array with predictions for the upcoming laps
    """
    # Obtain predictions using the ensemble_predict function.
    pred_result, weights = ensemble_predict(
        X=input_tensor,
        compound_id=compound_id,
        global_model=model,              # Global model
        specialized_models=specialized_models,  # Dictionary with specialized models
        device=device
    )

    # If the first dimension is 1 (i.e., batch_size=1), extract the result.
    if len(pred_result.shape) > 1 and pred_result.shape[0] == 1:
        pred_result = pred_result[0]

    return pred_result


def process_raw_lap_data(df):
    """
    Applies all necessary transformations to the raw data
    to obtain the variables expected by the model.

    Args:
        df: A pandas DataFrame or a CSV file path.

    Returns:
        A processed DataFrame with derived features.
    """
    # Check if 'df' is a file path (string) or already a DataFrame.
    if isinstance(df, str):
        # If it's a file path, load the data from CSV.
        print(f"Loading data from: {df}")
        df = pd.read_csv(df)

    # Create a copy of the DataFrame to work on.
    processed_df = df.copy()

    # Constant to adjust for fuel effect (improvement per lap)
    LAP_TIME_IMPROVEMENT_PER_LAP = 0.055  # seconds per lap

    # 1. Find the baseline for degradation.
    # Check if there are any laps with TyreAge == 1.
    if 1 in processed_df['TyreAge'].values:
        baseline_data = processed_df[processed_df['TyreAge'] == 1]
        baseline_lap_time = baseline_data['LapTime'].mean()
        baseline_tire_age = 1
    else:
        # If no TyreAge of 1 is found, use the minimum TyreAge available.
        min_age = processed_df['TyreAge'].min()
        baseline_data = processed_df[processed_df['TyreAge'] == min_age]
        baseline_lap_time = baseline_data['LapTime'].mean()
        baseline_tire_age = min_age

    # 2. Calculate derived variables.
    # Compute how many laps have passed since the baseline.
    processed_df['LapsFromBaseline'] = processed_df['TyreAge'] - \
        baseline_tire_age
    # Calculate the effect of fuel improvement over these laps.
    processed_df['FuelEffect'] = processed_df['LapsFromBaseline'] * \
        LAP_TIME_IMPROVEMENT_PER_LAP

    # 3. Calculate fuel-adjusted lap times.
    # Adjust the raw lap times by adding the fuel effect.
    processed_df['FuelAdjustedLapTime'] = processed_df['LapTime'] + \
        processed_df['FuelEffect']

    # 4. Compute degradation percentage based on fuel-adjusted lap times.
    processed_df['FuelAdjustedDegPercent'] = (
        processed_df['FuelAdjustedLapTime'] / baseline_lap_time - 1) * 100

    # 5. Calculate degradation rate.
    # Group the data by TyreAge and compute the average fuel-adjusted lap time.
    avg_laptimes = processed_df.groupby(
        'TyreAge')['FuelAdjustedLapTime'].mean()
    # Compute the difference between consecutive average lap times.
    deg_rates = avg_laptimes.diff()

    # Assign degradation rates to each lap according to its TyreAge.
    processed_df['DegradationRate'] = 0.0  # Default value
    for age, rate in zip(deg_rates.index, deg_rates.values):
        # Create a mask for rows where TyreAge equals the current age.
        mask = processed_df['TyreAge'] == age
        # Assign the computed rate if available, otherwise 0.0.
        processed_df.loc[mask, 'DegradationRate'] = rate if not pd.isna(
            rate) else 0.0

    return processed_df


def predict_tire_degradation_live(input_df, window_size=5, prediction_horizon=3):
    """
    Simulates real-time predictions for the input data.

    Args:
        input_df: Input DataFrame containing raw lap data.
        window_size: Number of consecutive laps to use as input.
        prediction_horizon: Number of future laps to predict.

    Returns:
        A DataFrame containing the predictions along with related information.
    """
    # 1. Process the raw data using the transformation function.
    processed_df = process_raw_lap_data(input_df)

    # 2. Define the columns required by the model.
    model_columns = [
        'FuelAdjustedLapTime', 'FuelAdjustedDegPercent', 'DegradationRate',
        'TyreAge', 'CompoundID', 'DriverNumber', 'Stint', 'Sector1Time',
        'Sector2Time', 'Sector3Time', 'Position', 'FuelLoad',
        'SpeedI1', 'SpeedI2', 'SpeedFL'
    ]
    # Create a fictitious lap number column by copying TyreAge (used as a replacement).
    processed_df['LapNumberFicticio'] = processed_df['TyreAge'].copy()
    model_columns.append('LapNumberFicticio')

    # Check for any missing columns and initialize them to 0 if not present.
    missing = [col for col in model_columns if col not in processed_df.columns]
    if missing:
        print(
            f"Warning: Missing columns that will be initialized to 0: {missing}")
        for col in missing:
            processed_df[col] = 0.0

    # 3. Prepare a list to store prediction results.
    results = []

    # 4. Iterate through the DataFrame to simulate live processing.
    # This loop moves a window of size 'window_size' over the data.
    for i in range(len(processed_df) - window_size + 1):
        # Get a window of consecutive data rows.
        window = processed_df.iloc[i:i+window_size].copy()

        # Skip the window if the data comes from different stints or tire compounds.
        if window['CompoundID'].nunique() > 1 or window['Stint'].nunique() > 1:
            continue

        # Prepare the input for the model.
        compound_id = window['CompoundID'].iloc[0]
        sequence = window[model_columns].values

        # Convert the sequence to a tensor and add a batch dimension.
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        # Get the predicted degradation for the upcoming laps using the ensemble function.
        pred_degradation = predict_with_ensemble(input_tensor, compound_id)

        # Store the result for the current window.
        result_row = {
            # Use TyreAge instead of LapNumber
            'TyreAge': window['TyreAge'].iloc[-1],
            'CompoundID': compound_id,
            'CurrentDegradationRate': window['DegradationRate'].iloc[-1]
        }

        # Add predictions for each future lap based on the prediction horizon.
        for j in range(prediction_horizon):
            result_row[f'PredictedDegRate_Lap{j+1}'] = pred_degradation[j]

        results.append(result_row)

    # Return the results as a DataFrame.
    return pd.DataFrame(results)


def simulate_tire_degradation_live(data_path, driver_numbers=[44, 1, 16], n_laps=20, update_interval=1.0):
    """
    Simulates real-time tire degradation for multiple drivers,
    showing actual values and predictions on a plot that updates each lap.

    IPython.display is used to update the figure at each iteration.

    Args:
        data_path: Path to the CSV file containing data.
        driver_numbers: List of driver numbers to follow.
        n_laps: Number of laps to simulate.
        update_interval: Seconds between updates.
    """
    import time
    from IPython.display import display, clear_output  # For updating output display

    # Load and process raw data from the CSV file.
    raw_data = pd.read_csv(data_path)
    processed_df = process_raw_lap_data(raw_data)

    # Map each driver to its tire compound.
    driver_compounds = {}
    for driver in driver_numbers:
        driver_data = processed_df[processed_df['DriverNumber'] == driver]
        if len(driver_data) > 0:
            compound_id = driver_data['CompoundID'].iloc[0]
            driver_compounds[driver] = compound_id

    # Define styles (colors and markers) based on the tire compound.
    styles = {
        'Soft': {'color': 'red', 'marker': 'o'},
        'Medium': {'color': 'yellow', 'marker': 's'},
        'Hard': {'color': 'gray', 'marker': '^'}
    }

    # Define compound names based on their IDs.
    compound_names = {1: 'Soft', 2: 'Medium', 3: 'Hard'}

    # Create a history dictionary to store lap numbers, actual degradation rates,
    # predicted lap numbers, and predicted degradation rates for each driver.
    history = {
        driver: {
            'laps': [],
            'deg_rates': [],     # Actual degradation rate
            'pred_laps': [],     # Lap numbers for predictions
            'pred_rates': []     # Predicted degradation rates
        } for driver in driver_numbers
    }

    # Base degradation parameters for each compound (tuned manually).
    deg_params = {1: 0.15, 2: 0.08, 3: 0.04}

    # Simulate lap-by-lap evolution.
    for lap in range(1, n_laps + 1):
        # Update data for each driver.
        for driver in driver_numbers:
            if driver not in driver_compounds:
                continue  # Skip if driver has no compound data

            compound_id = driver_compounds[driver]
            compound = compound_names.get(
                compound_id, f"Compound {compound_id}")
            style = styles.get(compound, {'color': 'blue', 'marker': 'o'})

            # Calculate the degradation rate based on the compound and current lap.
            base_deg = deg_params.get(compound_id, 0.1)
            if compound == 'Soft':
                # For Soft compound, the degradation rate increases differently before and after lap 7.
                if lap <= 7:
                    deg_rate = base_deg * lap * 0.5
                else:
                    deg_rate = base_deg * (lap - 7) * 1.2
            elif compound == 'Medium':
                # For Medium compound, a fixed multiplier is applied.
                deg_rate = base_deg * lap * 0.7
            else:  # Hard compound
                deg_rate = base_deg * lap * 0.4

            # Add random noise to the degradation rate and avoid negative values.
            deg_rate += np.random.normal(0, 0.05)
            deg_rate = max(0, deg_rate)

            # Save the real degradation data in the history.
            history[driver]['laps'].append(lap)
            history[driver]['deg_rates'].append(deg_rate)

            # Generate predictions if there are enough laps (starting from lap 5).
            if lap >= 5:
                # Define prediction lap numbers for the next 3 laps.
                pred_laps = [lap + i + 1 for i in range(3)]
                pred_rates = []
                if compound == 'Soft':
                    for pl in pred_laps:
                        if pl <= 7:
                            pr = base_deg * pl * 0.55
                        else:
                            pr = base_deg * (pl - 7) * 1.3
                        pred_rates.append(pr)
                elif compound == 'Medium':
                    for pl in pred_laps:
                        pr = base_deg * pl * 0.75
                        pred_rates.append(pr)
                else:  # Hard compound
                    for pl in pred_laps:
                        pr = base_deg * pl * 0.45
                        pred_rates.append(pr)

                # Add noise to predictions as well.
                pred_rates = [max(0, r + np.random.normal(0, 0.07))
                              for r in pred_rates]
                for i, pl in enumerate(pred_laps):
                    history[driver]['pred_laps'].append(pl)
                    history[driver]['pred_rates'].append(pred_rates[i])

        # Visualize real-time evolution starting from lap 3.
        if lap >= 3:
            plt.figure(figsize=(14, 8))
            # Plot for each driver.
            for driver in driver_numbers:
                if driver not in driver_compounds:
                    continue
                compound = compound_names.get(
                    driver_compounds[driver], 'Unknown')
                style = styles.get(compound, {'color': 'blue', 'marker': 'o'})

                # Plot the actual degradation data.
                plt.plot(history[driver]['laps'], history[driver]['deg_rates'],
                         color=style['color'], marker=style['marker'],
                         label=f"Driver {driver} ({compound})")

                # Plot the predictions if available.
                if history[driver]['pred_laps']:
                    plt.plot(history[driver]['pred_laps'], history[driver]['pred_rates'],
                             color=style['color'], linestyle='--', alpha=0.7,
                             label=f"Driver {driver} - Predictions")

            plt.title("Real-Time Tire Degradation Evolution")
            plt.xlabel("Lap Number")
            plt.ylabel("Degradation Rate (s/lap)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            display(plt.gcf())  # Display the current figure.
            plt.close()

        # Print lap information to the console.
        print(f"\n--- LAP {lap} ---")
        for driver in driver_numbers:
            if history[driver]['deg_rates']:
                deg = history[driver]['deg_rates'][-1]
                compound = compound_names.get(
                    driver_compounds.get(driver, 0), 'Unknown')
                level = "LOW"
                if deg > 0.8:
                    level = "HIGH"
                elif deg > 0.4:
                    level = "MEDIUM"
                print(
                    f"Driver {driver} ({compound}): {deg:.3f} s/lap - {level} degradation")

        # Wait for the specified update interval before the next lap,
        # then clear the output to update the display.
        if lap < n_laps:
            print(f"\nWaiting {update_interval} seconds...")
            time.sleep(update_interval)
            clear_output(wait=True)

    # At the end, display a final figure showing the complete evolution.
    plt.figure(figsize=(14, 8))
    for driver in driver_numbers:
        if driver not in driver_compounds:
            continue
        compound = compound_names.get(driver_compounds[driver], 'Unknown')
        style = styles.get(compound, {'color': 'blue', 'marker': 'o'})

        plt.plot(history[driver]['laps'], history[driver]['deg_rates'],
                 color=style['color'], marker=style['marker'],
                 label=f"Driver {driver} ({compound})")
        if history[driver]['pred_laps']:
            plt.plot(history[driver]['pred_laps'], history[driver]['pred_rates'],
                     color=style['color'], linestyle='--', alpha=0.7,
                     label=f"Driver {driver} - Predictions")

    plt.title("Final Tire Degradation Evolution")
    plt.xlabel("Lap Number")
    plt.ylabel("Degradation Rate (s/lap)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    display(plt.gcf())
    plt.close()


if __name__ == "__main__":
    # Section 2: Loading Dataframes
    data = pd.read_csv("../../outputs/week3/lap_prediction_data.csv")
    print("\nRegular data sample:")

    seq_data = pd.read_csv(
        "../../outputs/week3/sequential_lap_prediction_data.csv")
    print("\nSequential data sample:")

    # Display basic information
    print("Basic dataset information:")
    print(f"Regular data shape: {data.shape}")
    print(f"Sequential data shape: {seq_data.shape}")

    # Section 3: Locating Tire Related Columns
    tire_columns = ['CompoundID', 'TyreAge']
    print(f"\nTire-related columns: {tire_columns}")
    print("\nTire-related statistics:")

    # Section 4: Compound Mappings
    print("\nCompound mappings:")
    print(f"Compound names: {compound_names}")
    print(f"Compound colors: {compound_colors}")

    # Section 5: Analyzing Relationship between Tire Age and Lap Time by compound
    plt.figure(figsize=(12, 6))
    for compound_id in data['CompoundID'].unique():
        subset = data[data['CompoundID'] == compound_id]
        agg_data = subset.groupby('TyreAge')['LapTime'].agg(
            ['mean', 'std', 'count']).reset_index()
        if len(agg_data) > 1:
            color = compound_colors.get(compound_id, 'black')
            compound_name = compound_names.get(
                compound_id, f'Unknown ({compound_id})')
            plt.plot(agg_data['TyreAge'], agg_data['mean'],
                     'o-', color=color, label=f'{compound_name} Tire')
            if 'std' in agg_data.columns:
                plt.fill_between(agg_data['TyreAge'], agg_data['mean'] - agg_data['std'],
                                 agg_data['mean'] + agg_data['std'], color=color, alpha=0.2)
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Lap Time (s)')
    plt.title('Tire Degradation: Effect on Lap Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../../outputs/week5/tire_deg_curve.png')
    plt.show()

    # Section 6: Exploring Lap Times Deltas and Tire Ages
    if 'LapTime_Delta' in seq_data.columns:
        plt.figure(figsize=(12, 6))
        for compound_id in seq_data['CompoundID'].unique():
            subset = seq_data[seq_data['CompoundID'] == compound_id]
            agg_data = subset.groupby(
                'TyreAge')['LapTime_Delta'].mean().reset_index()
            if len(agg_data) > 1:
                color = compound_colors.get(compound_id, 'black')
                compound_name = compound_names.get(
                    compound_id, f'Unknown ({compound_id})')
                plt.plot(agg_data['TyreAge'], agg_data['LapTime_Delta'],
                         'o-', color=color, label=f'{compound_name} Tire')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Tire Age (laps)')
        plt.ylabel('Lap Time Delta (s) - Positive means getting slower')
        plt.title('Lap Time Degradation Rate by Tire Age')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('../../outputs/week5/tire_deg_rate.png')
        plt.show()

    # Section 7: Exploring if Tire Age affects Speed in different Sectors
    speed_columns = ['SpeedI1', 'SpeedI2', 'SpeedFL']
    plt.figure(figsize=(14, 8))
    for speed_col in speed_columns:
        subset = data[data['CompoundID'] == 2]
        agg_data = subset.groupby('TyreAge')[speed_col].mean().reset_index()
        if len(agg_data) > 1:
            plt.plot(agg_data['TyreAge'], agg_data[speed_col],
                     'o-', label=f'{speed_col}')
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Speed (kph)')
    plt.title(f'Effect of Tire Age on Speed - {compound_names.get(2)} Tires')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../../outputs/week5/tire_age_speed_effect.png')
    plt.show()

    # Section 8: Creating Tire Degradation Metrics
    LAP_TIME_IMPROVEMENT_PER_LAP = 0.055
    tire_deg_data = pd.DataFrame()
    for compound_id in data['CompoundID'].unique():
        compound_name = compound_names.get(
            compound_id, f"Unknown ({compound_id})")
        print(f"Processing {compound_name} tires (ID: {compound_id})...")
        compound_data = data[data['CompoundID'] == compound_id].copy()
        compound_data = compound_data.sort_values('TyreAge')
        if len(compound_data) < 5:
            print(f"  Not enough data for {compound_name} tires, skipping")
            continue

        # Find baseline information and calculations
        if 1 in compound_data['TyreAge'].values:
            baseline_data = compound_data[compound_data['TyreAge'] == 1]
            baseline_lap_time = baseline_data['LapTime'].mean()
            baseline_tire_age = 1
        else:
            min_age = compound_data['TyreAge'].min()
            baseline_data = compound_data[compound_data['TyreAge'] == min_age]
            baseline_lap_time = baseline_data['LapTime'].mean()
            baseline_tire_age = min_age
            print(
                f"  No laps with new tires for {compound_name}, using TyreAge={min_age} as baseline")

        # Calculate fuel adjustment and degradation metrics
        compound_data['LapsFromBaseline'] = compound_data['TyreAge'] - \
            baseline_tire_age
        compound_data['FuelEffect'] = compound_data['LapsFromBaseline'] * \
            LAP_TIME_IMPROVEMENT_PER_LAP
        compound_data['FuelAdjustedLapTime'] = compound_data['LapTime'] + \
            compound_data['FuelEffect']
        compound_data['TireDegAbsolute'] = compound_data['LapTime'] - \
            baseline_lap_time
        compound_data['TireDegPercent'] = (
            compound_data['LapTime'] / baseline_lap_time - 1) * 100
        baseline_adjusted_lap_time = baseline_lap_time
        compound_data['FuelAdjustedDegAbsolute'] = compound_data['FuelAdjustedLapTime'] - \
            baseline_adjusted_lap_time
        compound_data['FuelAdjustedDegPercent'] = (
            compound_data['FuelAdjustedLapTime'] / baseline_adjusted_lap_time - 1) * 100
        compound_data['CompoundName'] = compound_name
        tire_deg_data = pd.concat([tire_deg_data, compound_data])

        # Print statistics
        max_laps = compound_data['TyreAge'].max() - baseline_tire_age
        total_fuel_effect = max_laps * LAP_TIME_IMPROVEMENT_PER_LAP
        print(
            f"  Baseline lap time for {compound_name}: {baseline_lap_time:.3f}s")
        print(f"  Maximum laps from baseline: {max_laps:.0f}")
        print(f"  Estimated total fuel benefit: ~{total_fuel_effect:.2f}s")
        print(
            f"  Processed {len(compound_data)} laps with {compound_name} tires")

    print("\nComparison of regular vs. fuel-adjusted metrics (sample):")
    sample_comparison = tire_deg_data.groupby(['CompoundName', 'TyreAge'])[
        ['TireDegAbsolute', 'FuelAdjustedDegAbsolute', 'FuelEffect']
    ].mean().reset_index()

    # Section 8 continued: Plotting the difference between regular and Fuel Adjusted Degradation
    plt.figure(figsize=(16, 12))
    compound_ids = tire_deg_data['CompoundID'].unique()
    for i, compound_id in enumerate(compound_ids):
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]
        color = compound_colors.get(compound_id, 'black')
        compound_name = compound_names.get(
            compound_id, f'Unknown ({compound_id})')
        reg_agg = compound_subset.groupby('TyreAge')['TireDegAbsolute'].mean()
        adj_agg = compound_subset.groupby(
            'TyreAge')['FuelAdjustedDegAbsolute'].mean()
        plt.subplot(len(compound_ids), 1, i+1)
        plt.plot(reg_agg.index, reg_agg.values, 'o--', color=color,
                 alpha=0.5, label=f'{compound_name} (Regular)')
        plt.plot(adj_agg.index, adj_agg.values, 'o-', color=color,
                 linewidth=2, label=f'{compound_name} (Fuel Adjusted)')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.ylabel('Degradation (s)')
        plt.title(
            f'{compound_name} Tire Degradation: Regular vs. Fuel-Adjusted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        min_lap = reg_agg.index.min()
        max_lap = reg_agg.index.max()
        total_laps = max_lap - min_lap
        total_fuel_effect = total_laps * LAP_TIME_IMPROVEMENT_PER_LAP
        plt.annotate(f"Est. total fuel effect: ~{total_fuel_effect:.2f}s",
                     xy=(0.02, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        if i == len(compound_ids)-1:
            plt.xlabel('Tire Age (laps)')
    plt.tight_layout()
    plt.savefig('../../outputs/week5/regular_vs_adjusted_comparison.png')
    plt.show()

    # Section 8.1: Absolute Tire Degradation
    plt.figure(figsize=(14, 7))
    for compound_id in compound_ids:
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]
        color = compound_colors.get(compound_id, 'black')
        compound_name = compound_names.get(
            compound_id, f'Unknown ({compound_id})')
        agg_data = compound_subset.groupby('TyreAge')['FuelAdjustedDegAbsolute'].agg([
            'mean', 'std']).reset_index()
        plt.plot(agg_data['TyreAge'], agg_data['mean'], 'o-',
                 color=color, linewidth=2, label=f'{compound_name}')
        if 'std' in agg_data.columns and not agg_data['std'].isnull().all():
            plt.fill_between(agg_data['TyreAge'], agg_data['mean'] - agg_data['std'],
                             agg_data['mean'] + agg_data['std'], color=color, alpha=0.2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Fuel-Adjusted Absolute Degradation (s)')
    plt.title('Tire Degradation by Compound and Age (Fuel Effect Removed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../../outputs/week5/fuel_adjusted_deg_by_compound.png')
    plt.show()

    # Section 8.2: Tire Degradation Percentage
    plt.figure(figsize=(14, 7))
    for compound_id in compound_ids:
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]
        color = compound_colors.get(compound_id, 'black')
        compound_name = compound_names.get(
            compound_id, f'Unknown ({compound_id})')
        agg_data = compound_subset.groupby('TyreAge')['FuelAdjustedDegPercent'].agg([
            'mean', 'std']).reset_index()
        plt.plot(agg_data['TyreAge'], agg_data['mean'], 'o-',
                 color=color, linewidth=2, label=f'{compound_name}')
        if 'std' in agg_data.columns and not agg_data['std'].isnull().all():
            plt.fill_between(agg_data['TyreAge'], agg_data['mean'] - agg_data['std'],
                             agg_data['mean'] + agg_data['std'], color=color, alpha=0.2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Fuel-Adjusted Percentage Degradation (%)')
    plt.title(
        'Percentage Tire Degradation by Compound and Age (Fuel Effect Removed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../../outputs/week5/fuel_adjusted_deg_percent_by_compound.png')
    plt.show()

    # Section 8.3: Tire Degradation Rate
    for compound_id in compound_ids:
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]
        avg_laptimes = compound_subset.groupby(
            'TyreAge')['FuelAdjustedLapTime'].mean()
        deg_rates = avg_laptimes.diff()
        for age, rate in zip(deg_rates.index, deg_rates.values):
            mask = (tire_deg_data['CompoundID'] == compound_id) & (
                tire_deg_data['TyreAge'] == age)
            tire_deg_data.loc[mask, 'DegradationRate'] = rate

    print("\nFirst rows with DegradationRate:")

    plt.figure(figsize=(14, 7))
    for compound_id in compound_ids:
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]
        deg_stats = compound_subset.groupby('TyreAge')['DegradationRate'].agg([
            'mean', 'std']).reset_index()
        color = compound_colors.get(compound_id, 'black')
        compound_name = compound_names.get(
            compound_id, f'Unknown ({compound_id})')
        plt.plot(deg_stats['TyreAge'], deg_stats['mean'], marker='o',
                 linestyle='-', color=color, linewidth=2, label=compound_name)
        if 'std' in deg_stats.columns and not deg_stats['std'].isnull().all():
            plt.fill_between(deg_stats['TyreAge'], deg_stats['mean'] - deg_stats['std'],
                             deg_stats['mean'] + deg_stats['std'], color=color, alpha=0.)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Fuel-Adjusted Degradation Rate (s/lap)')
    plt.title('Tire Degradation Rate by Compound (Fuel Effect Removed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    num_nans = tire_deg_data['DegradationRate'].isna().sum()
    print(num_nans)
    tire_deg_data['DegradationRate'] = tire_deg_data['DegradationRate'].fillna(
        0)
    print(
        f"Number of NaN after sustitution: {tire_deg_data['DegradationRate'].isna().sum()}")

    # Section 9: Correlation Analysis
    compound_names_inv = {value: key for key, value in compound_names.items()}
    tire_deg_data["CompoundName"] = tire_deg_data["CompoundName"].replace(
        compound_names_inv)
    tire_deg_data = tire_deg_data.drop('Unnamed: 0', axis=1)
    correlation_matrix = tire_deg_data.corr()
    plt.figure(figsize=(24, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Section 10: Conclusions and Variable Cleaning
    columns_to_remove = [
        'LapTime', 'TireDegAbsolute', 'TireDegPercent', 'FuelAdjustedDegAbsolute',
        'CompoundName', 'LapsFromBaseline', 'FuelEffect'
    ]
    tire_deg_data = tire_deg_data.drop(columns=columns_to_remove)
    print(f"Removed {len(columns_to_remove)} redundant variables.")
    print(f"The dataframe now has {tire_deg_data.shape[1]} columns.")

    # Section 11: New Correlation Matrix
    updated_correlation_matrix = tire_deg_data.corr()
    plt.figure(figsize=(16, 8))
    sns.heatmap(updated_correlation_matrix,
                annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Updated Correlation Matrix after Removing Redundant Variables")
    plt.show()

    # Section 12: Saving Dataframe
    output_path = "../../outputs/week5/tire_degradation_fuel_adjusted.csv"
    tire_deg_data.to_csv(output_path, index=False)

    # Section 13: Creating Sequential Data
    df = pd.read_csv("../../outputs/week5/tire_degradation_fuel_adjusted.csv")
    sequences, targets = create_sequences(df)

    # Section 13.1: New Approach - Divide sequences by compound ID
    compound_sequences, compound_targets, unique_compounds = split_by_compound(
        df, sequences, targets)

    # Section 14: Verifying the Sequential Data
    verify_sequences_with_targets(sequences, targets)

    # Section 15: LSTM Data Preparation
    X, y = prepare_for_lstm(sequences, targets)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42)

    print("Data split:")
    print(f"X_train shape: {X_train.shape} ({len(X_train)/len(X):.1%})")
    print(f"X_val shape: {X_val.shape} ({len(X_val)/len(X):.1%})")
    print(f"X_test shape: {X_test.shape} ({len(X_test)/len(X):.1%})")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Section 16: Pytorch Data Preparation
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    BATCH_SIZE = 32
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Section 16.3: Prepare Data by Compound
    compound_data = prepare_compound_data(compound_sequences, compound_targets)

    # Section 17.1: Hyperparameters
    input_size = 16
    output_size = 3
    learning_rate = 0.0001
    weight_decay = 1e-6
    batch_size = 32
    num_epochs = 200
    patience = 25
    compound_epochs = 100
    compound_patience = 15

    # Section 17.2: Initializing Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedTCN(input_size, output_size)
    model.to(device)

    # Section 17.3: Loss Function and Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Section 19.2: TCN Training Loop
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []

    print("Starting Training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(
            f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(),
                       '../../outputs/week5/models/tire_degradation_tcn.pth')
            print(f"Saving model (improved val_loss: {best_val_loss:.6f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Section 19.3: Training models by Compound
    specialized_models = {}
    compound_performance = {}
    compound_loss_curves = {}

    for compound in compound_data:
        compound_name = compound_names.get(compound, f"Compound {compound}")
        print(f"\n=== Training model for {compound_name} ===")

        X, y = compound_data[compound]

        if len(X) > 10:
            model, metrics, loss_curves = train_compound_model(
                compound, X, y, device)

            specialized_models[compound] = model
            compound_performance[compound] = metrics
            compound_loss_curves[compound] = loss_curves
        else:
            print(f"Not enough data for {compound_name} tire")

    # Extract learning curves for plotting
    train_losses_combined = []
    val_losses_combined = []

    max_length = 0
    for compound in compound_loss_curves:
        train_curve, val_curve = compound_loss_curves[compound]
        max_length = max(max_length, len(train_curve))

    for i in range(max_length):
        train_sum = 0
        val_sum = 0
        count = 0

        for compound in compound_loss_curves:
            train_curve, val_curve = compound_loss_curves[compound]
            if i < len(train_curve):
                train_sum += train_curve[i]
                val_sum += val_curve[i]
                count += 1

        if count > 0:
            train_losses_combined.append(train_sum / count)
            val_losses_combined.append(val_sum / count)

    # Use these combined curves for plotting
    train_losses = train_losses_combined
    val_losses = val_losses_combined

    # Section 20: Load Best Model
    model.load_state_dict(torch.load(
        '../../outputs/week5/models/tire_degradation_tcn.pth'))

    # Section 21: Evaluating on test Set
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")

    # Making predictions on test set
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # Section 22: Calculating metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)

    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE: {mae:.6f}")

    # Section 22.1: Losses in training and validation
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('../../outputs/week5/training_validation_loss.png')
    plt.show()

    # Section 22.2: Predictions vs Real values
    plt.figure(figsize=(14, 7))
    sample_indices = np.random.choice(len(predictions), size=5, replace=False)
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        horizon = range(1, 4)
        plt.plot(horizon, actuals[idx], 'o-', label='Actual', color='blue')
        plt.plot(horizon, predictions[idx], 'o--',
                 label='Predicted', color='red')
        plt.title(f'Sample {idx}')
        plt.xlabel('Future Lap')
        plt.ylabel('Degradation (s/lap)')
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig('../../outputs/week5/predictions_vs_actual.png')
    plt.show()

    # Section 22.3: Performance Comparison by Compound
    print("\nPerformance by Compound:")
    print("{:<15} {:<12} {:<12} {:<12}".format(
        "Compound", "MSE", "RMSE", "MAE"))
    print("-" * 55)

    global_metrics = {"mse": mse, "rmse": rmse, "mae": mae}

    for compound, metrics in compound_performance.items():
        compound_name = compound_names.get(compound, f"Compound {compound}")
        print("{:<15} {:<12.6f} {:<12.6f} {:<12.6f}".format(
            compound_name,
            metrics['mse'],
            metrics['rmse'],
            metrics['mae']
        ))

    print("{:<15} {:<12.6f} {:<12.6f} {:<12.6f}".format(
        "GLOBAL MODEL",
        global_metrics['mse'],
        global_metrics['rmse'],
        global_metrics['mae']
    ))

    plt.figure(figsize=(12, 6))
    compounds = list(compound_performance.keys())
    rmse_values = [metrics['rmse']
                   for metrics in compound_performance.values()]
    compound_labels = [compound_names.get(
        c, f"Compound {c}") for c in compounds]
    colors = [compound_colors.get(c, 'gray') for c in compounds]

    plt.bar(compound_labels, rmse_values, color=colors)
    plt.axhline(y=rmse, color='red', linestyle='--',
                label=f'Global Model (RMSE={rmse:.3f})')
    plt.ylabel('RMSE (seconds/round)')
    plt.title('RMSE Comparison by Tire Type')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('../../outputs/week5/compound_specialized_performance.png')
    plt.show()

    # Section 22.4: Visualizations of Predictions vs Actual Values by Compound
    for compound in specialized_models:
        compound_name = compound_names.get(compound, f"Compound {compound}")
        print(f"\nVisualizing predictions for {compound_name}...")

        compound_indices_test = []
        for i, seq in enumerate(sequences):
            if i < len(X_test) and seq['CompoundID'].iloc[0] == compound:
                compound_indices_test.append(i)

        if len(compound_indices_test) > 0:
            X_compound_test = X_test[compound_indices_test]
            y_compound_test = y_test[compound_indices_test]

            visualize_compound_predictions(compound, specialized_models[compound],
                                           X_compound_test, y_compound_test, device)
        else:
            print(f"There is no test data for compound {compound_name}")

    # Section 22.5: Compare Learning Curves by Compound
    plt.figure(figsize=(12, 7))
    for compound in compound_loss_curves:
        compound_name = compound_names.get(compound, f"Compound {compound}")
        color = compound_colors.get(compound, 'black')

        train_curve, val_curve = compound_loss_curves[compound]

        plt.plot(train_curve, '--', color=color, alpha=0.7,
                 label=f'{compound_name} (Train)')
        plt.plot(val_curve, '-', color=color, label=f'{compound_name} (Val)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves by Tire Type')
    plt.legend()
    plt.grid(True)
    plt.savefig('../../outputs/week5/learning_curves_by_compound.png')
    plt.show()

    # Section 23: Ensemble System for Combined Predictions
    # Evaluate the performance of the ensemble system
    ensemble_predictions = []
    ensemble_weights_used = {}

    print("\nEvaluating Ensemble System:")
    test_indices_by_compound = {}

    for i in range(len(X_test)):
        test_sequence_idx = i % len(sequences)
        if test_sequence_idx < len(sequences):
            compound_id = sequences[test_sequence_idx]['CompoundID'].iloc[0]

            if compound_id not in test_indices_by_compound:
                test_indices_by_compound[compound_id] = []

            test_indices_by_compound[compound_id].append(i)

    for compound_id, indices in test_indices_by_compound.items():
        compound_name = compound_names.get(
            compound_id, f"Compound {compound_id}")
        X_compound = X_test[indices]
        y_compound = y_test[indices]

        compound_predictions = []
        weights_sum = [0, 0]

        for i, x in enumerate(X_compound):
            x_tensor = torch.FloatTensor(x.reshape(1, x.shape[0], x.shape[1]))
            pred, weights = ensemble_predict(
                x_tensor, compound_id, model, specialized_models, device)
            compound_predictions.append(pred[0])
            weights_sum[0] += weights[0]
            weights_sum[1] += weights[1]

        if len(X_compound) > 0:
            avg_weights = [w / len(X_compound) for w in weights_sum]
            ensemble_weights_used[compound_id] = avg_weights

        compound_predictions = np.array(compound_predictions)

        mse = mean_squared_error(y_compound, compound_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_compound, compound_predictions)

        global_rmse = 0.355017
        specialized_rmse = compound_performance.get(
            compound_id, {'rmse': float('inf')})['rmse']

        improvement_over_global = (global_rmse - rmse) / global_rmse * 100
        improvement_over_specialized = (
            specialized_rmse - rmse) / specialized_rmse * 100

        print(f"\n{compound_name}:")
        print(f"  Ensemble RMSE: {rmse:.6f}")
        print(
            f"  Global RMSE: {global_rmse:.6f} ({improvement_over_global:.2f}% improvement)")
        print(
            f"  Specialized RMSE: {specialized_rmse:.6f} ({improvement_over_specialized:.2f}% improvement)")
        print(
            f"  Weights used: Global={avg_weights[0]:.2f}, Specialized={avg_weights[1]:.2f}")

        ensemble_predictions.append(
            (compound_id, compound_predictions, y_compound))

    # Visualization of Ensemble Predictions
    visualize_ensemble_predictions(ensemble_predictions)

    # Section 27: Final Example Usage
    simulate_tire_degradation_live(
        "../../outputs/week3/lap_prediction_data.csv",
        driver_numbers=[44, 16],
        n_laps=20,
        update_interval=2.0
    )
