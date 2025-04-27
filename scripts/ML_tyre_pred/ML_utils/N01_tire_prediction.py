# N01_tire_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants for compound visualization
compound_colors = {
    1: 'red',      # Soft
    2: 'yellow',   # Medium
    3: 'gray'     # Hard
}

compound_names = {
    1: 'Soft',
    2: 'Medium',
    3: 'Hard'
}

# Default fuel effect constant
LAP_TIME_IMPROVEMENT_PER_LAP = 0.055  # seconds per lap


def calculate_fuel_adjusted_metrics(data, lap_time_improvement_per_lap=LAP_TIME_IMPROVEMENT_PER_LAP):
    """
    Calculate fuel-adjusted degradation metrics from raw lap time data.

    Args:
        data (pd.DataFrame): DataFrame with lap time data
        lap_time_improvement_per_lap (float): Seconds of improvement per lap from fuel burn

    Returns:
        pd.DataFrame: DataFrame with added fuel-adjusted metrics
    """
    tire_deg_data = pd.DataFrame()

    # Process each compound separately
    for compound_id in data['CompoundID'].unique():
        # Filter for this compound
        compound_data = data[data['CompoundID'] == compound_id].copy()

        # Sort by TyreAge to see the degradation trend
        compound_data = compound_data.sort_values('TyreAge')

        # Check if we have enough data
        if len(compound_data) < 5:
            continue

        # Find baseline information
        if 1 in compound_data['TyreAge'].values:
            # Get baseline data (TyreAge=1)
            baseline_data = compound_data[compound_data['TyreAge'] == 1]
            baseline_lap_time = baseline_data['LapTime'].mean()
            baseline_tire_age = 1
        else:
            # If no 'new tire' laps, use the minimum TyreAge available
            min_age = compound_data['TyreAge'].min()
            baseline_data = compound_data[compound_data['TyreAge'] == min_age]
            baseline_lap_time = baseline_data['LapTime'].mean()
            baseline_tire_age = min_age

        # Calculate fuel adjustment based on laps from baseline
        compound_data['LapsFromBaseline'] = compound_data['TyreAge'] - \
            baseline_tire_age
        compound_data['FuelEffect'] = compound_data['LapsFromBaseline'] * \
            lap_time_improvement_per_lap

        # Calculate fuel-adjusted lap time
        compound_data['FuelAdjustedLapTime'] = compound_data['LapTime'] + \
            compound_data['FuelEffect']

        # Calculate traditional degradation metrics
        compound_data['TireDegAbsolute'] = compound_data['LapTime'] - \
            baseline_lap_time
        compound_data['TireDegPercent'] = (
            compound_data['LapTime'] / baseline_lap_time - 1) * 100

        # Calculate fuel-adjusted degradation metrics
        # For new tires, no adjustment needed
        baseline_adjusted_lap_time = baseline_lap_time
        compound_data['FuelAdjustedDegAbsolute'] = compound_data['FuelAdjustedLapTime'] - \
            baseline_adjusted_lap_time
        compound_data['FuelAdjustedDegPercent'] = (
            compound_data['FuelAdjustedLapTime'] / baseline_adjusted_lap_time - 1) * 100

        # Add compound info for later aggregation
        compound_data['CompoundName'] = compound_names.get(
            compound_id, f'Unknown ({compound_id})')

        # Add to the combined DataFrame
        tire_deg_data = pd.concat([tire_deg_data, compound_data])

    # Calculate degradation rate
    tire_deg_data = calculate_degradation_rate(tire_deg_data)

    return tire_deg_data


def calculate_degradation_rate(tire_deg_data):
    """
    Calculate the degradation rate for each compound and tire age.

    Args:
        tire_deg_data (pd.DataFrame): DataFrame with tire degradation data

    Returns:
        pd.DataFrame: DataFrame with added DegradationRate column
    """
    compound_ids = tire_deg_data['CompoundID'].unique()

    # Calculate degradation rate for each compound
    for compound_id in compound_ids:
        # Get data for this compound
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]

        # Calculate average lap times per tire age
        avg_laptimes = compound_subset.groupby(
            'TyreAge')['FuelAdjustedLapTime'].mean()

        # Calculate the difference in lap times (degradation rate)
        deg_rates = avg_laptimes.diff()

        # Assign values to the dataframe
        for age, rate in zip(deg_rates.index, deg_rates.values):
            mask = (tire_deg_data['CompoundID'] == compound_id) & (
                tire_deg_data['TyreAge'] == age)
            tire_deg_data.loc[mask, 'DegradationRate'] = rate

    # Fill NaN values with 0 (no degradation yet for new tires)
    tire_deg_data['DegradationRate'] = tire_deg_data['DegradationRate'].fillna(
        0)

    return tire_deg_data


def plot_lap_time_deltas(seq_data, compound_colors=compound_colors, compound_names=compound_names):
    """
    Plot lap time deltas by tire age for each compound.

    Args:
        seq_data (pd.DataFrame): DataFrame with sequential lap data
        compound_colors (dict): Mapping of compound IDs to colors
        compound_names (dict): Mapping of compound IDs to display names

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
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
        return plt.gcf()  # Return the figure instead of showing it
    return None


def plot_speed_vs_tire_age(data, compound_id=2, compound_colors=compound_colors, compound_names=compound_names):
    """
    Plot sector speeds vs tire age for a specific compound.

    Args:
        data (pd.DataFrame): DataFrame with speed data by sector
        compound_id (int): ID of tire compound to analyze
        compound_colors (dict): Mapping of compound IDs to colors
        compound_names (dict): Mapping of compound IDs to display names

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    speed_columns = ['SpeedI1', 'SpeedI2', 'SpeedFL']
    plt.figure(figsize=(14, 8))
    subset = data[data['CompoundID'] == compound_id]

    for speed_col in speed_columns:
        if speed_col in subset.columns:
            agg_data = subset.groupby(
                'TyreAge')[speed_col].mean().reset_index()
            if len(agg_data) > 1:
                plt.plot(agg_data['TyreAge'], agg_data[speed_col],
                         'o-', label=f'{speed_col}')

    compound_name = compound_names.get(compound_id, f'Unknown ({compound_id})')
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Speed (kph)')
    plt.title(f'Effect of Tire Age on Speed - {compound_name} Tires')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def plot_regular_vs_adjusted_degradation(tire_deg_data, compound_colors=compound_colors,
                                         compound_names=compound_names, lap_time_improvement_per_lap=LAP_TIME_IMPROVEMENT_PER_LAP):
    """
    Plot comparison between regular and fuel-adjusted degradation.

    Args:
        tire_deg_data (pd.DataFrame): DataFrame with degradation metrics
        compound_colors (dict): Mapping of compound IDs to colors
        compound_names (dict): Mapping of compound IDs to display names
        lap_time_improvement_per_lap (float): Seconds improvement per lap due to fuel

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=(16, 12))
    compound_ids = tire_deg_data['CompoundID'].unique()

    for i, compound_id in enumerate(compound_ids):
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]
        color = compound_colors.get(compound_id, 'black')
        compound_name = compound_names.get(
            compound_id, f'Unknown ({compound_id})')

        # Calculate aggregated values
        reg_agg = compound_subset.groupby('TyreAge')['TireDegAbsolute'].mean()
        adj_agg = compound_subset.groupby(
            'TyreAge')['FuelAdjustedDegAbsolute'].mean()

        # Create subplot for this compound
        plt.subplot(len(compound_ids), 1, i + 1)

        # Plot regular degradation
        plt.plot(reg_agg.index, reg_agg.values, 'o--', color=color,
                 alpha=0.5, label=f'{compound_name} (Regular)')

        # Plot fuel-adjusted degradation
        plt.plot(adj_agg.index, adj_agg.values, 'o-', color=color,
                 linewidth=2, label=f'{compound_name} (Fuel Adjusted)')

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.ylabel('Degradation (s)')
        plt.title(
            f'{compound_name} Tire Degradation: Regular vs. Fuel-Adjusted')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Calculate fuel effect for annotation
        min_lap = reg_agg.index.min() if not reg_agg.empty else 0
        max_lap = reg_agg.index.max() if not reg_agg.empty else 0
        total_laps = max_lap - min_lap
        total_fuel_effect = total_laps * lap_time_improvement_per_lap

        plt.annotate(f"Est. total fuel effect: ~{total_fuel_effect:.2f}s",
                     xy=(0.02, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Add xlabel only to bottom subplot
        if i == len(compound_ids) - 1:
            plt.xlabel('Tire Age (laps)')

    plt.tight_layout()
    return plt.gcf()


def plot_fuel_adjusted_degradation(tire_deg_data, compound_colors=compound_colors, compound_names=compound_names):
    """
    Plot fuel-adjusted absolute degradation by compound.

    Args:
        tire_deg_data (pd.DataFrame): DataFrame with degradation metrics
        compound_colors (dict): Mapping of compound IDs to colors
        compound_names (dict): Mapping of compound IDs to display names

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=(14, 7))
    compound_ids = tire_deg_data['CompoundID'].unique()

    for compound_id in compound_ids:
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]
        color = compound_colors.get(compound_id, 'black')
        compound_name = compound_names.get(
            compound_id, f'Unknown ({compound_id})')

        # Calculate aggregated values with error bands
        agg_data = compound_subset.groupby('TyreAge')['FuelAdjustedDegAbsolute'].agg([
            'mean', 'std']).reset_index()

        # Plot mean line
        plt.plot(agg_data['TyreAge'], agg_data['mean'], 'o-',
                 color=color, linewidth=2, label=f'{compound_name}')

        # Add error bands if standard deviation is available
        if 'std' in agg_data.columns and not agg_data['std'].isnull().all():
            plt.fill_between(agg_data['TyreAge'],
                             agg_data['mean'] - agg_data['std'],
                             agg_data['mean'] + agg_data['std'],
                             color=color, alpha=0.2)

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Fuel-Adjusted Absolute Degradation (s)')
    plt.title('Tire Degradation by Compound and Age (Fuel Effect Removed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def plot_fuel_adjusted_percentage_degradation(tire_deg_data, compound_colors=compound_colors, compound_names=compound_names):
    """
    Plot fuel-adjusted percentage degradation by compound.

    Args:
        tire_deg_data (pd.DataFrame): DataFrame with degradation metrics
        compound_colors (dict): Mapping of compound IDs to colors
        compound_names (dict): Mapping of compound IDs to display names

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=(14, 7))
    compound_ids = tire_deg_data['CompoundID'].unique()

    for compound_id in compound_ids:
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]
        color = compound_colors.get(compound_id, 'black')
        compound_name = compound_names.get(
            compound_id, f'Unknown ({compound_id})')

        # Calculate aggregated values with error bands
        agg_data = compound_subset.groupby('TyreAge')['FuelAdjustedDegPercent'].agg([
            'mean', 'std']).reset_index()

        # Plot mean line
        plt.plot(agg_data['TyreAge'], agg_data['mean'], 'o-',
                 color=color, linewidth=2, label=f'{compound_name}')

        # Add error bands if standard deviation is available
        if 'std' in agg_data.columns and not agg_data['std'].isnull().all():
            plt.fill_between(agg_data['TyreAge'],
                             agg_data['mean'] - agg_data['std'],
                             agg_data['mean'] + agg_data['std'],
                             color=color, alpha=0.2)

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Fuel-Adjusted Percentage Degradation (%)')
    plt.title(
        'Percentage Tire Degradation by Compound and Age (Fuel Effect Removed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def plot_degradation_rate(tire_deg_data, compound_colors=compound_colors, compound_names=compound_names):
    """
    Plot degradation rate for each tire compound.

    Args:
        tire_deg_data (pd.DataFrame): DataFrame with tire degradation metrics
        compound_colors (dict): Mapping of compound IDs to colors
        compound_names (dict): Mapping of compound IDs to display names

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Check if required columns exist
    required_columns = ['CompoundID', 'TyreAge', 'DegradationRate']
    if not all(col in tire_deg_data.columns for col in required_columns):
        missing = [
            col for col in required_columns if col not in tire_deg_data.columns]
        raise ValueError(
            f"Missing required columns in tire_deg_data: {missing}")

    plt.figure(figsize=(14, 7))
    compound_ids = tire_deg_data['CompoundID'].unique()

    for compound_id in compound_ids:
        compound_subset = tire_deg_data[tire_deg_data['CompoundID']
                                        == compound_id]

        # Calculate aggregated metrics
        deg_stats = compound_subset.groupby('TyreAge')['DegradationRate'].agg([
            'mean', 'std']).reset_index()

        color = compound_colors.get(compound_id, 'black')
        compound_name = compound_names.get(
            compound_id, f'Unknown ({compound_id})')

        # Plot mean line
        plt.plot(deg_stats['TyreAge'], deg_stats['mean'], marker='o',
                 linestyle='-', color=color, linewidth=2, label=compound_name)

        # Add error bands if standard deviation is available
        if 'std' in deg_stats.columns and not deg_stats['std'].isnull().all():
            plt.fill_between(deg_stats['TyreAge'],
                             deg_stats['mean'] - deg_stats['std'],
                             deg_stats['mean'] + deg_stats['std'],
                             color=color, alpha=0.2)

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Fuel-Adjusted Degradation Rate (s/lap)')
    plt.title('Tire Degradation Rate by Compound (Fuel Effect Removed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()  # Return the figure instead of showing it
