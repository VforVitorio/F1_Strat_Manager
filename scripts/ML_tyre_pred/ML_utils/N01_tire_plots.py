import matplotlib.pyplot as plt


def plot_lap_time_deltas(seq_data, compound_colors, compound_names):
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
        plt.show()


def plot_speed_vs_tire_age(data, compound_colors, compound_names, compound_id=2):
    speed_columns = ['SpeedI1', 'SpeedI2', 'SpeedFL']
    plt.figure(figsize=(14, 8))
    subset = data[data['CompoundID'] == compound_id]
    for speed_col in speed_columns:
        agg_data = subset.groupby('TyreAge')[speed_col].mean().reset_index()
        if len(agg_data) > 1:
            plt.plot(agg_data['TyreAge'], agg_data[speed_col],
                     'o-', label=f'{speed_col}')
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Speed (kph)')
    plt.title(
        f'Effect of Tire Age on Speed - {compound_names.get(compound_id)} Tires')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_regular_vs_adjusted_degradation(tire_deg_data, compound_colors, compound_names, lap_time_improvement_per_lap):
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
        plt.subplot(len(compound_ids), 1, i + 1)
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
        total_fuel_effect = total_laps * lap_time_improvement_per_lap
        plt.annotate(f"Est. total fuel effect: ~{total_fuel_effect:.2f}s", xy=(
            0.02, 0.05), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        if i == len(compound_ids) - 1:
            plt.xlabel('Tire Age (laps)')
    plt.tight_layout()
    plt.show()


def plot_fuel_adjusted_degradation(tire_deg_data, compound_colors, compound_names):
    plt.figure(figsize=(14, 7))
    compound_ids = tire_deg_data['CompoundID'].unique()
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
    plt.show()


def plot_fuel_adjusted_percentage_degradation(tire_deg_data, compound_colors, compound_names):
    plt.figure(figsize=(14, 7))
    compound_ids = tire_deg_data['CompoundID'].unique()
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
    plt.show()


def plot_degradation_rate(tire_deg_data, compound_colors, compound_names):
    plt.figure(figsize=(14, 7))
    compound_ids = tire_deg_data['CompoundID'].unique()
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
                             deg_stats['mean'] + deg_stats['std'], color=color, alpha=0.2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Tire Age (laps)')
    plt.ylabel('Fuel-Adjusted Degradation Rate (s/lap)')
    plt.title('Tire Degradation Rate by Compound (Fuel Effect Removed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
