"""
OpenF1 Intervals Data Extraction Script
---------------------------------------
This script specifically extracts interval data between drivers
from the OpenF1 API and stores it in Parquet format for later use
in strategy analysis, undercuts/overcuts, etc.

Complements the existing FastF1 extraction script to provide
more precise data about gaps between cars.
"""

import requests
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

# Create necessary directories
Path("data/raw").mkdir(parents=True, exist_ok=True)


def get_session_key(year, gp_name):
    """
    Gets the session_key for the specified race.

    Args:
        year (int): Race year
        gp_name (str): Grand Prix name

    Returns:
        int: session_key to use in OpenF1 API
    """
    # For GP Spain 2023, the correct session_key is 9102
    if year == 2023 and gp_name.lower() == "spain":
        return 9102

    # If we need to dynamically get session_keys for other races,
    # we could implement a search using the OpenF1 sessions API
    # For now, we only implement for Spain 2023

    print(f"⚠️ We only have the session_key implemented for Spain 2023")
    print(f"⚠️ For other races, you need to find the correct session_key")
    return None


def fetch_openf1_intervals(year, gp_name, max_interval=None):
    """
    Extracts interval data between cars from OpenF1 API for a specific race.

    Args:
        year (int): Race year (e.g.: 2023)
        gp_name (str): Grand Prix name (e.g.: 'Spain')
        max_interval (float, optional): Filter intervals smaller than this value (in seconds)

    Returns:
        pandas.DataFrame: DataFrame with interval data between cars
    """
    base_url = "https://api.openf1.org/v1/intervals"

    # Get the session_key for this race
    session_key = get_session_key(year, gp_name)
    if session_key is None:
        print("❌ Could not get session_key for this race")
        return pd.DataFrame()

    print(
        f"Searching for data in OpenF1 for {gp_name} {year} (session_key: {session_key})")

    # Build the URL directly, exactly like the one that works in the browser
    # can change max_interval to min_interval if we change the < sign to >
    if max_interval is not None:
        url = f"{base_url}?session_key={session_key}&interval<{max_interval}"
        print(f"URL: {url}")
    else:
        url = f"{base_url}?session_key={session_key}"
        print(f"URL: {url}")

    try:
        print("Making request to OpenF1 API...")
        response = requests.get(url)
        response.raise_for_status()

        print(f"Response status: {response.status_code}")

        # Check if we have received valid data
        if response.text and response.text.strip():
            try:
                # Try to parse the JSON
                intervals_data = response.json()
                print(
                    f"✓ Found {len(intervals_data)} records for session_key={session_key}")

                # Convert to DataFrame
                df_intervals = pd.DataFrame(intervals_data)

                # Basic data processing
                if 'date' in df_intervals.columns:
                    # Handle different date formats
                    df_intervals['date'] = pd.to_datetime(
                        df_intervals['date'], format='mixed')                # Ensure the column name is consistent
                if 'interval' in df_intervals.columns:
                    df_intervals.rename(
                        columns={'interval': 'interval_in_seconds'}, inplace=True)

                if 'interval_in_seconds' in df_intervals.columns:
                    # Mark intervals in strategic zones
                    df_intervals['undercut_window'] = df_intervals['interval_in_seconds'] < 1.5
                    df_intervals['drs_window'] = df_intervals['interval_in_seconds'] < 1.0

                return df_intervals
            except Exception as e:
                print(f"Error processing data: {e}")
                print("Trying to process manually...")

                # Process manually without using pandas for datetime
                try:
                    intervals_data = json.loads(response.text)
                    df_intervals = pd.DataFrame(intervals_data)

                    # Save dates as strings to avoid format issues
                    if 'date' in df_intervals.columns:
                        # Save as string
                        df_intervals['date_str'] = df_intervals['date']
                        # Don't convert to datetime to avoid errors

                    # Ensure the column name is consistent
                    if 'interval' in df_intervals.columns:
                        df_intervals.rename(
                            columns={'interval': 'interval_in_seconds'}, inplace=True)

                    if 'interval_in_seconds' in df_intervals.columns:
                        # Mark intervals in strategic zones
                        df_intervals['undercut_window'] = df_intervals['interval_in_seconds'] < 1.5
                        df_intervals['drs_window'] = df_intervals['interval_in_seconds'] < 1.0

                    print(
                        f"✓ Manual processing successful. Got {len(df_intervals)} records.")
                    return df_intervals
                except Exception as e2:
                    print(f"Error in manual processing: {e2}")
                    return pd.DataFrame()
        else:
            print("Response is empty")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error querying OpenF1: {e}")
        if 'response' in locals():
            print(
                f"Status code: {response.status_code if hasattr(response, 'status_code') else 'N/A'}")
            print(
                f"First 500 characters of response: {response.text[:500] if hasattr(response, 'text') else 'N/A'}")
        return pd.DataFrame()


def extract_openf1_intervals(year, gp_name, max_interval=None):
    """
    Main function to extract and save OpenF1 interval data.

    Args:
        year (int): Race year
        gp_name (str): Grand Prix name
        max_interval (float, optional): Filter intervals smaller than this value (in seconds)
    """
    intervals_df = fetch_openf1_intervals(year, gp_name, max_interval)

    if not intervals_df.empty:
        # Handle mixed data types in columns - ERROR SOLUTION
        # Identify and handle problematic columns with mixed data (like gap_to_leader)
        if 'gap_to_leader' in intervals_df.columns:
            # Option 1: Convert entire column to string
            intervals_df['gap_to_leader'] = intervals_df['gap_to_leader'].astype(
                str)

            # Option 2 (alternative): Create separate columns for numeric values and categories
            # Allows easier subsequent numeric analysis
            intervals_df['gap_to_leader_numeric'] = pd.to_numeric(
                intervals_df['gap_to_leader'], errors='coerce')
            intervals_df['is_lapped'] = intervals_df['gap_to_leader'].astype(
                str).str.contains('LAP')

        # Review other potentially problematic columns with mixed types
        for col in intervals_df.columns:
            if intervals_df[col].dtype == 'object' and col != 'date' and col != 'date_str':
                # If it's not a date, we try to convert to numeric or ensure it's string
                try:
                    # Try to convert to numeric
                    intervals_df[col] = pd.to_numeric(intervals_df[col])
                except:
                    # If it fails, ensure it's string
                    intervals_df[col] = intervals_df[col].astype(str)

        # Save data in Parquet format
        output_path = f"data/raw/{gp_name}_{year}_openf1_intervals.parquet"
        intervals_df.to_parquet(output_path)

        # Also save in CSV for greater compatibility/diagnosis
        csv_path = f"data/raw/{gp_name}_{year}_openf1_intervals.csv"
        intervals_df.to_csv(csv_path, index=False)
        # Summary information
        print(f"Datos de respaldo guardados en CSV: {csv_path}")
        print(f"Interval data saved at {output_path}")
        print(f"Total records: {len(intervals_df)}")
        if 'interval_in_seconds' in intervals_df.columns:
            print(
                f"Interval range: {intervals_df['interval_in_seconds'].min():.2f}s - {intervals_df['interval_in_seconds'].max():.2f}s")
            undercut_opportunities = intervals_df['undercut_window'].sum()
            print(
                f"Undercut opportunities (<1.5s): {undercut_opportunities} ({undercut_opportunities/len(intervals_df)*100:.1f}%)")

        # Crear archivo de metadatos
        metadata = {
            "source": "OpenF1 API",
            "race": {
                "year": year,
                "name": gp_name,
                "session_key": get_session_key(year, gp_name)
            },
            "extraction_date": datetime.now().isoformat(),
            "record_count": len(intervals_df),
            "fields": list(intervals_df.columns),
            "filter_applied": {"max_interval": max_interval} if max_interval is not None else None,
            "column_fixes": ["gap_to_leader converted to string and separate numeric column created"]
        }

        with open(f"data/raw/{gp_name}_{year}_openf1_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        return True
    else:
        print("No se guardaron datos porque no se encontraron registros.")
        return False


if __name__ == "__main__":
    # Extract interval data for the 2023 Spanish GP
    # Using session_key 9102 and filtering intervals smaller than 1.75 seconds
    extract_openf1_intervals(2023, "Spain", max_interval=1.75)
