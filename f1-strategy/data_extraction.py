"""
Archive for extracting data from Spanish Grand Prix 2023
"""

import fastf1 as ff1
import pandas as pd
from pathlib import Path


def extract_f1_data(year: int, gp: str, session_type: str = 'R'):
    """Extrae datos de FastF1 y los guarda en Parquet."""
    Path("f1_cache").mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache("f1_cache")

    try:
        session = ff1.get_session(year, gp, session_type)
        session.load()

        Path("data/raw").mkdir(parents=True, exist_ok=True)

        # Obtain pitstops from laps
        pit_stops = session.laps[session.laps["PitInTime"].notna()]
        laps = session.laps
        weather = session.weather_data

        # Store data
        laps.to_parquet(f"data/raw/{gp}_{year}_laps.parquet")
        pit_stops.to_parquet(f"data/raw/{gp}_{year}_pitstops.parquet")
        weather.to_parquet(f"data/raw/{gp}_{year}_weather.parquet")

        print(f"Datos de {gp} {year} extraídos exitosamente!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Execute only for GP España 2023 (initial prioriry)
    extract_f1_data(2023, "Spain")
