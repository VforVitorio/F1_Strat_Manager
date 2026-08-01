"""First-iteration FastF1 wrapper, archived (see ``src/shared/README.md``).

Superseded by ``src/data_extraction/fastf1/session_extractor.py`` and
``scripts/download_data.py``. ``extract_f1_data`` takes any year/GP/session,
but the ``__main__`` block below only ever ran it against the 2023 Spanish
GP, the first race this project pulled data for. No code imports this module
today -- ``notebooks/data_engineering/N01_data_download.ipynb`` only
LINK-references it in a markdown cell ("Legacy extraction: [fastf1_extractor.py]
(../../src/shared/data_extraction/fastf1_extractor.py)"), it does not import it
in any code cell. Kept per ``src/shared/README.md``'s stated reason: deleting it
would break that historical link, not any running code.
"""

import fastf1 as ff1
import pandas as pd
from pathlib import Path


def extract_f1_data(year: int, gp: str, session_type: str = 'R'):
    """Load one FastF1 session and write its laps, pit stops, and weather to Parquet.

    Pit stops are derived from ``laps`` by filtering on a non-null
    ``PitInTime``, not fetched separately, so they are always a strict subset
    of the laps file. Any failure (bad GP name, missing session, network
    error) is caught and printed rather than raised, so a batch of calls over
    several GPs does not stop at the first one that fails to load.
    """
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

        print(f"Data from {gp} {year} extracted successfully!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Execute only for GP Spain 2023 (initial priority)
    extract_f1_data(2023, "Spain")
