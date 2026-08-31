"""Fallbacks shared by the agent modules when session_meta or weather arrives incomplete.

A leaf module (no other agent internals, no heavy imports), so pulling these in never
drags in model weights (same reasoning as ``tire_parsing.py``).
"""

from __future__ import annotations

from typing import Any, Mapping

# RaceStateManager.get_session_meta() always supplies total_laps (CLAUDE.md section 6,
# "lap_state is the single contract"; race_state_manager.py's own get_session_meta sets
# it unconditionally). This constant only guards a hand-built session_meta -- a test
# fixture, or a partially populated state -- from resolving a missing key to a
# different number at every call site. 57 is the median/mode race length across the
# 2023-2025 dataset (70 races) -- NOT "2022-2025": there is no 2022 season anywhere in
# data/ (CLAUDE.md section 1). Previously restated as a bare literal in
# pit_strategy_agent.py (x3), race_situation_agent.py (x2) and tire_agent.py (x1);
# consolidated here so a future change updates every caller at once instead of drifting
# one site at a time.
DEFAULT_TOTAL_LAPS: int = 57

# How many times an agent's LLM client retries a recoverable failure. The SDK retries
# only connection errors, 408, 409, 429 and 5xx, and it waits what the provider asks:
# ``openai._base_client.BaseClient._calculate_retry_timeout`` honours ``Retry-After``
# when it is between 0 and 60 s, and otherwise backs off exponentially with jitter. So
# the budget is spent in provider-declared waits rather than in guesses.
#
# It was 1, which is two attempts, and a single 429 mid-burst cost a whole lap of an LLM
# run: the API reported 178831 of 200000 TPM used and asked for a retry, but a TPM
# window drains over a MINUTE and both attempts landed inside it (#1153). Five covers
# one window. It does NOT make a long enough burst safe; pacing the run still does that.
#
# Restated as a bare ``1`` in twelve places across six agent modules before this, which
# is the same drift ``DEFAULT_TOTAL_LAPS`` above was consolidated to stop.
LLM_MAX_RETRIES: int = 5


def reading_or_default(source: Mapping[str, Any], key: str, default: float) -> float:
    """Read a numeric reading that may be ABSENT or PRESENT-AND-``None``.

    ``dict.get(key, default)`` only fires its default when the KEY is missing. Our
    producers deliberately report an unmeasured reading as the key present with a
    ``None`` value -- ``_safe_none`` in the telemetry backend and
    ``race_state_manager.get_weather_state`` both do this on purpose, so that an absent
    measurement never becomes a searchable sentinel (#465). The two conventions meet
    badly: ``wx.get('air_temp', 28.0)`` returns ``None``, and the ``float()`` one layer
    down raises ``TypeError``.

    That is not hypothetical. Every 2025 laps parquet ships without weather columns, so
    the backend's producer emits ``None`` for all four readings on every 2025 lap. It
    crashed ``/recommend`` with a 422 for the whole default season (#788) via
    ``race_situation_agent``, and it silently moved ``tire_agent``'s cliff estimate 2.3
    laps in the optimistic direction -- the dangerous one, since it delays the pit call.

    The bug shape is a twin that never got the fix: ``pace_agent`` had guarded this read
    with an inline conditional and a comment describing the exact crash, while the
    identical reads in ``tire_agent`` and ``race_situation_agent`` had not. Hence one
    named function rather than a fourth inline copy.

    ``default`` stayed a per-caller argument because the agents disagreed on the
    fallback temperatures when this was written (pace 25/35, tire and race_situation
    28/38), and reconciling those numbers was a modelling decision rather than something
    to smuggle in behind a crash fix. #789 then reconciled them, so every TEMPERATURE call
    site now passes the two constants below. The parameter stays because the signature is
    the general one and because the other quantities never went through that
    consolidation: humidity is 50.0 at every caller, and rainfall is ``0`` at
    ``pace_agent.py:974`` and ``0.0`` at ``tire_agent.py:1647``. Those two are
    per-caller numbers still, not evidence that the temperature disagreement survives.
    """
    value = source.get(key)
    if value is None:
        return default
    return value


# The air and track temperature a consumer reads when the reading is genuinely absent.
#
# MEASURED, not chosen: the medians over the 66,924 laps of 2023-2025, taken through
# ``augment_featured_laps``. AirTemp median 24.6, TrackTemp median 34.7.
#
# One pair, because there were FIVE for the same two quantities (#789), three of them
# feeding models: pace read 25.0/35.0, tire and race_situation read 28.0/38.0, and the
# backend producer read 25/40 in one route and 28/38 in another. The 28.0/38.0 pair sat
# 3.8 degrees above the median in both quantities, and pace's 35.0 matches the TrackTemp
# MEAN (35.22) rather than its median, which is how two plausible numbers for one
# quantity survive next to each other.
#
# WHAT UNIFYING THEM DOES NOT DO is move any model input on a replay. Measured through
# ``RaceReplayEngine.replay()`` over four races across three seasons, 229 lap states
# carried a real air and track temperature on 100% of laps, so these defaults never
# fire there. (Measured through ``rsm.get_lap_state(lap)`` WITHOUT the weather frame
# they appear to fire on every lap, which is an artefact of calling the state manager
# directly rather than through the engine that supplies it -- the same wrong-call
# reading CLAUDE.md section 11 already records for the arcade temperatures.)
#
# So this is a dedup of a LATENT disagreement, and the paths where it does bite are the
# ones with no weather frame at all: a hand-built session_meta, a test fixture, or the
# backend producer that emits None per key (#788).
# Re-measured after the 2023 Spanish GP duplicate left the dataset. That race was in the
# featured files twice, so the earlier 24.2 / 34.2 described a sample of 68,122 laps that
# counted one weekend's weather double. The move is small and its direction is the point:
# the constants are a claim about the dataset, and it was the dataset that was wrong.
DEFAULT_AIR_TEMP_C: float = 24.6
DEFAULT_TRACK_TEMP_C: float = 34.7
