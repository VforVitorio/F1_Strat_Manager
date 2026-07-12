"""Friendly GP name → on-disk corpus slug mapping (single source of truth).

The static OpenF1 radio builder writes per-GP folders under
``data/processed/race_radios/{year}/{slug}/`` and
``data/raw/radio_audio/{year}/{slug}/`` where ``slug`` is computed by
``RadioDatasetBuilder._compute_slug``: lowercased country for single-race
countries (``bahrain``, ``united_kingdom``, ...) and ``country_circuit``
for the multi-race countries Italy and United States (``italy_imola``,
``italy_monza``, ``united_states_miami``, ``united_states_austin``,
``united_states_las_vegas``).

The CLI / featured-laps parquet, however, uses **friendly** GP names that
do not necessarily match the country (``Sakhir``, ``Imola``, ``Marina Bay``,
``Yas Island``, ...). This module is the *one* place that translates from
those friendly names to the on-disk slugs, so the runner, the FastAPI
endpoints, and the lazy first-run downloader (``data_cache``) all stay in
sync. Both ``src/nlp/radio_runner.py`` and
``src/f1_strat_manager/data_cache.py`` import the resolver from here —
keeping it in this module (not under ``src/agents/`` or ``src/nlp/``)
avoids dragging the radio-NLP package init (Whisper, librosa, the
sentiment / intent / NER classifiers) into the lightweight data-bootstrap
path on first run.
"""

from __future__ import annotations

# Map from the friendly GP names used by the CLI / featured-laps parquet to
# the on-disk slug produced by ``RadioDatasetBuilder._compute_slug``. Single
# entries for single-race countries, distinct entries per circuit for the
# two double-header countries (Italy = Imola + Monza, United States = Miami
# + Austin + Las Vegas). Adding another double-header country in a future
# season is a one-line change here followed by a Phase 0 rebuild of the
# affected GPs — no agent code needs to know.
COUNTRY_SLUG_BY_GP: dict[str, str] = {
    # Single-race countries — slug is just the lowercased country name
    "Sakhir": "bahrain",
    "Jeddah": "saudi_arabia",
    "Melbourne": "australia",
    "Suzuka": "japan",
    "Shanghai": "china",
    "Monaco": "monaco",
    "Barcelona": "spain",
    "Montréal": "canada",
    "Montreal": "canada",  # ASCII fallback for CLI input
    "Spielberg": "austria",
    "Silverstone": "united_kingdom",
    "Budapest": "hungary",
    "Spa-Francorchamps": "belgium",
    "Zandvoort": "netherlands",
    "Baku": "azerbaijan",
    "Marina Bay": "singapore",
    "Mexico City": "mexico",
    "São Paulo": "brazil",
    "Sao Paulo": "brazil",  # ASCII fallback for CLI input
    "Lusail": "qatar",
    "Yas Island": "united_arab_emirates",
    # Multi-race countries — slug carries the circuit suffix from Phase 0
    "Imola": "italy_imola",
    "Monza": "italy_monza",
    "Miami": "united_states_miami",
    "Austin": "united_states_austin",
    "Las Vegas": "united_states_las_vegas",
}


# On-disk raw folder names that differ from the friendly key by more than the
# underscore substitution — a mid-season circuit rename. Keyed by the folder
# name under ``data/raw/{year}/``, value is the canonical friendly name used by
# both :data:`COUNTRY_SLUG_BY_GP` and ``data/tire_compounds_by_race.json``. The
# space-vs-underscore forms (``Las_Vegas`` → ``Las Vegas``) are handled generically
# by :func:`canonical_gp_name` and do NOT belong here.
FOLDER_ALIASES: dict[str, str] = {
    "Miami_Gardens": "Miami",  # 2025 raw folder; 2023/2024 used "Miami"
}


def canonical_gp_name(name: str) -> str:
    """Normalise any GP identifier form to the canonical friendly name.

    A single GP is referred to by several strings across the project: the
    friendly name the featured-laps parquet uses (``"Las Vegas"``), the raw
    on-disk folder name with underscores (``"Las_Vegas"``, ``"Marina_Bay"``),
    and the occasional renamed folder (``"Miami_Gardens"`` for what the tables
    call ``"Miami"``). Both :data:`COUNTRY_SLUG_BY_GP` and
    ``data/tire_compounds_by_race.json`` are keyed by the *friendly* name, so
    every caller that starts from a folder name must funnel through here first
    or it silently misses the lookup (~6 GPs/season had no radio and no
    compound labels before this existed).

    Resolution order: exact friendly name, then an explicit folder alias, then
    the generic underscore→space form. Unknown inputs are returned unchanged so
    the caller keeps control of the miss (the radio resolver raises; the
    compound lookup degrades to a short label).
    """
    if name in COUNTRY_SLUG_BY_GP:
        return name
    if name in FOLDER_ALIASES:
        return FOLDER_ALIASES[name]
    spaced = name.replace("_", " ")
    if spaced in COUNTRY_SLUG_BY_GP:
        return spaced
    return name


def resolve_gp_slug(gp_name: str) -> str:
    """Translate a friendly or on-disk GP name into the corpus slug.

    Accepts the names the CLI passes (``"Sakhir"``, ``"Imola"``,
    ``"Marina Bay"``, ...) *and* the raw folder names with underscores
    (``"Las_Vegas"``, ``"Miami_Gardens"``, ...) via
    :func:`canonical_gp_name`, and returns the slug used by the static
    builder for the corpus directories under
    ``data/processed/race_radios/{year}/`` and
    ``data/raw/radio_audio/{year}/``. Falls through silently when the
    input is *already* a slug, which keeps callers reentrant — passing
    the canonical form a second time is a no-op instead of an error,
    which matters for ``ensure_radio_corpus`` retrying after a partial
    download.

    Raises :class:`ValueError` listing the known GP names whenever the
    input matches neither a friendly/folder name nor an existing slug, so
    a typo at the CLI surfaces immediately instead of producing a silent
    zero-radio simulation.
    """
    canonical = canonical_gp_name(gp_name)
    if canonical in COUNTRY_SLUG_BY_GP:
        return COUNTRY_SLUG_BY_GP[canonical]
    if gp_name in set(COUNTRY_SLUG_BY_GP.values()):
        return gp_name
    raise ValueError(f"Unknown GP {gp_name!r}. Known: {sorted(COUNTRY_SLUG_BY_GP)}")
