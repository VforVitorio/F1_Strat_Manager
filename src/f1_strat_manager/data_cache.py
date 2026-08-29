"""First-run data cache resolver + HuggingFace Hub downloader.

The CLI ships as two console scripts (``f1-strat``, ``f1-sim``) that can be
installed globally via ``uv tool install git+https://…`` without cloning
the repository. In that install mode the code arrives on disk but the
~15 GB of trained models and FastF1 race dumps do not: they live on the
HuggingFace Dataset ``VforVitorio/f1-strategy-dataset``. This module is
the bridge: it resolves which on-disk directory should hold those assets
(preferring an editable-dev checkout when one is present) and downloads
the missing pieces from HF Hub on the first run with a Rich progress bar.

All resolution goes through :func:`get_data_root` so that sub-agents,
scripts, and the CLI land on the same directory regardless of how the
user installed the project. The order of precedence is:

    1. ``$F1_STRAT_DATA_ROOT``: explicit override for power users who
       want the cache on a different volume.
    2. ``<repo>/data/``: when the running module sits inside a git
       checkout, the walker finds the repo root and prefers the existing
       ``data/`` tree so editable-dev workflows never trigger downloads.
    3. ``~/.f1-strat/data/``: user cache directory for the global
       ``uv tool install`` scenario where there is no repo to live next to.

Environment variables respected
-------------------------------
F1_STRAT_DATA_ROOT
    Absolute path to use as the data root. Overrides the walker and the
    user cache. Useful for putting the cache on a fast NVMe volume.
F1_STRAT_OFFLINE
    Set to ``"1"`` to disable every HF Hub call. :func:`ensure_setup`
    becomes a no-op instead of raising; downstream callers are expected
    to handle missing files with clear error messages.
F1_STRAT_NO_FIRST_RUN
    Set to ``"1"`` to skip the first-run setup flow entirely. Intended
    for CI runs where the data is mounted from a cache volume.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

# ── Rich colour palette mirrored from scripts/run_simulation_cli.py ────────────
# Kept verbatim so the first-run UX blends into the rest of the CLI visual
# language (gold watch / green ok / red alert / dim grey labels).
COL_OK = "green3"
COL_WATCH = "gold1"
COL_ALERT = "red3"
COL_LABEL = "grey70"
COL_DIM = "grey50"
COL_HEADLINE = "bright_white"

# ── HuggingFace dataset identity ───────────────────────────────────────────────
# The canonical remote where race data and model weights live. Kept as a
# module-level constant so tests can monkey-patch it and advanced users can
# point to a fork by editing a single line.
HF_DATASET_REPO_ID: str = "VforVitorio/f1-strategy-dataset"
HF_DATASET_REVISION: str = "main"

# ── Sentinel race for first-run smoke tests ────────────────────────────────────
# We always pull at least one full GP folder during the default setup so the
# user can immediately run ``f1-sim Melbourne NOR McLaren --no-llm`` to verify
# the install. Melbourne 2025 is the smallest guaranteed-present race folder
# on the Hub audit (~500 KB of parquet files).
DEFAULT_SENTINEL_RACE: tuple[int, str] = (2025, "Melbourne")

# ── Critical files used to detect whether a setup has already happened ────────
# If any of these are missing we consider the install "fresh" and trigger the
# download flow. Kept deliberately small: only the load-bearing artefacts
# that every sub-agent touches, so that a partially-populated cache still
# boots as long as it has the essentials.
_CRITICAL_MODEL_FILES: tuple[str, ...] = (
    "tire_degradation/tiredeg_modelA_v4.pt",
    "tire_degradation/mc_dropout_calibration.json",
    "overtake_probability/lgbm_overtake_v1.pkl",
    "safety_car_probability/lgbm_sc_v1.pkl",
    "pit_prediction/hist_pit_p50_v1.pkl",
    "pit_prediction/lgbm_undercut_v1.pkl",
    "nlp/pipeline_config_v1.json",
    "agents/pace_agent_config_v1.json",
    "agents/strategy_orchestrator_config_v1.json",
)

# ── Default HF Hub allow_patterns ─────────────────────────────────────────────
# snapshot_download accepts glob-style patterns; these pull only the
# load-bearing assets plus the sentinel race so the first-run footprint is
# around 7-8 GB instead of the full 31.7 GB (which includes 6 redundant
# DeBERTa intent checkpoints that N24's production path never touches).
_DEFAULT_MODEL_PATTERNS: tuple[str, ...] = (
    "data/models/tire_degradation/**",
    "data/models/overtake_probability/**",
    "data/models/safety_car_probability/**",
    "data/models/pit_prediction/**",
    "data/models/lap_time/**",
    "data/models/k_means_circuit_clustering/**",
    "data/models/agents/**",
    "data/models/nlp/pipeline_config_v1.json",
    "data/models/nlp/sentiment_classifier_v1/**",
    "data/models/nlp/intent_setfit_modernbert_v1/**",
    "data/models/nlp/ner_v1/bert_bio_v1/**",
    "data/models/nlp/rcm_parser_v1/**",
    # The RoBERTa sentiment weights, read at MODULE IMPORT by
    # `radio_agent.py` (a bare `torch.load`, not a lazy accessor). Without
    # them the orchestrator cannot be imported at all, so the CLI, the
    # arcade pipeline and every eval tier are unreachable on a clean
    # install - a harder failure than a degraded feature (#837).
    "data/models/nlp/bert_sentiment_v1/**",
    "data/models/xgb_laptime_final.json",
    "data/models/xgb_laptime_final_feature_names.json",
    "data/models/xgb_laptime_global_v1.json",
    "data/models/model_registry.json",
    # Featured parquet + supporting configs: the CLI loads these directly.
    #
    # ALL FOUR, not just 2025. The per-year 2023/2024 files are what
    # `pace_agent._load_circuit_mean_sector_speed` reads to build its (Year, GP) speed map,
    # and the combined file is what N07 trained the tyre TCN on. A clean install pulled only
    # the 2025 one, so those two lookups silently resolved against nothing on exactly the
    # machines that had never run the notebooks — invisible to every developer whose disk
    # already held them (the same shape as the #798 undercut gap below).
    "data/processed/laps_featured.parquet",
    "data/processed/laps_featured_2023.parquet",
    "data/processed/laps_featured_2024.parquet",
    "data/processed/laps_featured_2025.parquet",
    "data/processed/feature_manifest_laptime.json",
    "data/processed/tiredeg_feature_manifest.json",
    "data/processed/tiredeg_sequence_config.json",
    "data/processed/circuit_clustering/**",
    # N16's undercut holdout. NOT optional and not an eval-only artefact:
    # `PitStrategyAgent.__init__` reads it unconditionally to pre-aggregate the
    # per-circuit and per-team undercut rates, so without it the pit agent cannot be
    # CONSTRUCTED and every surface that reaches it raises FileNotFoundError. It was
    # missing from this list and from the dataset itself, which stayed invisible
    # because every machine that has run the notebook already had the file (#798).
    "data/processed/undercut_labeled/**",
    # N13's safety-car labels, read at MODULE IMPORT by
    # `race_situation_agent.py` to build the per-circuit SC base-rate map.
    # Same class as the undercut gap above and as the sentiment weights
    # higher up: import-time, so its absence is not a missing feature but
    # an orchestrator that cannot be constructed (#837). Three artefacts,
    # one pattern list, and only one of them had ever been noticed.
    "data/processed/sc_labeled/**",
    # Radio corpus metadata: small parquets (~430 KB total for the full
    # 2025 calendar) that the runner reads to enumerate per-lap team-radio
    # rows and FIA race-control messages. The matching MP3 audio tree under
    # data/raw/radio_audio/** is intentionally NOT pulled by default
    # (~80 MB); :func:`ensure_radio_corpus` downloads it lazily per GP
    # only when the simulation actually targets that race.
    "data/processed/race_radios/**",
    # RAG index: optional, the Hub may not have it yet. snapshot_download
    # ignores missing patterns silently.
    "data/rag/**",
)


# ==============================================================================
# Path resolution
# ==============================================================================


def _find_repo_root() -> Path | None:
    """Walk up from this file looking for a ``.git`` sibling.

    Returns the repo root path when the package is running from an editable
    checkout, or ``None`` when it is not (i.e. installed via
    ``uv tool install`` into an isolated tool venv). The walker is the same
    pattern used by ``src/agents/strategy_orchestrator.py`` so behaviour
    stays consistent across the codebase.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def get_data_root() -> Path:
    """Resolve the on-disk directory that holds race data and model weights.

    Precedence is documented at the module top. The returned directory is
    guaranteed to exist (created on demand) so callers can append subpaths
    without having to mkdir themselves. All other helpers in this module
    route through this function, which keeps the "which directory are we
    using" decision in exactly one place.
    """
    override = os.environ.get("F1_STRAT_DATA_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    repo = _find_repo_root()
    if repo is not None:
        root = repo / "data"
        root.mkdir(parents=True, exist_ok=True)
        return root

    root = Path.home() / ".f1-strat" / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_models_root() -> Path:
    """Resolve the models directory using the same precedence as data root.

    Returns ``<data_root>/models``: the same directory whether running from
    a repo checkout (``<repo>/data/models``) or the user cache
    (``~/.f1-strat/data/models``), since :func:`get_data_root` has already
    resolved which one applies. Models sit inside the data root rather than
    beside it, so a single HF ``snapshot_download`` call populates both
    trees in one pass.
    """
    # Both editable-dev and user-cache layouts keep models under data/models/
    root = get_data_root() / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ==============================================================================
# First-run detection
# ==============================================================================


def is_first_run() -> bool:
    """Return True when the essential data and models are not on disk yet.

    The check is intentionally permissive: it only looks for one race
    folder under ``data/raw/<year>/`` and the handful of model files listed
    in ``_CRITICAL_MODEL_FILES``. Partial installs where only some of the
    extras are missing (e.g. the RAG Qdrant index) do NOT trigger the
    first-run flow, because those failures have their own dedicated error
    paths downstream and we do not want to re-download gigabytes just
    because one sub-agent is misconfigured.
    """
    data_root = get_data_root()
    models_root = get_models_root()

    # At least one race folder must exist under raw/<year>/
    raw_root = data_root / "raw"
    has_any_race = False
    if raw_root.exists():
        for year_dir in raw_root.iterdir():
            if year_dir.is_dir():
                for gp_dir in year_dir.iterdir():
                    if gp_dir.is_dir() and any(gp_dir.iterdir()):
                        has_any_race = True
                        break
            if has_any_race:
                break

    if not has_any_race:
        return True

    # At least one critical model file must be missing for us to call it
    # a first run. If every critical file is present we assume the install
    # is warm.
    for rel in _CRITICAL_MODEL_FILES:
        if not (models_root / rel).exists():
            return True

    return False


# ==============================================================================
# Download helpers
# ==============================================================================


def _build_allow_patterns(races: Iterable[tuple[int, str]] | None) -> list[str]:
    """Build the snapshot_download allow_patterns list.

    Takes the static set of model patterns and appends one
    ``data/raw/<year>/<gp>/**`` glob per requested race. When ``races`` is
    None we include the sentinel race so the user always ends up with at
    least one runnable GP after setup.
    """
    patterns = list(_DEFAULT_MODEL_PATTERNS)
    race_list: list[tuple[int, str]]
    if races is None:
        race_list = [DEFAULT_SENTINEL_RACE]
    else:
        race_list = list(races)
    for year, gp in race_list:
        patterns.append(f"data/raw/{year}/{gp}/**")
    return patterns


def _snapshot_download(
    allow_patterns: list[str],
    show_progress: bool,
) -> Path:
    """Thin wrapper over ``huggingface_hub.snapshot_download``.

    Keeps the import local so the CLI startup cost is not paid when the
    cache is already warm. Forces the download destination to live inside
    :func:`get_data_root` so that subsequent lookups via the same helper
    find the files regardless of where ``huggingface_hub`` would otherwise
    store them.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - import-time failure
        raise RuntimeError(
            "huggingface_hub is not installed. Reinstall the project with "
            "`uv sync` or `pip install -e .` to pull the declared dependency."
        ) from exc

    data_root = get_data_root()
    # snapshot_download writes into local_dir/ preserving the repo layout,
    # which already matches ``data/…`` + ``models/…`` on the remote, so we
    # point local_dir at the data_root parent (so ``models/`` lands next to
    # ``data/`` on disk) when running in user-cache mode, or at the repo
    # root when running from a clone.
    repo = _find_repo_root()
    if repo is not None and data_root == repo / "data":
        local_dir = repo
    else:
        # User cache layout: ~/.f1-strat/ contains both data/ and models/
        local_dir = data_root.parent

    local_dir.mkdir(parents=True, exist_ok=True)

    # tqdm on/off: ``snapshot_download`` does not accept a progress arg
    # directly but respects the HF_HUB_DISABLE_PROGRESS_BARS env var.
    if not show_progress:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        snapshot_download(
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset",
            revision=HF_DATASET_REVISION,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
        )
    finally:
        if not show_progress:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

    return local_dir


def _resolve_repo() -> None:
    """Ask the Hub for the dataset's metadata, and nothing else.

    This is what the spinner in :func:`ensure_data` is actually waiting on: is
    the Hub reachable, does the revision exist, do the credentials work. One
    API call answers all three, and it answers them before any bytes are
    fetched, which is the fail-fast the caller wants.

    It replaces a full silent `snapshot_download` that was standing in for the
    probe (#168). That call downloaded the whole 7-8 GB with progress bars
    switched off, under a spinner reading "Resolving...", and only then did the
    second call run with progress on, find every file present and return at
    once. A first-time user watched a spinner for the entire download and then
    a progress bar that completed instantly, and the Hub metadata sweep ran
    twice.
    """
    from huggingface_hub import HfApi

    HfApi().repo_info(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        revision=HF_DATASET_REVISION,
    )


def _render_header(console, data_root: Path) -> None:
    """Print the first-run banner using the same Rich palette as the CLI.

    Kept separate from :func:`ensure_setup` so tests can import it without
    pulling in ``huggingface_hub`` and so the visual style stays consistent
    with the lap-by-lap panels rendered by ``run_simulation_cli.py``.
    """
    from rich.panel import Panel
    from rich.table import Table

    grid = Table.grid(padding=(0, 2), expand=False)
    grid.add_column(style=COL_DIM, justify="right", min_width=10)
    grid.add_column(justify="left")
    grid.add_row("Dataset", f"[{COL_HEADLINE}]{HF_DATASET_REPO_ID}[/{COL_HEADLINE}]")
    grid.add_row("Cache", f"[{COL_HEADLINE}]{data_root}[/{COL_HEADLINE}]")
    grid.add_row(
        "Note",
        f"[{COL_DIM}]Downloads ~7-8 GB on first run — models, configs, and one sentinel race.[/{COL_DIM}]",
    )
    grid.add_row(
        "Override",
        f"[{COL_DIM}]$F1_STRAT_DATA_ROOT / $F1_STRAT_OFFLINE=1 / $F1_STRAT_NO_FIRST_RUN=1[/{COL_DIM}]",
    )

    console.print(
        Panel(
            grid,
            title=f"[bold {COL_WATCH}]F1 StratLab — first-run setup[/bold {COL_WATCH}]",
            title_align="center",
            border_style=COL_WATCH,
            padding=(1, 2),
            expand=False,
        )
    )


def ensure_setup(
    races: list[tuple[int, str]] | None = None,
    skip_if_offline: bool = True,
    show_progress: bool = True,
) -> None:
    """Run the first-run setup: download data + models from HF Hub.

    Invoked by the CLI entry points when :func:`is_first_run` returns True.
    Honours ``F1_STRAT_NO_FIRST_RUN`` (exits silently) and ``F1_STRAT_OFFLINE``
    (exits silently when ``skip_if_offline`` is True, otherwise raises a
    descriptive error). Downloads via ``huggingface_hub.snapshot_download``
    with a curated ``allow_patterns`` list so the footprint stays at roughly
    7-8 GB instead of the full 31.7 GB dataset.

    The parameters exist for callers that want to customise the flow:
    ``races`` lets CI runs hand-pick which GPs to cache up-front, and
    ``show_progress`` toggles the Rich progress bar for headless logs.

    Raises a ``RuntimeError`` with an actionable hint when the HF Hub call
    fails for network or auth reasons.
    """
    if os.environ.get("F1_STRAT_NO_FIRST_RUN") == "1":
        return

    if os.environ.get("F1_STRAT_OFFLINE") == "1":
        if skip_if_offline:
            return
        raise RuntimeError(
            "F1_STRAT_OFFLINE=1 is set but the data cache is not populated. "
            "Unset the variable or pre-populate the cache manually."
        )

    from rich.console import Console

    console = Console()

    data_root = get_data_root()
    _render_header(console, data_root)

    patterns = _build_allow_patterns(races)

    try:
        with console.status(
            f"[{COL_DIM}]Resolving {HF_DATASET_REPO_ID} from HuggingFace Hub…[/{COL_DIM}]",
            spinner="dots",
        ):
            _resolve_repo()
        local_dir = _snapshot_download(patterns, show_progress=show_progress)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download F1 StratLab assets from HuggingFace Hub.\n"
            f"  Dataset: {HF_DATASET_REPO_ID}\n"
            f"  Cache  : {data_root}\n"
            "  Check your internet connection, set HF_TOKEN if the dataset is\n"
            "  private, or set $F1_STRAT_OFFLINE=1 to skip the download and\n"
            "  provide the files manually.\n"
            f"  Original error: {exc}"
        ) from exc

    console.print(f"[{COL_OK}][OK] Setup complete. Cached under {local_dir}[/{COL_OK}]")


# Races the dataset stores under a name the calendar does not use. Keyed by
# season because the same race is not named the same way in every one: Miami is
# `Miami` in 2023 and 2024 and `Miami_Gardens` in 2025, after the circuit's
# location rather than the city. Everything else is the label with its spaces
# turned into underscores, measured against the repository listing: with this
# table, 70 of the 70 rounds the menu offers across 2023-2025 resolve.
_RACE_FOLDER_ALIASES: dict[tuple[int, str], str] = {
    (2025, "Miami"): "Miami_Gardens",
}


def race_folder(year: int, gp_name: str) -> str:
    """The dataset folder holding this GP, from the name the menu shows.

    `ensure_race` used to glob the label verbatim. `snapshot_download` on a
    pattern that matches nothing returns quietly, so picking "Mexico City"
    fetched zero files, raised nothing, and left the strategy layer degrading
    against an empty directory: the same defect the lazy fetch was added to
    close, surviving it for six of the 2025 rounds (#1116).
    """
    alias = _RACE_FOLDER_ALIASES.get((year, gp_name))
    if alias is not None:
        return alias
    return gp_name.replace(" ", "_")


def ensure_race(year: int, gp_name: str, show_progress: bool = True) -> Path:
    """Download a single race folder when it is missing and return its path.

    Used by higher-level callers (e.g. the interactive picker in
    ``scripts/cli/pickers.py``) that already know which GP they need but
    want to be robust to a partially-populated cache. Skips the download
    entirely when the folder already exists or when
    ``F1_STRAT_OFFLINE=1``: in the offline case the caller receives the
    (possibly empty) path and can decide whether to raise.
    """
    data_root = get_data_root()
    folder = race_folder(year, gp_name)
    race_dir = data_root / "raw" / str(year) / folder

    if race_dir.exists() and any(race_dir.iterdir()):
        return race_dir

    if os.environ.get("F1_STRAT_OFFLINE") == "1":
        return race_dir

    patterns = [f"data/raw/{year}/{folder}/**"]
    _snapshot_download(patterns, show_progress=show_progress)
    return race_dir


def ensure_radio_corpus(
    year: int,
    gp_name: str,
    show_progress: bool = True,
) -> Path:
    """Download the static OpenF1 radio corpus for a single GP on demand.

    The metadata parquets under ``data/processed/race_radios/**`` are
    pulled by the default first-run setup because they are tiny, but the
    matching MP3 audio tree (~3 MB per GP) only lands on disk when the
    user actually runs a simulation against that race. This helper is the
    lazy bridge: it resolves the friendly GP name to the on-disk slug via
    :mod:`src.f1_strat_manager.gp_slugs`, checks whether the audio folder
    is already populated, and triggers a focused
    ``snapshot_download`` for just that one slug when it is not.

    Returns the resolved audio directory path so callers can immediately
    pass it to :class:`src.nlp.radio_runner.RadioPipelineRunner`. Skips the
    network entirely when ``F1_STRAT_OFFLINE=1`` is set, returning the
    (possibly empty) directory and letting the runner decide whether to
    raise or downgrade to the synthetic fallback.

    Idempotent: re-running on a populated cache short-circuits before
    importing huggingface_hub, so the hot path of an already-warm install
    pays no startup cost beyond the slug lookup itself.
    """
    # Lightweight import: gp_slugs has zero heavy deps so this stays cheap
    # even when the rest of src.agents has not been touched yet.
    from src.f1_strat_manager.gp_slugs import resolve_gp_slug

    try:
        slug = resolve_gp_slug(gp_name)
    except ValueError:
        # Unknown name even after canonical normalisation: a genuine typo or an
        # as-yet-unmapped GP. We do NOT raise here (the runner may accept the raw
        # gp_name path and gives a clearer error), but warn so the miss is not
        # fully silent: a zero-radio simulation used to look identical to a
        # healthy one (#243).
        print(
            f"[warn] ensure_radio_corpus: no radio-corpus mapping for {gp_name!r} "
            f"({year}); simulating with no team radio.",
            file=sys.stderr,
        )
        return get_data_root() / "raw" / "radio_audio" / str(year) / gp_name

    data_root = get_data_root()
    audio_dir = data_root / "raw" / "radio_audio" / str(year) / slug

    if audio_dir.exists() and any(audio_dir.iterdir()):
        return audio_dir

    if os.environ.get("F1_STRAT_OFFLINE") == "1":
        return audio_dir

    # Pull both the audio tree and the matching parquets in one shot. The
    # parquets are usually already on disk from the default first-run pull
    # but listing them here makes ensure_radio_corpus self-contained for
    # the case where the user manually deleted data/processed/race_radios/.
    patterns = [
        f"data/raw/radio_audio/{year}/{slug}/**",
        f"data/processed/race_radios/{year}/{slug}/**",
    ]
    _snapshot_download(patterns, show_progress=show_progress)
    return audio_dir


def ensure_models(show_progress: bool = True) -> Path:
    """Download the full model tree when any critical file is missing.

    Cheaper than :func:`ensure_setup` because it does not touch race data.
    Useful when the user already pre-populated ``data/raw/`` manually
    (e.g. by cloning the HF dataset with ``git clone`` or by mounting a
    shared volume) but still needs the model weights on the local disk.
    """
    models_root = get_models_root()
    missing = [rel for rel in _CRITICAL_MODEL_FILES if not (models_root / rel).exists()]
    if not missing:
        return models_root

    if os.environ.get("F1_STRAT_OFFLINE") == "1":
        raise RuntimeError(
            "F1_STRAT_OFFLINE=1 but critical model files are missing: "
            f"{missing[:3]}…. Unset the variable or populate {models_root} manually."
        )

    _snapshot_download(list(_DEFAULT_MODEL_PATTERNS), show_progress=show_progress)
    return models_root


__all__ = [
    "HF_DATASET_REPO_ID",
    "HF_DATASET_REVISION",
    "DEFAULT_SENTINEL_RACE",
    "get_data_root",
    "get_models_root",
    "is_first_run",
    "ensure_setup",
    "ensure_race",
    "race_folder",
    "ensure_radio_corpus",
    "ensure_models",
]
