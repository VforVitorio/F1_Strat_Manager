"""Arcade-local strategy driver.

Runs the full N31 multi-agent pipeline in a background thread against the
same ``RaceReplayEngine`` + featured-laps parquet the backend SSE endpoint
uses, and mutates a shared ``StrategyState`` so ``F1ArcadeView.on_draw``
and the dashboard subprocess can pick up the latest ``LapDecision`` plus
every raw sub-agent output without blocking. The arcade no longer depends
on the FastAPI backend at runtime: it owns its own simulation loop, which
keeps the TFG's CLI/Streamlit path isolated from any arcade change.

Lap loop order matches ``backend/services/simulation/simulator.py::simulate_race``
(the SSE producer). Kept separate so edits to the arcade path cannot
regress the CLI/Streamlit consumers that still depend on the backend.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import pandas as pd

from src.arcade.config import (
    ACCENT,
    DANGER,
    GP_TO_LOCATION,
    INFO,
    REPO_ROOT,
    SUCCESS,
    TEXT_SECONDARY,
    WARNING,
)
from src.f1_strat_manager.laps_augment import augment_featured_laps

logger = logging.getLogger(__name__)


# --- DTOs -----------------------------------------------------------------


@dataclass(frozen=True)
class SimulateRequestDTO:
    """Payload for the simulate endpoint. Mirrors `SimulateRequest` Pydantic
    in `src/telemetry/backend/api/v1/endpoints/strategy.py` without importing
    from the backend package."""

    year: int
    gp: str
    driver: str
    team: str
    driver2: str | None = None
    lap_range: tuple[int, int] | None = None
    risk_tolerance: float = 0.5
    no_llm: bool = False
    # Matches the agents' own preference: ChatOpenAI with the canonical
    # gpt-4.1-mini / orchestrator model names when ``F1_LLM_PROVIDER=openai``
    # (the documented TFG setup). Override to "lmstudio" for local dev
    # against an LM Studio server at ``http://localhost:1234/v1``.
    provider: str = "openai"
    interval_s: float = 0.0


@dataclass(frozen=True)
class StartEventDTO:
    """The run's fixed metadata, emitted once before the first lap.

    ``driver2`` used to sit here, carrying the launch rival's code onto the
    wire, and nothing on either window rendered it (#1052). The rival identity
    a consumer needs is `arcade.driver_rival` on every tick, which since
    schema v2 names one of twenty spans rather than one of two roles. The
    request still carries a `driver2`: that is the arcade's own pick, an input
    rather than a published fact.
    """

    gp: str = ""
    year: int = 0
    driver: str = ""
    team: str = ""
    lap_start: int = 1
    lap_end: int = 0
    total_laps: int = 0
    no_llm: bool = False
    provider: str = ""


@dataclass(frozen=True)
class PerAgentOutputsDTO:
    """Raw per-agent outputs for one lap, ready to be rendered by the
    dashboard. Each field is the dict form of the corresponding agent
    dataclass (``PaceOutput``, ``TireOutput``, ``RaceSituationOutput``,
    ``RadioOutput``, ``PitStrategyOutput``), obtained via
    ``dataclasses.asdict`` so the DTO stays pure-Python and
    JSON-serialisable without pulling ``src/agents/`` types into the
    dashboard process.

    ``regulation_context`` is the string from N30 RAG (empty when the
    agent did not fire). ``rag`` is the structured form of the same
    output (``question`` / ``answer`` / ``articles`` / ``chunks``) used
    by the dashboard's RAG card to render article references and the
    full chunk transcripts on hover; ``None`` when the agent did not
    fire. ``active`` lists the conditional agents routed this lap so
    the dashboard can dim the cards that are idle.
    """

    pace: dict[str, Any] | None = None
    tire: dict[str, Any] | None = None
    situation: dict[str, Any] | None = None
    radio: dict[str, Any] | None = None
    pit: dict[str, Any] | None = None
    regulation_context: str = ""
    rag: dict[str, Any] | None = None
    active: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LapDecisionDTO:
    """One lap's decision, as the wire carries it.

    **What this deliberately does NOT copy from ``StrategyRecommendation``, and
    why (#1046).** That object has fourteen fields and this takes ten. The four
    it leaves behind used to stop here silently, which reads as an oversight
    rather than a decision:

    - ``contingencies`` and ``key_risks`` ARRIVED with #1048's schema bump, which is
      what this paragraph used to promise. ``contingencies`` is the one the
      orchestrator memory audit calls load-bearing, and it is invisible in
      ``reasoning``, which is why it was given its own field instead of being left
      to the model's prose. They are carried as plain dicts and strings rather than
      as the orchestrator's ``Contingency`` model: the wire is JSON and the DTO is
      what ``asdict`` walks.
    - ``expected_stint_end`` stays out. The PLAN timeline already draws the stint
      boundary from ``pit_lap_target``, and a second source for one number on one
      surface is the twin shape this repo pays for most.
    - ``target_lap_time_s`` stays out. Under a safety car the rail sets it to
      ``None`` by Art. 55.7, because N06 predicts green-flag pace and publishing a
      target above the delta sends the driver at a penalty. A field whose absence
      is the load-bearing case needs a designed rendering before it is worth
      carrying, and nothing on either window asks for it.

    **And what it stopped carrying.** ``agent_alerts`` was a list of intent
    strings flattened out of ``radio_out.alerts`` for a Qt dashboard retired in
    ``7ea6a7a6``; it was a lossy copy of ``per_agent.radio.alerts``, which the same
    tick carries in full and which the RADIO console actually renders (#1040).
    """

    lap_number: int = 0
    compound: str = ""
    tyre_life: int = 0
    position: int = 0
    lap_time_s: float | None = None
    gap_ahead_s: float = 0.0
    action: str = "STAY_OUT"
    confidence: float = 0.0
    reasoning: str = ""
    scenario_scores: dict[str, float] = field(default_factory=dict)
    # Optional tactical fields (LLM mode only):
    pace_mode: str | None = None
    risk_posture: str | None = None
    pit_lap_target: int | None = None
    compound_next: str | None = None
    undercut_target: str | None = None
    # Conditional branches the orchestrator planned for upcoming laps, as plain
    # dicts (`trigger`, `switch_to`, `priority`, `rationale`), plus the risks it
    # chose to flag. Both are decision content that stopped at this boundary
    # until schema v2 (#1046).
    contingencies: list[dict[str, Any]] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    guardrail_reason: str | None = None
    # Raw per-agent outputs (populated by the arcade-local pipeline so the
    # dashboard can render predicted vs actual, CI bounds, cliff percentiles
    # and every other model detail that used to live only in the CLI panel).
    per_agent: PerAgentOutputsDTO | None = None
    # DecisionMemory prompt block as it was BEFORE this lap's pipeline call
    # (None on lap 1 or the no-llm profile, where no prompt is built at all)
    # and whether this lap's action differs from the previous one. The block
    # never shows up in `reasoning` even when it changes the call, so the
    # dashboard renders it directly rather than relying on the LLM to narrate
    # its own continuity.
    memory_block: str | None = None
    plan_changed: bool = False


# --- Shared state ---------------------------------------------------------


@dataclass
class StrategyState:
    """Mutable handoff between connector thread and render thread.

    Access is guarded by `_lock`; the render side takes the lock only for a
    fraction of a frame to snapshot `latest` + `error`."""

    start: StartEventDTO | None = None
    latest: LapDecisionDTO | None = None
    history: list[LapDecisionDTO] = field(default_factory=list)
    error: str | None = None
    finished: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> tuple[LapDecisionDTO | None, str | None, bool]:
        with self._lock:
            return self.latest, self.error, self.finished

    def snapshot_dict(self, history_tail: int = 30) -> dict:
        """JSON-serialisable view consumed by the dashboard over the TCP stream.

        ``latest`` carries the full ``LapDecisionDTO`` including the raw
        per-agent outputs so the dashboard can render predicted-vs-actual,
        cliff percentiles, overtake/SC probabilities, etc. ``history_tail``
        strips ``per_agent`` from each past decision to keep the wire
        payload small (charts accumulate their own per-agent history from
        successive ``latest`` updates; the backend does not need to
        replay 30 copies of the dataclass on every broadcast).
        """
        with self._lock:
            return {
                "start": asdict(self.start) if self.start is not None else None,
                "latest": asdict(self.latest) if self.latest is not None else None,
                "history_tail": [
                    {k: v for k, v in asdict(d).items() if k != "per_agent"}
                    for d in self.history[-history_tail:]
                ],
                "error": self.error,
                "finished": self.finished,
            }


# --- Connector ------------------------------------------------------------


class SimConnector(threading.Thread):
    """Arcade-local strategy driver.

    Owns a background thread that iterates the same ``RaceReplayEngine``
    the backend uses, builds a ``RaceState`` per lap, invokes
    ``run_strategy_pipeline`` (verbose wrapper that returns both the
    synthesised ``StrategyRecommendation`` and every raw sub-agent
    output), and pushes the merged decision into ``StrategyState`` so
    the arcade replay panel and the dashboard subprocess both get the
    full model telemetry without the arcade depending on the FastAPI
    backend at runtime.

    Class name kept for backwards-compatibility with ``F1ArcadeView``'s
    wiring (``self._strategy_connector = SimConnector(...)``); that call
    site does not need to know the driver is now local.
    """

    daemon = True

    def __init__(
        self,
        request: SimulateRequestDTO,
        state: StrategyState,
        backend_url: str = "",  # kept for backwards compat, unused
        current_lap_provider: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(name="SimConnector")
        self._request = request
        self._state = state
        self._stop_event = threading.Event()
        # Optional callback the arcade view supplies to publish the lap the
        # user is currently watching.  When set, the lap loop blocks until
        # arcade catches up before kicking the agents — so pausing the
        # replay also pauses the agentic flow.  When ``None`` (e.g. CLI
        # smoke tests) the loop runs as fast as the LLM allows, preserving
        # pre-existing behaviour.
        self._current_lap_provider = current_lap_provider
        # RadioPipelineRunner is materialised in _warmup_models after the
        # corpus is ensured on disk; ``None`` until then (graceful
        # degrade path — the agents run with empty radio_msgs if the
        # corpus cannot be loaded).
        self._radio_runner: Any = None
        # One Safety-Car tracker for the whole replay: a "SAFETY CAR DEPLOYED"
        # corpus message is announced once (at the deploy lap), so without this
        # the per-lap RCM window is empty on laps 8-10 of the same neutralisation
        # and the SC override drops mid-stint. Persists the state across laps and
        # re-asserts it in _build_race_state (NR-02, #305 → #398; same wiring the
        # CLI uses).
        from src.nlp.rcm_state import RaceControlStateTracker

        self._sc_tracker = RaceControlStateTracker()
        # One DecisionMemory for the whole replay, same lifetime and shape as the
        # SC tracker above: without it the Layer 3 prompt is stateless and
        # re-argues an unrelated case every lap instead of building on its own
        # previous call.
        from src.strategy.inference.decision_memory import DecisionMemory

        self._memory = DecisionMemory()

    def stop(self) -> None:
        self._stop_event.set()

    def _wait_for_arcade(self, target_lap: int, poll_interval_s: float = 0.2) -> bool:
        """Block until the arcade replay's current lap reaches ``target_lap``.

        Returns ``True`` when the wait succeeded (arcade caught up or no
        provider was wired) and ``False`` when ``stop()`` was called while
        we were waiting; the caller propagates that as a clean exit.
        Polls instead of using a condition variable because the arcade
        view's frame index is read from a different thread without a lock
        and we only need lap-level granularity, not frame-level.
        """
        if self._current_lap_provider is None:
            return True
        while self._current_lap_provider() < target_lap:
            if self._stop_event.wait(poll_interval_s):
                return False
        return True

    # Number of laps the agent loop is allowed to lag behind the arcade
    # before we consider the lap "stale" and skip the LLM call.  One lap
    # of buffer absorbs the natural drift between the ~5 s agent step and
    # the visual replay without making the dashboard miss the lap the
    # user is actively watching.
    _STALE_LAP_TOLERANCE: int = 1

    def _should_skip_stale(self, lap_num: int) -> bool:
        """True when arcade has seeked far enough ahead to drop this lap.

        Always ``False`` when no playback provider is wired: CLI / smoke
        tests must keep processing every lap end-to-end.
        """
        if self._current_lap_provider is None:
            return False
        return self._current_lap_provider() > lap_num + self._STALE_LAP_TOLERANCE

    @staticmethod
    def _lap_time_from_state(lap_state: dict[str, Any], fallback: float) -> float:
        """Pull the real lap time out of a skipped lap_state, falling back.

        Keeps the next agent call's ``prev_lap_time`` baseline accurate
        even when we skipped one or several intermediate laps; using the
        actual recorded lap time is more truthful than carrying the last
        predicted value forward.
        """
        driver = lap_state.get("driver") or {}
        return float(driver.get("lap_time_s") or fallback)

    @staticmethod
    def _lap_skip_reason(driver_st: dict[str, Any]) -> str | None:
        """Reason this lap must skip the strategy pipeline, or None when safe.

        Mirrors the two guards the CLI PMV applies before building a RaceState
        (scripts/run_simulation_cli.py L1551-1584):

        - DNF: ``RaceStateManager.get_driver_state`` returns an empty dict once
          the driver retires, so an empty ``driver`` state means the car is out.
        - Incomplete lap: FastF1 lands a NaN position / tyre_life / lap_time on
          some opening laps (RSM emits them as None). A None position would be
          coerced into a searchable number and a None lap_time into a physically
          impossible pace delta.

        Skipping keeps the arcade from fabricating a P10 MEDIUM car for a
        retired driver and invoking the pipeline once per remaining lap (#441).
        """
        if not driver_st:
            return "DNF"
        if driver_st.get("position") is None:
            return "incomplete lap (position is None)"
        if driver_st.get("tyre_life") is None:
            return "incomplete lap (tyre_life is None)"
        if driver_st.get("lap_time_s") is None:
            return "incomplete lap (lap_time is None)"
        return None

    def run(self) -> None:
        """Drive the local strategy loop and capture fatal errors.

        Top-level ``try`` turns any exception escaping ``_drive_pipeline``
        into a ``state.error`` message instead of killing the thread
        silently (the replay panel / dashboard need to surface the
        failure to the user)."""
        try:
            self._drive_pipeline()
        except Exception as exc:
            logger.exception("Arcade strategy driver crashed: %s", exc)
            with self._state._lock:
                self._state.error = f"driver error: {exc}"

    def _drive_pipeline(self) -> None:
        """One-shot replay loop: load data, emit start, iterate laps."""
        os.environ["F1_LLM_PROVIDER"] = self._request.provider

        laps_df = self._load_laps_df(self._request.year)
        if laps_df is None:
            with self._state._lock:
                self._state.error = f"laps_featured_{self._request.year}.parquet missing"
            return

        race_dir = self._resolve_race_dir(self._request.year, self._request.gp)
        if not race_dir.exists():
            with self._state._lock:
                self._state.error = f"race dir missing: {race_dir.name}"
            return

        from src.simulation.replay_engine import RaceReplayEngine

        engine = RaceReplayEngine(
            race_dir,
            driver_code=self._request.driver,
            team=self._request.team,
            interval_seconds=self._request.interval_s,
        )

        lap_start = self._request.lap_range[0] if self._request.lap_range else 1
        lap_end = self._request.lap_range[1] if self._request.lap_range else engine.total_laps
        self._emit_start(lap_start, lap_end, engine.total_laps)
        self._warmup_models()
        self._load_radio_corpus(laps_df)

        prev_lap_time = 0.0
        for lap_state in engine.replay():
            if self._stop_event.is_set():
                return
            lap_num = int(lap_state.get("lap_number") or 0)
            if lap_num < lap_start or lap_num > lap_end:
                continue
            # Block until the arcade replay reaches this lap.  Without this
            # gate the agent thread storms ahead of the visual replay (one
            # LLM call per lap, ~5-10 s), so pausing arcade in V2 used to
            # leave the dashboard rendering recommendations for V3, V4, …
            # ``_wait_for_arcade`` is a no-op when no provider is wired
            # (CLI / smoke tests preserve the as-fast-as-possible loop).
            if not self._wait_for_arcade(lap_num):
                return
            # Skip stale laps when the user has seeked the arcade well ahead
            # of where the agent loop sits.  Without this, a fast-forward
            # from V2 to V20 would burn ~17 LLM calls for laps the user
            # never sees again.  We still keep ``prev_lap_time`` accurate
            # so the next processed lap sees a sensible baseline.
            if self._should_skip_stale(lap_num):
                prev_lap_time = self._lap_time_from_state(lap_state, prev_lap_time)
                continue
            # DNF + incomplete-lap guard (mirrors the CLI, run_simulation_cli.py
            # L1551-1584). Without it _build_race_state defaults an empty driver
            # state to a P10 MEDIUM car and the loop keeps invoking the pipeline
            # for a car that retired, and a None position / lap_time reaches
            # RaceState (#441).
            skip_reason = self._lap_skip_reason(lap_state.get("driver", {}))
            if skip_reason is not None:
                logger.info("Lap %d skipped (%s): no strategy pipeline call", lap_num, skip_reason)
                continue
            try:
                prev_lap_time = self._step_once(laps_df, lap_state, prev_lap_time)
            except Exception as exc:
                logger.exception("Lap %d pipeline failed: %s", lap_num, exc)
                with self._state._lock:
                    self._state.error = f"lap {lap_num}: {exc}"

        with self._state._lock:
            self._state.finished = True
        logger.info("Arcade strategy driver finished (lap_end=%d)", lap_end)

    def _step_once(
        self,
        laps_df: pd.DataFrame,
        lap_state: dict[str, Any],
        prev_lap_time: float,
    ) -> float:
        """Process one lap end-to-end and return the lap_time to carry forward."""
        from src.arcade.strategy_pipeline import run_strategy_pipeline

        race_state = self._build_race_state(lap_state, prev_lap_time)
        # Captured BEFORE the pipeline call: this is the block as the model
        # actually saw it. `record()` below mutates the accumulator, so reading
        # it after would show this lap's own decision back to the dashboard
        # instead of the plan the model was given.
        memory_block = self._memory.block()
        rec, agent_outputs = run_strategy_pipeline(
            race_state, laps_df, lap_state, memory=self._memory
        )
        self._memory.record(race_state.lap, rec)
        plan_changed = self._memory.last_call_changed()
        lap_time_s = lap_state.get("driver", {}).get("lap_time_s")
        decision = _build_decision(
            rec, race_state, lap_time_s, agent_outputs, memory_block, plan_changed
        )
        with self._state._lock:
            self._state.latest = decision
            self._state.history.append(decision)
            self._state.error = None
        return float(lap_time_s) if lap_time_s else prev_lap_time

    def _warmup_models(self) -> None:
        """Force-load the strategy pipeline and every sub-agent singleton
        before the first lap so the user sees a clear "warming up" banner
        in the dashboard instead of an empty card grid for 20 seconds.

        - Importing ``src.arcade.strategy_pipeline`` triggers the chain of
          ``src.agents.*`` imports (xgboost, torch, transformers, etc.).
        - Calling ``_get_default_*_agent()`` on the four agents that expose
          a singleton accessor materialises their model weights on GPU.
        - Radio / RAG have no simple accessor and warm up naturally on the
          first lap; still relatively cheap.
        - The warmup runs after ``_emit_start`` so the dashboard already
          has the StartEventDTO and can render the header immediately."""
        with self._state._lock:
            self._state.error = "Warming up strategy models…"
        try:
            import src.arcade.strategy_pipeline  # noqa: F401 — import-for-side-effects
            from src.agents.pace_agent import _get_default_pace_agent
            from src.agents.pit_strategy_agent import _get_default_pit_agent
            from src.agents.race_situation_agent import _get_default_situation_agent
            from src.agents.tire_agent import _get_default_tire_agent

            _get_default_pace_agent()
            _get_default_tire_agent()
            _get_default_situation_agent()
            _get_default_pit_agent()
            logger.info("Strategy models warmed up")
        except Exception as exc:
            logger.warning("Warmup failed: %s — first lap will bear the cost", exc)
        finally:
            with self._state._lock:
                self._state.error = None

    def _load_radio_corpus(self, laps_df: pd.DataFrame) -> None:
        """Mount the OpenF1 radio + RCM corpus for this GP and wire the NLP
        pipeline the CLI uses.

        Mirrors ``scripts/run_simulation_cli.py`` L1600-1638: ensure the
        per-GP audio tree is on disk (lazy download on first run),
        build a ``RadioPipelineRunner`` that the per-lap loop queries
        for ``(radios, rcms)`` dicts shaped as the radio agent expects.
        Without this step the Radio card / alerts feed stay silent for
        the whole race and N28/N30 never fire on radio triggers.

        Graceful degrade: on any failure (corpus missing, Whisper
        unavailable, transcript cache corrupt) we log a warning and
        leave ``_radio_runner`` as ``None``; ``_build_race_state`` then
        falls back to empty ``radio_msgs`` / ``rcm_events`` just like
        before.
        """
        try:
            from src.f1_strat_manager.data_cache import (
                ensure_radio_corpus,
                get_data_root,
            )
            from src.nlp.radio_runner import RadioPipelineRunner
        except Exception as exc:
            logger.warning(
                "Radio corpus deps unavailable (%s) — radio agent will see no events", exc
            )
            return

        with self._state._lock:
            self._state.error = "Loading radio corpus…"
        try:
            ensure_radio_corpus(self._request.year, self._request.gp)
            self._radio_runner = RadioPipelineRunner(
                year=self._request.year,
                gp_name=self._request.gp,
                laps_df=laps_df,
                data_root=get_data_root(),
                whisper_model_name="turbo",
                eager_transcribe=True,
            )
            logger.info(
                "Radio corpus loaded: %d radios + %d rcms (%s)",
                self._radio_runner.total_radios(),
                self._radio_runner.total_rcms(),
                self._radio_runner.slug,
            )
        except Exception as exc:
            logger.warning(
                "Radio corpus load failed (%s: %s) — falling back to empty radios",
                exc.__class__.__name__,
                exc,
            )
            self._radio_runner = None
        finally:
            with self._state._lock:
                self._state.error = None

    def _emit_start(self, lap_start: int, lap_end: int, total_laps: int) -> None:
        with self._state._lock:
            self._state.start = StartEventDTO(
                gp=self._request.gp,
                year=self._request.year,
                driver=self._request.driver,
                team=self._request.team,
                lap_start=lap_start,
                lap_end=lap_end,
                total_laps=total_laps,
                no_llm=self._request.no_llm,
                provider=self._request.provider,
            )
            self._state.error = None
        logger.info(
            "Arcade strategy driver started: %s %d %s (laps %d-%d)",
            self._request.gp,
            self._request.year,
            self._request.driver,
            lap_start,
            lap_end,
        )

    def _load_laps_df(self, year: int) -> pd.DataFrame | None:
        """Load the featured laps for `year`, augmented, for the agents to consume.

        Never `read_parquet` this file raw. N04 drops `Time` and no published featured
        parquet carries a `Time_s`, which is the column N11 trains its overtake gap on,
        so a direct read silently degrades that gap to a lap-time delta: at Lusail the
        model reads a 0.49 s mean gap where the truth is 3.29 s, and 90% of pairs look
        like they are in the DRS window when 20% are.

        The backend's loader had this fix and the CLI did not; the arcade did not either.
        `augment_featured_laps` is the one place that owns it now.
        """
        path = REPO_ROOT / "data" / "processed" / f"laps_featured_{year}.parquet"
        if not path.exists():
            logger.error("Featured laps parquet missing: %s", path)
            return None
        return augment_featured_laps(pd.read_parquet(path), year)

    @staticmethod
    def _resolve_race_dir(year: int, gp: str):
        """Map a friendly GP name (``Australia``) to the on-disk folder
        (``Melbourne``).

        The arcade menu / CLI propagate the country-style labels in
        ``GP_NAMES``, but the race data folders under ``data/raw/<year>/``
        follow the FastF1 Location convention. ``GP_TO_LOCATION`` is the
        single translation table; falls back to the raw name when already
        a Location so ``--gp Melbourne`` shortcuts keep working. Also
        tries the underscore variant because FastF1 emits ``Marina Bay``
        / ``São Paulo`` with spaces but the raw folders use underscores."""
        base = REPO_ROOT / "data" / "raw" / str(year)
        folder = GP_TO_LOCATION.get(gp, gp)
        candidate = base / folder
        if candidate.exists():
            return candidate
        alt = base / folder.replace(" ", "_")
        if alt.exists():
            return alt
        return candidate  # report the primary miss so error messaging stays clean

    def _build_race_state(self, lap_state: dict[str, Any], prev_lap_time: float):
        """Delegate to the canonical builder, keeping only the arcade-side inputs.

        The lap_state -> RaceState mapping lives in
        ``src.agents.race_state_builder.build_race_state`` (#784): one
        implementation shared by all three surfaces — this arcade, the CLI, and
        the telemetry backend, which reaches it through a re-export shim (#786)
        rather than keeping the copy it used to carry. So the defaults the
        models receive no longer drift per surface.
        The #465 position guard, the gap fallback rationale and
        the #750 pace-delta axis are all documented there now. What stays here
        is exactly what needs arcade instance state: sourcing ``radio_msgs`` /
        ``rcm_events`` from the ``RadioPipelineRunner`` corpus so the Radio
        agent sees the real OpenF1 team messages (same as the CLI; empty lists
        when the corpus could not load), and the stateful Safety-Car
        re-injection. Both are passed to the builder as parameters.

        ``prev_lap_time`` is accepted but no longer read here: ``pace_delta_s``
        used to be computed from it and that was the wrong axis (#750, now
        enforced by the canonical builder). Left in the signature rather than
        removed, since the caller's per-lap bookkeeping that produces it
        (``_step_once`` / ``_lap_time_from_state``) serves no other purpose
        today and dropping it is a small cleanup outside this fix's scope.
        """
        from src.agents.race_state_builder import build_race_state

        lap_num = int(lap_state.get("lap_number", 1) or 1)
        radio_msgs: list[dict] = []
        rcm_events: list[dict] = []
        if self._radio_runner is not None:
            try:
                radio_msgs, rcm_events = self._radio_runner.radios_for_lap(lap_num)
            except (KeyError, ValueError, TypeError) as exc:
                # radios_for_lap does plain pandas row indexing + int/str
                # casts (see RadioPipelineRunner._radio_row_to_dict /
                # _rcm_row_to_dict) — a malformed lap_number or a missing
                # column are the only realistic failure modes here.
                logger.debug("radios_for_lap(%d) failed: %s", lap_num, exc)

        # Re-assert an active Safety Car on the laps whose RCM window carries no
        # fresh deploy message. Ingest only the laps actually processed here; a
        # release landing on a skipped (stale) lap is bounded by the tracker's
        # safety valve rather than pinning the override (NR-02, #398 — mirrors
        # the CLI wiring in run_simulation_cli).
        self._sc_tracker.ingest(lap_num, rcm_events)
        if self._sc_tracker.should_inject(lap_num):
            rcm_events = list(rcm_events) + [self._sc_tracker.synthetic_event()]

        return build_race_state(
            lap_state,
            risk_tolerance=self._request.risk_tolerance,
            radio_msgs=radio_msgs,
            rcm_events=rcm_events,
        )


# --- Helpers exposed to the panel ----------------------------------------


_ACTION_STYLE: dict[str, tuple[tuple[int, int, int], str]] = {
    "STAY_OUT": (SUCCESS, "STAY OUT"),
    "PIT_NOW": (DANGER, "PIT NOW"),
    "UNDERCUT": (WARNING, "UNDERCUT"),
    "OVERCUT": (WARNING, "OVERCUT"),
    "ALERT": (INFO, "ALERT"),
    "DNF": (TEXT_SECONDARY, "DNF"),
    "ERROR": (DANGER, "ERROR"),
}


def classify_action(action: str) -> tuple[tuple[int, int, int], str]:
    """Map a raw action string to (colour, display-label) for the badge.

    ``action`` was typed as a plain ``str`` but neither this function's original
    body nor its theme.py twin (deduplicated 2026-08-01) actually guarded a
    ``None`` -- both called ``.upper()`` on it unguarded as the dict lookup key
    and crashed identically. No live caller currently passes ``None``
    (`orchestrator_card.py` sanitises with `str(... or "--")` first), but the
    signature invites it, so this is fixed now rather than left as a latent
    crash the next caller could trigger.
    """
    action = (action or "--").upper()
    return _ACTION_STYLE.get(action, (ACCENT, action))


# --- Private helpers -----------------------------------------------------


def _normalize_scores(raw: Any) -> dict[str, float]:
    """Flatten ``{"stay_out": {"score": 0.7}, ...}`` or ``{"STAY_OUT": 0.7}``.

    MC simulation returns the nested form; the orchestrator re-attaches it
    to ``StrategyRecommendation.scenario_scores`` without flattening. The
    dashboard wants a simple ``{UPPER: float}`` dict, so normalise here."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        key = str(k).upper()
        if isinstance(v, dict):
            v = v.get("score")
        # A candidate the projection engine declined to score arrives as None.
        # It used to become 0.0 here, which painted a full-height bar at the
        # zero line for a strategy that was never on the table — a numeric
        # sentinel by accident. Dropping the key instead leaves the dashboard
        # with three bars, and an absent bar cannot be misread as a score.
        if v is None:
            continue
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _dump_dataclass(obj: Any) -> dict[str, Any] | None:
    """Convert an agent-output dataclass to a plain dict, tolerating ``None``.

    ``dataclasses.asdict`` recurses into nested dataclasses, which is what
    we want for the per-agent serialisation: ``PaceOutput``, ``TireOutput``,
    etc. turn into JSON-ready dicts without hand-written field mappings."""
    if obj is None:
        return None
    from dataclasses import asdict as _asdict
    from dataclasses import is_dataclass

    if is_dataclass(obj):
        return _asdict(obj)
    return obj if isinstance(obj, dict) else None


# N27 computes these two from the RCM stream for the lap it was asked about.
# The tick already carries `track_status_label`, decoded from FastF1 TrackStatus
# for the lap on screen by the arcade's own priority rule, and BOTH windows
# render that one. Publishing the agent's pair as well would put two sources for
# one fact on one desk, free to disagree, which is the shape this repo pays for
# most - so the wire carries the decoded signal and the agent keeps its own
# copy for its own reasoning (#1043).
#
# They are filtered HERE rather than deleted from `RaceSituationOutput`, because
# the pair is read across `src/agents/`: N27's own `sc_active` property, N28's
# routing branches, and the orchestrator's safety-car handling including the
# rail that nulls `target_lap_time_s` under Art. 55.7.
_SITUATION_FIELDS_OFF_THE_WIRE = ("sc_currently_active", "vsc_active")


def _situation_for_the_wire(situation_out: Any) -> dict[str, Any] | None:
    """Dump N27's output minus the neutralisation booleans the tick decodes itself."""
    dumped = _dump_dataclass(situation_out)
    if dumped is None:
        return None
    return {k: v for k, v in dumped.items() if k not in _SITUATION_FIELDS_OFF_THE_WIRE}


def _build_per_agent(agent_outputs: dict[str, Any]) -> PerAgentOutputsDTO:
    """Package the pipeline's intermediate outputs into a DTO the
    ``StrategyState`` can broadcast to the dashboard."""
    return PerAgentOutputsDTO(
        pace=_dump_dataclass(agent_outputs.get("pace_out")),
        tire=_dump_dataclass(agent_outputs.get("tire_out")),
        situation=_situation_for_the_wire(agent_outputs.get("situation_out")),
        radio=_dump_dataclass(agent_outputs.get("radio_out")),
        pit=_dump_dataclass(agent_outputs.get("pit_out")),
        regulation_context=str(agent_outputs.get("regulation_context") or ""),
        rag=agent_outputs.get("rag"),
        active=list(agent_outputs.get("active") or []),
    )


def _contingency_dicts(contingencies: Any) -> list[dict[str, Any]]:
    """Flatten the orchestrator's ``Contingency`` models into wire dicts.

    Four keys each: ``trigger``, ``switch_to``, ``priority``, ``rationale``.
    Named rather than inlined because the orchestrator can hand this back as
    Pydantic models OR, on the dict-shaped result path, as dicts already, and a
    comprehension that assumed one of the two would produce a payload of empty
    objects on the other without failing anywhere.
    """
    packed: list[dict[str, Any]] = []
    for item in contingencies or []:
        if isinstance(item, dict):
            source = item
        else:
            source = {
                key: getattr(item, key, None)
                for key in ("trigger", "switch_to", "priority", "rationale")
            }
        packed.append(
            {
                key: str(source.get(key) or "")
                for key in ("trigger", "switch_to", "priority", "rationale")
            }
        )
    return packed


def _build_decision(
    rec: Any,
    race_state: Any,
    lap_time_s: float | None,
    agent_outputs: dict[str, Any],
    memory_block: str | None = None,
    plan_changed: bool = False,
) -> LapDecisionDTO:
    """Merge the synthesised ``StrategyRecommendation`` + raw agent outputs
    into the DTO consumed by ``StrategyState.history`` / the dashboard.

    ``contingencies`` is flattened to plain dicts because the wire is JSON and
    ``asdict`` walks this DTO: the orchestrator holds them as ``Contingency``
    models, which would not survive the trip typed anyway.

    ``memory_block`` / ``plan_changed`` come from ``DecisionMemory`` and are
    just carried onto the DTO here; the caller (``_step_once``) is the one
    that decides when each is captured relative to ``record()``.
    """
    return LapDecisionDTO(
        lap_number=race_state.lap,
        compound=str(race_state.compound),
        tyre_life=int(race_state.tyre_life),
        position=int(race_state.position),
        lap_time_s=float(lap_time_s) if lap_time_s else None,
        # The TCP wire contract pins this slot as float
        # (tests/surfaces/test_arcade_wire_contract.py) and #857 owns the wire's
        # future. 0.0 is the wire's existing absent-marker, kept AT THE BOUNDARY
        # on purpose - not a producer writing 0.0 back into the strategy path,
        # which is the defect #878 is about. Remove with the wire redesign.
        gap_ahead_s=(float(race_state.gap_ahead_s) if race_state.gap_ahead_s is not None else 0.0),
        action=str(getattr(rec, "action", "ERROR")),
        confidence=float(getattr(rec, "confidence", 0.0) or 0.0),
        reasoning=str(getattr(rec, "reasoning", "")),
        scenario_scores=_normalize_scores(getattr(rec, "scenario_scores", {})),
        pace_mode=getattr(rec, "pace_mode", None),
        risk_posture=getattr(rec, "risk_posture", None),
        pit_lap_target=getattr(rec, "pit_lap_target", None),
        compound_next=getattr(rec, "compound_next", None),
        undercut_target=getattr(rec, "undercut_target", None),
        contingencies=_contingency_dicts(getattr(rec, "contingencies", None)),
        key_risks=[str(risk) for risk in (getattr(rec, "key_risks", None) or [])],
        guardrail_reason=None,
        per_agent=_build_per_agent(agent_outputs),
        memory_block=memory_block,
        plan_changed=plan_changed,
    )
