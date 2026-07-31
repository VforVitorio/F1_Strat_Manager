"""Pin the three horizons #742 found undocumented, so nobody "unifies" them (#742).

``position_projection`` holds three predicates about the same underlying fact,
whether a rival's outstanding pit stop should cost them time, and each is
correct for a DIFFERENT horizon rather than being three attempts at one rule:

    rival_time_deltas -> rival.is_pitting                              (inside the window)
    _terminal_gaps     -> rival.stop_pending is True and not is_pitting (race end)
    rank_targets        -> rival.stop_pending is True                   (after both pit cycles)

The module docstring now states this rule once, prominently. This file is the
enforcement: it constructs rivals who owe a stop under different ``is_pitting``
states and asserts the three functions treat them differently, naming the
horizon each assertion pins. Any patch that makes the three predicates agree,
in either direction, must fail one of these tests.

Hermetic by construction: no model weights, no ``data/`` beyond the measured
JSON tables that ``position_projection`` already degrades gracefully without
(see its ``measured_tables`` docstring), so this suite runs in any checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.position_projection import (
    DriverPlan,
    ProjectionConfig,
    RivalState,
    _terminal_gaps,
    rank_targets,
    rival_time_deltas,
)

# A rival's outstanding obligation, held constant across every test in this
# file so the only thing that varies between OWES_NOT_TAKING and
# OWES_TAKING_NOW is ``is_pitting`` -- the one bit the three horizons disagree
# about how to use.
STOP_LOSS_S = 22.0
GAP_S = -1.0
RACING_LAPS = 5.0

# Owes the Art. 30.5(m) stop but is not in the pit lane THIS lap. This is the
# exact rival the #742 acceptance criteria names: charged at the terminal
# horizon, not charged inside the window.
OWES_NOT_TAKING = RivalState(
    "OWES_NOT_TAKING",
    gap_s=GAP_S,
    is_pitting=False,
    stop_pending=True,
    stop_loss_s=STOP_LOSS_S,
)

# Owes the same stop and IS taking it this lap. Distinguishes _terminal_gaps
# (which must NOT charge them again, rival_time_deltas already did) from
# rank_targets (which charges every known obligation regardless of timing).
OWES_TAKING_NOW = RivalState(
    "OWES_TAKING_NOW",
    gap_s=GAP_S,
    is_pitting=True,
    stop_pending=True,
    stop_loss_s=STOP_LOSS_S,
)

STAY_OUT = DriverPlan("STAY_OUT", stops_in_window=False)


def test_window_horizon_charges_only_the_rival_pitting_this_lap():
    """rival_time_deltas: is_pitting is the only fact this horizon trusts.

    A rival who owes a stop but is racing this lap costs themselves nothing
    inside the window; a rival serving the stop right now pays the full loss.
    Unifying this predicate with ``stop_pending`` (matching the other two
    horizons) would charge OWES_NOT_TAKING inside the window and break the
    first assertion below.
    """
    config = ProjectionConfig(racing_laps=RACING_LAPS)
    deltas = rival_time_deltas([OWES_NOT_TAKING, OWES_TAKING_NOW], config, draws=1)

    assert deltas[0, 0] == pytest.approx(0.0), (
        "window horizon: owes a stop but is not taking it now, so is_pitting=False "
        "keeps them uncharged inside the window"
    )
    assert deltas[0, 1] == pytest.approx(STOP_LOSS_S), (
        "window horizon: in the pit lane this lap, so is_pitting=True charges the stop right now"
    )


def test_terminal_horizon_charges_an_owed_stop_exactly_once():
    """_terminal_gaps: race-end residual, and never a stop already priced.

    OWES_NOT_TAKING is uncharged in the window (previous test), so the
    terminal residual must land here or the obligation vanishes entirely.
    OWES_TAKING_NOW was already charged inside the window, so the terminal
    residual must be zero for them, or the same stop is paid twice.
    """
    config = ProjectionConfig(racing_laps=RACING_LAPS, mandatory_stop_pending=False)
    rivals = [OWES_NOT_TAKING, OWES_TAKING_NOW]

    window_deltas = rival_time_deltas(rivals, config, draws=1)
    current_gaps = np.array([[rival.gap_s for rival in rivals]])
    projected_gaps = current_gaps + window_deltas  # our own delta is 0 here

    pit_loss_s = np.zeros(1)
    terminal_gaps = _terminal_gaps(rivals, STAY_OUT, projected_gaps, pit_loss_s, config)

    assert terminal_gaps[0, 0] - projected_gaps[0, 0] == pytest.approx(STOP_LOSS_S), (
        "terminal horizon: still owes the stop, so the residual lands even though "
        "nothing charged them in the window"
    )
    assert terminal_gaps[0, 1] - projected_gaps[0, 1] == pytest.approx(0.0), (
        "terminal horizon: already charged inside the window (rival_time_deltas), "
        "so the terminal residual excludes them rather than paying for the same "
        "stop twice"
    )


def test_post_cycle_horizon_charges_every_known_obligation_alike():
    """rank_targets: stop_pending alone, deliberately ignoring is_pitting.

    Once both pit cycles have played out, whether the stop happened on THIS
    lap stops mattering, so rank_targets charges OWES_NOT_TAKING and
    OWES_TAKING_NOW identically. A patch that adds the terminal horizon's
    ``and not is_pitting`` guard here (making the "selector" agree with the
    "scorer") would zero out OWES_TAKING_NOW's charge and break the second
    assertion below, exactly the unification #742 warns against.
    """
    config = ProjectionConfig(racing_laps=RACING_LAPS)
    ranked = {
        target.driver: target
        for target in rank_targets([OWES_NOT_TAKING, OWES_TAKING_NOW], config, our_pit_loss_s=0.0)
    }

    expected_projected_gap_s = GAP_S + STOP_LOSS_S
    assert ranked["OWES_NOT_TAKING"].projected_gap_s == pytest.approx(expected_projected_gap_s)
    assert ranked["OWES_TAKING_NOW"].projected_gap_s == pytest.approx(expected_projected_gap_s), (
        "post-cycle horizon: charged the SAME as OWES_NOT_TAKING even though "
        "is_pitting distinguishes them at the other two horizons; rank_targets' "
        "predicate is stop_pending alone, by design"
    )
