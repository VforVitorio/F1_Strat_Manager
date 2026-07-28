"""N31 orchestrator card: the top-left decision panel.

Reads one ``latest`` LapDecision dict per update and renders:

- Action badge (big, coloured by ``classify_action``).
- Confidence bar (red → amber → green based on value).
- Plan strip: pace_mode · risk_posture chips + pit target + undercut
  target, matching the CLI's execution-plan table (``06_cli_inference_panel.md``
  §3) so the dashboard reads as a visual extension of the CLI view.
- Guardrail line: DANGER-coloured when ``guardrail_reason`` is set so
  the user sees *why* the orchestrator overrode the MC winner.

Idle state (``latest is None``) keeps the layout intact with ``"--"``
placeholders so the window does not reflow when the first decision
arrives.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
)

from src.arcade.dashboard.theme import (
    ACCENT,
    DANGER,
    SUCCESS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    WARNING,
    classify_action,
    compound_pill_html,
    hex_str,
)

# --- Regime colour maps (mirrors CLI §3 convention) ----------------------

_PACE_COLOURS: dict[str, tuple[int, int, int]] = {
    "PUSH": DANGER,
    "NEUTRAL": TEXT_SECONDARY,
    "MANAGE": WARNING,
    "LIFT_AND_COAST": WARNING,
}

_RISK_COLOURS: dict[str, tuple[int, int, int]] = {
    "AGGRESSIVE": DANGER,
    "BALANCED": TEXT_SECONDARY,
    "NEUTRAL": TEXT_SECONDARY,
    "CONSERVATIVE": WARNING,
    "DEFENSIVE": WARNING,
}


def _confidence_colour(conf: float) -> tuple[int, int, int]:
    """Traffic-light colour for the confidence bar."""
    if conf >= 0.66:
        return SUCCESS
    if conf >= 0.33:
        return WARNING
    return DANGER


class OrchestratorCard(QFrame):
    """Top-left card showing the synthesised N31 decision."""

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("card", True)
        # 170, not 220. The rows below add up to roughly 165px (a 70px badge, the
        # chip row, the plan line, the guardrail line, plus margins and spacing), so
        # a 220 floor reserved around 50px of dead space under the plan line and took
        # it from the only widget in this column that stretches: the reasoning tabs.
        # With the decision-memory block appended on a changed lap, those 50px were
        # the difference between the block fitting and being scrolled below the fold.
        # The vertical policy stays Fixed, so a long wrapped guardrail line still
        # grows the card past this floor. Lowering a MINIMUM cannot clip content.
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        # --- Row 1: action badge + confidence -----------------------------
        top_row = QHBoxLayout()
        top_row.setSpacing(14)

        self._badge = QLabel("--")
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setMinimumHeight(70)
        self._badge.setMinimumWidth(200)
        self._badge.setStyleSheet(self._badge_style(ACCENT))

        conf_col = QVBoxLayout()
        conf_col.setSpacing(4)
        self._conf_label = QLabel("Confidence: --")
        self._conf_label.setStyleSheet(f"color: {hex_str(TEXT_SECONDARY)}; font-size: 11px;")
        self._conf_bar = QLabel()
        self._conf_bar.setFixedHeight(14)
        self._conf_bar.setStyleSheet(self._bar_style(0.0, TEXT_TERTIARY))
        conf_col.addStretch()
        conf_col.addWidget(self._conf_label)
        conf_col.addWidget(self._conf_bar)
        conf_col.addStretch()

        top_row.addWidget(self._badge)
        top_row.addLayout(conf_col, 1)
        outer.addLayout(top_row)

        # --- Row 2: pace · risk chips -------------------------------------
        chip_row = QHBoxLayout()
        chip_row.setSpacing(8)
        self._pace_chip = self._make_chip("Pace: --")
        self._risk_chip = self._make_chip("Risk: --")
        chip_row.addWidget(self._pace_chip)
        chip_row.addWidget(self._risk_chip)
        chip_row.addStretch()
        outer.addLayout(chip_row)

        # --- Row 3: plan line ---------------------------------------------
        # Rich text so the ``Next: <compound>`` chunk can render a Pirelli
        # compound pill inline next to the rest of the plan.
        self._plan = QLabel("Pit: — · Next: — · UCUT: —")
        self._plan.setTextFormat(Qt.RichText)
        self._plan.setStyleSheet(f"color: {hex_str(TEXT_SECONDARY)}; font-size: 12px;")
        self._plan.setWordWrap(True)
        outer.addWidget(self._plan)

        # --- Row 4: guardrail line ----------------------------------------
        self._guardrail = QLabel("")
        self._guardrail.setStyleSheet(
            f"color: {hex_str(DANGER)}; font-size: 11px; font-weight: 600;"
        )
        self._guardrail.setWordWrap(True)
        self._guardrail.setVisible(False)
        outer.addWidget(self._guardrail)

        outer.addStretch()

    # --- Public update slot ----------------------------------------------

    def update_from(self, latest: dict[str, Any] | None) -> None:
        """Paint the card from a ``LapDecisionDTO`` dict. ``None`` clears."""
        if not latest:
            self._render_idle()
            return

        action = str(latest.get("action") or "--")
        conf = float(latest.get("confidence") or 0.0)
        pace_mode = latest.get("pace_mode")
        risk_posture = latest.get("risk_posture")
        pit_target = latest.get("pit_lap_target")
        compound_next = latest.get("compound_next")
        undercut_target = latest.get("undercut_target")
        guardrail = latest.get("guardrail_reason")

        badge_color, badge_label = classify_action(action)
        self._badge.setText(badge_label)
        self._badge.setStyleSheet(self._badge_style(badge_color))

        self._conf_label.setText(f"Confidence: {conf * 100:.0f}%")
        self._conf_bar.setStyleSheet(self._bar_style(conf, _confidence_colour(conf)))

        self._pace_chip.setText(f"Pace: {pace_mode or '--'}")
        self._pace_chip.setStyleSheet(
            self._chip_style(_PACE_COLOURS.get(str(pace_mode or "").upper(), TEXT_TERTIARY))
        )
        self._risk_chip.setText(f"Risk: {risk_posture or '--'}")
        self._risk_chip.setStyleSheet(
            self._chip_style(_RISK_COLOURS.get(str(risk_posture or "").upper(), TEXT_TERTIARY))
        )

        # Graceful empty state: on STAY_OUT with no tactical plan, render a
        # single "stint continues" line instead of three "--" chips — the
        # orchestrator intentionally leaves pit/next/UCUT blank when there
        # is no committed pit plan, so "--" on every field reads noisy.
        if not any((pit_target, compound_next, undercut_target)):
            if action.upper() == "STAY_OUT":
                self._plan.setText("stint continues · no pit window yet")
            else:
                self._plan.setText("Pit plan pending")
        else:
            plan_bits = []
            plan_bits.append(f"Pit: L{pit_target}" if pit_target else "Pit: —")
            plan_bits.append(
                f"Next: {compound_pill_html(compound_next)}" if compound_next else "Next: —"
            )
            plan_bits.append(f"UCUT: {undercut_target}" if undercut_target else "UCUT: —")
            self._plan.setText(" · ".join(plan_bits))

        if guardrail:
            self._guardrail.setText(f"⚠ Guardrail: {guardrail}")
            self._guardrail.setVisible(True)
        else:
            self._guardrail.setVisible(False)

    # --- Rendering helpers -----------------------------------------------

    def _render_idle(self) -> None:
        self._badge.setText("--")
        self._badge.setStyleSheet(self._badge_style(TEXT_TERTIARY))
        self._conf_label.setText("Confidence: --")
        self._conf_bar.setStyleSheet(self._bar_style(0.0, TEXT_TERTIARY))
        self._pace_chip.setText("Pace: --")
        self._pace_chip.setStyleSheet(self._chip_style(TEXT_TERTIARY))
        self._risk_chip.setText("Risk: --")
        self._risk_chip.setStyleSheet(self._chip_style(TEXT_TERTIARY))
        self._plan.setText("Pit: -- · Next: -- · UCUT: --")
        self._guardrail.setVisible(False)

    @staticmethod
    def _badge_style(rgb: tuple[int, int, int]) -> str:
        return (
            f"background-color: {hex_str(rgb)}; color: {hex_str(TEXT_PRIMARY)}; "
            f"font-size: 26px; font-weight: 800; letter-spacing: 1px; "
            f"border-radius: 10px; padding: 14px 18px;"
        )

    @staticmethod
    def _bar_style(fraction: float, rgb: tuple[int, int, int]) -> str:
        # Simulate a progress bar with a linear-gradient between filled/empty.
        # stop colours: filled up to ``fraction``, transparent-ish after.
        pct = max(0.0, min(1.0, float(fraction)))
        # Clamp to 2 decimals for nicer CSS.
        cut = round(pct, 3)
        empty_col = hex_str((40, 40, 52))
        return (
            "QLabel { "
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            f"stop:0 {hex_str(rgb)}, stop:{cut} {hex_str(rgb)}, "
            f"stop:{cut + 0.001 if cut < 1 else 1} {empty_col}, "
            f"stop:1 {empty_col}); "
            "border-radius: 6px; "
            "}"
        )

    @staticmethod
    def _chip_style(rgb: tuple[int, int, int]) -> str:
        return (
            f"color: {hex_str(rgb)}; font-weight: 700; font-size: 12px; "
            f"padding: 4px 10px; border: 1px solid {hex_str(rgb)}; "
            f"border-radius: 10px;"
        )

    @staticmethod
    def _make_chip(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(OrchestratorCard._chip_style(TEXT_TERTIARY))
        return lbl
