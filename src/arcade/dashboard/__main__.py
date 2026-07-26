"""Dashboard entry point: ``python -m src.arcade.dashboard``.

One Qt subprocess, two QMainWindows: ``MainWindow`` for the strategy
surface (orchestrator / agents / reasoning / charts) and
``TelemetryWindow`` for the live speed/throttle/brake/DRS view. Both
subscribe to the same arcade TCP stream independently, so the user can
drag them across monitors and close one without affecting the other.

The arcade spawns this subprocess once when strategy mode is enabled;
both windows appear together. Running the module directly
(``python -m src.arcade.dashboard``) also opens both for standalone
development against any arcade listening on the stream port.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from src.arcade.dashboard.telemetry_window import TelemetryWindow
from src.arcade.dashboard.theme import apply_dark_palette
from src.arcade.dashboard.window import MainWindow


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("F1 Strategy Dashboard")
    apply_dark_palette(app)

    strategy = MainWindow()
    telemetry = TelemetryWindow()
    # Offset the telemetry window so it does not spawn exactly on top
    # of the strategy window by default. Users can drag it anywhere.
    strategy.move(40, 40)
    strategy.show()
    telemetry.move(strategy.x() + strategy.width() + 20, strategy.y())
    telemetry.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
