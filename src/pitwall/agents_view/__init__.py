"""The AGENTS window's content layer, computed in Python.

`host.get_agents_view` is the only caller. The package exists because the
alternative - reimplementing the Qt window's formatters in TypeScript -
would make "1:1" a claim to be checked by eye every sprint instead of a
property of the code.
"""

from src.pitwall.agents_view.builder import AGENTS_VIEW_VERSION, AgentsViewBuilder

__all__ = ["AGENTS_VIEW_VERSION", "AgentsViewBuilder"]
