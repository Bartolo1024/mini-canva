"""Private checkout-only raw-state matrix for historical reward regression tests.

Public tasks and replayable action trajectories live under ``data``. This single
test fixture preserves malformed states and older task variants for regression
tests and offline reward studies; it is not installed as application data.
"""

from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from marketcanvas_env.reward.tasks import TaskSpec


@cache
def _matrix() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "tests/fixtures/reward_regressions.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a regression matrix mapping")
    return data


def load_catalog(section: str) -> dict[str, tuple[dict[str, Any], TaskSpec]]:
    """Return fresh raw states and validated TaskSpecs in their archived order."""
    entries = _matrix().get(section)
    if not isinstance(entries, dict):
        raise ValueError(f"Unknown regression matrix section {section!r}")
    return {
        name: (deepcopy(entry["state"]), TaskSpec.model_validate(deepcopy(entry["task"])))
        for name, entry in entries.items()
    }
