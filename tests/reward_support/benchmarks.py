"""Compatibility helpers for the archival raw-state regression matrix.

Public tasks and action trajectories live under ``data``. These historical
snapshots preserve older tests and offline studies. Copy ``Benchmark.state``
before editing it; ``_element`` supports dynamically generated test cases.
"""

from dataclasses import dataclass
from typing import Any

from marketcanvas_env.reward.tasks import TaskSpec
from tests.reward_support.regression_data import load_catalog


@dataclass(frozen=True)
class Benchmark:
    """A human-readable task and one deliberately simple successful design."""

    task: TaskSpec
    state: dict[str, Any]


def _element(
    role: str,
    kind: str,
    x: float,
    y: float,
    width: float,
    height: float,
    content: str = "",
    color: str | None = None,
) -> dict[str, Any]:
    return {
        "id": role,
        "role": role,
        "type": kind,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "z_index": 2,
        "content": content,
        "color": color,
        "text_color": "#111111",
        "font_size": 24,
    }


BENCHMARKS = {
    name: Benchmark(task, state) for name, (state, task) in load_catalog("benchmarks").items()
}


def review_scenarios() -> dict[str, tuple[dict[str, Any], TaskSpec]]:
    """Load fresh canvases for the seven human reward-review comparisons."""
    return load_catalog("review")
