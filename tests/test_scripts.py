"""Public scripts select authored trajectories without falling back to another task."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from marketcanvas_env.task_data import replay_trajectory, trajectory_paths

ROOT = Path(__file__).resolve().parents[1]


def test_selected_trajectory_exports_the_same_result_as_direct_replay(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_examples.py",
            "--task",
            "webinar",
            "--trajectory",
            "missing_cta",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    source = next(p for p in trajectory_paths("webinar") if p.stem == "missing_cta")
    assert list(tmp_path.glob("*/*.json")) == [tmp_path / "webinar/missing_cta.json"]
    assert json.loads((tmp_path / "webinar/missing_cta.json").read_text()) == replay_trajectory(
        source
    )
    assert "webinar/missing_cta" in completed.stdout
    assert "well_done" not in completed.stdout


@pytest.mark.parametrize("script", ["run_examples", "mcp_client"])
@pytest.mark.parametrize(
    "args", [["--task", "unknown"], ["--task", "webinar", "--trajectory", "unknown"]]
)
def test_bad_selection_fails_before_replay(script, args):
    completed = subprocess.run(
        [sys.executable, f"scripts/{script}.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert not completed.stdout


def test_trajectory_filter_requires_a_task():
    completed = subprocess.run(
        [sys.executable, "scripts/run_examples.py", "--trajectory", "well_done"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--trajectory requires --task" in completed.stderr
