"""The recruiter's entry point runs from another directory and exports matching results."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from marketcanvas_env.rendering import render_rgb
from marketcanvas_env.reward import compute_reward_breakdown

DEMO = Path(__file__).resolve().parents[1] / "demo.py"


def run_demo(cwd, *args):
    return subprocess.run(
        [sys.executable, str(DEMO), "--json", *args], cwd=cwd, capture_output=True, text=True
    )


def test_default_demo_is_reproducible_and_reports_actual_terminal_reward(tmp_path):
    first, second = run_demo(tmp_path), run_demo(tmp_path)
    assert first.returncode == second.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    assert first.stderr == ""
    result = json.loads(first.stdout)
    state = result["state"]
    assert state["steps_taken"] == 4 and state["status"] == "finished"
    assert [step["reward"] for step in result["steps"]] == [0.0, 0.0, 0.0, 1.0]
    assert [step["terminated"] for step in result["steps"]] == [False, False, False, True]
    # The reported actions really reproduce the final state and episode return.
    from marketcanvas_env.env import MarketCanvasEnv

    env = MarketCanvasEnv()
    env.reset()
    for step in result["steps"]:
        transition = env.execute_action(step["action"])
        assert transition["reward"] == step["reward"]
    assert env.get_canvas_state() == state
    assert {e["type"] for e in state["elements"]} == {"text", "image", "shape"}
    assert result["terminated"] and not result["truncated"]
    assert result["reward_breakdown"] == compute_reward_breakdown(state, state["target"])
    assert result["final_reward"] == result["reward_breakdown"]["reward"] == 1.0
    assert not list(tmp_path.iterdir())


def test_default_yaml_and_png_match_reported_state(tmp_path):
    path = tmp_path / "images" / "sale.png"
    result = run_demo(tmp_path, "--png", str(path))
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)["state"]
    with Image.open(path) as image:
        assert image.size == (800, 600) and image.mode == "RGB"
        assert np.array_equal(np.array(image), render_rgb(state))
    before = path.read_bytes()
    assert run_demo(tmp_path, "--png", str(path)).returncode == 0
    assert path.read_bytes() == before


def test_unimplemented_prompt_reports_error_without_output_files(tmp_path):
    path = tmp_path / "unused" / "sale.png"
    result = run_demo(tmp_path, "--prompt", "Create an arbitrary design", "--png", str(path))
    assert result.returncode == 2
    assert "not_implemented" in result.stderr and "Traceback" not in result.stderr
    assert not result.stdout and not path.parent.exists()


def test_help_explains_input_and_export(tmp_path):
    result = run_demo(tmp_path, "--help")
    assert result.returncode == 0
    assert "--prompt" in result.stdout and "--png" in result.stdout
    assert "fixed example policy" in result.stdout


def test_export_failure_is_actionable(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("existing data")
    result = run_demo(tmp_path, "--png", str(blocker / "sale.png"))
    assert result.returncode == 1
    assert "demo:" in result.stderr and "Traceback" not in result.stderr
    assert not result.stdout and blocker.read_text() == "existing data"


def test_default_human_output_explains_episode_and_output_location(tmp_path):
    png = tmp_path / "sale.png"
    result = subprocess.run(
        [sys.executable, str(DEMO), "--png", str(png)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "Step 1: add_element text/headline | reward=0.000" in result.stdout
    assert "Step 4: finish | reward=1.000" in result.stdout
    assert "Final reward: 1.000 | terminated=True | truncated=False" in result.stdout
    assert "bounds=1.000" in result.stdout
    assert f"PNG: {png}" in result.stdout
    assert "Final state:" in result.stdout and "Reward breakdown" in result.stdout
    assert "professional visual quality" in result.stdout
