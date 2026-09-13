"""Rendering is an optional, observationally pure environment operation."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from PIL import Image

from marketcanvas_env.core import LifecycleError
from marketcanvas_env.env import MarketCanvasEnv


def test_gym_rgb_checker():
    # No spec is registered: explicitly check the supported render mode here.
    with pytest.warns(UserWarning) as warnings:
        check_env(MarketCanvasEnv(render_mode="rgb_array"))
    # Static, manually constructed environments have neither a playback FPS nor a registry spec.
    assert len(warnings) == 2
    assert "No render fps" in str(warnings[0].message)
    assert "alternative render modes" in str(warnings[1].message)


def test_modes_and_uninitialized_render(tmp_path):
    env = MarketCanvasEnv()
    assert env.render() is None
    with pytest.raises(LifecycleError, match="not_initialized"):
        env.save_png(tmp_path / "before.png")
    assert not (tmp_path / "before.png").exists()
    with pytest.raises(LifecycleError, match="not_initialized"):
        MarketCanvasEnv(render_mode="rgb_array").render()
    with pytest.raises(ValueError):
        MarketCanvasEnv(render_mode="human")
    env.reset()
    env.save_png(tmp_path / "empty.png")
    with Image.open(tmp_path / "empty.png") as image:
        assert image.mode == "RGB" and image.size == (800, 600)


def test_rendering_preserves_state_reward_rng_and_returned_arrays(tmp_path):
    env = MarketCanvasEnv(render_mode="rgb_array")
    env.reset(seed=7)
    env.execute_action(
        {
            "op": "add_element",
            "element": {
                "type": "text",
                "content": "Summer Sale",
                "role": "headline",
                "width": 400,
            },
        }
    )
    before = env.get_canvas_state()
    report = env.get_current_reward()
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    array = env.render()
    expected = array.copy()
    array[:] = 0
    assert np.array_equal(env.render(), expected)
    env.save_png(tmp_path / "first.png")
    env.save_png(tmp_path / "second.png")
    assert (tmp_path / "first.png").read_bytes() == (tmp_path / "second.png").read_bytes()
    assert env.get_canvas_state() == before
    assert env.get_current_reward() == report
    assert env.np_random.bit_generator.state == rng
    final = env.execute_action({"op": "finish"})
    assert final["reward"] == report["reward"]
    assert np.array_equal(env.render(), expected)
    assert env.get_canvas_state() == final["state"]


def test_steps_do_not_render(monkeypatch):
    import marketcanvas_env.rendering as rendering

    def fail(*args):
        raise AssertionError("rendering was called on a transition")

    monkeypatch.setattr(rendering, "render_rgb", fail)
    env = MarketCanvasEnv(render_mode="rgb_array")
    env.reset()
    env.execute_action({"op": "finish"})
    env.get_current_reward()


def test_example_cli_exports_public_trajectory_images_and_traces(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "examples.reward.render", "--all", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    images = sorted(tmp_path.glob("*/*.png"))
    assert len(images) == 15
    for path in images:
        report = json.loads(path.with_suffix(".json").read_text())
        assert report["steps"][-1]["action"] == {"op": "finish"}
        assert report["steps"][-1]["reward"] == report["reward_breakdown"]["reward"]
        assert -1 <= report["reward_breakdown"]["reward"] <= 1
        with Image.open(path) as image:
            assert image.mode == "RGB" and image.size == (800, 600)
