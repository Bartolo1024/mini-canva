"""Configuration errors fail early; configured limits drive schemas and transitions."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from marketcanvas_env.config import EnvironmentConfig, load_config


@pytest.mark.parametrize("value", ["true", "0", "-1", "1.5", '"32"'])
def test_invalid_limit_is_rejected(tmp_path, value):
    path = tmp_path / "config.yaml"
    data = load_config().model_dump()
    data["max_elements"] = yaml.safe_load(value)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="max_elements"):
        load_config(path)


def test_changed_yaml_drives_core_and_schema_in_fresh_process(tmp_path):
    # Copy the package to avoid changing the real process-wide configuration.
    source = Path(__file__).parents[1] / "src" / "marketcanvas_env"
    package = tmp_path / "marketcanvas_env"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__"))
    data = load_config().model_dump()
    data.update(max_elements=1, max_steps=3, max_content_length=5)
    (package / "config" / "environment_config.yaml").write_text(yaml.safe_dump(data))
    code = """
from marketcanvas_env import CanvasCore, RequestError
from marketcanvas_env.models import parse_action
core = CanvasCore()
core.reset(options={"target": {"headline": "Sale", "cta_text": "Shop", "cta_color": "#FFFF00"}})
assert core.get_canvas_state()["limits"] == dict(max_elements=1, max_steps=3, max_content_length=5)
try:
    core.apply_action({"op": "add_element", "element": {"type": "text", "content": "123456"}})
except RequestError:
    pass
else:
    raise AssertionError("content limit ignored")
assert core.get_canvas_state()["steps_taken"] == 0
core.apply_action({"op": "add_element", "element": {"type": "image"}})
assert core.apply_action({"op": "add_element", "element": {"type": "image"}}).error == "capacity_exceeded"
last = core.apply_action({"op": "delete_element", "id": 3})
assert last.terminated and last.end_reason == "budget_exhausted"
assert last.state["steps_remaining"] == 0
try:
    parse_action({"op": "delete_element", "id": 4})
except RequestError:
    pass
else:
    raise AssertionError("ID bound ignored")

from marketcanvas_env.env import MarketCanvasEnv
from marketcanvas_env.codec import decode_observation
env = MarketCanvasEnv()
obs, _ = env.reset(options={"target": {"prompt": "Tiny task", "constraints": []}})
assert env.observation_space.contains(obs)
assert len(obs["elements"]) == 1
assert decode_observation(obs) == env.get_canvas_state()
env.action_space.seed(19)
for _ in range(20):
    obs, reward, ended, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert not truncated
    if ended:
        env.reset(options={"target": {"prompt": "Tiny task", "constraints": []}})
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("key", ["alpha", "beta"])
@pytest.mark.parametrize("value", [0.5, 0, 1, None])
def test_removed_reward_weights_are_rejected(key, value):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EnvironmentConfig.model_validate({**load_config().model_dump(), key: value})


def test_default_configuration_has_no_reward_weights():
    config = load_config()
    assert "alpha" not in config.model_dump()
    assert "beta" not in config.model_dump()
    assert "alpha" not in type(config).model_json_schema()["properties"]
    assert "beta" not in type(config).model_json_schema()["properties"]


@pytest.mark.parametrize("length", [1, 2, 3])
def test_role_limit_must_fit_default_role(length):
    with pytest.raises(ValidationError):
        EnvironmentConfig.model_validate({**load_config().model_dump(), "max_role_length": length})


def test_content_limit_cannot_exceed_scene_support():
    with pytest.raises(ValidationError):
        EnvironmentConfig.model_validate({**load_config().model_dump(), "max_content_length": 4097})
