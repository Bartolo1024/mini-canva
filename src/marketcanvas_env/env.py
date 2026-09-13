"""Gymnasium lifecycle and shared scored semantic API over CanvasCore."""

import json
from typing import Any

import gymnasium as gym
from pydantic import ValidationError

from marketcanvas_env.codec import (
    decode_action,
    encode_observation,
    make_action_space,
    make_observation_space,
)
from marketcanvas_env.config import config
from marketcanvas_env.core import CanvasCore
from marketcanvas_env.models import RequestError, validate_seed
from marketcanvas_env.prompt_parser import parse_prompt
from marketcanvas_env.reward import compute_reward_breakdown
from marketcanvas_env.reward.tasks import TaskSpec
from marketcanvas_env.task_data import load_task


def _task_options(options: Any) -> tuple[dict[str, Any], str]:
    """Normalize default YAML and the existing target alias into task fields."""
    if options is None or options == {}:
        options, source = load_task().model_dump(mode="json"), "default"
    else:
        source = "structured"
    if not isinstance(options, dict) or options.keys() - {"prompt", "constraints", "target"}:
        raise RequestError("invalid_request", "options must contain prompt/constraints or target")
    if "target" in options:
        if len(options) != 1:
            raise RequestError(
                "invalid_request", "target cannot be combined with prompt/constraints"
            )
        raw = options["target"]
        options = raw.model_dump(mode="json") if isinstance(raw, TaskSpec) else raw
        if not isinstance(options, dict):
            raise RequestError("invalid_request", "target must be a TaskSpec")
    return {"prompt": "", **options}, source


def _resolve_task(options: Any) -> tuple[TaskSpec, str]:
    """Use supplied constraints, otherwise parse the prompt; validate before reset.

    An explicit empty list is still supplied constraints, so test key presence
    rather than truthiness. Prompt parsing currently raises NotImplementedError.
    """
    data, source = _task_options(options)
    if "constraints" in data:
        constraints = data["constraints"]
    else:
        constraints = parse_prompt(data["prompt"])
        source = "prompt"
    try:
        task = TaskSpec.model_validate({**data, "constraints": constraints})
    except ValidationError as error:
        raise RequestError("invalid_request", str(error)) from error
    encoded = json.dumps(task.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    if len(encoded) > config.max_task_json_length:
        raise RequestError("invalid_request", "serialized TaskSpec exceeds observation capacity")
    return task, source


class MarketCanvasEnv(gym.Env):
    """One design episode. step takes encoded actions; execute_action takes JSON dictionaries.

    Both paths use the same core transition and terminal scoring. Reads are detached and pure.
    Rendering is on demand. Configure limits before importing the package; seeds do not affect edits.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str | None = None) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.render_mode = render_mode
        self.action_space = make_action_space()
        self.observation_space = make_observation_space()
        self._core = CanvasCore()

    def reset(self, *, seed=None, options=None):
        validate_seed(seed)
        task, source = _resolve_task(options)
        state, _ = self._core.reset(options={"target": task})
        super().reset(seed=seed)
        return encode_observation(state), {"target_source": source}

    def get_canvas_state(self) -> dict[str, Any]:
        return self._core.get_canvas_state()

    def get_current_reward(self) -> dict[str, Any]:
        """Diagnostic evaluation does not award a return or advance the episode."""
        state = self.get_canvas_state()
        return compute_reward_breakdown(state, state["target"])

    def execute_action(self, action: Any) -> dict[str, Any]:
        """Shared canonical step for direct callers and the MCP adapter."""
        transition = self._core.apply_action(action)
        report = (
            compute_reward_breakdown(transition.state, transition.state["target"])
            if transition.terminated
            else None
        )
        return {
            "state": transition.state,
            "reward": report["reward"] if report is not None else 0.0,
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "info": {
                "action_applied": transition.action_applied,
                "error": transition.error,
                "end_reason": transition.end_reason,
                "reward_breakdown": report,
            },
        }

    def step(self, action):
        try:
            canonical = decode_action(action)
        except ValueError as error:
            raise RequestError("invalid_request", str(error)) from error
        result = self.execute_action(canonical)
        return (
            encode_observation(result["state"]),
            result["reward"],
            result["terminated"],
            result["truncated"],
            result["info"],
        )

    def render(self):
        """Return a fresh RGB array when rgb_array mode is selected; otherwise None."""
        if self.render_mode is None:
            return None
        from marketcanvas_env.rendering import render_rgb

        return render_rgb(self.get_canvas_state())

    def save_png(self, path):
        """Export the current state in either render mode, without stepping or scoring."""
        from marketcanvas_env.rendering import save_png

        save_png(self.get_canvas_state(), path)
