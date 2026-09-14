"""Load task and canvas data separately from runtime configuration and reward logic."""

from copy import deepcopy
from functools import cache
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import yaml

from marketcanvas_env.reward.tasks import TaskSpec


@cache
def tasks_directory() -> Path:
    """Locate checkout data or the wheel's installed data via its distribution manifest.

    No dependence on the current working directory or fallback to embedded task copies.
    """
    package = Path(__file__).resolve().parent
    checkout = package.parent.parent
    if package.parent.name == "src" and (checkout / "pyproject.toml").is_file():
        return checkout / "data" / "tasks"
    installed = distribution("marketcanvas-env")
    for entry in installed.files or ():
        if entry.parts[-2:] == ("tasks", "summer_sale.yaml"):
            return Path(installed.locate_file(entry)).resolve().parent
    raise FileNotFoundError(
        "MarketCanvas task data is missing; reinstall the package with its data"
    )


def _path(filename: str | Path) -> Path:
    path = Path(filename)
    return (tasks_directory() / path if path.parent == Path(".") else path).resolve()


@cache
def _document(path: Path) -> dict[str, Any]:
    """Private process-lifetime data cache; never expose its mutable dictionaries."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def load_task(filename: str | Path = "summer_sale.yaml") -> TaskSpec:
    """Read a plain TaskSpec YAML. A bare filename refers to bundled task data."""
    return TaskSpec.model_validate(deepcopy(_document(_path(filename))))


def trajectory_paths(task_name: str | None = None) -> tuple[Path, ...]:
    """Discover the compact public trajectory set, in stable task/filename order."""
    root = tasks_directory().parent / "trajectories"
    if task_name is not None:
        if not isinstance(task_name, str) or Path(task_name).name != task_name:
            raise ValueError("task name must be a filename stem")
        root /= task_name
        return tuple(sorted(root.glob("*.yaml")))
    return tuple(sorted(root.glob("*/*.yaml")))


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """Read a task reference and ordered semantic actions; never accept final-state snapshots."""
    from marketcanvas_env.models import parse_action, validate_seed

    source = Path(path).resolve()
    data = deepcopy(_document(source))
    if set(data) != {"task", "description", "seed", "actions"}:
        raise ValueError(f"{source}: expected task, description, seed, and actions")
    name = data["task"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{source}: task must name a task YAML stem")
    if not isinstance(data["description"], str):
        raise ValueError(f"{source}: description must be text")
    load_task(f"{name}.yaml")  # Fail explicitly for missing/invalid referenced tasks.
    validate_seed(data["seed"])
    actions = data["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{source}: actions must be a nonempty list ending in finish")
    for action in actions:
        parse_action(action)
    if actions[-1]["op"] != "finish" or any(a["op"] == "finish" for a in actions[:-1]):
        raise ValueError(f"{source}: finish must appear exactly once, as the final action")
    return data


def replay_trajectory(path: str | Path) -> dict[str, Any]:
    """Replay a complete legal episode through the same interface used by RL and MCP."""
    from marketcanvas_env.env import MarketCanvasEnv

    trajectory = load_trajectory(path)
    env = MarketCanvasEnv()
    try:
        env.reset(
            seed=trajectory["seed"],
            options=load_task(trajectory["task"] + ".yaml").model_dump(mode="json"),
        )
        steps = []
        for action in trajectory["actions"]:
            result = env.execute_action(action)
            if not result["info"]["action_applied"]:
                raise ValueError(f"{path}: action could not be applied: {result['info']['error']}")
            steps.append(
                {"action": action, "reward": result["reward"], "terminated": result["terminated"]}
            )
        return {
            "task": trajectory["task"],
            "description": trajectory["description"],
            "steps": steps,
            "state": result["state"],
            "reward_breakdown": result["info"]["reward_breakdown"],
        }
    finally:
        env.close()
