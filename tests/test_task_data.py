"""Five public tasks and a compact set of genuine action trajectories."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from marketcanvas_env.env import MarketCanvasEnv
from marketcanvas_env.reward import compute_reward_breakdown
from marketcanvas_env.task_data import (
    load_task,
    load_trajectory,
    replay_trajectory,
    tasks_directory,
    trajectory_paths,
)


def test_public_data_contains_five_tasks_and_three_runs_per_task():
    names = {p.stem for p in tasks_directory().glob("*.yaml")}
    assert names == {"default_task", "webinar", "newsletter", "event", "two_column"}
    for name in names:
        paths = trajectory_paths(name)
        assert len(paths) == 3
        assert "well_done" in {path.stem for path in paths}
        for path in paths:
            assert load_trajectory(path)["task"] == name
    assert len(trajectory_paths()) == 15


def test_data_is_separate_and_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert tasks_directory() == Path(__file__).resolve().parents[1] / "data" / "tasks"
    env = MarketCanvasEnv()
    env.reset()
    assert env.get_canvas_state()["target"] == load_task().model_dump(mode="json")


@pytest.mark.parametrize("path", trajectory_paths(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_public_trajectory_replays_deterministically_through_the_environment(path):
    first = replay_trajectory(path)
    assert replay_trajectory(path) == first
    assert all(step["reward"] == 0 and not step["terminated"] for step in first["steps"][:-1])
    assert first["steps"][-1]["action"] == {"op": "finish"}
    assert first["steps"][-1]["terminated"]
    assert first["state"]["steps_taken"] == len(first["steps"])
    assert first["reward_breakdown"] == compute_reward_breakdown(
        first["state"], first["state"]["target"]
    )
    assert first["steps"][-1]["reward"] == first["reward_breakdown"]["reward"]
    if path.stem == "well_done":
        assert first["reward_breakdown"]["reward"] == 1
    first["state"]["elements"].clear()
    assert replay_trajectory(path)["state"]["elements"]


def test_representative_attacks_show_both_mitigations_and_remaining_loopholes():
    scores = {
        (p.parent.name, p.stem): replay_trajectory(p)["reward_breakdown"]["reward"]
        for p in trajectory_paths()
    }
    for task, name in [
        ("default_task", "offscreen_headline"),
        ("default_task", "small_element_spam"),
        ("webinar", "missing_cta"),
        ("webinar", "hidden_headline"),
        ("newsletter", "forbidden_cta"),
        ("newsletter", "tiny_logo"),
        ("event", "center_stacking"),
    ]:
        assert scores[task, name] < 0
    assert scores["two_column", "centered_layout"] < scores["two_column", "well_done"]
    # Observations of known loopholes, not claims that all attacks have been mitigated.
    for task, name in [
        ("event", "duplicate_cta"),
        ("two_column", "contrast_patch"),
    ]:
        assert scores[task, name] == 1


def test_loaded_action_lists_are_independent():
    path = trajectory_paths("default_task")[0]
    first = load_trajectory(path)
    first["actions"].clear()
    assert load_trajectory(path)["actions"]


def test_new_task_and_trajectory_need_no_reward_code_change(tmp_path, monkeypatch):
    import marketcanvas_env.task_data as data_module

    monkeypatch.setattr(data_module, "tasks_directory", lambda: tmp_path)
    (tmp_path / "custom.yaml").write_text(
        yaml.safe_dump(
            {
                "prompt": "A custom seal",
                "constraints": [
                    {
                        "id": "seal",
                        "kind": "exists",
                        "hard": True,
                        "selector": {"role": "custom_seal", "type": "image"},
                    }
                ],
            }
        )
    )
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "task": "custom",
                "description": "Custom task replay",
                "seed": 0,
                "actions": [
                    {
                        "op": "add_element",
                        "element": {
                            "type": "image",
                            "role": "custom_seal",
                            "width": 100,
                            "height": 100,
                        },
                    },
                    {"op": "finish"},
                ],
            }
        )
    )
    assert replay_trajectory(path)["reward_breakdown"]["reward"] == 1


@pytest.mark.parametrize(
    "actions",
    [
        [],
        [{"op": "add_element", "element": {"type": "image"}}],
        [{"op": "finish"}, {"op": "finish"}],
    ],
)
def test_trajectory_requires_exactly_one_final_finish(tmp_path, actions):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(
            {"task": "default_task", "description": "invalid", "seed": 0, "actions": actions}
        )
    )
    with pytest.raises(ValueError):
        load_trajectory(path)


def test_raw_canvas_snapshot_is_not_accepted_as_a_trajectory(tmp_path):
    path = tmp_path / "snapshot.yaml"
    path.write_text("task_file: default_task.yaml\nstate: {}\n")
    with pytest.raises(ValueError, match="actions"):
        load_trajectory(path)


def test_strict_task_validation_still_rejects_weights(tmp_path):
    path = tmp_path / "weighted.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "prompt": "x",
                "constraints": [{"id": "x", "kind": "exists", "selector": {}, "weight": 2}],
            }
        )
    )
    with pytest.raises(ValidationError, match="weight"):
        load_task(path)
