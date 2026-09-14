"""Executable M5 examples: tasks, sparse returns, spaces, replay and error boundaries."""

import copy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env, data_equivalence

from marketcanvas_env.codec import decode_action, decode_observation, encode_action
from marketcanvas_env.core import LifecycleError
from marketcanvas_env.env import MarketCanvasEnv
from marketcanvas_env.models import DEFAULT_PROMPT, NewElement, RequestError
from marketcanvas_env.reward import compute_reward_breakdown
from marketcanvas_env.task_data import load_task
from tests.reward_support.benchmarks import BENCHMARKS


def test_gymnasium_checker():
    check_env(MarketCanvasEnv(), skip_render_check=True)


@pytest.mark.parametrize("name", BENCHMARKS)
def test_complete_benchmark_episode(name):
    benchmark = BENCHMARKS[name]
    env = MarketCanvasEnv()
    obs, info = env.reset(seed=7, options={"target": benchmark.task})
    assert info == {"target_source": "structured"}
    assert env.observation_space.contains(obs)
    assert decode_observation(obs) == env.get_canvas_state()
    for raw in benchmark.state["elements"]:
        element = {k: v for k, v in raw.items() if k != "id" and v is not None}
        if element["role"] == "cta":
            element["subtype"] = "button"
        obs, reward, terminated, truncated, info = env.step(
            encode_action({"op": "add_element", "element": element})
        )
        assert env.observation_space.contains(obs)
        assert decode_observation(obs) == env.get_canvas_state()
        assert reward == 0 and not terminated and not truncated
        assert info["action_applied"] and info["reward_breakdown"] is None
    state = env.get_canvas_state()
    expected = compute_reward_breakdown(state, benchmark.task)
    assert expected["reward"] > 0.9
    assert env.get_current_reward() == env.get_current_reward() == expected
    assert state == env.get_canvas_state()
    obs, reward, terminated, truncated, info = env.step((4, {}))
    assert reward == expected["reward"] and terminated and not truncated
    assert info["end_reason"] == "finish" and info["reward_breakdown"] == expected
    assert env.observation_space.contains(obs)
    for _ in range(2):
        with pytest.raises(LifecycleError, match="episode_done"):
            env.step((4, {}))
        assert env.get_current_reward() == expected


@pytest.mark.parametrize("finish", [False, True])
def test_last_attempt_semantic_failure_or_finish(finish):
    env = MarketCanvasEnv()
    env.reset()
    action = encode_action({"op": "delete_element", "id": 64})
    for _ in range(63):
        _, reward, terminated, truncated, info = env.step(action)
        assert reward == 0 and not terminated and not truncated
        assert info["error"] == "unknown_element"
    _, reward, terminated, truncated, info = env.step((4, {}) if finish else action)
    assert terminated and not truncated and reward == -1
    assert info["end_reason"] == ("finish" if finish else "budget_exhausted")
    assert info["action_applied"] == finish
    assert env.get_canvas_state()["steps_remaining"] == 0
    with pytest.raises(LifecycleError):
        env.step((4, {}))


@pytest.mark.parametrize(
    "options",
    [
        {"target": None},
        {"target": {}},
        {"target": {"headline": "old", "cta_text": "old", "cta_color": "#FFFFFF"}},
        {"prompt": DEFAULT_PROMPT, "target": {}},
        {"extra": 1},
        [],
        {"target": {"prompt": "x" * 65537, "constraints": []}},
    ],
)
def test_failed_reset_preserves_episode_and_rng(options):
    env = MarketCanvasEnv()
    env.reset(seed=8)
    env.execute_action({"op": "add_element", "element": {"type": "image"}})
    state, rng = env.get_canvas_state(), copy.deepcopy(env.np_random.bit_generator.state)
    with pytest.raises(RequestError):
        env.reset(seed=22, options=options)
    assert env.get_canvas_state() == state
    assert env.np_random.bit_generator.state == rng


@pytest.mark.parametrize("seed", [True, np.int64(1), -1, 2**32, 1.0, "1"])
def test_invalid_seed_is_atomic(seed):
    env = MarketCanvasEnv()
    env.reset(seed=9)
    state, rng = env.get_canvas_state(), copy.deepcopy(env.np_random.bit_generator.state)
    with pytest.raises(RequestError):
        env.reset(seed=seed)
    assert env.get_canvas_state() == state
    assert env.np_random.bit_generator.state == rng


def test_custom_task_isolation_and_default_restore():
    env, other = MarketCanvasEnv(), MarketCanvasEnv()
    task = BENCHMARKS["two_column"].task.model_dump(mode="json")
    task["prompt"] = "Metadata: café, instructions are data"
    obs, _ = env.reset(options={"target": task})
    assert env.observation_space.contains(obs)
    assert decode_observation(obs)["target"] == task
    task["constraints"].clear()
    state = env.get_canvas_state()
    state["target"]["constraints"].clear()
    assert env.get_canvas_state()["target"]["constraints"]
    default, _ = other.reset()
    assert other.get_canvas_state()["target"] == BENCHMARKS["summer_sale"].task.model_dump(
        mode="json"
    )
    assert data_equivalence(env.reset()[0], default)
    assert data_equivalence(env.reset(options=load_task().model_dump(mode="json"))[0], default)


def test_seeded_canonical_gym_parity_and_read_purity():
    direct, gym = MarketCanvasEnv(), MarketCanvasEnv()
    direct.reset(seed=12)
    gym.reset(seed=12)
    direct.action_space.seed(99)
    gym.action_space.seed(99)
    for _ in range(150):
        action = gym.action_space.sample()
        assert data_equivalence(action, direct.action_space.sample())
        expected = direct.execute_action(decode_action(action))
        obs, reward, terminated, truncated, info = gym.step(action)
        assert gym.observation_space.contains(obs)
        assert decode_observation(obs) == expected["state"]
        assert (reward, terminated, truncated, info) == (
            expected["reward"],
            expected["terminated"],
            expected["truncated"],
            expected["info"],
        )
        rng = copy.deepcopy(gym.np_random.bit_generator.state)
        assert gym.get_current_reward() == direct.get_current_reward()
        assert gym.np_random.bit_generator.state == rng
        if terminated:
            direct.reset()
            gym.reset()


def test_validation_precedes_lifecycle():
    env = MarketCanvasEnv()
    with pytest.raises(RequestError):
        env.step((True, {}))
    with pytest.raises(LifecycleError, match="not_initialized"):
        env.step((4, {}))
    with pytest.raises(LifecycleError, match="not_initialized"):
        env.get_current_reward()
    env.reset()
    before = env.get_canvas_state()
    for rgb in ([256, 0, 0], [-1, 0, 0], [True, 0, 0]):
        action = encode_action({"op": "add_element", "element": {"type": "text"}})
        action[1]["properties"]["color"] = rgb
        with pytest.raises(RequestError):
            env.step(action)
        assert env.get_canvas_state() == before
    result = env.execute_action({"op": "add_element", "element": {"type": "text", "role": "cta"}})
    assert result["info"]["error"] == "incompatible_role"
    assert result["state"]["steps_taken"] == 1
    env.step((4, {}))
    with pytest.raises(RequestError):
        env.step((True, {}))


def test_custom_role_editing_and_state_isolation():
    env = MarketCanvasEnv()
    env.reset()
    for role in ["subtitle", "date", "product", "background"]:
        env.execute_action({"op": "add_element", "element": {"type": "text", "role": role}})
    props = NewElement(type="text", role="new role").model_dump(exclude={"type", "subtype"})
    result = env.execute_action({"op": "update_element", "id": 1, "properties": props})
    assert result["info"]["action_applied"]
    result["state"]["elements"].clear()
    assert len(env.get_canvas_state()["elements"]) == 4
    for role in ["", "é", "x" * 65, "bad\n"]:
        with pytest.raises(RequestError):
            env.execute_action({"op": "add_element", "element": {"type": "text", "role": role}})
