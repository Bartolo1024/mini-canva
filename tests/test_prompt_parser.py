"""Explicit constraints bypass the future parser and preserve deterministic resets."""

from copy import deepcopy

import pytest

from marketcanvas_env.codec import decode_observation
from marketcanvas_env.env import MarketCanvasEnv
from marketcanvas_env.models import RequestError
from marketcanvas_env.prompt_parser import parse_prompt
from marketcanvas_env.task_data import load_task, tasks_directory


@pytest.mark.parametrize("path", sorted(tasks_directory().glob("*.yaml")), ids=lambda p: p.stem)
def test_yaml_constraints_reset_without_parsing_and_match_target_alias(path, monkeypatch):
    def unexpected_parse(prompt):
        pytest.fail("Explicit constraints must bypass prompt parsing")

    monkeypatch.setattr("marketcanvas_env.env.parse_prompt", unexpected_parse)
    task = load_task(path)
    options = task.model_dump(mode="json")
    direct, alias = MarketCanvasEnv(), MarketCanvasEnv()
    try:
        observation, info = direct.reset(seed=7, options=options)
        alias.reset(seed=7, options={"target": task})
        assert info == {"target_source": "structured"}
        assert decode_observation(observation) == alias.get_canvas_state()
        options["constraints"][0]["selector"]["role"] = "caller_edit"
        assert direct.get_canvas_state()["target"] == task.model_dump(mode="json")
        assert direct.execute_action({"op": "finish"}) == alias.execute_action({"op": "finish"})
    finally:
        direct.close()
        alias.close()


@pytest.mark.parametrize("prompt", [load_task().prompt, "A brand new design"])
def test_prompt_only_parsing_is_explicitly_unimplemented_and_reset_is_atomic(prompt):
    env = MarketCanvasEnv()
    try:
        env.reset(seed=8)
        env.execute_action({"op": "add_element", "element": {"type": "image"}})
        state, rng = env.get_canvas_state(), deepcopy(env.np_random.bit_generator.state)
        with pytest.raises(NotImplementedError, match="supply constraints"):
            env.reset(seed=22, options={"prompt": prompt})
        assert env.get_canvas_state() == state
        assert env.np_random.bit_generator.state == rng
    finally:
        env.close()


@pytest.mark.parametrize(
    "constraints", [[], [{"id": "image", "kind": "exists", "selector": {"type": "image"}}]]
)
def test_constraints_without_prompt_and_empty_list_are_explicit_inputs(constraints):
    env = MarketCanvasEnv()
    try:
        env.reset(options={"constraints": constraints})
        assert env.get_canvas_state()["target"]["prompt"] == ""
        assert len(env.get_canvas_state()["target"]["constraints"]) == len(constraints)
    finally:
        env.close()


@pytest.mark.parametrize("prompt", [None, True, 12, {}, "", " \n ", "x" * 1025])
def test_invalid_prompt_is_a_request_error(prompt):
    with pytest.raises(RequestError) as error:
        parse_prompt(prompt)
    assert error.value.code == "invalid_request"


@pytest.mark.parametrize(
    "options",
    [
        {"prompt": "x", "constraints": None},
        {"prompt": "x", "constraints": {}},
        {"prompt": "x", "constraints": [], "unknown": True},
        {"constraints": [], "target": {}},
        {"constranst": []},
    ],
)
def test_invalid_constraints_are_not_treated_as_missing(options):
    env = MarketCanvasEnv()
    try:
        env.reset(seed=8)
        before = env.get_canvas_state()
        with pytest.raises(RequestError):
            env.reset(options=options)
        assert env.get_canvas_state() == before
    finally:
        env.close()


def test_seed_only_initializes_gymnasium_rng_not_canvas():
    first, second = MarketCanvasEnv(), MarketCanvasEnv()
    try:
        first.reset(seed=7)
        second.reset(seed=8)
        assert first.get_canvas_state() == second.get_canvas_state()
        assert first.np_random.bit_generator.state != second.np_random.bit_generator.state
        expected_rng = deepcopy(first.np_random.bit_generator.state)
        action = {"op": "add_element", "element": {"type": "image"}}
        assert first.execute_action(action) == second.execute_action(action)
        assert first.get_current_reward() == second.get_current_reward()
        assert first.np_random.bit_generator.state == expected_rng
        first.reset(seed=7)
        assert first.np_random.bit_generator.state == expected_rng
    finally:
        first.close()
        second.close()
