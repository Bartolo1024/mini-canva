"""Space membership, strict decoding, and lossless canonical state encoding."""

import copy
import json

import numpy as np
import pytest

from marketcanvas_env.codec import (
    MAX_TASK_JSON_LENGTH,
    decode_action,
    decode_observation,
    encode_action,
    encode_observation,
    make_action_space,
    make_observation_space,
)
from marketcanvas_env.config import config
from marketcanvas_env.core import CanvasCore
from marketcanvas_env.models import parse_action
from marketcanvas_env.reward.tasks import TaskSpec


def test_seeded_samples_decode_to_schema_valid_canonical_actions():
    first, second = make_action_space(), make_action_space()
    first.seed(710)
    second.seed(710)
    branches = set()
    for _ in range(300):
        sample = first.sample()
        branches.add(int(sample[0]))
        canonical = decode_action(sample)
        assert canonical == decode_action(second.sample())
        assert parse_action(canonical).model_dump(mode="json") == canonical
        assert first.contains(encode_action(canonical))
    assert branches == set(range(5))


@pytest.mark.parametrize(
    "kind, subtype",
    [
        ("text", None),
        ("shape", "rectangle"),
        ("shape", "button"),
        ("image", None),
    ],
)
def test_add_defaults_kind_color_and_custom_role(kind, subtype):
    canonical = {
        "op": "add_element",
        "element": {
            "type": kind,
            "subtype": subtype,
            "role": "product image / 01",
            "color": "#aBcD09",
            "content": " ~",
            "text_align": "right",
        },
    }
    encoded = encode_action(canonical)
    assert make_action_space().contains(encoded)
    decoded = decode_action(encoded)
    assert decoded == parse_action(canonical).model_dump(mode="json")
    assert decoded["element"]["color"] == "#ABCD09"
    assert decoded["element"]["width"] == 200


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.0, np.float64(1), "1", np.array(1)])
@pytest.mark.parametrize("field", ["branch", "id", "new_x", "kind", "font_size"])
def test_strict_integer_types_rejected(bad, field):
    if field in {"kind", "font_size"}:
        branch, payload = encode_action({"op": "add_element", "element": {"type": "text"}})
        if field == "kind":
            payload[field] = bad
        else:
            payload["properties"][field] = bad
    else:
        branch, payload = 1, {"id": 1, "new_x": 0, "new_y": 0}
        if field == "branch":
            branch = bad
        else:
            payload[field] = bad
    with pytest.raises(ValueError):
        decode_action((branch, payload))


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.5, "1", -1, 256, np.uint64(256), 2**100])
def test_rgb_channels_cannot_coerce_or_overflow(bad):
    branch, payload = encode_action({"op": "add_element", "element": {"type": "text"}})
    payload["properties"]["color"] = [bad, 0, 0]
    with pytest.raises(ValueError):
        decode_action((branch, payload))


def test_numpy_integer_scalars_are_json_compatible():
    action = decode_action(
        (
            np.int64(1),
            {
                "id": np.int32(config.max_steps),
                "new_x": np.int64(-800),
                "new_y": np.int16(1199),
            },
        )
    )
    assert json.loads(json.dumps(action)) == action


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        (),
        (5, {}),
        (-1, {}),
        (4, {"id": 1}),
        (0, None),
        (0, {"kind": []}),
        (1, {"id": 0, "new_x": 0, "new_y": 0}),
        (1, {"id": 1, "new_x": 1600, "new_y": 0}),
    ],
)
def test_malformed_values_raise_value_error(invalid):
    with pytest.raises(ValueError):
        decode_action(invalid)


def test_round_trip_task_relationships_deletion_and_independent_arrays():
    core = CanvasCore()
    state, _ = core.reset()
    state["target"] = TaskSpec.model_validate(
        {
            "prompt": "Choose a café ☕",
            "constraints": [
                {
                    "id": "custom",
                    "kind": "exists",
                    "selector": {"role": "product"},
                }
            ],
        }
    ).model_dump(mode="json")
    target = state["target"]
    space = make_observation_space()
    encoded = encode_observation(state)
    assert space.contains(encoded)
    assert decode_observation(encoded) == state
    assert encoded["target"].isascii()
    assert encoded["target"] == json.dumps(
        target, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    assert len(encoded["elements"]) == config.max_elements
    for index in range(3):
        state = core.apply_action(
            {
                "op": "add_element",
                "element": {
                    "type": "shape",
                    "x": index * 10,
                    "y": 0,
                    "width": 100,
                    "height": 100,
                },
            }
        ).state
    state = core.apply_action({"op": "delete_element", "id": 2}).state
    state["target"] = target
    encoded = encode_observation(state)
    assert encoded["active"].dtype == np.int8
    assert encoded["relationships"].dtype == np.int8
    assert encoded["elements"][0]["id"] == 1
    assert encoded["elements"][1]["id"] == 3
    assert encoded["elements"][2]["id"] == 0
    assert np.count_nonzero(encoded["relationships"]) > 0
    assert not encoded["relationships"][2:].any()
    assert not encoded["relationships"][:, 2:].any()
    assert space.contains(encoded)
    assert decode_observation(encoded) == state
    original = copy.deepcopy(state)
    encoded["elements"][0]["properties"]["color"][0] = 0
    encoded["elements"][2]["properties"]["color"][0] = 0
    assert encoded["elements"][3]["properties"]["color"][0] == 255
    assert state == original


def test_capacity_and_terminal_observations_round_trip():
    core = CanvasCore()
    core.reset()
    for _ in range(config.max_elements):
        core.apply_action({"op": "add_element", "element": {"type": "image"}})
    for _ in range(config.max_steps - config.max_elements):
        state = core.apply_action({"op": "delete_element", "id": config.max_steps}).state
    encoded = encode_observation(state)
    assert make_observation_space().contains(encoded)
    assert encoded["active"].all()
    assert decode_observation(encoded) == state
    assert state["status"] == "budget_exhausted"


def test_target_json_bound_is_enforced():
    state, _ = CanvasCore().reset()
    state["target"] = {"prompt": "a" * MAX_TASK_JSON_LENGTH, "constraints": []}
    with pytest.raises(ValueError):
        encode_observation(state)


def test_spaces_follow_config_and_children_are_independent():
    space = make_observation_space()
    assert space["target"].max_length == config.max_task_json_length
    assert space["steps_taken"].n == config.max_steps + 1
    assert space["relationships"].shape == (config.max_elements, config.max_elements, 6)
    first, second = space["elements"].spaces[:2]
    assert first is not second
    assert first["properties"] is not second["properties"]
    assert first["properties"]["color"] is not second["properties"]["color"]
    assert first["properties"]["content"].max_length == config.max_content_length
    assert first["properties"]["role"].max_length == config.max_role_length
    assert first["properties"]["role"].min_length == 1
