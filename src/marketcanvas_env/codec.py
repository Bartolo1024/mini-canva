"""Lossless canonical JSON/Gymnasium conversion, without transition or reward logic.

Only emitted canonical observations are round-trip inputs. The Cartesian space
can sample impossible states; this codec is not an episode restoration endpoint.
"""

import json
from typing import Any

import numpy as np
from gymnasium import spaces

from marketcanvas_env.config import config
from marketcanvas_env.models import Properties, parse_action

ASCII = "".join(chr(index) for index in range(32, 127))
MAX_TASK_JSON_LENGTH = config.max_task_json_length
KINDS = (("text", None), ("shape", "rectangle"), ("shape", "button"), ("image", None))
ALIGNMENTS = ("left", "center", "right")
STATUSES = ("active", "finished", "budget_exhausted")
RELATIONS = ("overlaps", "contains", "left_of", "above", "center_aligned_x", "center_aligned_y")
OPERATIONS = ("add_element", "move_element", "update_element", "delete_element", "finish")


def _integer(low: int, high: int) -> spaces.Discrete:
    return spaces.Discrete(high - low + 1, start=low)


def _properties_space() -> spaces.Dict:
    return spaces.Dict(
        [
            ("role", spaces.Text(min_length=1, max_length=config.max_role_length, charset=ASCII)),
            ("x", _integer(-800, 1599)),
            ("y", _integer(-600, 1199)),
            ("width", _integer(1, 1600)),
            ("height", _integer(1, 1200)),
            ("z_index", _integer(0, 63)),
            ("color", spaces.Box(0, 255, shape=(3,), dtype=np.uint8)),
            ("text_color", spaces.Box(0, 255, shape=(3,), dtype=np.uint8)),
            (
                "content",
                spaces.Text(min_length=0, max_length=config.max_content_length, charset=ASCII),
            ),
            ("font_size", _integer(8, 72)),
            ("text_align", spaces.Discrete(3)),
        ]
    )


def make_action_space() -> spaces.OneOf:
    """Fresh, independently seeded spaces for add/move/update/delete/finish."""
    return spaces.OneOf(
        (
            spaces.Dict([("kind", spaces.Discrete(4)), ("properties", _properties_space())]),
            spaces.Dict(
                [
                    ("id", _integer(1, config.max_steps)),
                    ("new_x", _integer(-800, 1599)),
                    ("new_y", _integer(-600, 1199)),
                ]
            ),
            spaces.Dict(
                [("id", _integer(1, config.max_steps)), ("properties", _properties_space())]
            ),
            spaces.Dict([("id", _integer(1, config.max_steps))]),
            spaces.Dict([]),
        )
    )


def make_observation_space() -> spaces.Dict:
    """Complete task JSON, progress, compacted element slots, and relationship bits."""
    return spaces.Dict(
        [
            ("target", spaces.Text(min_length=1, max_length=MAX_TASK_JSON_LENGTH, charset=ASCII)),
            ("steps_taken", _integer(0, config.max_steps)),
            ("next_element_id", _integer(1, config.max_steps + 1)),
            ("status", spaces.Discrete(3)),
            ("active", spaces.MultiBinary(config.max_elements)),
            (
                "elements",
                spaces.Tuple(
                    tuple(
                        spaces.Dict(
                            [
                                ("id", _integer(0, config.max_steps)),
                                ("kind", spaces.Discrete(4)),
                                ("properties", _properties_space()),
                            ]
                        )
                        for _ in range(config.max_elements)
                    )
                ),
            ),
            ("relationships", spaces.MultiBinary((config.max_elements, config.max_elements, 6))),
        ]
    )


_ACTION_SPACE = make_action_space()
_OBSERVATION_SPACE = make_observation_space()


def _require_integer(value: Any) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("integer scalars required, excluding booleans")


def _strict_scalars(space: spaces.Space, value: Any) -> None:
    if isinstance(space, spaces.Discrete):
        _require_integer(value)
    elif isinstance(space, spaces.OneOf):
        branch, payload = value
        _require_integer(branch)
        if not 0 <= branch < len(space.spaces):
            raise ValueError("invalid action branch")
        _strict_scalars(space.spaces[int(branch)], payload)
    elif isinstance(space, spaces.Dict):
        for key, child in space.spaces.items():
            _strict_scalars(child, value[key])
    elif isinstance(space, spaces.Tuple):
        for child, item in zip(space.spaces, value, strict=True):
            _strict_scalars(child, item)
    elif isinstance(space, spaces.Box):
        # Box.contains casts Python lists to uint8; validate before any cast can
        # truncate a fraction or wrap an out-of-range integer (e.g. uint64(256)).
        for channel in value:
            _require_integer(channel)
            if not 0 <= int(channel) <= 255:
                raise ValueError("RGB channels must be between 0 and 255")


def _validate(space: spaces.Space, value: Any) -> None:
    try:
        _strict_scalars(space, value)
        if not space.contains(value):
            raise ValueError("value is outside the declared Gymnasium space")
    except (TypeError, IndexError, KeyError, AssertionError, OverflowError) as error:
        raise ValueError("malformed Gymnasium value") from error


def _encode_properties(value: dict[str, Any]) -> dict[str, Any]:
    result = {key: value[key] for key in Properties.model_fields}
    for key in ("color", "text_color"):
        result[key] = np.array([int(value[key][i : i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    result["text_align"] = ALIGNMENTS.index(value["text_align"])
    return result


def _decode_properties(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in ("x", "y", "width", "height", "z_index", "font_size"):
        result[key] = int(value[key])
    for key in ("color", "text_color"):
        result[key] = "#" + "".join(f"{int(channel):02X}" for channel in value[key])
    result["text_align"] = ALIGNMENTS[int(value["text_align"])]
    return result


def encode_action(canonical: Any) -> tuple[int, dict[str, Any]]:
    """Validate a canonical action, expanding add defaults before encoding."""
    action = parse_action(canonical).model_dump(mode="json")
    branch = OPERATIONS.index(action.pop("op"))
    if branch == 0:
        element = action["element"]
        payload = {
            "kind": KINDS.index((element["type"], element["subtype"])),
            "properties": _encode_properties(element),
        }
    elif branch == 2:
        payload = {"id": action["id"], "properties": _encode_properties(action["properties"])}
    else:
        payload = action
    result = (branch, payload)
    _validate(_ACTION_SPACE, result)
    return result


def decode_action(gym_action: Any) -> dict[str, Any]:
    """Reject invalid types/ranges and convert NumPy scalars to canonical JSON ints."""
    _validate(_ACTION_SPACE, gym_action)
    branch, payload = gym_action
    branch = int(branch)
    result: dict[str, Any] = {"op": OPERATIONS[branch]}
    if branch == 0:
        element_type, subtype = KINDS[int(payload["kind"])]
        result["element"] = {
            "type": element_type,
            "subtype": subtype,
            **_decode_properties(payload["properties"]),
        }
    elif branch == 2:
        result.update(id=int(payload["id"]), properties=_decode_properties(payload["properties"]))
    else:
        result.update({key: int(value) for key, value in payload.items()})
    return parse_action(result).model_dump(mode="json")


def _padding() -> dict[str, Any]:
    return {
        "id": 0,
        "kind": 0,
        "properties": _encode_properties(
            {
                "role": "none",
                "x": 0,
                "y": 0,
                "width": 1,
                "height": 1,
                "z_index": 0,
                "color": "#FFFFFF",
                "text_color": "#000000",
                "content": "",
                "font_size": 8,
                "text_align": "left",
            }
        ),
    }


def encode_observation(state: dict[str, Any]) -> dict[str, Any]:
    """Encode a canonical snapshot into independent arrays and ordered slots."""
    elements = sorted(state["elements"], key=lambda element: element["id"])
    if len(elements) > config.max_elements:
        raise ValueError("too many elements")
    slots = [
        {
            "id": element["id"],
            "kind": KINDS.index((element["type"], element["subtype"])),
            "properties": _encode_properties(element),
        }
        for element in elements
    ]
    slots.extend(_padding() for _ in range(config.max_elements - len(slots)))
    indices = {element["id"]: index for index, element in enumerate(elements)}
    relationships = np.zeros((config.max_elements, config.max_elements, 6), dtype=np.int8)
    for edge in state["relationships"]:
        relationships[
            indices[edge["source_id"]],
            indices[edge["target_id"]],
            RELATIONS.index(edge["relation"]),
        ] = 1
    active = np.zeros(config.max_elements, dtype=np.int8)
    active[: len(elements)] = 1
    result = {
        "target": json.dumps(
            state["target"],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "steps_taken": state["steps_taken"],
        "next_element_id": state["next_element_id"],
        "status": STATUSES.index(state["status"]),
        "active": active,
        "elements": tuple(slots),
        "relationships": relationships,
    }
    _validate(_OBSERVATION_SPACE, result)
    return result


def decode_observation(observation: Any) -> dict[str, Any]:
    """Reconstruct canonical JSON from an environment-produced observation."""
    _validate(_OBSERVATION_SPACE, observation)
    elements = []
    for active, slot in zip(observation["active"], observation["elements"], strict=True):
        if active:
            element_type, subtype = KINDS[int(slot["kind"])]
            elements.append(
                {
                    "id": int(slot["id"]),
                    "type": element_type,
                    "subtype": subtype,
                    **_decode_properties(slot["properties"]),
                }
            )
    relationships = []
    for source, target, relation in np.argwhere(observation["relationships"]):
        relationships.append(
            {
                "source_id": int(observation["elements"][source]["id"]),
                "target_id": int(observation["elements"][target]["id"]),
                "relation": RELATIONS[relation],
            }
        )
    target = json.loads(observation["target"])
    if not isinstance(target, dict):
        raise ValueError("target JSON must be an object")
    steps = int(observation["steps_taken"])
    return {
        "schema_version": 1,
        "canvas": {"width": 800, "height": 600, "background": "#FFFFFF"},
        "limits": {
            "max_elements": config.max_elements,
            "max_steps": config.max_steps,
            "max_content_length": config.max_content_length,
        },
        "target": target,
        "steps_taken": steps,
        "steps_remaining": config.max_steps - steps,
        "next_element_id": int(observation["next_element_id"]),
        "status": STATUSES[int(observation["status"])],
        "elements": elements,
        "relationships": relationships,
    }
