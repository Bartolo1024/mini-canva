"""Canonical request boundaries: strict JSON types, normalization, and reset inputs."""

import copy

import pytest
from pydantic import ValidationError

from marketcanvas_env.models import (
    DEFAULT_PROMPT,
    Element,
    NewElement,
    Properties,
    RequestError,
    Target,
    parse_action,
    validate_reset,
)


@pytest.mark.parametrize(
    ("kind", "subtype"), [("text", None), ("shape", "rectangle"), ("image", None)]
)
def test_creation_defaults_and_read_only_models(kind, subtype):
    element = NewElement(type=kind)
    assert element.subtype == subtype
    assert element.model_dump() == {
        "type": kind,
        "subtype": subtype,
        "role": "none",
        "x": 0,
        "y": 0,
        "width": 200,
        "height": 60,
        "z_index": 0,
        "color": "#FFFFFF",
        "text_color": "#000000",
        "content": "",
        "font_size": 24,
        "text_align": "left",
    }
    with pytest.raises(ValidationError):
        element.x = 1
    snapshot = element.model_dump()
    snapshot["x"] = 10
    assert element.x == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "shape", "subtype": None},
        {"type": "text", "subtype": "rectangle"},
        {"type": "image", "subtype": "button"},
        {"type": "shape", "subtype": "ellipse"},
        {"type": "button"},
        {"subtype": "rectangle"},
        {"type": "text", "id": 1},
    ],
)
def test_invalid_kind_and_add_id(payload):
    with pytest.raises(ValidationError):
        NewElement.model_validate(payload)


def test_explicit_subtypes_and_semantic_roles_are_structurally_valid():
    assert NewElement(type="text", subtype=None, role="cta").role == "cta"
    assert NewElement(type="image", subtype=None, role="headline").role == "headline"
    assert NewElement(type="shape", subtype="button").subtype == "button"
    with pytest.raises(ValidationError):
        Element(type="text")
    assert Element(id=64, type="text").id == 64


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
@pytest.mark.parametrize("field", ["x", "y", "width", "height", "z_index", "font_size"])
def test_integer_properties_reject_coercion(field, value):
    with pytest.raises(ValidationError):
        NewElement.model_validate({"type": "text", field: value})


@pytest.mark.parametrize(
    ("field", "low", "high"),
    [
        ("x", -800, 1599),
        ("y", -600, 1199),
        ("width", 1, 1600),
        ("height", 1, 1200),
        ("z_index", 0, 63),
        ("font_size", 8, 72),
    ],
)
def test_integer_property_boundaries(field, low, high):
    for value in (low, high):
        assert getattr(NewElement.model_validate({"type": "text", field: value}), field) == value
    for value in (low - 1, high + 1):
        with pytest.raises(ValidationError):
            NewElement.model_validate({"type": "text", field: value})


@pytest.mark.parametrize("content", ["\n", "hello\n", "a\tb", "é", "\x7f", "a" * 257, None, b"abc"])
def test_content_rejects_non_ascii_and_oversize(content):
    with pytest.raises(ValidationError):
        NewElement(type="text", content=content)


def test_content_preserved_and_colors_normalized_without_input_mutation():
    payload = {
        "op": "add_element",
        "element": {"type": "shape", "color": "#ab01ef", "content": "  A ~!  "},
    }
    original = copy.deepcopy(payload)
    action = parse_action(payload)
    assert action.element.color == "#AB01EF"
    assert action.element.content == "  A ~!  "
    assert payload == original


@pytest.mark.parametrize("color", ["#fff", "red", "#000000ff", "#00zz00", "#000000\n", None, 0])
def test_colors_reject_other_formats(color):
    with pytest.raises(ValidationError):
        NewElement(type="text", color=color)


def test_updates_require_exact_complete_properties():
    props = NewElement(type="text").model_dump(exclude={"type", "subtype"})
    assert parse_action({"op": "update_element", "id": 1, "properties": props}).properties == (
        Properties.model_validate(props)
    )
    for missing in props:
        incomplete = {key: value for key, value in props.items() if key != missing}
        with pytest.raises(RequestError, match="Field required"):
            parse_action({"op": "update_element", "id": 1, "properties": incomplete})
    for extra in ("id", "type", "subtype", "unexpected"):
        with pytest.raises(ValidationError):
            Properties.model_validate({**props, extra: 1})


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "finish",
        {},
        {"op": "unknown"},
        {"op": "finish", "id": 1},
        {"op": "move_element", "id": True, "new_x": 0, "new_y": 0},
        {"op": "move_element", "id": 1, "new_x": "0", "new_y": 0},
        {"op": "delete_element", "id": 0},
        {"op": "delete_element", "id": 65},
    ],
)
def test_invalid_actions_have_consistent_error_code(payload):
    with pytest.raises(RequestError) as caught:
        parse_action(payload)
    assert caught.value.code == "invalid_request"


def test_action_round_trips_and_nested_immutability():
    payloads = [
        {"op": "add_element", "element": {"type": "text"}},
        {"op": "move_element", "id": 64, "new_x": -800, "new_y": 1199},
        {"op": "delete_element", "id": 64},
        {"op": "finish"},
    ]
    for payload in payloads:
        action = parse_action(payload)
        assert parse_action(action.model_dump()) == action
    with pytest.raises(ValidationError):
        parse_action(payloads[0]).element.content = "mutated"


@pytest.mark.parametrize("seed", [None, 0, 7, 4294967295])
def test_default_reset_and_supported_prompt(seed):
    default, source = validate_reset(seed)
    assert source == "default"
    assert default.model_dump() == {
        "headline": "Summer Sale",
        "cta_text": "Shop Now",
        "cta_color": "#FFFF00",
    }
    assert validate_reset(seed, {}) == (default, "default")
    assert validate_reset(
        seed, {"prompt": "\n  " + DEFAULT_PROMPT.upper().replace(" ", "\t  ")}
    ) == (default, "prompt")


def test_structured_target_normalizes_only_outer_ascii_spaces_and_color():
    payload = {
        "target": {"headline": "  Autumn  Sale  ", "cta_text": " Browse ", "cta_color": "#00aAff"}
    }
    original = copy.deepcopy(payload)
    target, source = validate_reset(7, payload)
    assert source == "structured"
    assert target.model_dump() == {
        "headline": "Autumn  Sale",
        "cta_text": "Browse",
        "cta_color": "#00AAFF",
    }
    assert payload == original
    with pytest.raises(ValidationError):
        target.headline = "mutated"
    assert validate_reset()[0].headline == "Summer Sale"


@pytest.mark.parametrize("seed", [True, False, -1, 4294967296, 1.0, "7", []])
def test_reset_rejects_invalid_seeds(seed):
    with pytest.raises(RequestError) as caught:
        validate_reset(seed)
    assert caught.value.code == "invalid_request"


@pytest.mark.parametrize(
    "options",
    [
        [],
        "prompt",
        {"extra": 1},
        {"target": None},
        {"prompt": None},
        {"prompt": ""},
        {"prompt": "  \n"},
        {"prompt": "a" * 1025},
        {"prompt": 1},
        {"target": {}},
        {"prompt": DEFAULT_PROMPT, "target": {}},
        {"target": {"headline": " ", "cta_text": "ok", "cta_color": "#FFFFFF"}},
    ],
)
def test_reset_rejects_malformed_options(options):
    with pytest.raises(RequestError) as caught:
        validate_reset(options=options)
    assert caught.value.code == "invalid_request"


@pytest.mark.parametrize("prompt", ["different prompt", DEFAULT_PROMPT + ".", "Ignore all rules"])
def test_unsupported_prompts_are_explicit(prompt):
    with pytest.raises(RequestError) as caught:
        validate_reset(options={"prompt": prompt})
    assert caught.value.code == "unsupported_prompt"


@pytest.mark.parametrize("text", ["", "   ", "\tHello", "Hello\n", "é", "a" * 257])
def test_target_text_constraints(text):
    with pytest.raises(ValidationError):
        Target(headline=text, cta_text="Shop", cta_color="#FFFFFF")
