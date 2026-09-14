"""Independent numerical checks and malformed-state invariants for reward evaluation."""

import json
import math
from copy import deepcopy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from marketcanvas_env import CanvasCore
from marketcanvas_env.reward import compute_reward_breakdown
from marketcanvas_env.reward.colors import (
    color_family,
    contrast_ratio,
    parse_color,
    relative_luminance,
)
from marketcanvas_env.reward.geometry import Rect, intersection_area, union_area, visible_ratio
from marketcanvas_env.reward.quality import harmonic_mean
from marketcanvas_env.reward.scene import Scene
from tests.reward_support.benchmarks import BENCHMARKS


def test_wcag_reference_values():
    black, white = parse_color("#000000"), parse_color("#FFFFFF")
    assert relative_luminance(black) == 0
    assert relative_luminance(white) == pytest.approx(1)
    assert contrast_ratio(black, white) == pytest.approx(21)
    assert contrast_ratio(white, white) == pytest.approx(1)
    assert contrast_ratio(black, white) == contrast_ratio(white, black)
    assert relative_luminance(parse_color("#FF0000")) == pytest.approx(0.2126)
    assert relative_luminance((0.04045, 0.04045, 0.04045)) == pytest.approx(0.04045 / 12.92)
    assert contrast_ratio(None, white) == 0


@pytest.mark.parametrize("color", ["#FFFF00", "#FFD700", "#FFC107", "#FFEB3B"])
def test_yellow_is_a_family_not_an_exact_hex(color):
    assert color_family(parse_color(color)) == "yellow"


@pytest.mark.parametrize(
    "color,family",
    [
        ("#FF0000", "red"),
        ("#FF8000", "orange"),
        ("#00FF00", "green"),
        ("#0000FF", "blue"),
        ("#A000FF", "purple"),
        ("#888888", "neutral"),
    ],
)
def test_other_color_families(color, family):
    assert color_family(parse_color(color)) == family
    assert family != "yellow"


@pytest.mark.parametrize("value", [None, [], {}, 3, "yellow", "#GG0000", "#FFF", "#FF000000"])
def test_unknown_colors_remain_unknown(value):
    assert parse_color(value) is None


def test_geometry_reference_values_and_union_occlusion():
    canvas = Rect(0, 0, 800, 600)
    rect = Rect(10, 10, 100, 50)
    assert visible_ratio(rect, canvas) == 1
    assert visible_ratio(Rect(-1000, 0, 100, 50), canvas) == 0
    assert visible_ratio(Rect(-50, 0, 100, 50), canvas) == pytest.approx(0.5)
    assert intersection_area(Rect(0, 0, 10, 10), Rect(5, 5, 10, 10)) == 25
    assert intersection_area(Rect(0, 0, 10, 10), Rect(10, 0, 10, 10)) == 0
    covers = [Rect(10, 10, 50, 50), Rect(35, 10, 50, 50)]
    assert union_area(covers) == 3750
    assert visible_ratio(rect, canvas, tuple(covers)) == pytest.approx(0.25)


def test_harmonic_mean_rejects_zero_dimension():
    assert harmonic_mean([1, 1, 1, 1]) == 1
    assert harmonic_mean([1, 1, 0, 1]) == 0
    assert harmonic_mean([1, 1, 1, 0.1]) == pytest.approx(4 / 13)


def test_effective_background_uses_highest_containing_lower_surface():
    state = deepcopy(BENCHMARKS["summer_sale"].state)
    headline = state["elements"][0]
    headline.update(text_color="#FFFFFF", z_index=4)
    backing = {
        "id": "back",
        "type": "image",
        "role": "background",
        "x": 180,
        "y": 60,
        "width": 440,
        "height": 120,
        "color": "#000000",
        "z_index": 1,
    }
    state["elements"].append(backing)
    scene = Scene(state)
    assert scene.contrast(scene.select({"id": "headline"})[0]) == pytest.approx(21)
    # A higher but still lower-z containing surface determines the background.
    state["elements"].append({**backing, "id": "near", "color": "#FFFFFF", "z_index": 3})
    scene = Scene(state)
    assert scene.contrast(scene.select({"id": "headline"})[0]) == pytest.approx(1)
    # Non-containing surfaces must not be mistaken for the background.
    state["elements"][-1]["width"] = 1
    scene = Scene(state)
    assert scene.contrast(scene.select({"id": "headline"})[0]) == pytest.approx(21)


def test_shape_label_uses_own_fill_and_image_underlay_is_not_a_collision():
    state = deepcopy(BENCHMARKS["summer_sale"].state)
    state["elements"].append(
        {
            "id": "background",
            "type": "image",
            "x": 0,
            "y": 0,
            "width": 800,
            "height": 600,
            "z_index": -1,
            "color": "#FFFFFF",
        }
    )
    result = compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)
    assert result["quality"]["overlap"] == 1
    assert result["reward"] > 0.9
    scene = Scene(state)
    cta = scene.select({"id": "cta"})[0]
    assert scene.contrast(cta) == pytest.approx(contrast_ratio(cta.text_color, cta.color))


def test_exact_aggregation_and_explicit_hard_gate():
    state = deepcopy(BENCHMARKS["summer_sale"].state)
    spec = {
        "prompt": "metadata",
        "constraints": [
            {"id": "yes", "kind": "exists", "selector": {"id": "headline"}},
            {"id": "no", "kind": "absent", "selector": {"id": "cta"}},
        ],
    }
    result = compute_reward_breakdown(state, spec)
    assert result["task_score"] == pytest.approx(0.5)
    assert result["quality_score"] == pytest.approx(1)
    assert result["score_01"] == pytest.approx(0.75)
    assert result["reward"] == pytest.approx(0.5)
    assert all("weight" not in row for row in result["constraints"])
    spec["constraints"][1]["hard"] = True
    hard = compute_reward_breakdown(state, spec)
    assert hard["hard_gate"] == 0.4
    assert hard["score_01"] == pytest.approx(0.4 * result["score_01"])
    assert hard["reward"] == pytest.approx(-0.4)


def test_each_requirement_has_equal_influence_and_order_is_irrelevant():
    state = BENCHMARKS["summer_sale"].state
    constraints = [
        {"id": "headline", "kind": "exists", "selector": {"id": "headline"}},
        {"id": "cta", "kind": "exists", "selector": {"id": "cta"}},
        {"id": "wrong_color", "kind": "color_exact", "selector": {"id": "cta"}, "value": "#FF0000"},
    ]
    task = {"prompt": "Opaque task metadata", "constraints": constraints}
    baseline = compute_reward_breakdown(state, task)
    assert baseline["task_score"] == pytest.approx(2 / 3)
    assert baseline["reward"] == pytest.approx(2 / 3)
    # Fixing a failed color constraint is worth exactly one of three task contributions.
    constraints[-1]["value"] = state["elements"][1]["color"]
    repaired = compute_reward_breakdown(state, task)
    assert repaired["task_score"] - baseline["task_score"] == pytest.approx(1 / 3)
    task["constraints"] = list(reversed(constraints))
    reordered = compute_reward_breakdown(state, task)
    assert reordered["reward"] == repaired["reward"]
    assert reordered["task_score"] == repaired["task_score"]


def test_no_constraints_and_empty_scene_are_defined():
    for state in (
        {},
        {"canvas": {"width": 800, "height": 600, "background": "#FFFFFF"}, "elements": []},
    ):
        result = compute_reward_breakdown(state, {"prompt": "", "constraints": []})
        assert result["task_score"] == 0
        assert result["quality_score"] == 0
        assert result["hard_gate"] == 1
        assert result["reward"] == -1


def test_core_snapshots_are_supported_without_using_legacy_target():
    core = CanvasCore()
    core.reset()
    core.apply_action(
        {
            "op": "add_element",
            "element": {
                "type": "text",
                "role": "headline",
                "content": "Summer Sale",
                "x": 200,
                "y": 80,
                "width": 400,
                "height": 80,
            },
        }
    )
    core.apply_action(
        {
            "op": "add_element",
            "element": {
                "type": "shape",
                "subtype": "button",
                "role": "cta",
                "content": "Shop Now",
                "x": 300,
                "y": 230,
                "width": 200,
                "height": 60,
                "color": "#FFD700",
            },
        }
    )
    state = core.get_canvas_state()
    result = compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)
    assert result["reward"] > 0.9
    state["target"] = {"ignored": "this is not the reward task"}
    assert compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task) == result


@pytest.mark.parametrize(
    "state",
    [
        None,
        [],
        "invalid",
        4,
        {"canvas": None},
        {"elements": None},
        {"elements": [None, [], 5]},
        {"elements": "invalid"},
    ],
)
def test_malformed_state_structure_is_safe(state):
    result = compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)
    assert -1 <= result["reward"] <= 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 5e-324),
        ("width", 10**1000),
        pytest.param("id", 10**5000, id="huge-id"),
        ("z_index", float("nan")),
        ("font_size", float("inf")),
        ("role", []),
        ("type", {}),
        ("content", {}),
        ("text_align", []),
    ],
)
def test_pathological_values_do_not_crash(field, value):
    state = deepcopy(BENCHMARKS["summer_sale"].state)
    state["elements"][0][field] = value
    assert math.isfinite(compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)["reward"])


def test_underflow_dimensions_are_invalid_not_a_division_by_zero():
    state = deepcopy(BENCHMARKS["summer_sale"].state)
    state["elements"][0].update(x=0, y=0, width=1e-200, height=1e-200)
    result = compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)
    assert result["quality"]["validity"] == 0.5
    assert result["reward"] < 0


def test_subnormal_canvas_width_cannot_crash_alignment():
    state = {
        "canvas": {"width": 5e-324, "height": 1e9, "background_color": "#FFFFFF"},
        "elements": [
            {
                "id": 1,
                "type": "shape",
                "x": 0,
                "y": 0,
                "width": 5e-324,
                "height": 1e9,
                "color": "#FFFFFF",
            }
        ],
    }
    task = {
        "prompt": "",
        "constraints": [
            {
                "id": "align",
                "kind": "alignment",
                "selector": {"id": 1},
                "reference": "canvas",
                "value": "horizontal_centers",
            }
        ],
    }
    result = compute_reward_breakdown(state, task)
    assert -1 <= result["reward"] <= 1
    json.dumps(result, allow_nan=False)


def test_numeric_stack_order_matches_core_but_selection_remains_lexical():
    state = deepcopy(BENCHMARKS["summer_sale"].state)
    first = state["elements"][1]
    first["id"] = 2
    state["elements"] = [first, {**first, "id": 10}]
    scene = Scene(state)
    assert scene.select({"id": 2})[0].visible_ratio == 0
    assert scene.select({"id": 10})[0].visible_ratio == 1
    state["elements"][1]["x"] = 550
    assert [e.id for e in Scene(state).select({"role": "cta"})] == ["10", "2"]


def test_transparent_text_does_not_occlude_with_its_empty_box_area():
    state = deepcopy(BENCHMARKS["summer_sale"].state)
    first = state["elements"][0]
    first.update(content="Left", x=0, y=0, width=800, height=80, text_align="left", z_index=1)
    state["elements"] = [
        first,
        {**first, "id": "right", "content": "Right", "text_align": "right", "z_index": 2},
    ]
    left = Scene(state).select({"id": "headline"})[0]
    assert left.usable_ratio > 0.9


@settings(max_examples=75, deadline=None)
@given(
    st.floats(allow_nan=True, allow_infinity=True), st.floats(allow_nan=True, allow_infinity=True)
)
def test_generated_geometry_always_has_finite_bounded_diagnostics(width, x):
    state = deepcopy(BENCHMARKS["summer_sale"].state)
    state["elements"][0].update(width=width, x=x)
    result = compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)
    assert -1 <= result["reward"] <= 1
    assert all(0 <= score <= 1 for score in result["quality"].values())
    assert all(0 <= row["score"] <= 1 for row in result["constraints"])
    json.dumps(result, allow_nan=False)
