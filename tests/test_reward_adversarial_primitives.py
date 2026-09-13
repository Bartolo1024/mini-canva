"""Numerical safeguards and explicit diagnoses of current production scoring."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from examples.reward.benchmarks import BENCHMARKS
from marketcanvas_env.reward import compute_reward_breakdown
from marketcanvas_env.reward.colors import contrast_ratio, relative_luminance
from marketcanvas_env.reward.constraints import evaluate_constraint
from marketcanvas_env.reward.geometry import Rect, intersection_area, visible_ratio
from marketcanvas_env.reward.quality import collision_ratio
from marketcanvas_env.reward.scene import Scene
from marketcanvas_env.reward.tasks import Constraint, TaskSpec


def reference():
    return deepcopy(BENCHMARKS["summer_sale"].state)


def test_large_cover_small_object_normalization_is_not_iou():
    state = reference()
    state["elements"][1].update(x=100, y=20, width=600, height=400, z_index=3)
    scene = Scene(state)
    headline, cta = scene.elements
    assert intersection_area(headline.rect, cta.rect) == headline.rect.area
    assert collision_ratio(headline, cta) == 1
    assert headline.visible_ratio == 0


@pytest.mark.parametrize("fraction", [0, 0.1, 0.5, 0.79, 0.8, 1])
def test_clipping_fraction_and_task_configured_threshold(fraction):
    state = reference()
    item = state["elements"][1]
    item.update(type="image", role="product", x=-200 * (1 - fraction), y=400)
    scene = Scene(state)
    product = scene.select({"role": "product"})[0]
    assert product.visible_ratio == pytest.approx(fraction)
    for threshold in (0.5, 0.8, 1.0):
        constraint = Constraint(
            id="presence",
            kind="exists",
            selector={"role": "product", "type": "image"},
            min_visible_ratio=threshold,
        )
        score, _ = evaluate_constraint(constraint, scene)
        assert score == float(fraction >= threshold)


@pytest.mark.parametrize("distance,expected", [(200, 0.5), (100, 0.75), (50, 0.875), (0, 1)])
def test_requested_alignment_exact_decay(distance, expected):
    state = reference()
    state["elements"] = [state["elements"][1]]
    state["elements"][0]["x"] = 300 + distance
    task = TaskSpec(
        prompt="alignment only",
        constraints=[
            {
                "id": "center",
                "kind": "alignment",
                "selector": {"role": "cta"},
                "reference": "canvas",
                "value": "horizontal_centers",
            }
        ],
    )
    report = compute_reward_breakdown(state, task)
    assert report["constraints"][0]["score"] == pytest.approx(expected)
    assert report["quality_score"] == 1
    # No hard failure and Q=1: final reward equals this one task score.
    assert report["reward"] == pytest.approx(expected)


@pytest.mark.parametrize("gap,expected", [(-60, 0), (-30, 0.5), (0, 1), (12, 1), (72, 1), (300, 1)])
def test_relative_layout_uses_edges_but_has_no_preferred_gap(gap, expected):
    state = reference()
    # Separate columns keep visibility constant while isolating the relative-position formula.
    state["elements"][0].update(x=0, width=300)
    state["elements"][1].update(x=550, y=160 + gap)
    c = Constraint(
        id="below",
        kind="relative_position",
        subject={"role": "cta"},
        object={"role": "headline"},
        relation="below",
    )
    assert evaluate_constraint(c, Scene(state))[0] == pytest.approx(expected)


@pytest.mark.parametrize("ratio", [1.5, 2.5, 3.5, 4.5, 7, 21])
def test_wcag_continuous_score_separate_from_readability(ratio):
    # Analytically construct linear luminance against black, independent of 8-bit rounding.
    luminance = (ratio - 1) * 0.05
    channel = (
        12.92 * luminance if luminance <= 0.0031308 else 1.055 * luminance ** (1 / 2.4) - 0.055
    )
    rgb = (channel,) * 3
    assert relative_luminance(rgb) == pytest.approx(luminance)
    assert contrast_ratio((0, 0, 0), rgb) == pytest.approx(ratio)
    # Exercise the actual constraint evaluator with closest 8-bit gray as text on black.
    state = reference()
    state["canvas"]["background_color"] = "#000000"
    state["elements"] = [state["elements"][0]]
    byte = round(channel * 255)
    state["elements"][0]["text_color"] = f"#{byte:02X}{byte:02X}{byte:02X}"
    scene = Scene(state)
    c = Constraint(id="contrast", kind="contrast_min", selector={"role": "headline"}, value=4.5)
    actual = scene.contrast(scene.elements[0])
    assert evaluate_constraint(c, scene)[0] == pytest.approx(min(actual / 4.5, 1))


def test_contrast_scores_increase_then_cap_without_moving_text():
    state = reference()
    state["elements"] = [state["elements"][0]]
    state["canvas"]["background_color"] = "#000000"
    c = Constraint(id="contrast", kind="contrast_min", selector={"role": "headline"}, value=4.5)
    scores = []
    for gray in (40, 70, 100, 140, 200, 255):
        state["elements"][0]["text_color"] = f"#{gray:02x}{gray:02x}{gray:02x}"
        scores.append(evaluate_constraint(c, Scene(state))[0])
    assert scores == sorted(scores)
    assert scores[0] < scores[-1] == 1
    assert scores[-2] == 1


def test_tiny_text_has_real_contrast_but_zero_usable_ink():
    state = reference()
    state["elements"] = [state["elements"][0]]
    state["elements"][0].update(width=1, height=1, text_color="#000000")
    scene = Scene(state)
    assert scene.contrast(scene.elements[0]) == pytest.approx(21)
    assert scene.elements[0].visible_ratio == 1
    assert scene.elements[0].usable_ratio == 0
    report = compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)
    assert report["quality"]["contrast"] == report["quality"]["bounds"] == 0
    assert report["reward"] < 0


def test_missing_dependencies_are_zero_and_still_included_in_mean_observation():
    state = reference()
    state["elements"].pop()
    report = compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)
    scores = {c["id"]: c["score"] for c in report["constraints"]}
    assert scores["cta_exists"] == scores["cta_yellow"] == scores["good_contrast"] == 0
    assert report["task_score"] == pytest.approx(2 / 5)
    assert report["hard_gate"] == 0.4 and report["reward"] < 0
    assert all("applicable" not in row for row in report["constraints"])


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="RH-06: missing dependencies are zero, not N/A"
)
def test_missing_dependent_constraints_should_be_not_applicable():
    state = reference()
    state["elements"].pop()
    report = compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)
    dependent = next(c for c in report["constraints"] if c["id"] == "cta_yellow")
    assert dependent.get("applicable") is False


def test_overlapping_selectors_reuse_one_element_observation():
    state = reference()
    state["elements"] = [state["elements"][0]]
    task = TaskSpec(
        prompt="two required objects",
        constraints=[
            {"id": "headline", "kind": "exists", "hard": True, "selector": {"role": "headline"}},
            {"id": "another_text", "kind": "exists", "hard": True, "selector": {"type": "text"}},
        ],
    )
    report = compute_reward_breakdown(state, task)
    assert [row["score"] for row in report["constraints"]] == [1, 1]
    assert report["reward"] == 1  # No global one-to-one assignment exists.


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="RH-07: exists chooses visible box/ID, not best usable match",
)
def test_blank_duplicate_should_not_hide_a_valid_presence_match():
    state = reference()
    state["elements"].append(
        {**state["elements"][0], "id": "a_blank", "x": 0, "y": 400, "content": ""}
    )
    c = Constraint(id="exists", kind="exists", selector={"role": "headline"})
    assert evaluate_constraint(c, Scene(state))[0] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_width_ratio", 0.02),
        ("min_height_ratio", 0.02),
    ],
)
def test_normalized_dimension_authoring_is_currently_unsupported(field, value):
    with pytest.raises(ValidationError, match="not applicable|Extra inputs"):
        Constraint.model_validate(
            {"id": "size", "kind": "exists", "selector": {"role": "logo"}, field: value}
        )


def test_missing_content_is_semantically_empty_but_structurally_valid():
    state = reference()
    del state["elements"][0]["content"]
    report = compute_reward_breakdown(state, BENCHMARKS["summer_sale"].task)
    assert report["quality"]["validity"] == 1  # Missing content defaults to empty.
    assert report["constraints"][0]["score"] == 0
    assert report["reward"] < 0


def test_occlusion_union_is_not_double_counted_or_diluted():
    canvas = Rect(0, 0, 100, 100)
    covers = (Rect(0, 0, 50, 100), Rect(25, 0, 50, 100))
    assert visible_ratio(canvas, canvas, covers) == pytest.approx(0.25)
    assert visible_ratio(canvas, canvas, covers + (Rect(200, 200, 10, 10),)) == pytest.approx(0.25)


def test_diagnostic_results_do_not_alias_input_or_prior_results():
    state = reference()
    task = BENCHMARKS["summer_sale"].task
    before = deepcopy(state)
    report = compute_reward_breakdown(state, task)
    report["constraints"][0]["score"] = -999
    assert state == before
    assert compute_reward_breakdown(state, task)["constraints"][0]["score"] == 1
    assert np.isfinite(compute_reward_breakdown(state, task)["reward"])
