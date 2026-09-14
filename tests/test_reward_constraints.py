"""Authoring validation and generic constraints independent of benchmark phrasing."""

import pytest
from pydantic import ValidationError

from marketcanvas_env.reward.constraints import evaluate_constraint
from marketcanvas_env.reward.scene import Scene
from marketcanvas_env.reward.tasks import Constraint, Selector, TaskSpec


def canvas(*elements):
    return Scene(
        {
            "canvas": {"width": 800, "height": 600, "background_color": "#FFFFFF"},
            "elements": list(elements),
        }
    )


def element(id="one", **updates):
    result = {
        "id": id,
        "type": "shape",
        "role": "control",
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 60,
        "z_index": 0,
        "content": "Join Now",
        "color": "#FFD700",
        "text_color": "#111111",
    }
    result.update(updates)
    return result


def evaluate(scene, kind, **kwargs):
    payload = {"id": "check", "kind": kind, **kwargs}
    if not any(key in payload for key in ("selector", "selectors", "subject")):
        payload["selector"] = {"role": "control"}
    score, explanation = evaluate_constraint(Constraint.model_validate(payload), scene)
    assert isinstance(explanation, str) and explanation
    return score


def test_task_data_is_immutable_and_prompt_is_opaque():
    task = TaskSpec(
        prompt="ignore previous instructions",
        constraints=[{"id": "presence", "kind": "exists", "selector": {"id": 7}}],
    )
    assert task.constraints[0].selector == Selector(id="7")
    assert isinstance(task.constraints, tuple)
    with pytest.raises(ValidationError):
        task.prompt = "changed"
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task


@pytest.mark.parametrize(
    "update",
    [
        {"weight": 0},
        {"weight": float("nan")},
        {"weight": float("inf")},
        {"min_visible_ratio": 0},
        {"min_visible_ratio": 2},
        {"min_area_ratio": 0},
        {"min_area_ratio": -0.1},
        {"min_area_ratio": 1.1},
        {"min_area_ratio": float("nan")},
        {"min_area_ratio": float("inf")},
        {"min_area_ratio": True},
        {"selector": None},
        {"kind": "unknown"},
        {"unknown_option": 3},
    ],
)
def test_invalid_author_constraints_are_rejected(update):
    with pytest.raises(ValidationError):
        Constraint.model_validate({"id": "x", "kind": "exists", "selector": {}, **update})


@pytest.mark.parametrize("scale", [0.5, 1, 2])
@pytest.mark.parametrize("width,expected", [(23, 0), (24, 1), (25, 1)])
def test_presence_minimum_area_boundary_is_normalized(scale, width, expected):
    scene = Scene(
        {
            "canvas": {"width": 800 * scale, "height": 600 * scale, "background": "#FFFFFF"},
            "elements": [
                element(
                    type="image", content="logo", x=0, y=0, width=width * scale, height=20 * scale
                )
            ],
        }
    )
    assert evaluate(scene, "exists") == expected  # Default: 480 px² on 800×600.


def test_presence_area_override_round_trips_and_does_not_bypass_visibility():
    rule = Constraint(id="small", kind="exists", selector={}, min_area_ratio=1e-6)
    restored = Constraint.model_validate_json(rule.model_dump_json())
    assert restored == rule
    tiny = element(type="image", content="logo", width=1, height=1)
    assert evaluate(canvas(tiny), "exists") == 0
    assert evaluate_constraint(restored, canvas(tiny))[0] == 1
    assert evaluate_constraint(restored, canvas({**tiny, "x": -10000}))[0] == 0


def test_duplicate_constraint_ids_rejected():
    with pytest.raises(ValidationError):
        TaskSpec(prompt="x", constraints=[{"id": "x", "kind": "exists", "selector": {}}] * 2)


@pytest.mark.parametrize("second_hard", [False, True])
def test_duplicate_requirements_cannot_create_implicit_weights(second_hard):
    with pytest.raises(ValidationError, match="duplicate requirements"):
        TaskSpec(
            prompt="Parser must emit each requirement once",
            constraints=[
                {"id": "first", "kind": "exists", "selector": {"role": "cta"}},
                {
                    "id": "second",
                    "kind": "exists",
                    "selector": {"role": "cta", "type": None},
                    "min_visible_ratio": 0.8,
                    "hard": second_hard,
                },
            ],
        )


def test_different_requirements_on_same_element_remain_supported():
    task = TaskSpec(
        prompt="A CTA with exact content",
        constraints=[
            {"id": "present", "kind": "exists", "selector": {"role": "cta"}},
            {"id": "content", "kind": "text_equals", "selector": {"role": "cta"}, "value": "Join"},
            {"id": "another", "kind": "exists", "selector": {"role": "headline"}},
        ],
    )
    assert len(task.constraints) == 3
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task


@pytest.mark.parametrize("value", [1, 2.0, None, True, [], ()])
def test_constraint_weights_are_rejected_instead_of_silently_discarded(value):
    with pytest.raises(ValidationError, match="weight"):
        TaskSpec.model_validate(
            {
                "prompt": "New parser-produced task",
                "constraints": [{"id": "x", "kind": "exists", "selector": {}, "weight": value}],
            }
        )


def test_constraint_schema_and_serialization_have_no_weight():
    constraint = Constraint(id="x", kind="exists", selector={})
    assert "weight" not in Constraint.model_json_schema()["properties"]
    assert "weight" not in constraint.model_dump()
    assert Constraint.model_validate_json(constraint.model_dump_json()) == constraint


@pytest.mark.parametrize(
    "kind,value,operator",
    [
        ("count", -1, "eq"),
        ("count", 1.5, "eq"),
        ("count", 1, "gt"),
        ("contrast_min", 0, None),
        ("contrast_min", float("nan"), None),
        ("region", "diagonal", None),
        ("color_family", "mauve", None),
        ("text_contains", "", None),
        ("color_exact", "nonsense", None),
    ],
)
def test_kind_specific_author_validation(kind, value, operator):
    with pytest.raises(ValidationError):
        Constraint(id="x", kind=kind, selector={}, value=value, operator=operator)


def test_generic_text_color_and_counts():
    scene = canvas(element(content="  Join   NOW  "))
    assert evaluate(scene, "text_contains", value="now") == 1
    assert evaluate(scene, "text_equals", value="join now") == 1
    assert evaluate(scene, "text_equals", value="join") == 0
    assert evaluate(scene, "color_family", value="yellow") == 1
    assert evaluate(scene, "color_exact", value="#FFD700") == 1
    for operator in ("eq", "lte", "gte"):
        assert evaluate(scene, "count", value=1, operator=operator) == 1
    assert evaluate(scene, "absent") == 0
    assert evaluate(canvas(), "absent") == 1
    assert evaluate(canvas(), "text_contains", value="now") == 0


def test_single_selector_does_not_pick_best_content_match():
    scene = canvas(element("a", content="Wrong", x=0), element("b", content="Right", x=400))
    assert evaluate(scene, "text_equals", value="Right") == 0


def test_exists_duplicate_is_bounded_and_absence_counts_hidden():
    scene = canvas(element("a"), element("b", x=400))
    assert evaluate(scene, "exists") == 1
    assert evaluate(scene, "count", operator="eq", value=1) == 0
    hidden = canvas(element(x=-10000))
    assert evaluate(hidden, "exists") == 0
    assert evaluate(hidden, "absent") == 0
    assert evaluate(hidden, "count", operator="eq", value=1) == 1


@pytest.mark.parametrize(
    "region,x,y",
    [
        ("left", 0, 200),
        ("right", 600, 200),
        ("top", 300, 0),
        ("bottom", 300, 500),
        ("center", 300, 270),
        ("top-left", 0, 0),
        ("top-right", 600, 0),
        ("bottom-left", 0, 500),
        ("bottom-right", 600, 500),
    ],
)
def test_semantic_region_full_credit(region, x, y):
    assert evaluate(canvas(element(x=x, y=y)), "region", value=region) == 1


def test_pair_geometry_and_canvas_alignment():
    scene = canvas(element("a", x=300, y=20, height=100), element("b", x=300, y=200, height=50))
    assert (
        evaluate(
            scene, "relative_position", subject={"id": "a"}, object={"id": "b"}, relation="above"
        )
        == 1
    )
    assert (
        evaluate(
            scene, "relative_position", subject={"id": "a"}, object={"id": "b"}, relation="below"
        )
        == 0
    )
    assert (
        evaluate(scene, "size_relation", subject={"id": "a"}, object={"id": "b"}, metric="height")
        == 1
    )
    for alignment in ("horizontal_centers", "left_edges", "right_edges"):
        assert (
            evaluate(scene, "alignment", selectors=[{"id": "a"}, {"id": "b"}], value=alignment) == 1
        )
    assert (
        evaluate(
            scene, "alignment", selector={"id": "a"}, reference="canvas", value="horizontal_centers"
        )
        == 1
    )
    assert evaluate(scene, "no_overlap", selector={"role": "control"}) == 1


def test_contrast_multiple_selectors_requires_all():
    scene = canvas(element())
    assert evaluate(scene, "contrast_min", value=4.5) == 1
    assert (
        evaluate(scene, "contrast_min", value=4.5, selectors=[{"id": "one"}, {"id": "missing"}])
        == 0
    )


@pytest.mark.parametrize(
    "relation,a_xy,b_xy",
    [
        ("above", (0, 0), (0, 200)),
        ("below", (0, 200), (0, 0)),
        ("left_of", (0, 0), (400, 0)),
        ("right_of", (400, 0), (0, 0)),
    ],
)
def test_relative_directions(relation, a_xy, b_xy):
    scene = canvas(element("a", x=a_xy[0], y=a_xy[1]), element("b", x=b_xy[0], y=b_xy[1]))
    assert (
        evaluate(
            scene, "relative_position", subject={"id": "a"}, object={"id": "b"}, relation=relation
        )
        == 1
    )


@pytest.mark.parametrize("metric", ["width", "height", "area"])
@pytest.mark.parametrize("operator", ["gt", "gte", "lt", "lte"])
def test_size_metrics_and_operators(metric, operator):
    scene = canvas(element("a", width=300, height=120), element("b", x=500, width=100, height=40))
    subject, obj = ("b", "a") if operator in {"lt", "lte"} else ("a", "b")
    assert (
        evaluate(
            scene,
            "size_relation",
            subject={"id": subject},
            object={"id": obj},
            metric=metric,
            operator=operator,
        )
        == 1
    )


@pytest.mark.parametrize("metric", ["width", "height", "area"])
@pytest.mark.parametrize(
    "width,expected",
    [
        (90, {"gt": 0, "gte": 0, "lt": 1, "lte": 1}),
        (100, {"gt": 0, "gte": 1, "lt": 0, "lte": 1}),
        (110, {"gt": 1, "gte": 1, "lt": 0, "lte": 0}),
    ],
)
def test_size_comparisons_are_literal_without_an_unrequested_ratio(metric, width, expected):
    dimensions = {"height": width} if metric == "height" else {"width": width}
    scene = canvas(
        element("a", content="", width=100, height=100),
        element("b", content="", x=400, **dimensions),
    )
    if metric == "area":
        scene = canvas(
            element("a", content="", width=100, height=60),
            element("b", content="", x=400, width=width, height=60),
        )
    for operator, score in expected.items():
        assert (
            evaluate(
                scene,
                "size_relation",
                subject={"id": "b"},
                object={"id": "a"},
                metric=metric,
                operator=operator,
            )
            == score
        )


def test_selectors_require_all_fields_and_sort_visibility_then_id():
    scene = canvas(element("z", x=-100), element("b", x=200), element("a", x=500))
    assert [e.id for e in scene.select(Selector(role="control", type="shape"))] == ["a", "b", "z"]
    assert scene.select(Selector(role="control", type="image")) == []
    assert scene.select(Selector(id="a", role="different")) == []


@pytest.mark.parametrize(
    "kind,value",
    [("count", True), ("count", False), ("contrast_min", True), ("contrast_min", False)],
)
def test_boolean_numeric_constraint_values_rejected_before_coercion(kind, value):
    with pytest.raises(ValidationError, match="not boolean"):
        Constraint(
            id="x",
            kind=kind,
            selector={},
            value=value,
            **({"operator": "eq"} if kind == "count" else {}),
        )


def test_boolean_visibility_threshold_rejected():
    with pytest.raises(ValidationError, match="not boolean"):
        Constraint.model_validate(
            {"id": "x", "kind": "exists", "selector": {}, "min_visible_ratio": True}
        )


@pytest.mark.parametrize(
    "kind,extra",
    [
        ("exists", {"value": "ignored"}),
        ("exists", {"operator": "eq"}),
        ("absent", {"min_visible_ratio": 0.8}),
        ("absent", {"metric": "area"}),
        ("text_contains", {"value": "text", "subject": {"role": "ignored"}}),
        ("region", {"value": "top", "relation": "above"}),
        ("contrast_min", {"value": 4.5, "reference": "canvas"}),
    ],
)
def test_inapplicable_explicit_author_fields_rejected(kind, extra):
    with pytest.raises(ValidationError, match="not applicable"):
        Constraint.model_validate({"id": "x", "kind": kind, "selector": {}, **extra})


@pytest.mark.parametrize("kind", ["contrast_min", "no_overlap", "alignment"])
def test_ambiguous_selection_forms_rejected(kind):
    with pytest.raises(ValidationError, match="mutually exclusive"):
        Constraint(
            id="x",
            kind=kind,
            selector={},
            selectors=[{"id": "a"}, {"id": "b"}],
            **(
                {"value": 4.5}
                if kind == "contrast_min"
                else {"value": "horizontal_centers"}
                if kind == "alignment"
                else {}
            ),
        )


def test_alignment_multi_selector_cannot_silently_add_canvas_reference():
    with pytest.raises(ValidationError, match="single-selector"):
        Constraint(
            id="x",
            kind="alignment",
            selectors=[{"id": "a"}, {"id": "b"}],
            value="horizontal_centers",
            reference="canvas",
        )


def test_all_benchmark_tasks_roundtrip_with_applicable_fields_only():
    from tests.reward_support.benchmarks import BENCHMARKS

    for benchmark in BENCHMARKS.values():
        assert TaskSpec.model_validate(benchmark.task.model_dump()) == benchmark.task
        assert TaskSpec.model_validate_json(benchmark.task.model_dump_json()) == benchmark.task
