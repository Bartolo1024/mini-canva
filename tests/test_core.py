"""Observable canvas behavior, independent of reward and transport adapters."""

import json
import os
import subprocess
import sys
from copy import deepcopy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from marketcanvas_env.core import CanvasCore, LifecycleError
from marketcanvas_env.models import RequestError


def add(kind="text", **properties):
    return {"op": "add_element", "element": {"type": kind, **properties}}


def properties(record):
    return {k: v for k, v in record.items() if k not in {"id", "type", "subtype"}}


@pytest.fixture
def core():
    canvas = CanvasCore()
    canvas.reset()
    return canvas


def test_default_reset_snapshot(core):
    assert core.get_canvas_state() == {
        "schema_version": 1,
        "canvas": {"width": 800, "height": 600, "background": "#FFFFFF"},
        "limits": {"max_elements": 32, "max_steps": 64, "max_content_length": 256},
        "target": {"headline": "Summer Sale", "cta_text": "Shop Now", "cta_color": "#FFFF00"},
        "steps_taken": 0,
        "steps_remaining": 64,
        "next_element_id": 1,
        "status": "active",
        "elements": [],
        "relationships": [],
    }


@pytest.mark.parametrize(
    ("kind", "subtype", "role"),
    [
        ("text", None, "headline"),
        ("shape", "rectangle", "none"),
        ("shape", "button", "cta"),
        ("image", None, "none"),
    ],
)
def test_all_element_kinds_can_be_created_moved_and_edited(core, kind, subtype, role):
    result = core.apply_action(add(kind, subtype=subtype, role=role, content="Original"))
    element = result.state["elements"][0]
    assert element["id"] == 1
    assert element["subtype"] == subtype
    assert element["width"] == 200
    result = core.apply_action({"op": "move_element", "id": 1, "new_x": -20, "new_y": 110})
    updated = properties(result.state["elements"][0])
    updated.update(
        width=300,
        height=90,
        z_index=12,
        color="#abcDEF",
        text_color="#123abc",
        content="Revised content",
        font_size=36,
        text_align="right",
    )
    result = core.apply_action({"op": "update_element", "id": 1, "properties": updated})
    element = result.state["elements"][0]
    assert properties(element) == {**updated, "color": "#ABCDEF", "text_color": "#123ABC"}
    assert (element["type"], element["subtype"], element["id"]) == (kind, subtype, 1)
    assert element["x"] == -20
    assert result.action_applied and not result.terminated and not result.truncated


def test_banner_relationships_match_contract(core):
    core.apply_action(
        add(
            role="headline",
            x=100,
            y=100,
            width=600,
            height=80,
            content="Summer Sale",
            font_size=32,
            text_align="center",
        )
    )
    result = core.apply_action(
        add(
            "shape",
            subtype="button",
            role="cta",
            x=300,
            y=300,
            width=200,
            height=60,
            color="#FFFF00",
            content="Shop Now",
            text_align="center",
        )
    )
    assert result.state["relationships"] == [
        {"source_id": 1, "target_id": 2, "relation": "above"},
        {"source_id": 1, "target_id": 2, "relation": "center_aligned_x"},
        {"source_id": 2, "target_id": 1, "relation": "center_aligned_x"},
    ]


def test_deleted_ids_not_recycled_and_stacking_is_independent_of_serial_order(core):
    core.apply_action(add(z_index=5))
    core.apply_action(add("image", z_index=1))
    core.apply_action({"op": "delete_element", "id": 1})
    core.apply_action(add("shape", z_index=0))
    core.apply_action(add(z_index=1))
    state = core.get_canvas_state()
    assert [e["id"] for e in state["elements"]] == [2, 3, 4]
    assert state["next_element_id"] == 5
    assert [e["id"] for e in core.get_draw_order()] == [3, 2, 4]
    assert all(r["source_id"] != 1 and r["target_id"] != 1 for r in state["relationships"])


def test_capacity_failure_precedes_role_failure_and_does_not_allocate(core):
    for _ in range(32):
        core.apply_action(add())
    before = core.get_canvas_state()
    result = core.apply_action(add("image", role="headline"))
    assert result.error == "capacity_exceeded"
    assert not result.action_applied
    assert result.state == {**before, "steps_taken": 33, "steps_remaining": 31}
    core.apply_action({"op": "delete_element", "id": 8})
    result = core.apply_action(add("image"))
    assert result.state["elements"][-1]["id"] == 33


def test_semantic_update_failure_preserves_every_canvas_property(core):
    record = core.apply_action(add(content="Keep me")).state["elements"][0]
    before = core.get_canvas_state()
    replacement = {**properties(record), "role": "cta", "x": 100, "content": "Wrong"}
    result = core.apply_action({"op": "update_element", "id": 1, "properties": replacement})
    assert result.error == "incompatible_role"
    assert result.state == {**before, "steps_taken": 2, "steps_remaining": 62}
    result = core.apply_action({"op": "update_element", "id": 64, "properties": replacement})
    assert result.error == "unknown_element"


@pytest.mark.parametrize(
    "payload",
    [
        {"op": "move_element", "id": 1, "new_x": True, "new_y": 0},
        {"op": "move_element", "id": 1, "new_x": "0", "new_y": 0},
        {"op": "update_element", "id": 1, "properties": {"color": "#000000"}},
        {"op": "finish", "extra": 3},
        add("shape", subtype=None),
        add(width=0),
        add(content="Line\nbreak"),
        add(id=5),
        [],
    ],
)
def test_structural_errors_do_not_mutate_any_state(core, payload):
    core.apply_action(add())
    before = core.serialize_state()
    with pytest.raises(RequestError) as exc:
        core.apply_action(payload)
    assert exc.value.code == "invalid_request"
    assert core.serialize_state() == before


def test_validation_and_lifecycle_precedence():
    core = CanvasCore()
    with pytest.raises(RequestError):
        core.apply_action({"op": "not_an_action"})
    with pytest.raises(LifecycleError) as exc:
        core.apply_action({"op": "finish"})
    assert exc.value.code == "not_initialized"
    for read in (core.get_canvas_state, core.serialize_state, core.get_draw_order):
        with pytest.raises(LifecycleError):
            read()
    core.reset()
    core.apply_action({"op": "finish"})
    with pytest.raises(RequestError):
        core.apply_action({"op": "finish", "unexpected": 1})


@pytest.mark.parametrize("finish_last", [False, True])
def test_budget_and_finish_precedence_with_no_post_episode_mutation(core, finish_last):
    missing = {"op": "move_element", "id": 64, "new_x": 0, "new_y": 0}
    for _ in range(63):
        result = core.apply_action(missing)
        assert result.error == "unknown_element" and not result.terminated
    result = core.apply_action({"op": "finish"} if finish_last else missing)
    assert result.terminated and not result.truncated
    assert result.end_reason == ("finish" if finish_last else "budget_exhausted")
    assert result.action_applied is finish_last
    assert result.state["steps_remaining"] == 0
    assert result.state["next_element_id"] == 1
    snapshot = core.serialize_state()
    with pytest.raises(LifecycleError) as exc:
        core.apply_action({"op": "finish"})
    assert exc.value.code == "episode_done"
    assert core.serialize_state() == snapshot


def test_successful_mutation_on_last_attempt_is_in_terminal_snapshot(core):
    for _ in range(63):
        core.apply_action({"op": "delete_element", "id": 64})
    result = core.apply_action(add("image", content="Last"))
    assert result.terminated and result.action_applied
    assert result.state["elements"][0]["content"] == "Last"
    assert result.state["next_element_id"] == 2


def test_noop_is_an_attempt_and_early_finish_retains_remaining_budget(core):
    core.apply_action(add())
    core.apply_action({"op": "move_element", "id": 1, "new_x": 0, "new_y": 0})
    result = core.apply_action({"op": "finish"})
    assert result.state["steps_taken"] == 3
    assert result.state["steps_remaining"] == 61


def test_inputs_snapshots_and_instances_do_not_share_mutable_objects(core):
    other = CanvasCore()
    initial, _ = other.reset()
    payload = add(content="Original")
    result = core.apply_action(payload)
    core.apply_action(add("image"))
    before = core.serialize_state()
    payload["element"]["content"] = "Changed input"
    result.state["elements"][0]["content"] = "Changed old transition"
    snap = core.get_canvas_state()
    snap["target"]["headline"] = "Changed target"
    snap["canvas"]["width"] = 1
    snap["limits"]["max_steps"] = 999
    snap["elements"][0]["x"] = 999
    snap["relationships"][0]["relation"] = "broken"
    core.get_draw_order()[0]["content"] = "Changed draw record"
    assert core.serialize_state() == before
    assert other.get_canvas_state() == initial


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"seed": True}, "invalid_request"),
        ({"seed": -1}, "invalid_request"),
        ({"options": {"target": None}}, "invalid_request"),
        ({"options": {"prompt": "Unsupported"}}, "unsupported_prompt"),
        ({"options": {"target": {}, "prompt": "Unsupported"}}, "invalid_request"),
    ],
)
def test_invalid_reset_preserves_active_or_terminal_episode(core, kwargs, code):
    core.apply_action(add())
    for finished in (False, True):
        if finished:
            core.apply_action({"op": "finish"})
        before = core.serialize_state()
        with pytest.raises(RequestError) as exc:
            core.reset(**kwargs)
        assert exc.value.code == code
        assert core.serialize_state() == before


def test_reset_restores_default_and_does_not_retain_target_input(core):
    target = {"headline": " Autumn Sale ", "cta_text": "Browse Deals", "cta_color": "#00aaff"}
    state, info = core.reset(seed=7, options={"target": target})
    assert info == {"target_source": "structured"}
    assert state["target"] == {**target, "headline": "Autumn Sale", "cta_color": "#00AAFF"}
    target["headline"] = "Changed"
    core.apply_action(add())
    state, info = core.reset(seed=7)
    assert info == {"target_source": "default"}
    assert state["target"]["headline"] == "Summer Sale"
    assert state["elements"] == [] and state["next_element_id"] == 1
    assert state["steps_taken"] == 0 and state["status"] == "active"


positions = st.tuples(st.integers(-800, 1599), st.integers(-600, 1199))
actions = st.one_of(
    st.builds(
        lambda kind, xy: add(kind, x=xy[0], y=xy[1]),
        st.sampled_from(["text", "shape", "image"]),
        positions,
    ),
    st.builds(
        lambda eid, xy: {"op": "move_element", "id": eid, "new_x": xy[0], "new_y": xy[1]},
        st.integers(1, 64),
        positions,
    ),
    st.builds(lambda eid: {"op": "delete_element", "id": eid}, st.integers(1, 64)),
)


@settings(max_examples=40, deadline=None)
@given(st.lists(actions, min_size=1, max_size=64), st.integers(0, 4294967295))
def test_generated_replay_and_reset_are_deterministic(sequence, seed):
    left, right = CanvasCore(), CanvasCore()
    left.reset(seed=seed)
    right.reset(seed=(seed + 1) % 4294967296)
    first_pass = []
    for action in sequence:
        before = left.get_canvas_state()
        a, b = left.apply_action(deepcopy(action)), right.apply_action(deepcopy(action))
        assert a == b
        assert left.serialize_state() == right.serialize_state()
        if a.error:
            assert a.state["elements"] == before["elements"]
            assert a.state["next_element_id"] == before["next_element_id"]
        assert len(a.state["elements"]) <= 32
        assert len({e["id"] for e in a.state["elements"]}) == len(a.state["elements"])
        assert json.loads(left.serialize_state()) == a.state
        first_pass.append(left.serialize_state())
    left.reset(seed=seed)
    for action, expected in zip(sequence, first_pass, strict=True):
        left.apply_action(action)
        assert left.serialize_state() == expected


def test_serialization_is_stable_across_process_hash_seeds():
    code = """
from marketcanvas_env.core import CanvasCore
c = CanvasCore()
c.reset(seed=5)
c.apply_action({'op':'add_element','element':{'type':'text','content':'Hello'}})
c.apply_action({'op':'add_element','element':{'type':'image','x':200}})
print(c.serialize_state())
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}, text=True
        )
        for seed in ("1", "987")
    ]
    assert outputs[0] == outputs[1]
    state = json.loads(outputs[0])
    assert outputs[0].strip() == json.dumps(state, sort_keys=True, separators=(",", ":"))
