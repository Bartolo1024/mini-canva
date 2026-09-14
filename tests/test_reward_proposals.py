"""Validate offline comparison measurements, not a change to production reward semantics."""

import json
from copy import deepcopy

import pytest

from marketcanvas_env.reward.tasks import TaskSpec
from tests.reward_support.benchmarks import BENCHMARKS
from tests.reward_support.proposals import ProposalScene, compare, corrected, study_cases


@pytest.mark.parametrize("name", study_cases())
def test_experimental_results_are_repeatable_finite_and_do_not_mutate(name):
    state, task = study_cases()[name]
    before = deepcopy(state)
    first = compare(state, task)
    assert compare(state, task) == first
    assert state == before
    json.dumps(first, allow_nan=False)
    for variant in ("current", "product_only", "additive", "product", "minimum"):
        assert -1 <= first[variant] <= 1


@pytest.mark.parametrize("minimum", [0.01, 0.02, 0.04])
def test_size_sensitivity_preserves_references_but_rejects_microscopic_image(minimum):
    for benchmark in BENCHMARKS.values():
        assert compare(benchmark.state, benchmark.task, minimum_ratio=minimum)["additive"] == 1
    assert compare(*study_cases()["tiny_image"], minimum_ratio=minimum)["additive"] < 0


@pytest.mark.parametrize("cost", [0.025, 0.05, 0.1])
def test_duplicate_cost_sensitivity_keeps_spam_below_clean(cost):
    cases = study_cases()
    for name in (
        "one_duplicate",
        "two_duplicate",
        "color_hedge",
        "twenty_ctas",
        "readable_role_spam",
    ):
        assert compare(*cases[name], duplicate_cost=cost)["additive"] < 1


def test_reaggregation_alone_cannot_detect_missing_features():
    for name in ("ink_background", "readable_role_spam"):
        row = compare(*study_cases()[name])
        assert row["product_only"] == row["current"] == 1
        assert row["additive"] < row["current"]


def test_effective_contrast_uses_visible_not_buried_backgrounds():
    cases = study_cases()
    invisible = corrected(*cases["ink_background"])
    buried = corrected(*cases["buried_dark"])
    legitimate = corrected(*cases["giant_underlay"])
    assert invisible["quality"]["contrast"] == 0
    assert buried["quality"]["contrast"] == legitimate["quality"]["contrast"] == 1
    assert buried["penalty"] == pytest.approx(0.03)  # Extra panel, not contrast or large area.
    assert legitimate["penalty"] == 0


def test_joint_representative_prevents_color_content_collusion():
    state, task = study_cases()["split_properties"]
    report = corrected(state, task)
    scores = {c["id"]: c["score"] for c in report["constraints"]}
    assert scores["cta_yellow"] + scores["cta_text"] < 2
    assert report["duplicates"] == 1
    scene = ProposalScene(state, task)
    for c in task.constraints:
        if (
            c.selector is not None
            and c.selector.role == "cta"
            and c.kind not in {"count", "absent"}
        ):
            assert scene.select(c.selector)[0].id == report["matching"]["('role', 'cta')"]


def test_blank_first_does_not_block_useful_required_match():
    report = corrected(*study_cases()["blank_first"])
    assert report["matching"]["('role', 'headline')"] == "headline"
    assert report["task_score"] == 1
    assert report["quality_score"] == 0  # The blank duplicate still degrades the actual design.


def test_product_and_minimum_flatten_repair_that_additive_distinguishes():
    bad = compare(*study_cases()["repair_color_bad"])
    good = compare(*study_cases()["repair_color_good"])
    assert bad["components"]["quality_score"] == good["components"]["quality_score"] == 0
    assert bad["components"]["task_score"] < good["components"]["task_score"]
    assert bad["additive"] < good["additive"]
    assert bad["product"] == good["product"] == bad["minimum"] == good["minimum"] == -1


def test_explicit_count_budget_preserves_requested_duplicates():
    state, original = study_cases()["twenty_ctas_count"]
    task = original.model_dump(mode="json")
    next(c for c in task["constraints"] if c["kind"] == "count")["value"] = 20
    task = original.model_validate(task)
    report = corrected(state, task)
    assert report["duplicates"] == report["penalty"] == 0
    assert report["task_score"] == 1


def test_arbitrary_role_name_cannot_evade_normalized_size():
    state, original = study_cases()["tiny_logo"]
    state["elements"][0]["role"] = "custom_seal"
    task = original.model_dump(mode="json")
    task["constraints"][0]["selector"]["role"] = "custom_seal"
    assert compare(state, original.model_validate(task))["additive"] < 0


def test_binding_does_not_collapse_a_set_valued_no_overlap_constraint():
    state, original = study_cases()["tiny_logo"]
    state["elements"][0].update(width=100, height=100)
    state["elements"].append({**state["elements"][0], "id": "second"})
    data = original.model_dump(mode="json")
    data["constraints"].append(
        {"id": "separate", "kind": "no_overlap", "selector": {"role": "logo"}, "hard": True}
    )
    report = corrected(state, TaskSpec.model_validate(data))
    assert report["quality"]["overlap"] == 1  # Generic image exemption must not override the task.
    assert next(c["score"] for c in report["constraints"] if c["id"] == "separate") == 0
    assert report["gate"] == 0.4


def test_extra_visibility_cannot_compensate_for_below_threshold_size():
    state, task = study_cases()["tiny_logo"]
    state["elements"][0].update(width=800 * 0.016, height=600 * 0.016)
    report = corrected(state, task)
    assert report["constraints"][0]["score"] == pytest.approx(0.8)
    assert report["gate"] == 0.4
