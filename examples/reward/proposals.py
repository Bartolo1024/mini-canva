"""Offline reward proposals: production reward modules and TaskSpec are untouched."""

import argparse
import json
import math
from copy import copy, deepcopy
from dataclasses import replace
from itertools import pairwise

from examples.reward.adversarial import scenarios
from examples.reward.benchmarks import BENCHMARKS
from marketcanvas_env.reward import compute_reward_breakdown
from marketcanvas_env.reward.colors import contrast_ratio
from marketcanvas_env.reward.constraints import evaluate_constraint
from marketcanvas_env.reward.geometry import Rect, intersection
from marketcanvas_env.reward.quality import evaluate_quality, harmonic_mean
from marketcanvas_env.reward.scene import Scene, text_ink_box
from marketcanvas_env.reward.tasks import TaskSpec


def clip(value):
    return max(0.0, min(1.0, value))


def key(selector):
    """Primary requirements use distinct IDs or roles; type-only is a fallback."""
    for field in ("id", "role", "type"):
        if (value := getattr(selector, field, None)) is not None:
            return field, value
    return "all", "all"


def paint(element):
    if not element.valid or element.rect is None:
        return None
    if element.type != "text":
        return element.rect
    ink = text_ink_box(element)
    rect = element.rect
    if ink is None:
        return None
    return intersection(ink, Rect(rect.x + 4, rect.y + 4, rect.width - 8, rect.height - 8))


def visible_ink_contrast(scene, element):
    """Worst topmost background color beneath visible rectangular ink, without rasterizing."""
    if not element.valid or not element.text_bearing or not element.content.strip():
        return 0.0
    if element.type != "text":
        return contrast_ratio(element.text_color, element.color)
    region = paint(element)
    if region is None or scene.canvas is None:
        return 0.0
    region = intersection(region, scene.canvas)
    if region is None:
        return 0.0
    surfaces = []
    target_z = (element.z_index, element.stack_id)
    for other in scene._all:
        rect = paint(other)
        if other.id == element.id or rect is None:
            continue
        order = (other.z_index, other.stack_id)
        # Lower text does not become a solid background; upper text occludes approximated ink.
        if order < target_z and other.type == "text":
            continue
        if clipped := intersection(region, rect):
            surfaces.append((other, clipped, order))
    xs = sorted({region.x, region.right, *(x for _, r, _ in surfaces for x in (r.x, r.right))})
    ys = sorted({region.y, region.bottom, *(y for _, r, _ in surfaces for y in (r.y, r.bottom))})
    ratios = []
    for left, right in pairwise(xs):
        for top, bottom in pairwise(ys):
            x, y = (left + right) / 2, (top + bottom) / 2
            covering = [
                (e, order) for e, r, order in surfaces if r.x <= x < r.right and r.y <= y < r.bottom
            ]
            if any(order > target_z for _, order in covering):
                continue
            background = (
                max(covering, key=lambda item: item[1])[0].color if covering else scene.background
            )
            ratios.append(contrast_ratio(element.text_color, background))
    return min(ratios, default=0.0)


class ProposalScene(Scene):
    """Experimental signals with one representative per explicitly required ID/role."""

    def __init__(self, state, task, minimum_ratio=0.02):
        super().__init__(state)
        self.original_usability = {e.id: e.usable_ratio for e in self.elements}
        self.sizes = {
            e.id: min(
                1.0,
                e.rect.width / self.width / minimum_ratio,
                e.rect.height / self.height / minimum_ratio,
            )
            if e.valid and e.rect
            else 0.0
            for e in self.elements
        }
        self.elements = tuple(
            replace(e, usable_ratio=e.usable_ratio * self.sizes[e.id]) for e in self.elements
        )
        self._all = self.elements
        self.bound = {}
        self.binding = False
        self.contrasts = {
            e.id: visible_ink_contrast(self, e) for e in self.elements if e.text_bearing
        }
        anchors = [c for c in task.constraints if c.kind == "exists"]
        for anchor in anchors:
            identity = key(anchor.selector)
            if identity in self.bound:
                raise ValueError(
                    "Proposal requires distinct exists IDs/roles; use count for cardinality."
                )
            local = [
                c
                for c in task.constraints
                if c.selector is not None
                and key(c.selector) == identity
                and c.kind not in {"count", "absent", "no_overlap"}
            ]
            candidates = Scene.select(self, anchor.selector)

            def rank(element, local=local):
                singleton = copy(self)
                singleton.elements = (element,)
                score = math.fsum(singleton.score(c) for c in local) / len(local)
                return -score, -element.usable_ratio, -element.visible_ratio, element.id

            self.bound[identity] = min(candidates, key=rank) if candidates else None
        self.binding = True

    def select(self, selector):
        matches = Scene.select(self, selector)
        identity = key(selector)
        if self.binding and identity in self.bound:
            chosen = self.bound[identity]
            return [e for e in matches if chosen is not None and e.id == chosen.id]
        return sorted(matches, key=lambda e: (-e.usable_ratio, -e.visible_ratio, e.id))

    def contrast(self, element):
        return self.contrasts.get(element.id, 0.0)

    def score(self, constraint):
        if constraint.kind in {"count", "absent"} or (
            constraint.kind == "no_overlap" and constraint.selector is not None
        ):
            raw = copy(self)
            raw.binding = False
            return evaluate_constraint(constraint, raw)[0]
        if constraint.kind == "exists":
            matches = self.select(constraint.selector)
            return max(
                (
                    self.sizes[e.id]
                    * clip(self.original_usability[e.id] / constraint.min_visible_ratio)
                    for e in matches
                ),
                default=0.0,
            )
        return evaluate_constraint(constraint, self)[0]


def corrected(state, task, minimum_ratio=0.02, duplicate_cost=0.05, extra_cost=0.03):
    scene = ProposalScene(state, task, minimum_ratio)
    rows = [{"id": c.id, "score": scene.score(c), "hard": c.hard} for c in task.constraints]
    t = math.fsum(c["score"] for c in rows) / len(rows) if rows else 0.0
    _, quality = evaluate_quality(scene)
    # Keep the true WCAG ratio intact; normalize zero distinguishability to zero quality.
    quality["contrast"] = min(
        (
            clip((scene.contrast(e) - 1) / 3.5) * e.usable_ratio
            for e in scene.elements
            if e.text_bearing
        ),
        default=1.0,
    )
    q = harmonic_mean(list(quality.values()))
    used = {e.id for e in scene.bound.values() if e is not None}
    eligible = {
        e.id
        for c in task.constraints
        if c.kind == "exists"
        for e in Scene.select(scene, c.selector)
    }
    # Explicit cardinality admits that many candidates; it does not require picking one.
    for c in task.constraints:
        if c.kind == "count" and c.operator in {"eq", "gte"}:
            matches = Scene.select(scene, c.selector)
            eligible.update(e.id for e in matches)
            remaining = max(0, int(c.value) - sum(e.id in used for e in matches))
            for e in matches:
                if e.id not in used and remaining > 0:
                    used.add(e.id)
                    remaining -= 1
    duplicates = len(eligible - used)
    extras = sum(e.id not in eligible and e.id not in used for e in scene.elements)
    penalty = min(0.25, duplicate_cost * duplicates + extra_cost * max(0, extras - 1))
    gate = 0.4 if any(row["hard"] and row["score"] < 1 for row in rows) else 1.0
    return {
        "task_score": t,
        "quality_score": q,
        "quality": quality,
        "constraints": rows,
        "gate": gate,
        "penalty": penalty,
        "duplicates": duplicates,
        "extras": extras,
        "matching": {str(k): e.id if e else None for k, e in scene.bound.items()},
    }


def compare(state, task, **parameters):
    old = compute_reward_breakdown(state, task)
    fixed = corrected(state, task, **parameters)
    t, q, p, g = (fixed[k] for k in ("task_score", "quality_score", "penalty", "gate"))

    def reward(value, gate=g):
        return 2 * clip(gate * value) - 1

    return {
        "current": old["reward"],
        "product_only": reward(old["task_score"] * old["quality_score"], old["hard_gate"]),
        "additive": reward((t + q) / 2 - p),
        "product": reward(t * q - p),
        "minimum": reward(min(t, q) - p),
        "components": fixed,
    }


def study_cases():
    cases = scenarios()
    for name, benchmark in BENCHMARKS.items():
        cases["reference_" + name] = deepcopy(benchmark.state), benchmark.task
    base = BENCHMARKS["summer_sale"]
    for name in (
        "blank_first",
        "one_duplicate",
        "two_duplicate",
        "buried_dark",
        "repair_color_bad",
        "repair_color_good",
    ):
        state = deepcopy(base.state)
        if name == "blank_first":
            state["elements"].append(
                {**state["elements"][0], "id": "a_blank", "y": 400, "content": ""}
            )
        elif "duplicate" in name:
            for i in range(1 if name == "one_duplicate" else 2):
                state["elements"].append(
                    {**state["elements"][1], "id": f"extra{i}", "x": 30 + 300 * i, "y": 400}
                )
        elif name == "buried_dark":
            for i, color in enumerate(("#111111", "#FFFFFF")):
                state["elements"].append(
                    {
                        "id": f"panel{i}",
                        "role": "background",
                        "type": "shape",
                        "content": "",
                        "color": color,
                        "x": 200,
                        "y": 80,
                        "width": 400,
                        "height": 80,
                        "z_index": i,
                    }
                )
        else:
            # Unrelated collision keeps Q=0 while correcting the CTA still improves T.
            for i in range(2):
                state["elements"].append(
                    {
                        **state["elements"][0],
                        "id": f"collision{i}",
                        "role": "caption",
                        "x": 0,
                        "y": 450,
                        "width": 300,
                        "content": "Collision",
                    }
                )
            if name == "repair_color_bad":
                state["elements"][1]["color"] = "#4477FF"
        cases[name] = state, base.task
    # One yellow/wrong-text CTA and one blue/right-text CTA must not jointly satisfy both.
    state = deepcopy(base.state)
    state["elements"][1]["content"] = "Wrong"
    state["elements"].append(
        {
            **state["elements"][1],
            "id": "second",
            "x": 30,
            "y": 400,
            "color": "#4477FF",
            "content": "Shop Now",
        }
    )
    data = base.task.model_dump(mode="json")
    data["constraints"].append(
        {"id": "cta_text", "kind": "text_equals", "selector": {"role": "cta"}, "value": "Shop Now"}
    )
    cases["split_properties"] = state, TaskSpec.model_validate(data)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = {name: compare(*case) for name, case in study_cases().items()}
    if args.json:
        print(json.dumps(results, indent=2, allow_nan=False))
        return
    print("| Scenario | Current | Product only | Fixed additive | Fixed product | Fixed minimum |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, row in results.items():
        print(
            "| "
            + name
            + " | "
            + " | ".join(
                f"{row[k]:+.3f}"
                for k in ("current", "product_only", "additive", "product", "minimum")
            )
            + " |"
        )


if __name__ == "__main__":
    main()
