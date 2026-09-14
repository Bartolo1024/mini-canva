"""Measurements of the archival regression matrix; never changes runtime scoring.

Run ``python -m tests.reward_support.adversarial [--json | --markdown]``.
These are raw reward snapshots, not action traces. In particular, offscreen and
malformed snapshots deliberately exercise inputs rejected by the canvas core.
"""

import argparse
import json
import math
from typing import Any

from marketcanvas_env.reward import compute_reward_breakdown
from marketcanvas_env.reward.scene import Scene
from marketcanvas_env.reward.tasks import TaskSpec
from tests.reward_support.regression_data import load_catalog

Scenario = tuple[dict[str, Any], TaskSpec]


def scenarios() -> dict[str, Scenario]:
    """Load independent fresh snapshots from the private regression matrix."""
    return load_catalog("adversarial")


NOTES = {
    "scope": "Raw reward snapshots; offscreen/malformed cases need not be core-reachable. No reward changes.",
    "penalty": "n/a: current implementation has no extraneous-element penalty or matching assignment.",
    "quality": "quality.contrast combines contrast and text usability; separate readability is not emitted.",
    "presence": "Exists is a Boolean usable-visibility threshold; normalized minimum object size is absent.",
    "json": "Nonfinite input coordinates are encoded as strings NaN/Infinity in state for strict JSON; evaluated as floats.",
    "twenty_ctas": "Twenty readable, non-overlapping CTAs; compare twenty_ctas_count for explicit hard count.",
    "center_stack": "Raw centers coincide; task alignment also multiplies by occlusion-sensitive usability.",
    "collision_decoys": "32 elements total: catastrophic foreground pair and 30 disjoint decoys.",
    "giant_backgrounds": "Two full-canvas black underlays behind white headline text: contrast is valid, but complexity and image aesthetics are unscored.",
    "missing_content": "Missing content defaults to empty; geometry validity may remain perfect while usability fails.",
    "readable_role_spam": "Non-overlapping 145x40 boxes, font 18, readable role names; above glyph usability cutoff.",
}


def measurements() -> dict[str, dict[str, Any]]:
    """Return actual evaluator reports alongside reproducible scenario inputs."""
    rows = {}
    for name, (state, task) in scenarios().items():
        scene = Scene(state)
        rows[name] = {
            "state": state,
            "task": task.model_dump(mode="json"),
            "breakdown": compute_reward_breakdown(state, task),
            # Review-only measurements, not new reward components or matching assignments.
            "inspection": {
                "elements": [
                    {
                        "id": e.id,
                        "valid": e.valid,
                        "width_ratio": e.rect.width / scene.width if e.rect else None,
                        "height_ratio": e.rect.height / scene.height if e.rect else None,
                        "area_ratio": e.rect.area / scene.canvas.area
                        if e.rect and scene.canvas
                        else None,
                        "visible_ratio": e.visible_ratio,
                        "usable_ratio": e.usable_ratio,
                        "raw_contrast_ratio": scene.contrast(e) if e.text_bearing else None,
                    }
                    for e in scene.elements
                ],
                "single_selector_candidate_order": {
                    c.id: [e.id for e in scene.select(c.selector)]
                    for c in task.constraints
                    if c.selector is not None
                },
            },
        }
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--json", action="store_true", help="Include input states, TaskSpecs and complete reports"
    )
    group.add_argument(
        "--markdown", action="store_true", help="Print the measured table in Markdown"
    )
    args = parser.parse_args()
    rows = measurements()
    if args.json:
        print(
            json.dumps(_json_safe({"notes": NOTES, "scenarios": rows}), indent=2, allow_nan=False)
        )
        return
    if args.markdown:
        print("| Scenario | T | Q | Penalty | Gate | Reward |")
        print("| --- | ---: | ---: | --- | ---: | ---: |")
    else:
        print(f"{'Scenario':<25} {'T':>7} {'Q':>7} {'Penalty':>7} {'Gate':>7} {'Reward':>8}")
        print("-" * 68)
    for name, data in rows.items():
        r = data["breakdown"]
        if args.markdown:
            print(
                f"| {name} | {r['task_score']:.3f} | {r['quality_score']:.3f} | n/a | "
                f"{r['hard_gate']:.3f} | {r['reward']:+.3f} |"
            )
        else:
            print(
                f"{name:<25} {r['task_score']:7.3f} {r['quality_score']:7.3f} "
                f"{'n/a':>7} {r['hard_gate']:7.3f} {r['reward']:+8.3f}"
            )
    print("\nPenalty n/a: no clutter penalty is implemented. Gate is the actual multiplier.")
    print("Raw snapshots include malformed inputs; use --json for full evidence and scope notes.")


if __name__ == "__main__":
    main()
