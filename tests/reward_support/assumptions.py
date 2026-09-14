"""Export reproductions of accepted reward limitations; this is not a scoring test."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from marketcanvas_env.rendering import render_rgb, save_png
from marketcanvas_env.reward import compute_reward_breakdown
from marketcanvas_env.reward.scene import Scene
from tests.reward_support.benchmarks import BENCHMARKS


def main() -> None:
    benchmark = BENCHMARKS["summer_sale"]
    states = {}
    patch = deepcopy(benchmark.state)
    patch["elements"].append(
        {
            "id": "patch",
            "type": "shape",
            "role": "background",
            "x": 204,
            "y": 84,
            "width": 144,
            "height": 17,
            "z_index": 1,
            "color": "#111111",
            "content": "",
        }
    )
    states["ink_background"] = patch
    without_headline = deepcopy(patch)
    without_headline["elements"].pop(0)
    print(
        "Headline contributes no pixels:",
        np.array_equal(render_rgb(patch), render_rgb(without_headline)),
    )

    for role in ("decoration", "background"):
        state = deepcopy(benchmark.state)
        for index in range(2):
            state["elements"].append(
                {
                    "id": f"extra_{index}",
                    "type": "shape",
                    "role": role,
                    "x": 50,
                    "y": 400,
                    "width": 100,
                    "height": 100,
                    "z_index": index,
                    "color": "#FF0000",
                    "content": "",
                }
            )
        states["overlapping_" + role] = state

    wrong = deepcopy(benchmark.state)
    wrong["elements"][1]["color"] = "#4477FF"
    states["blue_cta"] = wrong
    duplicate = deepcopy(wrong)
    extra = deepcopy(benchmark.state["elements"][1])
    extra.update(id="a_correct_duplicate", x=50, y=350)
    duplicate["elements"].append(extra)
    states["selected_duplicate"] = duplicate

    clipped = deepcopy(benchmark.state)
    clipped["elements"][0]["width"] = 80
    states["clipped_headline"] = clipped

    directory = Path("artifacts/reward")
    directory.mkdir(parents=True, exist_ok=True)
    for name, state in states.items():
        state["target"] = benchmark.task.model_dump(mode="json")
        report = compute_reward_breakdown(state, benchmark.task)
        selected = [element.id for element in Scene(state).select({"role": "cta"})]
        path = directory / f"assumption_{name}.png"
        save_png(state, path)
        path.with_suffix(".json").write_text(
            json.dumps(
                {"state": state, "reward_breakdown": report, "cta_selection_order": selected},
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{path}: reward={report['reward']:.3f}, quality={report['quality']}")


if __name__ == "__main__":
    main()
