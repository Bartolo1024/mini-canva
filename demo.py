"""Run a reproducible Summer Sale canvas episode and print its state and reward."""

import argparse
import json
from pathlib import Path

from marketcanvas_env.codec import encode_action
from marketcanvas_env.env import MarketCanvasEnv
from marketcanvas_env.models import DEFAULT_PROMPT, RequestError
from marketcanvas_env.task_data import load_task


def run_demo(prompt: str | None = None, png: Path | None = None) -> dict:
    """Use the Gymnasium interface; rendering is optional and never changes the score."""
    env = MarketCanvasEnv()
    try:
        options = load_task().model_dump(mode="json") if prompt is None else {"prompt": prompt}
        env.reset(options=options)
        # A fixed Summer Sale policy using the default YAML's authored constraints.
        elements = [
            {
                "type": "text",
                "role": "headline",
                "content": "Summer Sale",
                "x": 100,
                "y": 80,
                "width": 600,
                "height": 80,
                "font_size": 40,
                "text_align": "center",
            },
            {
                "type": "image",
                "role": "product",
                "content": "Product image placeholder",
                "x": 300,
                "y": 200,
                "width": 200,
                "height": 100,
                "color": "#AAAAAA",
            },
            {
                "type": "shape",
                "subtype": "button",
                "role": "cta",
                "content": "Shop Now",
                "x": 300,
                "y": 350,
                "width": 200,
                "height": 60,
                "color": "#FFFF00",
                "text_color": "#000000",
                "text_align": "center",
            },
        ]
        actions = [{"op": "add_element", "element": element} for element in elements]
        actions.append({"op": "finish"})
        trace = []
        for step, action in enumerate(actions, start=1):
            _, reward, terminated, truncated, info = env.step(encode_action(action))
            trace.append(
                {
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            if not info["action_applied"] or (terminated and action["op"] != "finish"):
                raise RuntimeError("Demo requires capacity for three elements and four attempts.")
        if png is not None:
            png.parent.mkdir(parents=True, exist_ok=True)
            env.save_png(png)
        return {
            "final_reward": reward,
            "steps": trace,
            "state": env.get_canvas_state(),
            "reward_breakdown": info["reward_breakdown"],
            "terminated": terminated,
            "truncated": truncated,
        }
    finally:
        env.close()


def print_summary(result: dict, png: Path | None) -> None:
    """Make sparse rewards visible before the full inspectable state and report."""
    print("MarketCanvas demo — fixed example policy")
    print("Task:", result["state"]["target"]["prompt"])
    for step in result["steps"]:
        action = step["action"]
        label = action["op"]
        if "element" in action:
            element = action["element"]
            label += f" {element['type']}/{element['role']}"
        print(f"Step {step['step']}: {label} | reward={step['reward']:.3f}")
    report = result["reward_breakdown"]
    print(
        f"Final reward: {result['final_reward']:.3f} | "
        f"terminated={result['terminated']} | truncated={result['truncated']}"
    )
    print(
        f"T={report['task_score']:.3f} Q={report['quality_score']:.3f} "
        f"gate={report['hard_gate']:.3f}"
    )
    print("Quality:", ", ".join(f"{key}={value:.3f}" for key, value in report["quality"].items()))
    print("Reward measures the implemented task/quality checks, not professional visual quality.")
    if png is not None:
        print("PNG:", png)
    print("\nFinal state:")
    print(json.dumps(result["state"], indent=2, allow_nan=False))
    print("\nReward breakdown (including per-constraint explanations):")
    print(json.dumps(report, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "This demo uses a fixed example policy: the same Summer Sale layout. "
            "Default constraints come from YAML. Prompt-only parsing is not implemented."
        ),
    )
    parser.add_argument(
        "--prompt",
        help=f'future prompt parsing (not implemented); default YAML describes "{DEFAULT_PROMPT}"',
    )
    parser.add_argument(
        "--png",
        type=Path,
        metavar="PATH",
        help="also export a PNG; creates parent directories and overwrites the file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object with action trace, final state, and reward breakdown",
    )
    args = parser.parse_args()
    try:
        result = run_demo(args.prompt, args.png)
    except RequestError as error:
        parser.error(f"{error.code}: {error}")
    except NotImplementedError as error:
        parser.error(f"not_implemented: {error}")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"demo: {error}\n")
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print_summary(result, args.png)


if __name__ == "__main__":
    main()
