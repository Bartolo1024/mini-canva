"""Replay public trajectories and export their final images and computed traces."""

import argparse
import json
from pathlib import Path

from marketcanvas_env.rendering import save_png
from marketcanvas_env.task_data import replay_trajectory, tasks_directory, trajectory_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument(
        "--benchmark",
        choices=[p.stem for p in sorted(tasks_directory().glob("*.yaml"))],
        help="task to render (default: default_task)",
    )
    choice.add_argument("--all", action="store_true", help="all 15 representative trajectories")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reward"))
    args = parser.parse_args()
    paths = trajectory_paths() if args.all else trajectory_paths(args.benchmark or "default_task")
    for source in paths:
        result = replay_trajectory(source)
        output = args.output_dir / result["task"] / f"{source.stem}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        save_png(result["state"], output)
        output.with_suffix(".json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"{output}: reward={result['reward_breakdown']['reward']:.3f}")


if __name__ == "__main__":
    main()
