"""Replay data/trajectories, print rewards, and optionally save PNGs and JSON traces."""

import argparse
import json
from pathlib import Path

from marketcanvas_env.rendering import save_png
from marketcanvas_env.task_data import replay_trajectory, tasks_directory, trajectory_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=[p.stem for p in sorted(tasks_directory().glob("*.yaml"))],
        help="task filename stem (default: all tasks)",
    )
    parser.add_argument("--trajectory", help="trajectory filename stem; requires --task")
    parser.add_argument(
        "--output-dir", type=Path, help="save PNG and JSON files under this directory"
    )
    args = parser.parse_args()
    if args.trajectory and not args.task:
        parser.error("--trajectory requires --task")
    paths = trajectory_paths(args.task)
    if args.trajectory:
        paths = tuple(p for p in paths if p.stem == args.trajectory)
    if not paths:
        parser.error("no matching trajectories")

    print(f"{'Task / trajectory':<40} {'T':>7} {'Q':>7} {'Gate':>7} {'Reward':>8}")
    print("-" * 73)
    for path in paths:
        result = replay_trajectory(path)
        report = result["reward_breakdown"]
        label = f"{result['task']}/{path.stem}"
        print(
            f"{label:<40} {report['task_score']:7.3f} {report['quality_score']:7.3f} "
            f"{report['hard_gate']:7.3f} {report['reward']:8.3f}"
        )
        if args.output_dir is not None:
            output = args.output_dir / result["task"] / f"{path.stem}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            save_png(result["state"], output)
            output.with_suffix(".json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
