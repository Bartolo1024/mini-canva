"""Replay the compact public task/trajectory set and print computed rewards."""

from marketcanvas_env.task_data import replay_trajectory, trajectory_paths


def main() -> None:
    print(f"{'Task / trajectory':<40} {'T':>7} {'Q':>7} {'Gate':>7} {'Reward':>8}")
    print("-" * 73)
    for path in trajectory_paths():
        result = replay_trajectory(path)
        report = result["reward_breakdown"]
        label = f"{result['task']}/{path.stem}"
        print(
            f"{label:<40} {report['task_score']:7.3f} {report['quality_score']:7.3f} "
            f"{report['hard_gate']:7.3f} {report['reward']:8.3f}"
        )


if __name__ == "__main__":
    main()
