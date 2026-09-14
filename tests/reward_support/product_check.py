"""Compare literal and normalized product rewards using the same production measurements."""

import argparse
import json
from copy import deepcopy

from marketcanvas_env.reward import compute_reward_breakdown
from tests.reward_support.proposals import study_cases

# Constructed behavioral expectations, not labeled design-quality accuracy.
CHECKS = [
    ("valid_summer", ">", "wrong_cta_color"),
    ("valid_summer", ">", "bad_contrast"),
    ("valid_required", ">", "tiny_required"),
    ("valid_summer", ">", "tiny_roles"),
    ("valid_summer", ">", "readable_role_spam"),
    ("valid_summer", ">", "twenty_ctas"),
    ("single_cta_count", ">", "twenty_ctas_count"),
    ("valid_summer", ">", "color_hedge"),
    ("valid_summer", ">", "offscreen"),
    ("valid_summer", ">", "mostly_offscreen"),
    ("valid_summer", ">", "giant_cta"),
    ("valid_event", ">", "center_stack"),
    ("collision", "=", "collision_decoys"),
    ("valid_summer", ">", "missing_cta"),
    ("valid_summer", ">", "wrong_type"),
    ("two_column", ">", "centered_two_column"),
    ("newsletter", ">", "newsletter_cta"),
    ("normal_image", ">", "tiny_image"),
    ("normal_logo", ">", "tiny_logo"),
    ("valid_summer", ">", "ink_background"),
    ("repair_color_good", ">", "repair_color_bad"),
]
VARIANTS = ("current", "literal_product", "normalized_product")


def evaluate():
    cases = study_cases()
    for kind in ("image", "logo"):
        state, task = cases["tiny_" + kind]
        state = deepcopy(state)
        state["elements"][0].update(width=100, height=80)
        cases["normal_" + kind] = state, task
    results = {}
    for name, (state, task) in cases.items():
        report = compute_reward_breakdown(state, task)
        t, q, g = (report[k] for k in ("task_score", "quality_score", "hard_gate"))
        results[name] = {
            "T": t,
            "Q": q,
            "G": g,
            "current": report["reward"],
            "literal_product": g * t * q - 1,
            "normalized_product": 2 * g * t * q - 1,
        }
    checks = {}
    for variant in VARIANTS:
        failures = []
        for left, relation, right in CHECKS:
            a, b = results[left][variant], results[right][variant]
            passed = a > b + 1e-12 if relation == ">" else abs(a - b) <= 1e-12
            if not passed:
                failures.append(f"{left} {relation} {right}: {a:.6f} vs {b:.6f}")
        checks[variant] = {
            "passed": len(CHECKS) - len(failures),
            "total": len(CHECKS),
            "failures": failures,
        }
    return {"cases": results, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print("All variants use the plain mean of TaskSpec constraint scores for T.\n")
    print("| Scenario | Current | G T Q − 1 | 2 G T Q − 1 |")
    print("| --- | ---: | ---: | ---: |")
    for name, row in result["cases"].items():
        print("| " + name + " | " + " | ".join(f"{row[v]:+.3f}" for v in VARIANTS) + " |")
    print("\nConstructed behavioral checks (not statistical accuracy):")
    for variant, report in result["checks"].items():
        print(f"\n{variant}: {report['passed']}/{report['total']}")
        for failure in report["failures"]:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
