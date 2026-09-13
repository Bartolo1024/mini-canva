"""Reward equations, in execution order. Start review at compute_reward_breakdown.

(1) s_i = clip(evaluate_constraint(c_i, scene), 0, 1)
(2) T = mean(s_i)
(3) Q = harmonic_mean(B, O, C, V)
(4) G = 1 if all hard constraints score 1, otherwise configured failure multiplier
(5) R = clip(G * (T + Q) - 1, -1, 1)

TaskSpec supplies requirements; prompt prose is never interpreted by scoring.
See docs/REWARD_EQUATIONS.md for the equation-to-function map and edge conventions.
"""

import math
from typing import Any

from marketcanvas_env.config import config
from marketcanvas_env.reward.constraints import evaluate_constraint
from marketcanvas_env.reward.quality import evaluate_quality
from marketcanvas_env.reward.scene import Scene
from marketcanvas_env.reward.tasks import TaskSpec


def compute_reward_breakdown(state: Any, task: TaskSpec | dict) -> dict[str, Any]:
    """Compute s_i -> T -> Q -> G -> R, retaining each component for inspection.

    R = clip(G * (T + Q) - 1, -1, 1).
    Malformed canvas data is scored defensively; invalid TaskSpecs raise ValueError.
    This read is pure and does not issue an episode return or advance the canvas.
    """
    spec = task if isinstance(task, TaskSpec) else TaskSpec.model_validate(task)
    scene = Scene(state)

    constraints = evaluate_task_constraints(spec, scene)
    task_score = task_satisfaction(constraints)
    quality_score, quality = evaluate_quality(scene)
    gate = hard_constraint_gate(constraints)
    score_01, reward = aggregate_reward(task_score, quality_score, gate)

    return {
        "reward": reward,
        "score_01": score_01,
        "task_score": task_score,
        "quality_score": quality_score,
        "hard_gate": gate,
        "constraints": constraints,
        "quality": quality,
    }


def evaluate_task_constraints(spec: TaskSpec, scene: Scene) -> list[dict[str, Any]]:
    """Equation (1): s_i = clip(e_i, 0, 1); nonfinite e_i becomes 0.

    Evaluate every authored constraint once, in TaskSpec order. Preserve its hard
    flag and explanation; neither missing dependencies nor zero scores are omitted.
    """
    reports = []
    for constraint in spec.constraints:
        score, explanation = evaluate_constraint(constraint, scene)
        score = max(0.0, min(1.0, score)) if math.isfinite(score) else 0.0
        reports.append(
            {
                "id": constraint.id,
                "kind": constraint.kind,
                "score": score,
                "hard": constraint.hard,
                "explanation": explanation,
            }
        )
    return reports


def task_satisfaction(constraints: list[dict[str, Any]]) -> float:
    """Equation (2): T = sum(s_i) / n, or 0 when n = 0.

    Each distinct authored requirement contributes equally; no task weights.
    """
    return math.fsum(row["score"] for row in constraints) / len(constraints) if constraints else 0.0


def hard_constraint_gate(constraints: list[dict[str, Any]]) -> float:
    """Equation (4): G = 1 if every hard s_i = 1; otherwise G = g_fail.

    With no hard constraints the condition is satisfied. YAML restricts g_fail
    to [0, 0.5], so a hard failure cannot earn a positive reward.
    """
    all_hard_pass = all(row["score"] == 1.0 for row in constraints if row["hard"])
    return 1.0 if all_hard_pass else config.reward.hard_failure_gate


def aggregate_reward(task_score: float, quality_score: float, gate: float) -> tuple[float, float]:
    """Equation (5): R = clip(G * (T + Q) - 1, -1, 1).

    The diagnostic normalized score is S = G * (T + Q) / 2. Return (S, R).
    Keep the same arithmetic order as the documented additive reward.
    """
    gated_total = gate * (task_score + quality_score)
    score_01 = gated_total / 2
    reward = max(-1.0, min(1.0, gated_total - 1))
    return score_01, reward


def compute_reward(state: Any, task: TaskSpec | dict) -> float:
    """Return R = clip(G * (T + Q) - 1, -1, 1), without the diagnostic report."""
    return compute_reward_breakdown(state, task)["reward"]
