"""Deterministic canvas state machine shared by future RL and MCP adapters.

This layer produces unscored transitions. Reward evaluation is a separate concern;
no placeholder score is emitted for an unfinished reward implementation.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

from marketcanvas_env.config import config
from marketcanvas_env.geometry import spatial_relationships, stacking_order
from marketcanvas_env.models import (
    AddAction,
    DeleteAction,
    Element,
    FinishAction,
    MoveAction,
    Target,
    UpdateAction,
    parse_action,
    validate_reset,
)
from marketcanvas_env.reward.tasks import TaskSpec

SemanticError = Literal["unknown_element", "capacity_exceeded", "incompatible_role"]
EndReason = Literal["finish", "budget_exhausted"]
Status = Literal["active", "finished", "budget_exhausted"]


class LifecycleError(RuntimeError):
    """A valid operation cannot run in the current episode lifecycle."""

    def __init__(self, code: Literal["not_initialized", "episode_done"]) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CanvasTransition:
    """Detached state plus transition metadata, before the scoring layer is applied."""

    state: dict[str, Any]
    action_applied: bool
    error: SemanticError | None
    end_reason: EndReason | None

    @property
    def terminated(self) -> bool:
        return self.end_reason is not None

    @property
    def truncated(self) -> bool:
        return False


def _compatible_role(element: Element) -> bool:
    return (
        element.role not in {"headline", "cta"}
        or (element.role == "headline" and element.type == "text")
        or (element.role == "cta" and element.type == "shape" and element.subtype == "button")
    )


class CanvasCore:
    """Own one episode with strict requests, bounded attempts, and isolated snapshots.

    Call reset before using the core. apply_action accepts canonical JSON-compatible
    dictionaries, not Gymnasium values. This object is synchronous; adapters must
    serialize access when sharing an instance between callers.
    """

    def __init__(self) -> None:
        self._target: Target | TaskSpec | None = None
        self._elements: dict[int, Element] = {}
        self._steps_taken = 0
        self._next_element_id = 1
        self._status: Status = "active"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Validate first, then reset; omitted options always restore the default target.

        The seed is validated for interface parity but the core has no randomness.
        Initializing Gymnasium's own RNG belongs to its future adapter.
        """
        target, source = validate_reset(seed, options)
        self._target = target
        self._elements = {}
        self._steps_taken = 0
        self._next_element_id = 1
        self._status = "active"
        return self.get_canvas_state(), {"target_source": source}

    def _require_initialized(self) -> None:
        if self._target is None:
            raise LifecycleError("not_initialized")

    def get_canvas_state(self) -> dict[str, Any]:
        """Return a fresh canonical snapshot, including derived relationships."""
        self._require_initialized()
        assert self._target is not None
        return {
            "schema_version": 1,
            "canvas": {"width": 800, "height": 600, "background": "#FFFFFF"},
            "limits": config.model_dump(
                include={"max_elements", "max_steps", "max_content_length"}
            ),
            "target": self._target.model_dump(mode="json"),
            "steps_taken": self._steps_taken,
            "steps_remaining": config.max_steps - self._steps_taken,
            "next_element_id": self._next_element_id,
            "status": self._status,
            "elements": [
                self._elements[key].model_dump(mode="json") for key in sorted(self._elements)
            ],
            "relationships": spatial_relationships(self._elements.values()),
        }

    def serialize_state(self) -> str:
        """Return canonical ASCII JSON, stable across runs and Python hash seeds."""
        return json.dumps(
            self.get_canvas_state(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    def get_draw_order(self) -> list[dict[str, Any]]:
        """Return detached element records back to front, without rendering anything."""
        self._require_initialized()
        return [e.model_dump(mode="json") for e in stacking_order(self._elements.values())]

    def apply_action(self, payload: Any) -> CanvasTransition:
        """Apply one unscored attempt, preserving the canvas on semantic failure.

        Structural errors raise RequestError before lifecycle checks and consume
        no attempts. Semantic errors are returned and consume one attempt.
        """
        action = parse_action(payload)
        self._require_initialized()
        if self._status != "active":
            raise LifecycleError("episode_done")

        error: SemanticError | None = None
        candidate: Element | None = None
        if isinstance(action, (MoveAction, UpdateAction, DeleteAction)):
            if action.id not in self._elements:
                error = "unknown_element"
        if isinstance(action, AddAction):
            if len(self._elements) >= config.max_elements:
                error = "capacity_exceeded"
            else:
                candidate = Element.model_validate(
                    {**action.element.model_dump(), "id": self._next_element_id}
                )
        elif isinstance(action, UpdateAction) and error is None:
            existing = self._elements[action.id]
            candidate = Element.model_validate(
                {
                    **action.properties.model_dump(),
                    "id": existing.id,
                    "type": existing.type,
                    "subtype": existing.subtype,
                }
            )
        elif isinstance(action, MoveAction) and error is None:
            candidate = Element.model_validate(
                {
                    **self._elements[action.id].model_dump(),
                    "x": action.new_x,
                    "y": action.new_y,
                }
            )

        if error is None and candidate is not None and not _compatible_role(candidate):
            error = "incompatible_role"

        if error is None:
            if candidate is not None:
                self._elements[candidate.id] = candidate
                if isinstance(action, AddAction):
                    self._next_element_id += 1
            elif isinstance(action, DeleteAction):
                del self._elements[action.id]

        self._steps_taken += 1
        reason: EndReason | None = None
        if isinstance(action, FinishAction):
            self._status = "finished"
            reason = "finish"
        elif self._steps_taken == config.max_steps:
            self._status = "budget_exhausted"
            reason = "budget_exhausted"

        return CanvasTransition(
            state=self.get_canvas_state(),
            action_applied=error is None,
            error=error,
            end_reason=reason,
        )
