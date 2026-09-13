"""Immutable benchmark authoring models; prompts are descriptive data only."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_serializer, model_validator

from marketcanvas_env.config import config

_COMMON_FIELDS = {"id", "kind", "hard"}
_KIND_FIELDS = {
    "exists": {"selector", "min_visible_ratio", "min_area_ratio"},
    "absent": {"selector"},
    "count": {"selector", "value", "operator"},
    "text_contains": {"selector", "value"},
    "text_equals": {"selector", "value"},
    "color_family": {"selector", "value"},
    "color_exact": {"selector", "value"},
    "region": {"selector", "value"},
    "relative_position": {"subject", "object", "relation"},
    "alignment": {"selector", "selectors", "value", "reference"},
    "size_relation": {"subject", "object", "metric", "operator"},
    "contrast_min": {"selector", "selectors", "value"},
    "no_overlap": {"selector", "selectors"},
}


class Selector(BaseModel):
    """All supplied fields match; omitted fields are wildcards."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str | None = None
    role: str | None = None
    type: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else value


class Constraint(BaseModel):
    """A validated deterministic constraint, independent of any particular prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    kind: Literal[
        "exists",
        "absent",
        "count",
        "text_contains",
        "text_equals",
        "color_family",
        "color_exact",
        "region",
        "relative_position",
        "alignment",
        "size_relation",
        "contrast_min",
        "no_overlap",
    ]
    hard: bool = False
    selector: Selector | None = None
    selectors: tuple[Selector, ...] = ()
    subject: Selector | None = None
    object: Selector | None = None
    value: str | float | int | None = None
    operator: Literal["eq", "lte", "gte", "gt", "lt"] | None = None
    relation: Literal["above", "below", "left_of", "right_of"] | None = None
    metric: Literal["width", "height", "area"] = "area"
    reference: Literal["canvas"] | None = None
    min_visible_ratio: float = config.reward.min_visible_ratio
    min_area_ratio: float = config.reward.min_area_ratio

    @model_validator(mode="before")
    @classmethod
    def validate_author_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        kind = data.get("kind")
        if not isinstance(kind, str) or kind not in _KIND_FIELDS:
            return data  # Pydantic reports invalid/unknown kinds.
        numeric_fields = {"min_visible_ratio", "min_area_ratio"}
        if kind in {"count", "contrast_min"}:
            numeric_fields.add("value")
        for field in numeric_fields:
            if isinstance(data.get(field), bool):
                raise ValueError(f"{field} must be numeric, not boolean")
        allowed = _COMMON_FIELDS | _KIND_FIELDS[kind]
        for field, value in data.items():
            if field not in allowed and value is not None and value != [] and value != ():
                raise ValueError(f"{field} is not applicable to {kind}")
        if data.get("selector") is not None and data.get("selectors"):
            raise ValueError("selector and selectors are mutually exclusive")
        if kind == "alignment" and data.get("selectors") and data.get("reference") is not None:
            raise ValueError("canvas reference only applies to the single-selector alignment form")
        return data

    @model_serializer(mode="wrap")
    def serialize_applicable_fields(self, handler: Any) -> dict[str, Any]:
        """Emit only meaningful fields so serialized tasks round-trip strictly."""
        allowed = _COMMON_FIELDS | _KIND_FIELDS[self.kind]
        return {key: value for key, value in handler(self).items() if key in allowed}

    @model_validator(mode="after")
    def validate_contract(self) -> "Constraint":
        if not self.id.strip():
            raise ValueError("constraint id must be nonempty")
        if not math.isfinite(self.min_visible_ratio) or not 0 < self.min_visible_ratio <= 1:
            raise ValueError("min_visible_ratio must be in (0, 1]")
        if not math.isfinite(self.min_area_ratio) or not 0 < self.min_area_ratio <= 1:
            raise ValueError("min_area_ratio must be in (0, 1]")
        if self.kind in {"relative_position", "size_relation"}:
            if self.subject is None or self.object is None:
                raise ValueError("pair constraints require subject and object")
        elif self.kind == "alignment":
            if not (
                len(self.selectors) >= 2
                or (self.selector is not None and self.reference == "canvas")
            ):
                raise ValueError("alignment requires selectors or selector and canvas reference")
        elif self.kind in {"contrast_min", "no_overlap"}:
            if self.selector is None and not self.selectors:
                raise ValueError("constraint requires selector or selectors")
        elif self.selector is None:
            raise ValueError("constraint requires selector")
        enums = {
            "color_family": {"red", "orange", "yellow", "green", "blue", "purple", "neutral"},
            "region": {
                "left",
                "center",
                "right",
                "top",
                "bottom",
                "top-left",
                "top-right",
                "bottom-left",
                "bottom-right",
            },
            "alignment": {"horizontal_centers", "vertical_centers", "left_edges", "right_edges"},
        }
        if self.kind in enums and self.value not in enums[self.kind]:
            raise ValueError(f"unsupported {self.kind} value")
        if self.kind in {"text_contains", "text_equals", "color_exact"}:
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError("text/color value must be a nonempty string")
            if self.kind == "color_exact":
                from marketcanvas_env.reward.colors import parse_color

                if parse_color(self.value) is None or not self.value.startswith("#"):
                    raise ValueError("color_exact requires a valid hex color")
        if self.kind == "count":
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(self.value)
                or self.value < 0
                or int(self.value) != self.value
            ):
                raise ValueError("count value must be a nonnegative integer")
            if self.operator not in {"eq", "lte", "gte"}:
                raise ValueError("count requires eq, lte, or gte operator")
        if self.kind == "contrast_min":
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(self.value)
                or not 1 <= self.value <= 21
            ):
                raise ValueError("contrast_min value must be between 1 and 21")
        if self.kind == "relative_position" and self.relation is None:
            raise ValueError("relative_position requires relation")
        if self.kind == "size_relation" and self.operator not in {None, "gt", "lt", "gte", "lte"}:
            raise ValueError("size_relation requires gt, lt, gte, or lte")
        return self


class TaskSpec(BaseModel):
    """A benchmark prompt and its complete machine-readable scoring contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    prompt: str
    constraints: tuple[Constraint, ...] = ()

    @model_validator(mode="after")
    def unique_ids(self) -> "TaskSpec":
        ids = [constraint.id for constraint in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("constraint ids must be unique")
        requirements = [
            constraint.model_dump_json(exclude={"id", "hard"}) for constraint in self.constraints
        ]
        if len(requirements) != len(set(requirements)):
            raise ValueError("duplicate requirements cannot reweight task satisfaction")
        return self
