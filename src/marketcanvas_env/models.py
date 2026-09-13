"""Strict, immutable canonical JSON schemas and deterministic target resolution."""

from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from marketcanvas_env.config import config
from marketcanvas_env.reward.tasks import TaskSpec

DEFAULT_PROMPT = (
    "Create a Summer Sale email banner with a headline, a yellow CTA button, and good contrast"
)


class RequestError(ValueError):
    """A malformed canonical request or an unsupported mock prompt."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _uppercase(value: str) -> str:
    return value.upper()


Color = Annotated[
    str,
    Field(pattern=r"^#[0-9A-Fa-f]{6}$", min_length=7, max_length=7),
    AfterValidator(_uppercase),
]
Content = Annotated[str, Field(pattern=r"^[\x20-\x7e]*$", max_length=config.max_content_length)]
ElementId = Annotated[int, Field(ge=1, le=config.max_steps)]
XPosition = Annotated[int, Field(ge=-800, le=1599)]
YPosition = Annotated[int, Field(ge=-600, le=1199)]
Role = Annotated[str, Field(pattern=r"^[\x20-\x7e]+$", max_length=config.max_role_length)]
Alignment = Literal["left", "center", "right"]


class CanonicalModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
    )


class Properties(CanonicalModel):
    """The complete mutable property set; updates require every field."""

    role: Role
    x: XPosition
    y: YPosition
    width: Annotated[int, Field(ge=1, le=1600)]
    height: Annotated[int, Field(ge=1, le=1200)]
    z_index: Annotated[int, Field(ge=0, le=63)]
    color: Color
    text_color: Color
    content: Content
    font_size: Annotated[int, Field(ge=8, le=72)]
    text_align: Alignment


class NewElement(Properties):
    """Add payload, with defaults and structural kind validation only."""

    type: Literal["text", "shape", "image"]
    subtype: Literal["rectangle", "button"] | None = None
    role: Role = "none"
    x: XPosition = 0
    y: YPosition = 0
    width: Annotated[int, Field(ge=1, le=1600)] = 200
    height: Annotated[int, Field(ge=1, le=1200)] = 60
    z_index: Annotated[int, Field(ge=0, le=63)] = 0
    color: Color = "#FFFFFF"
    text_color: Color = "#000000"
    content: Content = ""
    font_size: Annotated[int, Field(ge=8, le=72)] = 24
    text_align: Alignment = "left"

    @model_validator(mode="before")
    @classmethod
    def default_shape_subtype(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("type") == "shape" and "subtype" not in value:
            return {**value, "subtype": "rectangle"}
        return value

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        if self.type == "shape":
            if self.subtype is None:
                raise ValueError("shape requires rectangle or button subtype")
        elif self.subtype is not None:
            raise ValueError("text and image require null subtype")
        return self


class Element(NewElement):
    """Stored element with a core-assigned, immutable identifier."""

    id: ElementId


class Target(CanonicalModel):
    headline: Annotated[Content, Field(min_length=1)]
    cta_text: Annotated[Content, Field(min_length=1)]
    cta_color: Color

    @field_validator("headline", "cta_text", mode="before")
    @classmethod
    def trim_ascii_spaces(cls, value: Any) -> Any:
        return value.strip(" ") if isinstance(value, str) else value


class AddAction(CanonicalModel):
    op: Literal["add_element"]
    element: NewElement


class MoveAction(CanonicalModel):
    op: Literal["move_element"]
    id: ElementId
    new_x: XPosition
    new_y: YPosition


class UpdateAction(CanonicalModel):
    op: Literal["update_element"]
    id: ElementId
    properties: Properties


class DeleteAction(CanonicalModel):
    op: Literal["delete_element"]
    id: ElementId


class FinishAction(CanonicalModel):
    op: Literal["finish"]


Action = Annotated[
    AddAction | MoveAction | UpdateAction | DeleteAction | FinishAction,
    Field(discriminator="op"),
]
ACTION_ADAPTER = TypeAdapter(Action)
_SEED_ADAPTER = TypeAdapter(Annotated[int, Field(strict=True, ge=0, le=4294967295)])


def parse_action(payload: Any) -> Action:
    """Validate canonical action data without applying semantic preconditions."""
    try:
        return ACTION_ADAPTER.validate_python(payload)
    except ValidationError as error:
        raise RequestError("invalid_request", str(error)) from error


def validate_seed(seed: Any) -> None:
    """Check the shared seed boundary without resolving a target."""
    try:
        if seed is not None:
            _SEED_ADAPTER.validate_python(seed)
    except ValidationError as error:
        raise RequestError("invalid_request", str(error)) from error


def validate_reset(seed: Any = None, options: Any = None) -> tuple[Target | TaskSpec, str]:
    """Resolve reset inputs without reading or modifying any episode state."""
    try:
        validate_seed(seed)
        if options is None:
            options = {}
        if not isinstance(options, dict) or options.keys() - {"prompt", "target"}:
            raise RequestError("invalid_request", "options must contain only prompt or target")
        if "prompt" in options and "target" in options:
            raise RequestError("invalid_request", "prompt and target are mutually exclusive")
        if "target" in options:
            target = options["target"]
            if isinstance(target, TaskSpec):
                target = target.model_dump(mode="json")
            if isinstance(target, dict) and ("constraints" in target or "prompt" in target):
                return TaskSpec.model_validate(target), "structured"
            return Target.model_validate(options["target"]), "structured"
        source = "default"
        if "prompt" in options:
            prompt = options["prompt"]
            if not isinstance(prompt, str) or not 1 <= len(prompt) <= 1024 or not prompt.strip():
                raise RequestError("invalid_request", "prompt must be a nonempty string up to 1024")
            if " ".join(prompt.split()).casefold() != DEFAULT_PROMPT.casefold():
                raise RequestError(
                    "unsupported_prompt", "use the supported mock prompt or a target"
                )
            source = "prompt"
        return Target(headline="Summer Sale", cta_text="Shop Now", cta_color="#FFFF00"), source
    except ValidationError as error:
        raise RequestError("invalid_request", str(error)) from error
