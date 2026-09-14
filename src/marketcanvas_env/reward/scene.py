"""Read-only normalized scene views; malformed elements remain visible to validity scoring."""

import math
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from PIL import ImageFont

from marketcanvas_env.config import config
from marketcanvas_env.reward.colors import RGB, contrast_ratio, parse_color
from marketcanvas_env.reward.geometry import (
    Rect,
    finite_number,
    intersection,
    rectangle,
    visible_ratio,
)


@lru_cache(maxsize=65)
def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Pinned embedded font shared by reward metrics and optional rendering."""
    return ImageFont.load_default(size=size)


@dataclass(frozen=True)
class ElementView:
    id: str
    type: str
    role: str
    content: str
    rect: Rect | None
    color: RGB | None
    text_color: RGB | None
    z_index: float
    stack_id: tuple[int, int | str]
    font_size: int
    text_align: str
    text_bearing: bool
    valid: bool
    visible_ratio: float = 0.0
    usable_ratio: float = 0.0


def _element(value: object, index: int) -> ElementView:
    """Normalize one record, retaining invalid records with ``valid=False``.

    Validity is the conjunction of ID, type, role, content, geometry, colors,
    stacking, and (for text-bearing elements) font validation. Safe defaults
    permit diagnostics to inspect malformed input without treating it as valid.
    """
    data = value if isinstance(value, dict) else {}
    identifier = data.get("id")
    id_valid = (isinstance(identifier, str) and bool(identifier)) or (
        type(identifier) is int and 0 < identifier <= config.reward.max_geometry_magnitude
    )
    kind = data.get("type", "")
    role = data.get("role", "none")
    content = data.get("content", "")
    content_valid = isinstance(content, str) and len(content) <= 4096
    kind_valid = isinstance(kind, str) and kind in ("text", "shape", "image")
    text_bearing = kind == "text" or (
        kind == "shape"
        and (
            data.get("subtype") == "button" or (isinstance(content, str) and bool(content.strip()))
        )
    )
    color, text_color = parse_color(data.get("color")), parse_color(data.get("text_color"))
    z = finite_number(data.get("z_index", 0))
    font_size = finite_number(data.get("font_size", 24))
    font_valid = font_size is not None and font_size.is_integer() and 8 <= font_size <= 72
    rect = rectangle(data)
    colors_valid = color is not None or (kind == "text" and data.get("color") is None)
    if text_bearing:
        colors_valid = colors_valid and text_color is not None
    valid = bool(
        id_valid
        and kind_valid
        and isinstance(role, str)
        and content_valid
        and rect is not None
        and colors_valid
        and z is not None
        and (not text_bearing or font_valid)
    )
    return ElementView(
        id=str(identifier) if id_valid else f"~invalid-{index}",
        type=kind if isinstance(kind, str) else "",
        role=role if isinstance(role, str) else "",
        content=content if content_valid else "",
        rect=rect,
        color=color,
        text_color=text_color,
        z_index=z if z is not None else 0,
        stack_id=(0, identifier)
        if type(identifier) is int and id_valid
        else (1, str(identifier) if id_valid else f"~invalid-{index}"),
        font_size=int(font_size) if font_valid else 24,
        text_align=data.get("text_align")
        if data.get("text_align") in ("left", "center", "right")
        else "left",
        text_bearing=text_bearing,
        valid=valid,
    )


def text_content_box(rect: Rect) -> Rect:
    """Return ``(x+p, y+p, w-2p, h-2p)`` for configured text inset ``p``.

    Scoring and rendering share this clipping rectangle.
    """
    padding = config.reward.text_inset
    return Rect(
        rect.x + padding,
        rect.y + padding,
        rect.width - 2 * padding,
        rect.height - 2 * padding,
    )


def text_ink_box(element: ElementView) -> Rect | None:
    """Position single-line font ink bounds using the element's alignment.

    For ink width ``w``, x is ``left+p``, ``center_x-w/2``, or ``right-p-w``;
    the renderer and scorer both use ``floor(x)`` and ``top+p``. This rectangle
    is not yet clipped to the element or canvas.
    """
    rect = element.rect
    if rect is None or not element.content.strip():
        return None
    padding = config.reward.text_inset
    if rect.width <= 2 * padding or rect.height <= 2 * padding:
        return None
    try:
        left, top, right, bottom = get_font(element.font_size).getbbox(element.content, anchor="lt")
    except (ValueError, OSError, UnicodeError):
        return None
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return None
    x = rect.x + padding
    if element.text_align == "center":
        x = rect.cx - width / 2
    elif element.text_align == "right":
        x = rect.right - padding - width
    return Rect(math.floor(x), rect.y + padding, width, height)


def _text_usable(element: ElementView, canvas: Rect, occluders: tuple[Rect, ...]) -> float:
    """Return ``area((ink ∩ inset ∩ canvas) \\ occluders) / area(ink)``.

    Missing ink or a font below the configured readable size receives zero.
    The geometry helper subtracts the union of occluders, avoiding double count.
    """
    rect = element.rect
    ink = text_ink_box(element)
    if rect is None or ink is None or element.font_size < config.reward.min_readable_font_size:
        return 0.0
    inset = text_content_box(rect)
    clip = intersection(inset, canvas)
    return visible_ratio(ink, clip, occluders)


def _normalize_elements(raw_elements: object) -> tuple[ElementView, ...]:
    """Normalize records in input order; a malformed collection becomes empty."""
    if not isinstance(raw_elements, (list, tuple)):
        return ()
    return tuple(_element(element, index) for index, element in enumerate(raw_elements))


def _invalidate_duplicate_ids(elements: tuple[ElementView, ...]) -> tuple[ElementView, ...]:
    """Set ``valid=False`` for every record whose normalized ID occurs twice."""
    ids = Counter(element.id for element in elements)
    return tuple(
        replace(element, valid=False) if ids[element.id] > 1 else element for element in elements
    )


def _painted_regions(elements: tuple[ElementView, ...]) -> tuple[tuple[ElementView, Rect], ...]:
    """Return opaque regions used by the visibility calculation.

    Valid solid non-text elements paint their rectangle. Transparent text
    paints only ``ink ∩ inset``; its stored fill color does not paint a box.
    Text ink is approximated by one rectangle, not individual glyph pixels.
    """
    painted = []
    for element in elements:
        if not element.valid or element.rect is None:
            continue
        paint = element.rect if element.color is not None and element.type != "text" else None
        if element.type == "text" and (ink := text_ink_box(element)) is not None:
            paint = intersection(ink, text_content_box(element.rect))
        if paint is not None:
            painted.append((element, paint))
    return tuple(painted)


def _visible_elements(
    elements: tuple[ElementView, ...],
    canvas: Rect | None,
    painted: tuple[tuple[ElementView, Rect], ...],
) -> tuple[ElementView, ...]:
    """Attach ``v = area((rect ∩ canvas) \\ higher_paint) / area(rect)``.

    Usability is ``u=v`` for non-text and ``u=min(v, text_usable)`` for text.
    Invalid elements receive zero. Higher paint is ordered by ``(z, stack_id)``;
    geometry subtracts its union, so overlapping occluders count only once.
    """
    views = []
    for element in elements:
        higher = tuple(
            paint
            for other, paint in painted
            if (other.z_index, other.stack_id) > (element.z_index, element.stack_id)
        )
        ratio = visible_ratio(element.rect, canvas, higher) if element.valid else 0.0
        usable = ratio
        if element.text_bearing and canvas is not None and element.valid:
            usable = min(ratio, _text_usable(element, canvas, higher))
        views.append(replace(element, visible_ratio=ratio, usable_ratio=usable))
    return tuple(views)


class Scene:
    """Normalize arbitrary JSON-like state without mutation or model calls."""

    def __init__(self, state: Any):
        """Read canvas → normalize elements → validate IDs → paint → visibility."""
        state = state if isinstance(state, dict) else {}
        canvas = state.get("canvas", {})
        canvas = canvas if isinstance(canvas, dict) else {}
        self.canvas = rectangle(
            {"x": 0, "y": 0, "width": canvas.get("width"), "height": canvas.get("height")}
        )
        self.width = self.canvas.width if self.canvas else 1.0
        self.height = self.canvas.height if self.canvas else 1.0
        self.background = parse_color(canvas.get("background_color", canvas.get("background")))
        raw_elements = state.get("elements", [])
        self.structure_valid = isinstance(raw_elements, (list, tuple))
        self.canvas_valid = self.canvas is not None and self.background is not None
        elements = _normalize_elements(raw_elements)
        elements = _invalidate_duplicate_ids(elements)
        painted = _painted_regions(elements)
        self.elements = _visible_elements(elements, self.canvas, painted)

    def select(self, selector: Any) -> list[ElementView]:
        """Filter exact selector fields, then order by ``(-visible_ratio, id)``."""
        fields = (
            selector.model_dump(exclude_none=True) if hasattr(selector, "model_dump") else selector
        )
        fields = fields or {}
        return sorted(
            (
                e
                for e in self.elements
                if all(
                    getattr(e, k, None) == (str(v) if k == "id" else v) for k, v in fields.items()
                )
            ),
            key=lambda e: (-e.visible_ratio, e.id),
        )

    def contrast(self, element: ElementView) -> float:
        """Return ``(max(L_text, L_bg)+0.05)/(min(L_text, L_bg)+0.05)``.

        Resolve the effective background first. Invalid, non-text-bearing, or
        blank elements receive zero; this is contrast, not text usability.
        """
        if not element.valid or not element.text_bearing or not element.content.strip():
            return 0.0
        background = self.effective_background(element)
        return contrast_ratio(element.text_color, background)

    def effective_background(self, element: ElementView) -> RGB | None:
        """Return ``argmin_bg contrast(text, bg)`` over exposed ink backgrounds.

        For clipped ink I, lower solid layer j contributes iff
        ``area((I ∩ rect_j) \\ higher_lower_layers) > 0``. Walk front to back;
        include canvas color only where no layer covers I. Controls use their
        own fill. Every positive-area patch counts; ink remains a bounding-box
        approximation, not a glyph mask.
        """
        if element.type != "text":
            return element.color
        ink = text_ink_box(element)
        if ink is None or element.rect is None or self.canvas is None:
            return self.background
        ink = intersection(ink, text_content_box(element.rect))
        ink = intersection(ink, self.canvas) if ink is not None else None
        if ink is None:
            return self.background
        lower = [
            other
            for other in self.elements
            if other.valid
            and other.rect
            and other.color is not None
            and other.type != "text"
            and (other.z_index, other.stack_id) < (element.z_index, element.stack_id)
            and intersection(other.rect, ink) is not None
        ]
        backgrounds = []
        covered = ()
        for other in sorted(lower, key=lambda other: (other.z_index, other.stack_id), reverse=True):
            if visible_ratio(ink, other.rect, covered) > 0:
                backgrounds.append(other.color)
            covered += (other.rect,)
        if visible_ratio(ink, self.canvas, covered) > 0:
            backgrounds.append(self.background)
        return min(
            backgrounds,
            key=lambda color: contrast_ratio(element.text_color, color),
            default=self.background,
        )
