"""On-demand raster output; rendering never changes canvas state or computes reward.

Scene supplies the same stacking, font, and text placement as semantic visibility.
Actual antialiased glyphs have holes; reward visibility approximates their ink box.
Rectangles use half-open bounds, rounding fractional snapshots out to touched pixels.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from marketcanvas_env.reward.colors import RGB
from marketcanvas_env.reward.geometry import Rect, intersection
from marketcanvas_env.reward.scene import (
    ElementView,
    Scene,
    get_font,
    text_content_box,
    text_ink_box,
)


def _rgb_bytes(color: RGB) -> tuple[int, int, int]:
    return tuple(round(channel * 255) for channel in color)


def _pixel_box(rect: Rect) -> tuple[int, int, int, int]:
    return (
        math.floor(rect.x),
        math.floor(rect.y),
        math.ceil(rect.right),
        math.ceil(rect.bottom),
    )


def _draw_text(image: Image.Image, element: ElementView, canvas: Rect) -> None:
    rect, ink = element.rect, text_ink_box(element)
    if rect is None or ink is None:
        return
    inset = text_content_box(rect)
    clip = intersection(inset, canvas)
    if clip is None or intersection(ink, clip) is None:
        return
    box = _pixel_box(clip)
    # Allocate only the visible inset, even for very large external snapshots.
    layer = image.crop(box)
    font = get_font(element.font_size)
    left, top, _, _ = font.getbbox(element.content, anchor="lt")
    ImageDraw.Draw(layer).text(
        (ink.x - left - box[0], ink.y - top - box[1]),
        element.content,
        font=font,
        fill=_rgb_bytes(element.text_color),
        anchor="lt",
    )
    image.paste(layer, box)


def _render_image(state: Any) -> Image.Image:
    scene = Scene(state)
    if (
        not scene.canvas_valid
        or not scene.structure_valid
        or any(not element.valid for element in scene.elements)
    ):
        raise ValueError(
            "Rendering requires a valid canvas and valid, uniquely identified elements"
        )
    if (scene.width, scene.height) != (800, 600):
        raise ValueError("Rendering currently supports only an 800×600 canvas")
    if any(element.text_bearing and "\n" in element.content for element in scene.elements):
        raise ValueError("Rendering supports single-line text only")
    image = Image.new("RGB", (800, 600), _rgb_bytes(scene.background))
    for element in sorted(scene.elements, key=lambda e: (e.z_index, e.stack_id)):
        clipped = intersection(element.rect, scene.canvas)
        if clipped is None:
            continue
        if element.type != "text":
            # Pillow's paste box has exclusive right/bottom, unlike rectangle().
            image.paste(_rgb_bytes(element.color), _pixel_box(clipped))
        if element.text_bearing:
            _draw_text(image, element, scene.canvas)
    return image


def render_rgb(state: Any) -> np.ndarray:
    """Return an independent uint8 (600, 800, 3) RGB array from a semantic snapshot."""
    return np.array(_render_image(state), dtype=np.uint8)


def save_png(state: Any, path: str | Path) -> None:
    """Save deterministic PNG bytes; the caller must create the parent directory."""
    _render_image(state).save(path, format="PNG")
