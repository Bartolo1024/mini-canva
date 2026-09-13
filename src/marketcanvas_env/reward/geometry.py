"""Safe floating-point geometry for untrusted evaluation snapshots."""

import math
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real

from marketcanvas_env.config import config


def finite_number(value: object) -> float | None:
    """Reject non-numbers, booleans, and huge values before arithmetic can overflow."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return (
        number
        if math.isfinite(number) and abs(number) <= config.reward.max_geometry_magnitude
        else None
    )


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2


def rectangle(data: dict) -> Rect | None:
    """Build (x, y, w, h) only for bounded finite values with w > 0, h > 0, w*h > 0."""
    values = [finite_number(data.get(k)) for k in ("x", "y", "width", "height")]
    if any(v is None for v in values):
        return None
    x, y, width, height = values
    if width <= 0 or height <= 0 or width * height <= 0:
        return None
    return Rect(x, y, width, height)


def intersection(a: Rect, b: Rect) -> Rect | None:
    """Intersect axes: [max(lefts), min(rights)] and [max(tops), min(bottoms)].

    Return None when either intersection span is nonpositive (including touching).
    """
    left, top = max(a.x, b.x), max(a.y, b.y)
    width, height = min(a.right, b.right) - left, min(a.bottom, b.bottom) - top
    return Rect(left, top, width, height) if width > 0 and height > 0 else None


def intersection_area(a: Rect, b: Rect) -> float:
    """A_intersection = max(0, min(rights)-max(lefts)) * max(0, min(bottoms)-max(tops))."""
    overlap = intersection(a, b)
    return overlap.area if overlap else 0.0


def contains(outer: Rect, inner: Rect) -> bool:
    """Containment requires both inner axis intervals to lie within the outer intervals."""
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and outer.right >= inner.right
        and outer.bottom >= inner.bottom
    )


def union_area(rectangles: list[Rect]) -> float:
    """A_union = sum_x (strip_width * union_length_of_y_intervals).

    Sweep sorted rectangle x boundaries and merge sorted y intervals within each
    strip. Overlapping occluders contribute area once, never once per rectangle.
    """
    xs = sorted({x for r in rectangles for x in (r.x, r.right)})
    area = 0.0
    for left, right in pairwise(xs):
        spans = sorted((r.y, r.bottom) for r in rectangles if r.x < right and r.right > left)
        height, end = 0.0, -math.inf
        for low, high in spans:
            height += max(0.0, high - max(low, end))
            end = max(end, high)
        area += (right - left) * height
    return area


def visible_ratio(
    rect: Rect | None, canvas: Rect | None, occluders: tuple[Rect, ...] = ()
) -> float:
    """v = clip((area(rect intersect canvas) - area(union(clipped_occluders))) / area(rect)).

    Clip every occluder to the visible rectangle before taking its union. Missing
    geometry or zero visible intersection returns 0. The output range is [0, 1].
    """
    if rect is None or canvas is None:
        return 0.0
    clipped = intersection(rect, canvas)
    if clipped is None:
        return 0.0
    covered = [overlap for other in occluders if (overlap := intersection(clipped, other))]
    return max(0.0, min(1.0, (clipped.area - union_area(covered)) / rect.area))
