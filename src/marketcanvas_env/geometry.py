"""Integer rectangle relationships, independent of rendering and scoring."""

from collections.abc import Iterable

from marketcanvas_env.models import Element

RELATION_NAMES = (
    "overlaps",
    "contains",
    "left_of",
    "above",
    "center_aligned_x",
    "center_aligned_y",
)
ALIGNMENT_TOLERANCE = 2


def relationship_names(a: Element, b: Element) -> tuple[str, ...]:
    """Return true predicates in contract order using raw, half-open rectangles."""
    right_a, bottom_a = a.x + a.width, a.y + a.height
    right_b, bottom_b = b.x + b.width, b.y + b.height
    predicates = (
        min(right_a, right_b) > max(a.x, b.x) and min(bottom_a, bottom_b) > max(a.y, b.y),
        a.x <= b.x and a.y <= b.y and right_a >= right_b and bottom_a >= bottom_b,
        right_a <= b.x,
        bottom_a <= b.y,
        abs((2 * a.x + a.width) - (2 * b.x + b.width)) <= 2 * ALIGNMENT_TOLERANCE,
        abs((2 * a.y + a.height) - (2 * b.y + b.height)) <= 2 * ALIGNMENT_TOLERANCE,
    )
    return tuple(name for name, matches in zip(RELATION_NAMES, predicates, strict=True) if matches)


def spatial_relationships(elements: Iterable[Element]) -> list[dict[str, int | str]]:
    """Build deterministic directed relationships, excluding each element's self-pair."""
    ordered = sorted(elements, key=lambda element: element.id)
    return [
        {"source_id": a.id, "target_id": b.id, "relation": name}
        for a in ordered
        for b in ordered
        if a.id != b.id
        for name in relationship_names(a, b)
    ]


def stacking_order(elements: Iterable[Element]) -> tuple[Element, ...]:
    """Return elements back to front, breaking equal z-index ties by stable ID."""
    return tuple(sorted(elements, key=lambda element: (element.z_index, element.id)))
